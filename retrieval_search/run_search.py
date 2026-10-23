# Peptide retrieval from MGF and FASTA inputs.

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from proteomicsML.base.constants import PROTON_MASS, WATER_MASS, aa_mono_masses  # noqa: E402
from proteomicsML.base.sequence import SequenceProcessor  # noqa: E402
from src.configs import DEFAULT_WEIGHT_REPO, ensure_weight, get_device, load_profiles, weight_path  # noqa: E402
from src.evaluate.evaluate import load_model_from_weights  # noqa: E402
from proteomicsML.base.mgf import lines, mgf_records  # noqa: E402

TEXT = h5py.string_dtype("utf-8")
PEAK_LIMIT = 200


def local_source(value: str, downloads: Path) -> Path:
    if not value.startswith("https://"):
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    name = Path(urlsplit(value).path).name
    if not name:
        raise ValueError(f"URL has no filename: {value}")
    downloads.mkdir(parents=True, exist_ok=True)
    path = downloads / name
    if not path.exists():
        temporary = path.with_name(path.name + ".part")
        with urlopen(value, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(path)
    return path


def mgf_to_h5(source: Path, target: Path) -> int:
    with h5py.File(target, "w") as handle:
        peak_ds = handle.create_dataset("peaks", (0, PEAK_LIMIT, 2), maxshape=(None, PEAK_LIMIT, 2),
                                        chunks=(128, PEAK_LIMIT, 2), dtype="f4")
        mz_ds = handle.create_dataset("precursor_mz", (0,), maxshape=(None,), chunks=True, dtype="f4")
        charge_ds = handle.create_dataset("charge", (0,), maxshape=(None,), chunks=True, dtype="i2")
        id_ds = handle.create_dataset("spec_id", (0,), maxshape=(None,), chunks=True, dtype=TEXT)
        batch = []

        def flush() -> None:
            if not batch:
                return
            start, stop = len(mz_ds), len(mz_ds) + len(batch)
            for dataset in (peak_ds, mz_ds, charge_ds, id_ds):
                dataset.resize(stop, axis=0)
            peak_ds[start:stop] = np.stack([row[0] for row in batch])
            mz_ds[start:stop] = [row[1] for row in batch]
            charge_ds[start:stop] = [row[2] for row in batch]
            id_ds[start:stop] = [row[3] for row in batch]
            batch.clear()

        for index, (meta, peaks) in enumerate(mgf_records(source)):
            precursor_fields = meta.get("PEPMASS", "").split()
            precursor = float(precursor_fields[0]) if precursor_fields else float("nan")
            charge_text = meta.get("CHARGE", "")
            charge_match = re.search(r"(\d+)\s*\+", charge_text)
            if not charge_match or not np.isfinite(precursor) or precursor <= 0:
                raise ValueError(f"Spectrum {index}: invalid PEPMASS or CHARGE")
            charge = int(charge_match.group(1))
            if charge < 1 or charge > 32767:
                raise ValueError(f"Spectrum {index}: invalid charge {charge}")
            selected = np.asarray(peaks, dtype=np.float32).reshape(-1, 2)
            if len(selected) > PEAK_LIMIT:
                selected = selected[np.argpartition(selected[:, 1], -PEAK_LIMIT)[-PEAK_LIMIT:]]
            selected = selected[np.argsort(selected[:, 0])]
            padded = np.zeros((PEAK_LIMIT, 2), dtype=np.float32)
            padded[: len(selected)] = selected
            batch.append((padded, precursor, charge, meta.get("TITLE") or meta.get("SCANS") or str(index)))
            if len(batch) == 512:
                flush()
        flush()
        if len(mz_ds) == 0:
            raise ValueError(f"No spectra found in {source}")
        return len(mz_ds)


def fasta_records(path: Path):
    header, parts = None, []
    for line in lines(path):
        line = line.strip()
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(parts)
            header, parts = line[1:], []
        elif line:
            if header is None:
                raise ValueError("FASTA sequence before first header")
            parts.append(line)
    if header is not None:
        yield header, "".join(parts)


def fasta_to_h5(source: Path, target: Path) -> int:
    peptides = {}
    for protein, sequence in fasta_records(source):
        clean, modifications = SequenceProcessor.parse_modifications(sequence)
        if not clean or any(residue not in aa_mono_masses for residue in clean):
            raise ValueError(f"Invalid peptide in FASTA: {sequence!r}")
        # Neutral peptide mass includes water.
        mass = sum(aa_mono_masses[residue] for residue in clean) + WATER_MASS
        mass += sum(shift for _, _, shift in modifications)
        peptides.setdefault(sequence, (protein, mass))
    if not peptides:
        raise ValueError(f"No peptides found in {source}")
    with h5py.File(target, "w") as handle:
        handle.create_dataset("sequence", data=list(peptides), dtype=TEXT)
        handle.create_dataset("protein", data=[item[0] for item in peptides.values()], dtype=TEXT)
        handle.create_dataset("neutral_mass", data=[item[1] for item in peptides.values()], dtype="f8")
    return len(peptides)


@torch.inference_mode()
def embed(images, model, device, batch_size: int = 64):
    vectors = []
    for start in range(0, len(images), batch_size):
        tensor = torch.as_tensor(np.stack(images[start:start + batch_size]), device=device).float()
        vectors.append(F.normalize(model(tensor).float(), dim=1).cpu())
    return torch.cat(vectors)


def search(observed: Path, database: Path, output: Path, model_name: str, batch_size: int,
           device_name: str, ppm_tolerance: float = 20.0):
    profiles = load_profiles()
    device = get_device(device_name)
    profile = profiles["models"][model_name]
    encoding, model = load_model_from_weights(model_name, profile, weight_path(profile), device)
    with h5py.File(observed) as spectra, h5py.File(database) as candidates:
        sequences = candidates["sequence"].asstr()[:]
        proteins = candidates["protein"].asstr()[:]
        neutral_masses = candidates["neutral_mass"][:]
        charges = spectra["charge"][:]
        precursor_mz = spectra["precursor_mz"][:]
        spec_ids = spectra["spec_id"].asstr()[:]
        best_score = np.full(len(charges), np.nan, dtype=np.float32)
        best_index = np.full(len(charges), -1, dtype=np.int32)

        for charge in np.unique(charges):
            query_indices = np.flatnonzero(charges == charge)
            peptide_mz = (neutral_masses + charge * PROTON_MASS) / charge
            order = np.argsort(peptide_mz)
            sorted_mz = peptide_mz[order]
            fraction = ppm_tolerance * 1e-6
            for start in range(0, len(query_indices), batch_size):
                rows = query_indices[start:start + batch_size]
                mz = precursor_mz[rows].astype(np.float64)
                # Bounds use theoretical m/z as the ppm-error denominator.
                left = np.searchsorted(sorted_mz, mz / (1 + fraction), side="left")
                right = np.searchsorted(sorted_mz, mz / (1 - fraction), side="right")
                eligible = [order[lo:hi] for lo, hi in zip(left, right)]
                active = [i for i, ids in enumerate(eligible) if len(ids)]
                if not active:
                    continue
                needed = np.unique(np.concatenate([eligible[i] for i in active]))
                peptide_batches = []
                for offset in range(0, len(needed), batch_size):
                    images = [encoding.spectrum_encoding_theoretical(sequences[int(i)],
                              meta={"charge": int(charge)})[0] for i in needed[offset:offset + batch_size]]
                    peptide_batches.append(embed(images, model, device, batch_size))
                peptide_vectors = torch.cat(peptide_batches).to(device)
                images = [encoding.spectrum_encoding(spectra["peaks"][int(rows[i])]) for i in active]
                query_vectors = embed(images, model, device, batch_size).to(device)
                for query_position, batch_position in enumerate(active):
                    ids = eligible[batch_position]
                    offsets = torch.as_tensor(np.searchsorted(needed, ids), device=device)
                    scores = peptide_vectors[offsets] @ query_vectors[query_position]
                    winner = int(scores.argmax())
                    row = rows[batch_position]
                    best_index[row] = int(ids[winner])
                    best_score[row] = float(scores[winner])
            scored = np.count_nonzero(best_index[query_indices] >= 0)
            print(f"charge {charge}: scored {scored:,}/{len(query_indices):,} spectra within "
                  f"{ppm_tolerance:g} ppm", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["spectrum_index", "spec_id", "charge", "precursor_mz", "best_peptide",
                         "protein", "score", "peptide_precursor_mz", "precursor_error_ppm"])
        for index, candidate in enumerate(best_index):
            # Spectra without a mass-eligible peptide have no assignment row.
            if candidate < 0:
                continue
            charge = int(charges[index])
            peptide_mz = (neutral_masses[candidate] + charge * PROTON_MASS) / charge
            error_ppm = (float(precursor_mz[index]) - peptide_mz) / peptide_mz * 1e6
            writer.writerow([index, spec_ids[index], charge, float(precursor_mz[index]),
                             sequences[candidate], proteins[candidate], float(best_score[index]),
                             peptide_mz, error_ppm])
    return int(np.count_nonzero(best_index >= 0))


def main() -> None:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Search MGF spectra against peptide FASTA candidates.")
    parser.add_argument("--mgf", default=str(folder / "sample_data/sample_data.mgf"))
    parser.add_argument("--fasta", default=str(folder / "sample_data/sample_fasta.fasta"))
    parser.add_argument("--model", choices=tuple(load_profiles()["models"]), default="SpectraVision-3C-CNN")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ppm-tolerance", type=float, default=20.0,
                        help="Maximum absolute precursor mass error in ppm (default: 20)")
    parser.add_argument("--output-dir", type=Path, default=folder / "output")
    parser.add_argument("--download_weight", "--download-weight", choices=("true", "false"), default="false")
    parser.add_argument("--weight-repo", default=DEFAULT_WEIGHT_REPO)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not np.isfinite(args.ppm_tolerance) or not 0 < args.ppm_tolerance < 1_000_000:
        parser.error("--ppm-tolerance must be between 0 and 1,000,000")
    if args.download_weight == "true":
        ensure_weight(args.model, args.weight_repo)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mgf = local_source(args.mgf, args.output_dir / "downloads")
    fasta = local_source(args.fasta, args.output_dir / "downloads")
    observed = args.output_dir / "observed.h5"
    database = args.output_dir / "peptides.h5"
    count = mgf_to_h5(mgf, observed)
    peptide_count = fasta_to_h5(fasta, database)
    print(f"Converted {count:,} spectra and {peptide_count:,} unique peptides", flush=True)
    output = args.output_dir / "matches.csv"
    matches = search(observed, database, output, args.model, args.batch_size, args.device, args.ppm_tolerance)
    print(f"Saved peptide assignments to {output}: {matches:,}")


if __name__ == "__main__":
    main()
