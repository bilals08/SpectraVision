# Labeled MGF to H5 preprocessing.

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proteomicsML.base.constants import aa_mono_masses, mod_dict  # noqa: E402
from proteomicsML.base.sequence import SequenceProcessor  # noqa: E402
from proteomicsML.base.mgf import mgf_records  # noqa: E402

PEAK_LIMIT = 200
TEXT = h5py.string_dtype("utf-8")
MASS_CODES = {float(mass): code for mass, code in mod_dict.items()}
NAME_CODES = {
    "carbamidomethyl": "c", "carbamidomethylation": "c",
    "oxidation": "o", "carbamyl": "k", "carbamylation": "k",
    "deamidated": "d", "deamidation": "d",
    "acetyl": "a", "acetylation": "a",
    "pyroglu": "q", "pyroglutamate": "q", "glnpyroglu": "q",
    "phospho": "p", "phosphorylation": "p",
}
MASS = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
SIGNED_MASS = re.compile(r"[+-](?:\d+(?:\.\d*)?|\.\d+)")


def mod_code(label: str) -> str:
    label = label.strip()
    if MASS.fullmatch(label):
        value = float(label)
        nearest = min(MASS_CODES, key=lambda mass: abs(value - mass))
        if abs(value - nearest) <= 0.01:
            return MASS_CODES[nearest]
    else:
        code = NAME_CODES.get(re.sub(r"[\s_-]", "", label.lower()))
        if code:
            return code
    raise ValueError(f"Unknown modification {label!r}")


def sequence_codes(sequence: str) -> str:
    # Convert inline modification names or masses to lowercase codes.
    sequence = sequence.strip()
    nterm, body = [], []
    residues = 0
    pos = 0
    while pos < len(sequence):
        char = sequence[pos]
        if char in "[(":
            closing = "]" if char == "[" else ")"
            end = sequence.find(closing, pos + 1)
            if end < 0:
                raise ValueError(f"Unclosed modification in {sequence!r}")
            code = mod_code(sequence[pos + 1:end])
            if residues == 0:
                if end + 1 >= len(sequence) or sequence[end + 1] != "-":
                    raise ValueError("N-terminal bracket modification must end with '-'")
                nterm.append(code)
                pos = end + 2
                continue
            if code == "q":
                if residues != 1 or body[0] != "Q":
                    raise ValueError("Pyroglutamate requires an N-terminal Q")
                # Store pyroglutamate before Q even when annotated after it.
                nterm.append(code)
            else:
                body.append(code)
            pos = end + 1
            continue
        mass = SIGNED_MASS.match(sequence, pos) if residues else None
        if mass:
            code = mod_code(mass.group())
            if code == "q":
                if residues != 1 or body[0] != "Q":
                    raise ValueError("Pyroglutamate requires an N-terminal Q")
                nterm.append(code)
            else:
                body.append(code)
            pos = mass.end()
            continue
        if "A" <= char <= "Z":
            body.append(char)
            residues += 1
        elif "a" <= char <= "z":
            if char not in MASS_CODES.values():
                raise ValueError(f"Unknown modification code {char!r}")
            if residues == 0:
                nterm.append(char)
            elif char == "q":
                if residues != 1 or body[0] != "Q":
                    raise ValueError("Pyroglutamate requires an N-terminal Q")
                nterm.append(char)
            else:
                body.append(char)
        else:
            raise ValueError(f"Unsupported sequence character {char!r} in {sequence!r}")
        pos += 1
    result = "".join(nterm + body)
    clean, _ = SequenceProcessor.parse_modifications(result)
    if not clean or any(residue not in aa_mono_masses for residue in clean):
        raise ValueError(f"Invalid peptide sequence {sequence!r}")
    if "q" in nterm and clean[0] != "Q":
        raise ValueError("Pyroglutamate requires an N-terminal Q")
    return result


def convert(source: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".h5", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with h5py.File(temporary, "w") as handle:
            datasets = {
                "peaks": handle.create_dataset("peaks", (0, PEAK_LIMIT, 2), maxshape=(None, PEAK_LIMIT, 2),
                                               chunks=(128, PEAK_LIMIT, 2), dtype="f4"),
                "sequence": handle.create_dataset("sequence", (0,), maxshape=(None,), chunks=True, dtype=TEXT),
                "charge": handle.create_dataset("charge", (0,), maxshape=(None,), chunks=True, dtype="i2"),
                "precursor_mass": handle.create_dataset("precursor_mass", (0,), maxshape=(None,), chunks=True, dtype="f4"),
                "protein": handle.create_dataset("protein", (0,), maxshape=(None,), chunks=True, dtype=TEXT),
                "spec_id": handle.create_dataset("spec_id", (0,), maxshape=(None,), chunks=True, dtype=TEXT),
            }
            batch = {key: [] for key in datasets}

            def flush():
                if not batch["sequence"]:
                    return
                stop = len(datasets["sequence"]) + len(batch["sequence"])
                for key, dataset in datasets.items():
                    dataset.resize(stop, axis=0)
                    dataset[stop - len(batch[key]):stop] = batch[key]
                    batch[key].clear()

            for index, (meta, peaks) in enumerate(mgf_records(source)):
                try:
                    if any(meta.get(key) for key in ("MODS", "IT_MODS", "FIXED_MODS")):
                        raise ValueError("separate MODS fields are unsupported; annotate SEQ inline")
                    raw = meta.get("SEQ") or meta.get("PEPTIDE")
                    if not raw:
                        raise ValueError("missing SEQ or PEPTIDE")
                    if meta.get("SEQ") and meta.get("PEPTIDE") and meta["SEQ"] != meta["PEPTIDE"]:
                        raise ValueError("SEQ and PEPTIDE disagree")
                    sequence = sequence_codes(raw)
                    precursor = float(meta.get("PEPMASS", "").split()[0])
                    charge_match = re.fullmatch(r"(\d+)\+", meta.get("CHARGE", "").strip())
                    if not charge_match or not np.isfinite(precursor) or precursor <= 0:
                        raise ValueError("invalid PEPMASS or CHARGE")
                    charge = int(charge_match.group(1))
                    if charge < 1 or charge > 32767:
                        raise ValueError("charge is out of range")
                    selected = np.asarray(peaks, dtype=np.float32).reshape(-1, 2)
                    if not len(selected):
                        raise ValueError("no positive peaks")
                    if len(selected) > PEAK_LIMIT:
                        selected = selected[np.argpartition(selected[:, 1], -PEAK_LIMIT)[-PEAK_LIMIT:]]
                    selected = selected[np.argsort(selected[:, 0])]
                    padded = np.zeros((PEAK_LIMIT, 2), dtype=np.float32)
                    padded[:len(selected)] = selected
                except (ValueError, IndexError) as error:
                    raise ValueError(f"Spectrum {index} ({meta.get('TITLE', 'untitled')}): {error}") from error
                batch["peaks"].append(padded)
                batch["sequence"].append(sequence)
                batch["charge"].append(charge)
                batch["precursor_mass"].append(precursor)
                batch["protein"].append(meta.get("PROTEIN", ""))
                batch["spec_id"].append(meta.get("TITLE") or meta.get("SCANS") or str(index))
                if len(batch["sequence"]) == 512:
                    flush()
            flush()
            count = len(datasets["sequence"])
            if not count:
                raise ValueError(f"No spectra found in {source}")
        os.replace(temporary, target)
        return count
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Convert labeled MGF spectra to SpectraVision H5.")
    parser.add_argument("mgf", type=Path, help="Input MGF or .mgf.gz with SEQ/PEPTIDE on every spectrum")
    parser.add_argument("h5", type=Path, help="Output H5 file")
    args = parser.parse_args()
    if not args.mgf.is_file():
        parser.error(f"Input file does not exist: {args.mgf}")
    if args.mgf.resolve() == args.h5.resolve():
        parser.error("Input and output paths must differ")
    print(f"Wrote {convert(args.mgf, args.h5):,} spectra to {args.h5}")


if __name__ == "__main__":
    main()
