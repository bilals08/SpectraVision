# H5 spectra, theoretical candidates, and decoys.
from typing import List, Tuple
import math
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

class HDF5PeaksDataset(Dataset):
    # Read labeled H5 spectra and generate paired candidate images.

    def __init__(self, SpectrumEncodingObject, file_path: str):
        self.file_path = file_path
        self.h5_file = None
        self.SpectrumEncodingObject = SpectrumEncodingObject

        with h5py.File(file_path, 'r') as f:
            missing = {'sequence', 'peaks'} - set(f)
            if missing:
                raise ValueError(f"{file_path}: missing HDF5 datasets {sorted(missing)}")
            self.length = f["sequence"].shape[0]
            if f['peaks'].shape[0] != self.length:
                raise ValueError(f"{file_path}: peaks and sequence lengths differ")

    def __len__(self) -> int:
        return self.length

    def _open_file(self):
        # Open H5 lazily for worker-safe dataset access.
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, "r")

    def _to_tensor(self, arr) -> torch.Tensor:
        if isinstance(arr, dict):
            return {key: self._to_tensor(value) for key, value in arr.items()}
        tensor = torch.from_numpy(arr)
        # Keep integer token IDs intact; model images use float32.
        if not torch.is_floating_point(tensor) and tensor.dtype not in [torch.int8, torch.int16, torch.int32, torch.int64]:
            return tensor.long()
        elif torch.is_floating_point(tensor):
            return tensor.float()
        else:
            return tensor

    def __getitem__(self, idx: int) -> Tuple:
        # Load experimental, theoretical, and decoy images with metadata.
        self._open_file()

        seq = self.h5_file["sequence"][idx]
        if isinstance(seq, bytes):
            seq = seq.decode()

        peaks = self.h5_file["peaks"][idx]
        meta = {'sequence': str(seq), 'spec_id': f'{Path(self.file_path).name}:{idx}'}
        if "charge" in self.h5_file:
            meta["charge"] = int(self.h5_file["charge"][idx])
        # Older H5 files name precursor m/z "precursor_mass".
        precursor_key = 'precursor_mz' if 'precursor_mz' in self.h5_file else 'precursor_mass' if 'precursor_mass' in self.h5_file else None
        if precursor_key:
            meta["precursor_mz"] = float(self.h5_file[precursor_key][idx])
        if "protein" in self.h5_file:
            protein = self.h5_file["protein"][idx]
            meta['protein'] = protein.decode() if isinstance(protein, bytes) else str(protein)
        if "spec_id" in self.h5_file:
            spec_id = self.h5_file["spec_id"][idx]
            meta['spec_id'] = spec_id.decode() if isinstance(spec_id, bytes) else str(spec_id)


        if getattr(self.SpectrumEncodingObject, "requires_precursor_metadata", False):
            if "precursor_mz" not in meta:
                raise ValueError(
                    "precursor encoding requires HDF5 precursor_mass (m/z)"
                )
            precursor_mz = meta["precursor_mz"]
            if not math.isfinite(precursor_mz) or precursor_mz <= 0:
                raise ValueError("invalid precursor m/z in HDF5 sample")
            exp_encoding = self.SpectrumEncodingObject.spectrum_encoding(peaks, meta=meta)
        else:
            exp_encoding = self.SpectrumEncodingObject.spectrum_encoding(peaks)
        theo_encoding, theo_meta = self.SpectrumEncodingObject.spectrum_encoding_theoretical(seq, meta=meta)

        decoy_encoding , decoy_meta = self.SpectrumEncodingObject.spectrum_encoding_decoy(seq, meta=meta)

        return (
            self._to_tensor(exp_encoding),
            self._to_tensor(theo_encoding),
            self._to_tensor(decoy_encoding),
            meta,
            theo_meta,
            decoy_meta,

        )

    def __del__(self):
        # Close an open H5 file when the dataset is discarded.
        if getattr(self, "h5_file", None) is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state['h5_file'] = None
        return state


class MultiHDF5Dataset(Dataset):
    # Combine multiple H5 datasets.

    def __init__(self, SpectrumEncodingObject, file_paths: List[str]):
        self.datasets = []
        self.cumulative = []
        total = 0

        for path in file_paths:
            dataset = HDF5PeaksDataset(SpectrumEncodingObject, path)
            n = len(dataset)
            self.datasets.append(dataset)
            total += n
            self.cumulative.append(total)

        if total == 0:
            raise ValueError("No spectra found in HDF5 files.")

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        # Translate a global index to its source H5 file.
        for i, cumulative_count in enumerate(self.cumulative):
            if idx < cumulative_count:
                local_idx = idx if i == 0 else idx - self.cumulative[i - 1]
                return self.datasets[i][local_idx]
        raise IndexError("Index out of range")
