# Stream MGF spectra, including gzip files.

import gzip
from pathlib import Path

import numpy as np


def lines(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        yield from stream


def mgf_records(path: Path):
    metadata, peaks = None, []
    for line in lines(path):
        line = line.strip()
        if line == "BEGIN IONS":
            metadata, peaks = {}, []
        elif line == "END IONS":
            if metadata is not None:
                yield metadata, peaks
            metadata, peaks = None, []
        elif metadata is not None and line:
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key.strip().upper()] = value.strip()
            else:
                fields = line.split()
                if len(fields) >= 2:
                    mz, intensity = float(fields[0]), float(fields[1])
                    if np.isfinite(mz) and np.isfinite(intensity) and mz > 0 and intensity > 0:
                        peaks.append((mz, intensity))
