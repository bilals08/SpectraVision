# H5 path and glob resolution.

from glob import glob
from pathlib import Path


def resolve_h5_files(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            matches = sorted(path.rglob("*.h5")) + sorted(path.rglob("*.hdf5"))
        else:
            matches = [Path(found) for found in sorted(glob(str(path)))]
        files.extend(match for match in matches if match.is_file() and match.suffix in {".h5", ".hdf5"})
    files = list(dict.fromkeys(files))
    if not files:
        raise FileNotFoundError(f"No HDF5 files matched: {paths}")
    return files
