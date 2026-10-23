# Convert published PyTorch state dictionaries to safetensors.

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    weights_dir = Path(__file__).resolve().parents[1] / "weights"
    for source in sorted(weights_dir.glob("SpectraVision-*.pt")):
        state = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError(f"Expected a model state dictionary in {source}")
        target = source.with_suffix(".safetensors")
        save_file({key: value.contiguous() for key, value in state.items()}, target)
        restored = load_file(str(target))
        if state.keys() != restored.keys() or any(not torch.equal(value, restored[key]) for key, value in state.items()):
            raise ValueError(f"Conversion verification failed for {target}")
        print(f"Verified {target.name} ({len(state)} tensors)")


if __name__ == "__main__":
    main()
