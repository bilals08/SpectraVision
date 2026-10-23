# Model profiles and Hugging Face weights.

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from proteomicsML.encodings import SpectraVision1C, SpectraVision3C
from src.models import SpectraVision1CCNN, SpectraVision3CCNN, SpectraVisionPeakViT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES = PROJECT_ROOT / "configs" / "models.json"
DEFAULT_WEIGHT_REPO = "SaeedLab/SpectraVision"

ENCODINGS = {
    "SpectraVision-1C": SpectraVision1C,
    "SpectraVision-3C": SpectraVision3C,
}
MODELS = {
    "SpectraVision-1C-CNN": SpectraVision1CCNN,
    "SpectraVision-3C-CNN": SpectraVision3CCNN,
    "SpectraVision-PeakViT": SpectraVisionPeakViT,
}
MODEL_ENCODINGS = {
    "SpectraVision-1C-CNN": "SpectraVision-1C",
    "SpectraVision-3C-CNN": "SpectraVision-3C",
    "SpectraVision-PeakViT": "SpectraVision-1C",
}


def load_profiles(path=None):
    # Load the model, image encoding, weight, and evaluation settings.
    path = Path(path) if path else DEFAULT_PROFILES
    with path.open(encoding="utf-8") as stream:
        profiles = json.load(stream)
    if not isinstance(profiles, dict) or not isinstance(profiles.get("models"), dict):
        raise ValueError(f"{path}: expected a models mapping")
    for name, profile in profiles["models"].items():
        if name not in MODELS:
            raise ValueError(f"Unknown model in {path}: {name}")
        if profile.get("encoding") != MODEL_ENCODINGS[name]:
            raise ValueError(f"{name} requires {MODEL_ENCODINGS[name]}")
        if not profile.get("weights"):
            raise ValueError(f"{name} has no weights path")
    return profiles


def build_components(model_name, profile):
    # Build the matching encoder and model, validating image dimensions.
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS)}")
    encoding_name = profile.get("encoding")
    if encoding_name != MODEL_ENCODINGS[model_name]:
        raise ValueError(f"{model_name} requires {MODEL_ENCODINGS[model_name]}")
    encoding = ENCODINGS[encoding_name](**profile.get("encoding_params", {}))
    params = dict(profile.get("model_params", {}))
    image_size = (encoding.img_size, encoding.img_size)
    requested = params.get("img_size")
    if requested is not None:
        requested = (requested, requested) if isinstance(requested, int) else tuple(requested)
        if requested != image_size:
            raise ValueError(f"{model_name}: image size {requested} does not match encoder {image_size}")
    params["img_size"] = image_size
    if model_name == "SpectraVision-PeakViT":
        params.setdefault("in_channels", 1)
        params.setdefault("patch_size", 16)
        params.setdefault("num_layers", 2)
        params.setdefault("nhead", 8)
        params.setdefault("intensity_channel", 0)
        params.setdefault("min_mz_wavelength", 0.2)
        params.setdefault("max_mz_wavelength", 1000)
        params.setdefault("max_mz", encoding.max_mz)
        params.setdefault("bin_multiply_factor", encoding.bin_multiply_factor)
    return encoding, MODELS[model_name](**params)


def weight_path(profile, root=PROJECT_ROOT, weights_dir=None):
    # Resolve the profile's exact weight filename.
    declared = Path(profile["weights"])
    if weights_dir:
        return Path(weights_dir) / declared.name
    return declared if declared.is_absolute() else Path(root) / declared


def ensure_weight(model_name, repo_id=DEFAULT_WEIGHT_REPO, weights_dir=None):
    # Download a missing bundled checkpoint with the Hugging Face Hub API.
    profile = load_profiles()["models"][model_name]
    target = weight_path(profile, weights_dir=weights_dir)
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(repo_id=repo_id, filename=target.name, local_dir=target.parent)
    if not target.is_file():
        raise FileNotFoundError(f"Downloaded checkpoint was not found: {target}")
    return target


def get_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device
