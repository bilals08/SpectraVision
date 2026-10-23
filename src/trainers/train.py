# H5 triplet training and checkpoint saving.

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.configs import build_components, get_device, load_profiles
from src.datasets import MultiHDF5Dataset, resolve_h5_files
from src.evaluate import evaluate_model
from src.losses import multi_positive_info_nce_with_decoy


def read_train_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict) or not config.get("model"):
        raise ValueError("Training config needs a model name")
    return config


def make_loader(paths, encoding, batch_size, num_workers, shuffle):
    files = resolve_h5_files(paths)
    dataset = MultiHDF5Dataset(encoding, files)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


def _embeddings(model, batch, device):
    images = [image.to(device, non_blocking=True).float() for image in batch[:3]]
    batch_size = images[0].shape[0]
    embeddings = model(torch.cat(images, dim=0))
    return embeddings.split(batch_size, dim=0)


def train(config, profiles_path=None):
    # Train the model and save best/last checkpoints.
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    name = config["model"]
    profile = {
        "encoding": config["encoding"],
        "encoding_params": config.get("encoding_params", {}),
        "model_params": config.get("model_params", {}),
    }
    encoding, model = build_components(name, profile)
    device = get_device(config.get("device", "auto"))
    model.to(device)
    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 0))
    epochs = int(config.get("epochs", 10))
    if batch_size <= 0 or num_workers < 0 or epochs <= 0:
        raise ValueError("batch_size and epochs must be positive; num_workers cannot be negative")
    train_loader = make_loader(config["train_data"], encoding, batch_size, num_workers, True)
    validation_loader = (
        make_loader(config["validation_data"], encoding, batch_size, num_workers, False)
        if config.get("validation_data") else None
    )
    defaults = load_profiles(profiles_path).get("loss", {})
    if defaults.get("name") != "multi_positive_info_nce_with_decoy":
        raise ValueError("Unsupported loss in model profiles")
    loss_params = {key: defaults[key] for key in ("temperature", "threshold", "decoy_weight") if key in defaults}
    loss_params.update(config.get("loss_params", {}))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-4)))
    output_dir = Path(config.get("output_dir", f"outputs/{name}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            anchor, positive, decoy = _embeddings(model, batch, device)
            sequences = batch[3]["sequence"]
            positive_mask = torch.tensor(
                [[left == right for right in sequences] for left in sequences],
                dtype=torch.bool,
                device=device,
            )
            loss, _ = multi_positive_info_nce_with_decoy(
                anchor.float(), positive.float(), decoy.float(),
                pos_mask=positive_mask, **loss_params,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(sequences)
            total_samples += len(sequences)

        metrics = {"epoch": epoch, "train_loss": total_loss / total_samples}
        if validation_loader is not None:
            metrics["validation"] = evaluate_model(model, validation_loader, device)
        checkpoint = {"model": model.state_dict(), "config": config, "epoch": epoch}
        torch.save(checkpoint, output_dir / "last.pt")
        score = metrics.get("validation", {}).get("top1_accuracy", -metrics["train_loss"])
        if score > best_score:
            best_score = score
            torch.save(checkpoint, output_dir / "best.pt")
        print(json.dumps(metrics), flush=True)
    return metrics
