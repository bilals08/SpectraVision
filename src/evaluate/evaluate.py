# Batch peptide retrieval evaluation.

import csv
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.configs import build_components, get_device, load_profiles, weight_path
from src.datasets import MultiHDF5Dataset, resolve_h5_files


def load_model_from_weights(model_name, profile, path, device):
    # Create the configured model.
    encoding, model = build_components(model_name, profile)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Weights for {model_name} were not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return encoding, model


def _prediction_file(output, model_name, multiple):
    if output is None:
        return None
    path = Path(output)
    if multiple or path.suffix.lower() != ".csv":
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{model_name}_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def rank_unique_peptides(similarities, candidate_sequences, topk=3):
    #Return the best-scoring row for each of the highest-ranked peptides.


    peptides = sorted(set(candidate_sequences))
    peptide_to_group = {peptide: index for index, peptide in enumerate(peptides)}
    batch_size, candidate_count = similarities.shape
    groups = torch.tensor(
        [peptide_to_group[sequence] for sequence in candidate_sequences],
        device=similarities.device, dtype=torch.long,
    ).unsqueeze(0).expand(batch_size, -1)

    scores = similarities.new_full((batch_size, len(peptides)), float("-inf"))
    scores.scatter_reduce_(1, groups, similarities, reduce="amax", include_self=True)

    # The row index is retained only for the prediction CSV, never for accuracy.
    rows = torch.arange(candidate_count, device=similarities.device).expand(batch_size, -1)
    best_rows = torch.where(similarities == scores.gather(1, groups), rows, candidate_count)
    representatives = torch.full(
        (batch_size, len(peptides)), candidate_count, device=similarities.device, dtype=torch.long,
    )
    representatives.scatter_reduce_(1, groups, best_rows, reduce="amin", include_self=True)

    ranked_groups = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :topk]
    return (
        ranked_groups,
        representatives.gather(1, ranked_groups),
        scores.gather(1, ranked_groups),
        peptides,
    )


@torch.inference_mode()
def evaluate_model(model, loader, device, predictions_path=None, forward_batch_size=None, cuda_autocast=False):
    # Rank theoretical and decoy candidates.
    device = torch.device(device)
    if forward_batch_size is not None and forward_batch_size <= 0:
        raise ValueError("forward_batch_size must be positive")
    model.eval()
    top1 = top3 = samples = 0
    output_file = None
    writer = None
    if predictions_path is not None:
        path = Path(predictions_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        output_file = path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            output_file,
            fieldnames=["spec_id", "sequence", "predicted_sequence", "candidate_type", "candidate_index", "score", "correct"],
        )
        writer.writeheader()
    try:
        for batch in loader:
            experimental, theoretical, decoy, meta, theoretical_meta, decoy_meta = batch
            embeddings = []
            for image in (experimental, theoretical, decoy):
                chunks = []
                for image_chunk in image.split(forward_batch_size or len(image)):
                    image_chunk = image_chunk.to(device, non_blocking=True).float()
                    autocast_context = (
                        torch.autocast(device_type="cuda")
                        if cuda_autocast and device.type == "cuda" else nullcontext()
                    )
                    with autocast_context:
                        chunks.append(model(image_chunk).float())
                embeddings.append(torch.cat(chunks, dim=0))
            anchors, positives, negatives = [F.normalize(embedding, dim=1) for embedding in embeddings]
            similarities = torch.cat((anchors @ positives.T, anchors @ negatives.T), dim=1)
            batch_size = anchors.shape[0]
            query_sequences = meta["sequence"]
            target_sequences = theoretical_meta["sequence"]
            decoy_sequences = decoy_meta["sequence"]
            candidate_sequences = list(target_sequences) + list(decoy_sequences)
            ranked_groups, ranked_indices, ranked_scores, peptides = rank_unique_peptides(
                similarities, candidate_sequences
            )
            group_rows = ranked_groups.cpu().tolist()
            index_rows = ranked_indices.cpu().tolist()
            score_rows = ranked_scores.cpu().tolist()
            spec_ids = meta["spec_id"]

            for row, candidates in enumerate(group_rows):
                first = index_rows[row][0]
                predicted = peptides[candidates[0]]
                hit = predicted == query_sequences[row]
                top1 += int(hit)
                top3 += int(any(peptides[group] == query_sequences[row] for group in candidates))
                samples += 1
                if writer is not None:
                    writer.writerow({
                        "spec_id": spec_ids[row],
                        "sequence": query_sequences[row],
                        "predicted_sequence": predicted,
                        "candidate_type": "theoretical" if first < batch_size else "decoy",
                        "candidate_index": first,
                        "score": score_rows[row][0],
                        "correct": hit,
                    })
    finally:
        if output_file is not None:
            output_file.close()
    if samples == 0:
        raise ValueError("Evaluation data contained no spectra")
    return {
        "samples": samples,
        "top1_accuracy": top1 / samples,
        "top3_accuracy": top3 / samples,
        "batch_size": loader.batch_size,
    }


def evaluate_models(
    data,
    model_names=None,
    config_path=None,
    weights_dir=None,
    output=None,
    batch_size=None,
    num_workers=None,
    device=None,
):
    # Evaluate one or all named models against the same HDF5 data.
    profiles = load_profiles(config_path)
    selected = list(profiles["models"]) if not model_names or model_names == ["all"] else list(model_names)
    unknown = set(selected) - set(profiles["models"])
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    settings = profiles.get("evaluation", {})
    resolved_device = get_device(device or settings.get("device", "auto"))
    resolved_files = resolve_h5_files(data)
    batch_size = int(batch_size or settings.get("batch_size", 32))
    num_workers = int(num_workers if num_workers is not None else settings.get("num_workers", 0))
    forward_batch_size = int(settings.get("forward_batch_size", batch_size))
    cuda_autocast = bool(settings.get("cuda_autocast", False))
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if forward_batch_size <= 0:
        raise ValueError("forward_batch_size must be positive")
    torch.backends.cudnn.enabled = bool(settings.get("cudnn_enabled", True))
    torch.backends.cudnn.benchmark = bool(settings.get("cudnn_benchmark", False))
    torch.backends.cudnn.deterministic = bool(settings.get("cudnn_deterministic", False))
    results = {}
    for name in selected:
        profile = profiles["models"][name]
        path = weight_path(profile, weights_dir=weights_dir)
        encoding, model = load_model_from_weights(name, profile, path, resolved_device)
        dataset = MultiHDF5Dataset(encoding, resolved_files)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        prediction_file = _prediction_file(output, name, len(selected) > 1)
        metrics = evaluate_model(
            model, loader, resolved_device, prediction_file,
            forward_batch_size=forward_batch_size, cuda_autocast=cuda_autocast,
        )
        metrics.update({
            "weights": str(path),
            "data_files": [str(file) for file in resolved_files],
            "device": str(resolved_device),
            "num_workers": num_workers,
            "forward_batch_size": forward_batch_size,
            "cuda_autocast": cuda_autocast and resolved_device.type == "cuda",
        })
        if prediction_file is not None:
            metrics["predictions"] = str(prediction_file)
        results[name] = metrics
    return results
