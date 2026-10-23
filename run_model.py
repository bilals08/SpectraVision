# Configured model training and inference.

import argparse
import json
from pathlib import Path

from src.configs import DEFAULT_WEIGHT_REPO, ensure_weight
from src.configs.model_configs import MODELS
from src.evaluate import evaluate_models
from src.trainers.train import read_train_config, train


ROOT = Path(__file__).resolve().parent


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description="Run a configured SpectraVision model for training or inference.")
    parser.add_argument("command", choices=("train", "infer"))
    parser.add_argument("model", choices=tuple(MODELS))
    parser.add_argument("--data", help="Override the configured training or test H5 path")
    parser.add_argument("--config", default=str(ROOT / "configs/run.json"))
    parser.add_argument("--download_weight", "--download-weight", choices=("true", "false"), default="false")
    parser.add_argument("--weight-repo", default=DEFAULT_WEIGHT_REPO)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        settings = json.load(stream)

    if args.command == "train":
        if args.download_weight == "true":
            parser.error("--download_weight applies to inference only")
        config = read_train_config(project_path(settings["training_configs"][args.model]))
        if config["model"] != args.model:
            raise ValueError(f"Training config model does not match {args.model}")
        config["train_data"] = [str(project_path(args.data or settings["train_data"]))]
        validation = settings.get("validation_data")
        config["validation_data"] = [str(project_path(validation))] if validation else None
        config["output_dir"] = str(project_path(config["output_dir"]))
        train(config)
    else:
        if args.download_weight == "true":
            ensure_weight(args.model, args.weight_repo)
        data = project_path(args.data or settings["test_data"])
        results = evaluate_models(
            [str(data)], model_names=[args.model],
            output=project_path(settings["inference_output"]),
        )
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
