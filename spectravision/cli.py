# SpectraVision command-line interface.

import argparse
import json

from src.configs import DEFAULT_WEIGHT_REPO, ensure_weight
from src.configs.model_configs import MODELS
from src.evaluate import evaluate_models
from src.trainers.train import read_train_config, train


def main(argv=None):
    parser = argparse.ArgumentParser(prog="spectravision")
    commands = parser.add_subparsers(dest="command", required=True)

    training = commands.add_parser("train", help="Train a model from a JSON config")
    training.add_argument("--config", required=True, help="Training JSON config")
    training.add_argument("--profiles", help="Model and loss profiles; defaults to configs/models.json")

    assessment = commands.add_parser("evaluate", help="Evaluate bundled weights against HDF5 spectra")
    assessment.add_argument("--data", nargs="+", required=True, help="H5 files, directories, or glob patterns")
    assessment.add_argument("--model", choices=["all", *MODELS], default="all")
    assessment.add_argument("--profiles", help="Model and weight profiles; defaults to configs/models.json")
    assessment.add_argument("--weights-dir", help="Override the directory holding the named weight files")
    assessment.add_argument("--output", help="Predictions CSV for one model or directory for all models")
    assessment.add_argument("--batch-size", type=int)
    assessment.add_argument("--num-workers", type=int)
    assessment.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    assessment.add_argument("--download_weight", "--download-weight", choices=("true", "false"), default="false")
    assessment.add_argument("--weight-repo", default=DEFAULT_WEIGHT_REPO)

    args = parser.parse_args(argv)
    if args.command == "train":
        train(read_train_config(args.config), args.profiles)
    else:
        names = None if args.model == "all" else [args.model]
        if args.download_weight == "true":
            for name in names or MODELS:
                ensure_weight(name, args.weight_repo, args.weights_dir)
        results = evaluate_models(
            data=args.data,
            model_names=names,
            config_path=args.profiles,
            weights_dir=args.weights_dir,
            output=args.output,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
