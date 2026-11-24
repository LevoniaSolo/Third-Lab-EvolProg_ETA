import argparse
import json
from pathlib import Path

from experiments import run_experiment


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="LR3, variant 17: NSGA-II feature selection")
    parser.add_argument("--config", type=Path, default=directory / "config.json")
    parser.add_argument("--runs", type=int, help="Override the number of independent runs")
    parser.add_argument("--output", type=Path, default=directory / "results")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.runs is not None:
        config["search"]["runs"] = args.runs
    run_experiment(config, args.output)
    print(f"Results saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
