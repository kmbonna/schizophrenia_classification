"""Command-line interface.

    schizo-eeg evaluate --config configs/logreg_bandpower.yaml
    schizo-eeg train    --config configs/logreg_bandpower.yaml --out artifacts/logreg_bandpower
    schizo-eeg predict  --model artifacts/logreg_bandpower recording.edf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import ExperimentConfig


def _format_metrics(m: dict) -> str:
    lo, hi = m["accuracy_95ci"]
    return (
        f"accuracy {m['accuracy']:.3f} (95% CI {lo:.2f}-{hi:.2f}), "
        f"sensitivity {m['sensitivity']:.3f}, specificity {m['specificity']:.3f}, "
        f"AUC {m['roc_auc']:.3f}, n={m['n']}"
    )


def cmd_evaluate(args) -> int:
    from .training import evaluate, load_dataset, write_report

    config = ExperimentConfig.from_yaml(args.config)
    result = evaluate(config, load_dataset(config, args.data))
    print(f"[{config.name}] {config.evaluation.cv}")
    print("  subject level:", _format_metrics(result.subject_metrics))
    print("  window level: ", _format_metrics(result.window_metrics))
    report = write_report(result, config, Path(args.reports) / f"{config.name}.json")
    print(f"  report: {report}")
    return 0


def cmd_train(args) -> int:
    from .training import evaluate, load_dataset, train

    config = ExperimentConfig.from_yaml(args.config)
    dataset = load_dataset(config, args.data)
    cv_result = None
    if args.evaluate:
        cv_result = evaluate(config, dataset)
        print("  subject level:", _format_metrics(cv_result.subject_metrics))
    out = Path(args.out or Path("artifacts") / config.name)
    train(config, dataset, out, cv_result)
    print(f"Saved model to {out}")
    return 0


def cmd_predict(args) -> int:
    from .pipeline import SchizophreniaClassifier

    classifier = SchizophreniaClassifier.load(args.model)
    results = []
    for path in args.recordings:
        prediction = classifier.predict_file(path)
        record = {"recording": str(path), **prediction.to_dict()}
        if not args.windows:
            record.pop("window_probabilities")
        results.append(record)
        if not args.json:
            print(
                f"{path}: {prediction.label_name} "
                f"(p={prediction.probability:.3f}, {prediction.n_windows} windows)"
            )
    if args.json:
        print(json.dumps(results, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="schizo-eeg", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    sub = parser.add_subparsers(dest="command", required=True)

    data_help = "folder with h*.edf / s*.edf (default: $DATASET_PATH from .env)"

    p = sub.add_parser("evaluate", help="cross-validate a config, scored per subject")
    p.add_argument("--config", required=True)
    p.add_argument("--data", help=data_help)
    p.add_argument("--reports", default="reports", help="where to write the JSON report")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("train", help="fit on all subjects and save a model artifact")
    p.add_argument("--config", required=True)
    p.add_argument("--data", help=data_help)
    p.add_argument("--out", help="output folder (default: artifacts/<config name>)")
    p.add_argument("--evaluate", action="store_true",
                   help="also run cross-validation and store the scores in the artifact")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="classify one or more EDF recordings")
    p.add_argument("--model", required=True, help="artifact folder created by 'train'")
    p.add_argument("recordings", nargs="+")
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    p.add_argument("--windows", action="store_true", help="include per-window probabilities")
    p.set_defaults(func=cmd_predict)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
