"""Evaluate an MLP checkpoint on a dataset split and save metrics JSON."""

import argparse
import json
import pickle

from common import load_and_register_tasks
from train_model import evaluate_model
from tvm.auto_scheduler.cost_model.mlp_model import MLPModelInternal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-scheme", choices=["within_task", "by_task", "by_target"], default="within_task")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    load_and_register_tasks()
    dataset = pickle.load(open(args.dataset, "rb"))
    if args.split_scheme == "within_task":
        _train, test = dataset.random_split_within_task(args.train_ratio)
    elif args.split_scheme == "by_task":
        _train, test = dataset.random_split_by_task(args.train_ratio)
    else:
        _train, test = dataset.random_split_by_target(args.train_ratio)

    model = MLPModelInternal(device=args.device)
    model.load(args.checkpoint)
    metrics = evaluate_model(model, test if len(test) > 0 else dataset)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
