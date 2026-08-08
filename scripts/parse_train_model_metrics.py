"""Parse printed metrics from scripts/train_model.py output into JSON."""

import argparse
import json
import re


PATTERN = re.compile(r"^(RMSE|R\^2|pairwise comparision accuracy|mape|average peak score@1|average peak score@5):\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    metrics = {}
    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = PATTERN.match(line.strip())
            if m:
                metrics[m.group(1)] = float(m.group(2))

    if not metrics:
        raise RuntimeError("No metrics found in train_model output.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
