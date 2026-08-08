"""Write a markdown row artifact for E5 bootstrap results."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--method", default="TenSetMLP")
    parser.add_argument("--out", required=True)
    parser.add_argument("--paper-ready", action="store_true")
    args = parser.parse_args()

    with open(args.metrics, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(args.coverage, "r", encoding="utf-8") as f:
        coverage = json.load(f)

    paper_ready = "true" if args.paper_ready else "false"
    notes = (
        f"bootstrap artifact; paper_ready={paper_ready}; "
        f"expanded_logs={coverage.get('expanded_log_count')}; "
        f"retained_tasks={coverage.get('retained_task_count')}; "
        f"dropped_tasks={coverage.get('dropped_task_count_min_sample_size')}; "
        "metric family is sample-weighted peak_score, not Top-k accuracy"
    )

    row = (
        "| Method | Platform | Peak Score@1 | Peak Score@5 | Pairwise Accuracy | Notes |\n"
        "| --- | --- | ---: | ---: | ---: | --- |\n"
        f"| {args.method} | {args.platform} | "
        f"{metrics.get('average peak score@1', float('nan')):.4f} | "
        f"{metrics.get('average peak score@5', float('nan')):.4f} | "
        f"{metrics.get('pairwise comparision accuracy', float('nan')):.4f} | "
        f"{notes} |\n"
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(row)


if __name__ == "__main__":
    main()
