"""Summarize auto-scheduler measure records for a task index range."""

import argparse
import math
from collections import Counter

from tvm import auto_scheduler

from common import load_and_register_tasks, MEASURE_RECORD_FOLDER, clean_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--start-idx", type=int, required=True)
    parser.add_argument("--end-idx", type=int, required=True)
    args = parser.parse_args()

    tasks = load_and_register_tasks()
    end_idx = min(args.end_idx, len(tasks))

    error_counter = Counter()
    ok_costs = []
    total_records = 0
    missing_files = []

    for i in range(args.start_idx, end_idx):
        task = tasks[i]
        task_key = (task.workload_key, str(task.target.kind))
        filename = f"{MEASURE_RECORD_FOLDER}/{args.model}/{clean_name(task_key)}.json"
        try:
            inputs, results = auto_scheduler.RecordReader(filename).read_lines()
        except Exception:
            missing_files.append((i, filename))
            continue

        _ = inputs
        for res in results:
            total_records += 1
            error_counter[int(res.error_no)] += 1
            if int(res.error_no) == 0 and len(res.costs) > 0:
                v = float(res.costs[0].value)
                if math.isfinite(v):
                    ok_costs.append(v)

    print(f"model={args.model}")
    print(f"task_range=[{args.start_idx}, {end_idx})")
    print(f"total_records={total_records}")
    print(f"error_hist={dict(sorted(error_counter.items()))}")
    if ok_costs:
        ok_costs.sort()
        n = len(ok_costs)
        p50 = ok_costs[n // 2]
        p90 = ok_costs[min(n - 1, int(n * 0.9))]
        mean = sum(ok_costs) / n
        print(f"ok_count={n}")
        print(f"ok_latency_mean_s={mean:.8f}")
        print(f"ok_latency_p50_s={p50:.8f}")
        print(f"ok_latency_p90_s={p90:.8f}")
    else:
        print("ok_count=0")

    if missing_files:
        print(f"missing_files={len(missing_files)}")
        for idx, filename in missing_files[:5]:
            print(f"missing_example task_idx={idx} file={filename}")


if __name__ == "__main__":
    main()
