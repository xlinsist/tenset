"""Build a TVM auto-scheduler dataset from measure logs in parallel.

This is a thin parallel wrapper around `make_dataset_from_log_file` semantics:
- reuse per-file `.dataset_cache/*.feature_cache` when present
- compute missing per-file caches concurrently
- merge all task data into one Dataset
- drop tasks with too few samples
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import pickle
import traceback

from tqdm import tqdm

from common import load_and_register_tasks
from tvm.auto_scheduler.dataset import Dataset, input_to_learning_task
from tvm.auto_scheduler.feature import get_per_store_features_from_measure_pairs
from tvm.auto_scheduler.measure_record import RecordReader


def process_one(args):
    filename, cache_folder = args
    cache_file = f"{cache_folder}/{filename.replace('/', '_')}.feature_cache"
    try:
        if os.path.exists(cache_file):
            features, throughputs, min_latency = pickle.load(open(cache_file, "rb"))
            return {
                "filename": filename,
                "features": features,
                "throughputs": throughputs,
                "min_latency": min_latency,
                "error": None,
            }

        measure_records = {}
        for inp, res in RecordReader(filename):
            task = input_to_learning_task(inp)
            if task not in measure_records:
                measure_records[task] = [[], []]
            measure_records[task][0].append(inp)
            measure_records[task][1].append(res)

        features = {}
        throughputs = {}
        min_latency = {}
        for task, (inputs, results) in measure_records.items():
            features_, normalized_throughputs, task_ids, min_latency_ = (
                get_per_store_features_from_measure_pairs(inputs, results)
            )
            assert not task_ids.any()
            if len(min_latency_) == 0:
                continue
            assert len(min_latency_) == 1
            features[task] = features_
            throughputs[task] = normalized_throughputs
            min_latency[task] = min_latency_[0]

        pickle.dump((features, throughputs, min_latency), open(cache_file, "wb"))
        return {
            "filename": filename,
            "features": features,
            "throughputs": throughputs,
            "min_latency": min_latency,
            "error": None,
        }
    except Exception as err:
        return {
            "filename": filename,
            "features": {},
            "throughputs": {},
            "min_latency": {},
            "error": {
                "type": type(err).__name__,
                "message": str(err),
                "traceback": traceback.format_exc(),
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--coverage-out")
    parser.add_argument("--bad-files-out")
    parser.add_argument("--min-sample-size", type=int, default=48)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--cache-folder", default=".dataset_cache")
    args = parser.parse_args()

    log_files = []
    for pattern in args.logs:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            log_files.extend(expanded)
        elif os.path.exists(pattern):
            log_files.append(pattern)

    if not log_files:
        raise RuntimeError("No log files found after expansion.")

    os.makedirs(args.cache_folder, exist_ok=True)
    load_and_register_tasks()

    dataset = Dataset()
    dataset.raw_files = log_files
    bad_files = []

    work_items = [(filename, args.cache_folder) for filename in log_files]
    with mp.get_context("fork").Pool(processes=args.jobs) as pool:
        for item in tqdm(
            pool.imap_unordered(process_one, work_items),
            total=len(work_items),
        ):
            if item["error"] is not None:
                bad_files.append(
                    {
                        "filename": item["filename"],
                        **item["error"],
                    }
                )
                continue
            for task in item["features"]:
                dataset.load_task_data(
                    task,
                    item["features"][task],
                    item["throughputs"][task],
                    item["min_latency"][task],
                )

    before_task_count = len(dataset.features)
    dropped = []
    for task, feature in list(dataset.features.items()):
        if len(feature) < args.min_sample_size:
            dropped.append(task)
            del dataset.features[task]
            del dataset.throughputs[task]
            del dataset.min_latency[task]

    pickle.dump(dataset, open(args.out_file, "wb"))

    if args.coverage_out:
        coverage = {
            "expanded_log_count": len(log_files),
            "dataset_raw_files_count": len(dataset.raw_files or []),
            "retained_task_count": len(dataset.features),
            "dropped_task_count_min_sample_size": len(dropped),
            "bad_file_count": len(bad_files),
            "min_sample_size": args.min_sample_size,
            "jobs": args.jobs,
        }
        with open(args.coverage_out, "w", encoding="utf-8") as f:
            json.dump(coverage, f, indent=2, ensure_ascii=False)

    if args.bad_files_out:
        with open(args.bad_files_out, "w", encoding="utf-8") as f:
            json.dump(bad_files, f, indent=2, ensure_ascii=False)

    print(f"A dataset file is saved to {args.out_file}")
    print(f"Expanded logs: {len(log_files)}")
    print(f"Retained tasks: {len(dataset.features)}")
    print(f"Dropped tasks: {len(dropped)}")
    print(f"Bad files: {len(bad_files)}")


if __name__ == "__main__":
    main()
