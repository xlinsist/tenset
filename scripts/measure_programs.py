"""Measure all programs

Usage:
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=e5-2666"
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=e5-2673"
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=epyc-7452"
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=epyc-7r32"
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=i7-8750h"
python3 measure_programs.py --target "llvm -mcpu=skylake-avx512 -model=platinum-8124m"
python3 measure_programs.py --target "llvm -mcpu=skylake-avx512 -model=platinum-8272l"
python3 measure_programs.py --target "llvm -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod -model=graviton2"
python3 measure_programs.py --target "llvm -mtriple=aarch64-linux-gnu -mattr=+neon -model=a72" --other-args "--rpc-device-key rasp4b-64 --rpc-host kraken --rpc-port 9191 --rpc-n-parallel 4"
"""

import argparse
import glob
import os
import pickle
import time

from tqdm import tqdm

import tvm
from tvm import auto_scheduler

from common import (
    load_and_register_tasks,
    get_measure_record_filename,
    get_to_measure_filename,
)


def make_measurer(
    build_timeout,
    build_n_parallel,
    run_timeout,
    repeat,
    number,
    enable_cpu_cache_flush,
    verbose,
    log_filename,
):
    if build_n_parallel is None:
        build_n_parallel = max(1, (os.cpu_count() or 1))

    builder = auto_scheduler.measure.LocalBuilder(
        timeout=build_timeout,
        n_parallel=build_n_parallel,
    )
    runner = auto_scheduler.measure.LocalRunner(
        timeout=run_timeout,
        repeat=repeat,
        number=number,
        enable_cpu_cache_flush=enable_cpu_cache_flush,
    )
    measurer = auto_scheduler.measure.ProgramMeasurer(
        builder,
        runner,
        [auto_scheduler.RecordToFile(log_filename)],
        verbose=verbose,
    )
    return measurer


def remeasure_file(task_idx, task, target, target_host, batch_size, measurer_kwargs):
    # Make folder and log filename
    target = tvm.target.Target(target)
    log_filename = get_measure_record_filename(task, target)
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    # Make measuer
    measurer_kwargs['log_filename'] = log_filename
    measurer = make_measurer(**measurer_kwargs)

    # Read reference measurement inputs
    to_measure_filename = get_to_measure_filename(task)
    inputs, _ = auto_scheduler.RecordReader(to_measure_filename).read_lines()
    task = auto_scheduler.measure.recover_measure_input(inputs[0]).task
    task = auto_scheduler.SearchTask(
        workload_key=task.workload_key,
        target=target,
        target_host=target_host,
        hardware_params=task.hardware_params,
        layout_rewrite_option=task.layout_rewrite_option,
    )
    empty_policy = auto_scheduler.search_policy.EmptyPolicy(task)

    # Do measurement
    for i in range(0, len(inputs), batch_size):
        print(f"===== task: {task_idx}\t programs: {i}/{len(inputs)} =====")
        inp_batch = []
        for inp in inputs[i:min(len(inputs), i + batch_size)]:
            inp_batch.append(auto_scheduler.MeasureInput(task, inp.state))
        res_batch = measurer.measure(task, empty_policy, inp_batch)

        timeout_ct = 0
        for res in res_batch:
            if res.error_no == auto_scheduler.measure.MeasureErrorNo.BUILD_TIMEOUT:
                timeout_ct += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--target-host", type=str)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=1000000)
    parser.add_argument("--step-idx", type=int, default=1)
    parser.add_argument("--build-timeout", type=int, default=15)
    parser.add_argument("--build-n-parallel", type=int, default=None)
    parser.add_argument("--run-timeout", type=int, default=5)
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--disable-cpu-cache-flush", action="store_true")
    args = parser.parse_args()

    # Load task registry
    print("Load all tasks...")
    tasks = load_and_register_tasks()
    target = tvm.target.Target(args.target)
    is_gpu_target = str(target.kind) in {"cuda", "rocm", "opencl", "vulkan", "metal"}

    end_idx = min(args.end_idx, len(tasks))

    # Remeasure all tasks
    for i in range(args.start_idx, end_idx, args.step_idx):
        with open("progress.txt", "a") as fout:
            fout.write(f"Begin {i}/{len(tasks)}: {time.time():.2f}\n")
        task = tasks[i]

        # Set measurement arguments
        measurer_kwargs = {
            "build_timeout": args.build_timeout,
            "build_n_parallel": args.build_n_parallel,
            "run_timeout": args.run_timeout,
            "number": args.number,
            "enable_cpu_cache_flush": not args.disable_cpu_cache_flush,
            "verbose": args.verbose,
        }
        if args.repeat is not None:
            measurer_kwargs["repeat"] = args.repeat
        elif is_gpu_target:
            # GPU measurements are more sensitive to timeout with larger repeats.
            # Use conservative repeats by default for robustness.
            measurer_kwargs["repeat"] = 1
            if args.build_timeout == 15:
                measurer_kwargs["build_timeout"] = 120
            if args.build_n_parallel is None:
                measurer_kwargs["build_n_parallel"] = 1
            if args.run_timeout == 5:
                measurer_kwargs["run_timeout"] = 20
            if not args.disable_cpu_cache_flush:
                measurer_kwargs["enable_cpu_cache_flush"] = False
        elif task.compute_dag.flop_ct >= 2416443392.0:
            measurer_kwargs["repeat"] = 4
        elif task.compute_dag.flop_ct >= 834928640.0:
            measurer_kwargs["repeat"] = 6
        elif task.compute_dag.flop_ct <= 2097152.0:
            measurer_kwargs["repeat"] = 10
        else:
            measurer_kwargs["repeat"] = 8

        # Run measurement
        remeasure_file(i, task, target, args.target_host, args.batch_size, measurer_kwargs)

        with open("progress.txt", "a") as fout:
            fout.write(f"End {i}/{len(tasks)}: {time.time():.2f}\n")
