"""Measure all programs."""

import argparse
import json
import os
import subprocess
import sys
import time

import tvm
from tvm import auto_scheduler

from common import (
    get_measure_record_filename,
    get_to_measure_filename_compat,
    load_and_register_tasks,
)

GPU_KINDS = {"cuda", "rocm", "opencl", "vulkan", "metal"}

def _set_cuda_arch_env(target):
    if str(target.kind) != "cuda":
        return
    arch = str(target.attrs.get("arch", "")).strip()
    if arch:
        os.environ["TVM_CUDA_TARGET_ARCH"] = arch


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


def _truncate_err(msg, limit):
    if msg is None:
        return ""
    s = str(msg).replace("\n", "\\n")
    return s if len(s) <= limit else s[:limit] + "...[truncated]"


def _error_sig(msg):
    if not msg:
        return ""
    head = str(msg).strip().splitlines()[0] if str(msg).strip() else ""
    return head[:256]


def _append_error_row(error_log_file, task_idx, sched_idx, error_no, err_msg, max_error_msg):
    existed = os.path.exists(error_log_file)
    with open(error_log_file, "a") as f:
        if not existed:
            f.write("ts\ttask_idx\tsched_idx\terror_no\terror_sig\terror_msg\n")
        now = time.strftime("%F %T")
        sig = _error_sig(err_msg).replace("\t", " ")
        msg = _truncate_err(err_msg, max_error_msg).replace("\t", " ")
        f.write(f"{now}\t{task_idx}\t{sched_idx}\t{int(error_no)}\t{sig}\t{msg}\n")


def _build_run_task(base_task, target, target_host):
    return auto_scheduler.SearchTask(
        workload_key=base_task.workload_key,
        target=target,
        target_host=target_host,
        hardware_params=base_task.hardware_params,
        layout_rewrite_option=base_task.layout_rewrite_option,
    )


def remeasure_single(task_idx, sched_idx, target, target_host, measurer_kwargs, max_error_msg):
    tasks = load_and_register_tasks()
    task = tasks[task_idx]
    target = tvm.target.Target(target)

    to_measure_filename = get_to_measure_filename_compat(task, target)
    inputs, _ = auto_scheduler.RecordReader(to_measure_filename).read_lines()
    inputs = list(inputs)
    if sched_idx < 0 or sched_idx >= len(inputs):
        raise IndexError(
            f"sched_idx {sched_idx} out of range for task={task_idx}, total={len(inputs)}"
        )

    recovered_task = auto_scheduler.measure.recover_measure_input(inputs[sched_idx]).task
    run_task = _build_run_task(recovered_task, target, target_host)

    out_record = get_measure_record_filename(task, target)
    os.makedirs(os.path.dirname(out_record), exist_ok=True)

    single_kwargs = dict(measurer_kwargs)
    single_kwargs["log_filename"] = out_record
    measurer = make_measurer(**single_kwargs)
    empty_policy = auto_scheduler.search_policy.EmptyPolicy(run_task)

    recovered = auto_scheduler.measure.recover_measure_input(inputs[sched_idx])
    measure_input = auto_scheduler.MeasureInput(run_task, recovered.state)
    res = measurer.measure(run_task, empty_policy, [measure_input])[0]

    err_msg = getattr(res, "error_msg", "")
    if int(res.error_no) != 0:
        _append_error_row(out_record + ".errors.tsv", task_idx, sched_idx, res.error_no, err_msg, max_error_msg)

    payload = {
        "task_idx": task_idx,
        "sched_idx": sched_idx,
        "error_no": int(res.error_no),
        "error_sig": _error_sig(err_msg),
        "error_msg": _truncate_err(err_msg, max_error_msg),
    }
    print(json.dumps(payload), flush=True)


def remeasure_file(
    task_idx,
    task,
    target,
    target_host,
    batch_size,
    measurer_kwargs,
    isolate_per_input,
    max_error_msg,
):
    target = tvm.target.Target(target)
    out_record = get_measure_record_filename(task, target)
    os.makedirs(os.path.dirname(out_record), exist_ok=True)

    to_measure_filename = get_to_measure_filename_compat(task, target)
    inputs, _ = auto_scheduler.RecordReader(to_measure_filename).read_lines()
    inputs = list(inputs)
    if not inputs:
        return

    if isolate_per_input:
        for sched_idx in range(len(inputs)):
            print(f"===== task: {task_idx}\t programs: {sched_idx}/{len(inputs)} =====")
            cmd = [
                sys.executable,
                __file__,
                "--mode",
                "single",
                "--task-idx",
                str(task_idx),
                "--sched-idx",
                str(sched_idx),
                "--target",
                str(target),
                "--target-host",
                str(target_host or ""),
                "--build-timeout",
                str(measurer_kwargs["build_timeout"]),
                "--build-n-parallel",
                str(measurer_kwargs["build_n_parallel"]),
                "--run-timeout",
                str(measurer_kwargs["run_timeout"]),
                "--number",
                str(measurer_kwargs["number"]),
                "--repeat",
                str(measurer_kwargs["repeat"]),
                "--verbose",
                str(measurer_kwargs["verbose"]),
                "--max-error-msg",
                str(max_error_msg),
            ]
            if not measurer_kwargs["enable_cpu_cache_flush"]:
                cmd.append("--disable-cpu-cache-flush")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                _append_error_row(
                    out_record + ".errors.tsv",
                    task_idx,
                    sched_idx,
                    4,
                    f"subprocess_rc={proc.returncode}; stderr={proc.stderr}",
                    max_error_msg,
                )
                continue
            line = ""
            for ln in reversed(proc.stdout.splitlines()):
                ln = ln.strip()
                if ln.startswith("{") and ln.endswith("}"):
                    line = ln
                    break
            if not line:
                _append_error_row(
                    out_record + ".errors.tsv",
                    task_idx,
                    sched_idx,
                    4,
                    f"missing_json_output; stdout={proc.stdout}; stderr={proc.stderr}",
                    max_error_msg,
                )
                continue
            try:
                payload = json.loads(line)
                if int(payload.get("error_no", 4)) != 0:
                    _append_error_row(
                        out_record + ".errors.tsv",
                        task_idx,
                        sched_idx,
                        payload.get("error_no", 4),
                        payload.get("error_msg", ""),
                        max_error_msg,
                    )
            except Exception as err:  # pylint: disable=broad-except
                _append_error_row(
                    out_record + ".errors.tsv",
                    task_idx,
                    sched_idx,
                    4,
                    f"json_parse_error={err}; raw={line}",
                    max_error_msg,
                )
        return

    single_kwargs = dict(measurer_kwargs)
    single_kwargs["log_filename"] = out_record
    measurer = make_measurer(**single_kwargs)

    recovered_task = auto_scheduler.measure.recover_measure_input(inputs[0]).task
    run_task = _build_run_task(recovered_task, target, target_host)
    empty_policy = auto_scheduler.search_policy.EmptyPolicy(run_task)

    for i in range(0, len(inputs), batch_size):
        print(f"===== task: {task_idx}\t programs: {i}/{len(inputs)} =====")
        inp_batch = []
        batch_end = min(len(inputs), i + batch_size)
        for inp in inputs[i:batch_end]:
            recovered = auto_scheduler.measure.recover_measure_input(inp)
            inp_batch.append(auto_scheduler.MeasureInput(run_task, recovered.state))
        res_batch = measurer.measure(run_task, empty_policy, inp_batch)

        for offset, res in enumerate(res_batch):
            if int(res.error_no) != 0:
                sched_idx = i + offset
                _append_error_row(
                    out_record + ".errors.tsv",
                    task_idx,
                    sched_idx,
                    res.error_no,
                    getattr(res, "error_msg", ""),
                    max_error_msg,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["batch", "single"], default="batch")
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--target-host", type=str, default=None)
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
    parser.add_argument("--isolate-per-input", action="store_true")
    parser.add_argument("--max-error-msg", type=int, default=8192)
    parser.add_argument("--task-idx", type=int, default=-1)
    parser.add_argument("--sched-idx", type=int, default=-1)
    args = parser.parse_args()

    target = tvm.target.Target(args.target)
    _set_cuda_arch_env(target)
    is_gpu_target = str(target.kind) in GPU_KINDS

    if args.mode == "single":
        if args.task_idx < 0 or args.sched_idx < 0:
            raise ValueError("--task-idx and --sched-idx are required in single mode")
        single_kwargs = {
            "build_timeout": args.build_timeout,
            "build_n_parallel": args.build_n_parallel,
            "run_timeout": args.run_timeout,
            "number": args.number,
            "repeat": args.repeat if args.repeat is not None else 1,
            "enable_cpu_cache_flush": not args.disable_cpu_cache_flush,
            "verbose": args.verbose,
        }
        if single_kwargs["build_n_parallel"] is None:
            single_kwargs["build_n_parallel"] = 1
        remeasure_single(
            task_idx=args.task_idx,
            sched_idx=args.sched_idx,
            target=args.target,
            target_host=(args.target_host or None),
            measurer_kwargs=single_kwargs,
            max_error_msg=args.max_error_msg,
        )
        sys.stdout.flush()
        os._exit(0)

    print("Load all tasks...")
    tasks = load_and_register_tasks()
    end_idx = min(args.end_idx, len(tasks))

    for i in range(args.start_idx, end_idx, args.step_idx):
        with open("progress.txt", "a") as fout:
            fout.write(f"Begin {i}/{len(tasks)}: {time.time():.2f}\n")

        task = tasks[i]
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

        isolate = args.isolate_per_input or is_gpu_target
        remeasure_file(
            i,
            task,
            target,
            args.target_host,
            args.batch_size,
            measurer_kwargs,
            isolate_per_input=isolate,
            max_error_msg=args.max_error_msg,
        )

        with open("progress.txt", "a") as fout:
            fout.write(f"End {i}/{len(tasks)}: {time.time():.2f}\n")

    sys.stdout.flush()
    os._exit(0)
