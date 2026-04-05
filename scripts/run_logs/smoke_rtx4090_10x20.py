#!/usr/bin/env python3
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


TARGET_STR = "cuda -arch=sm_89 -model=rtx4090"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def now():
    return time.strftime("%F %T")

def detect_gpu_count():
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.STDOUT)
        return sum(1 for ln in out.splitlines() if ln.strip().startswith("GPU "))
    except Exception:
        return 0


def run_worker(gpu_id, task_indices, per_task_sched, out_tsv):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TVM_NUM_THREADS"] = "1"

    target = tvm.target.Target(TARGET_STR)
    arch = str(target.attrs.get("arch", "")).strip()
    if arch:
        os.environ["TVM_CUDA_TARGET_ARCH"] = arch
    all_tasks = load_and_register_tasks()

    builder = auto_scheduler.measure.LocalBuilder(timeout=120, n_parallel=1)
    runner = auto_scheduler.measure.LocalRunner(
        timeout=20, repeat=1, number=1, enable_cpu_cache_flush=False
    )
    measurer = auto_scheduler.measure.ProgramMeasurer(builder, runner, callbacks=[], verbose=0)

    err_tsv = out_tsv.replace(".tsv", ".errors.tsv")
    with open(out_tsv, "w") as fout:
        fout.write(
            "ts\ttask_idx\tsched_idx\tgpu_id\terror_no\tall_cost\tcost0\twall_s\terror_sig\tout_record\n"
        )
        for task_idx in task_indices:
            task = all_tasks[task_idx]
            to_file = get_to_measure_filename_compat(task, target)
            inputs, _ = auto_scheduler.RecordReader(to_file).read_lines()
            inputs = list(inputs)[:per_task_sched]
            if not inputs:
                continue

            recovered_task = auto_scheduler.measure.recover_measure_input(inputs[0]).task
            run_task = auto_scheduler.SearchTask(
                workload_key=recovered_task.workload_key,
                target=target,
                target_host=None,
                hardware_params=recovered_task.hardware_params,
                layout_rewrite_option=recovered_task.layout_rewrite_option,
            )
            policy = auto_scheduler.search_policy.EmptyPolicy(run_task)

            ok_inputs, ok_results = [], []
            for sched_idx, inp in enumerate(inputs):
                st = time.time()
                recovered = auto_scheduler.measure.recover_measure_input(inp)
                m_inp = auto_scheduler.MeasureInput(run_task, recovered.state)
                res = measurer.measure(run_task, policy, [m_inp])[0]
                wall_s = time.time() - st

                error_no = int(res.error_no)
                all_cost = float(res.all_cost) if res.all_cost is not None else -1.0
                cost0 = float(res.costs[0].value) if res.costs else -1.0
                err_msg = getattr(res, "error_msg", "") or ""
                err_sig = (str(err_msg).strip().splitlines()[0] if str(err_msg).strip() else "")[:128]
                if error_no == 0:
                    ok_inputs.append(m_inp)
                    ok_results.append(res)
                else:
                    existed = os.path.exists(err_tsv)
                    with open(err_tsv, "a") as ef:
                        if not existed:
                            ef.write("ts\ttask_idx\tsched_idx\tgpu_id\terror_no\terror_sig\terror_msg\n")
                        msg = str(err_msg).replace("\n", "\\n").replace("\t", " ")
                        if len(msg) > 8192:
                            msg = msg[:8192] + "...[truncated]"
                        ef.write(
                            f"{now()}\t{task_idx}\t{sched_idx}\t{gpu_id}\t{error_no}\t{err_sig}\t{msg}\n"
                        )

                fout.write(
                    f"{now()}\t{task_idx}\t{sched_idx}\t{gpu_id}\t{error_no}\t"
                    f"{all_cost:.6f}\t{cost0:.6f}\t{wall_s:.6f}\t{err_sig}\t\n"
                )
                fout.flush()

            out_record = get_measure_record_filename(task, target)
            ensure_dir(os.path.dirname(out_record))
            if ok_inputs:
                auto_scheduler.save_records(out_record, ok_inputs, ok_results)
            fout.write(
                f"{now()}\t{task_idx}\t-1\t{gpu_id}\tTASK_SUMMARY\t{len(ok_inputs)}"
                f"\t{len(inputs)}\t0.0\t-\t{out_record}\n"
            )
            fout.flush()

    sys.stdout.flush()
    os._exit(0)


def launch(task_start, task_count, per_task_sched, num_gpus, out_dir):
    ensure_dir(out_dir)
    detected = detect_gpu_count()
    if detected > 0 and num_gpus > detected:
        print(
            f"[{now()}] requested num_gpus={num_gpus} but detected={detected}; clamp to {detected}",
            flush=True,
        )
        num_gpus = detected
    if num_gpus <= 0:
        raise ValueError("num_gpus must be > 0")

    task_end = task_start + task_count
    task_indices = list(range(task_start, task_end))
    shards = [task_indices[i::num_gpus] for i in range(num_gpus)]

    launcher_log = os.path.join(out_dir, "launcher.log")
    worker_logs = []
    procs = []
    for gpu_id, shard in enumerate(shards):
        tsv = os.path.join(out_dir, f"worker_gpu{gpu_id}.tsv")
        log = os.path.join(out_dir, f"worker_gpu{gpu_id}.log")
        worker_logs.append(tsv)
        cmd = [
            sys.executable,
            __file__,
            "--mode",
            "worker",
            "--gpu-id",
            str(gpu_id),
            "--task-indices",
            ",".join(str(x) for x in shard),
            "--per-task-sched",
            str(per_task_sched),
            "--out-tsv",
            tsv,
        ]
        with open(log, "w") as lf:
            p = subprocess.Popen(cmd, stdout=lf, stderr=lf)
        procs.append((gpu_id, shard, p, log, tsv))

    with open(launcher_log, "w") as lf:
        lf.write(f"[{now()}] launch start tasks={task_start}..{task_end-1}\n")
        for gpu_id, shard, p, log, tsv in procs:
            rc = p.wait()
            lf.write(
                f"[{now()}] gpu={gpu_id} rc={rc} tasks={shard} log={log} tsv={tsv}\n"
            )
            lf.flush()

    summarize(worker_logs, out_dir)


def summarize(worker_tsvs, out_dir):
    total = 0
    success = 0
    timeout = 0
    other_fail = 0
    valid_time = 0
    by_gpu = {}

    for tsv in worker_tsvs:
        if not os.path.exists(tsv):
            continue
        with open(tsv) as f:
            next(f, None)
            for ln in f:
                parts = ln.rstrip("\n").split("\t")
                if len(parts) < 10:
                    continue
                error = parts[4]
                gpu = parts[3]
                if error == "TASK_SUMMARY":
                    continue
                total += 1
                by_gpu[gpu] = by_gpu.get(gpu, 0) + 1
                try:
                    e = int(error)
                except Exception:
                    continue
                if e == 0:
                    success += 1
                    try:
                        if float(parts[5]) > 0:
                            valid_time += 1
                    except Exception:
                        pass
                elif e == 7:
                    timeout += 1
                else:
                    other_fail += 1

    summary = {
        "target": TARGET_STR,
        "total_measured": total,
        "success": success,
        "timeout_error7": timeout,
        "other_fail": other_fail,
        "success_rate": (success / total if total else 0.0),
        "valid_timing_count": valid_time,
        "gpu_load": by_gpu,
        "worker_tsvs": worker_tsvs,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["launcher", "worker"], default="launcher")
    p.add_argument("--task-start", type=int, default=0)
    p.add_argument("--task-count", type=int, default=10)
    p.add_argument("--per-task-sched", type=int, default=20)
    p.add_argument("--num-gpus", type=int, default=4)
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--task-indices", type=str, default="")
    p.add_argument("--out-tsv", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "launcher":
        out_dir = args.out_dir or os.path.join(
            "run_logs", f"rtx4090_smoke_10x20_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        launch(
            task_start=args.task_start,
            task_count=args.task_count,
            per_task_sched=args.per_task_sched,
            num_gpus=args.num_gpus,
            out_dir=out_dir,
        )
    else:
        task_indices = [int(x) for x in args.task_indices.split(",") if x]
        run_worker(
            gpu_id=args.gpu_id,
            task_indices=task_indices,
            per_task_sched=args.per_task_sched,
            out_tsv=args.out_tsv,
        )


if __name__ == "__main__":
    main()
