"""Dump programs for all tasks"""

import argparse
import pickle
import gc
import glob
import time
import os
import random

from tqdm import tqdm

from tvm import auto_scheduler

import tvm

from common import load_and_register_tasks, get_to_measure_filename


def _validate_states(
    task,
    states,
    build_timeout,
    run_timeout,
    build_n_parallel,
    number,
    repeat,
):
    if not states:
        return []

    builder = auto_scheduler.measure.LocalBuilder(
        timeout=build_timeout,
        n_parallel=build_n_parallel,
    )
    runner = auto_scheduler.measure.LocalRunner(
        timeout=run_timeout,
        number=number,
        repeat=repeat,
        enable_cpu_cache_flush=False,
    )
    measurer = auto_scheduler.measure.ProgramMeasurer(
        builder,
        runner,
        callbacks=[],
        verbose=0,
    )
    empty_policy = auto_scheduler.search_policy.EmptyPolicy(task)
    measure_inputs = [auto_scheduler.MeasureInput(task, s) for s in states]
    results = measurer.measure(task, empty_policy, measure_inputs)

    valid_states = []
    for s, res in zip(states, results):
        if int(res.error_no) == 0:
            valid_states.append(s)
    return valid_states


def dump_program(
    task,
    size,
    target=None,
    seed=42,
    max_retry_iter=10,
    force=False,
    validate_state=False,
    validate_build_timeout=120,
    validate_run_timeout=60,
    validate_build_n_parallel=1,
    validate_number=1,
    validate_repeat=1,
):
    if target is not None:
        task = auto_scheduler.SearchTask(
            workload_key=task.workload_key,
            target=target,
            target_host=None,
            hardware_params=task.hardware_params,
            layout_rewrite_option=task.layout_rewrite_option,
        )

    filename = get_to_measure_filename(task, target)
    if os.path.exists(filename) and not force:
        return
    if force and os.path.exists(filename):
        os.remove(filename)

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    policy = auto_scheduler.SketchPolicy(
        task,
        params={
            "evolutionary_search_num_iters": 1,
            "evolutionary_search_population": min(size * (4 if validate_state else 1), 2560),
        },
        seed=seed,
        verbose=0,
    )

    states = policy.sample_initial_population()

    # Generate unique states
    all_state_str_set = set()
    all_state_str_seen = set()
    all_state_list = []

    retry_ct = 0
    niter = 0

    while len(all_state_list) < size and retry_ct < max_retry_iter:
        ct_before = len(all_state_list)

        states = policy.evolutionary_search(states, len(states))
        candidates = []
        for s in states:
            str_s = str(s)
            if str_s in all_state_str_seen:
                continue
            all_state_str_seen.add(str_s)
            candidates.append(s)

        if validate_state:
            candidates = _validate_states(
                task,
                candidates,
                build_timeout=validate_build_timeout,
                run_timeout=validate_run_timeout,
                build_n_parallel=validate_build_n_parallel,
                number=validate_number,
                repeat=validate_repeat,
            )

        for s in candidates:
            str_s = str(s)
            if str_s in all_state_str_set:
                continue
            all_state_str_set.add(str_s)
            all_state_list.append(s)
            if len(all_state_list) >= size:
                break

        ct_after = len(all_state_list)

        if ct_before == ct_after:
            states = policy.sample_initial_population()
            retry_ct += 1
        else:
            retry_ct = 0

        print(niter, len(all_state_list))
        niter += 1
    all_state_list = all_state_list[:size]

    # Make measure inputs and results
    measure_inputs = []
    measure_results = []
    for state in all_state_list:
        measure_inputs.append(auto_scheduler.MeasureInput(task, state))
        measure_results.append(auto_scheduler.MeasureResult([0.0], 0, "", 0, time.time()))

    # Dump to file
    auto_scheduler.save_records(filename, measure_inputs, measure_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-idx", type=int)
    parser.add_argument("--end-idx", type=int)
    parser.add_argument("--size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42, help="random seed for evolutionary search")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-state", action="store_true")
    parser.add_argument("--validate-build-timeout", type=int, default=120)
    parser.add_argument("--validate-run-timeout", type=int, default=60)
    parser.add_argument("--validate-build-n-parallel", type=int, default=1)
    parser.add_argument("--validate-number", type=int, default=1)
    parser.add_argument("--validate-repeat", type=int, default=1)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    # TVM's SketchPolicy uses `seed or random.randint(...)`, so 0 would be treated as unset.
    # Convert to a non-zero deterministic seed.
    sketch_seed = args.seed if args.seed != 0 else 42
    random.seed(args.seed)

    tasks = load_and_register_tasks()

    start_idx = args.start_idx or 0
    end_idx = args.end_idx or len(tasks)

    # Dump programs for all tasks
    target = tvm.target.Target(args.target) if args.target else None
    for task in tqdm(tasks[start_idx:end_idx]):
        dump_program(
            task,
            size=args.size,
            target=target,
            seed=sketch_seed,
            force=args.force,
            validate_state=args.validate_state,
            validate_build_timeout=args.validate_build_timeout,
            validate_run_timeout=args.validate_run_timeout,
            validate_build_n_parallel=args.validate_build_n_parallel,
            validate_number=args.validate_number,
            validate_repeat=args.validate_repeat,
        )
        gc.collect()
