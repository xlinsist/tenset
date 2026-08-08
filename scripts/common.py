from collections import defaultdict, namedtuple
from pathlib import Path
import os
import pickle

import tvm
from tvm import relay, auto_scheduler
from tvm.auto_scheduler.utils import to_str_round

####################################
##### Network Utilities
####################################
def convert_to_nhwc(mod):
    """Convert to NHWC layout"""
    desired_layouts = {
        "nn.conv2d": ["NHWC", "default"],
        "nn.conv3d": ["NDHWC", "default"],
    }
    seq = tvm.transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod

# The format for a line in the results file
BenchmarkRecord = namedtuple("BenchmarkRecord",
                             ['device', 'backend', 'workload_type', 'workload_name',
                              'library', 'algorithm', 'value', 'time_stamp'])

def log_line(record, out_file):
    with open(out_file, 'a') as fout:
        fout.write("\t".join([to_str_round(x) for x in record]) + '\n')


####################################
##### Dataset Utilities
####################################

NETWORK_INFO_FOLDER = 'dataset/network_info'
TO_MEASURE_PROGRAM_FOLDER = 'dataset/to_measure_programs'
MEASURE_RECORD_FOLDER = 'dataset/measure_records'

def clean_name(x):
    x = str(x)
    x = x.replace(" ", "")
    x = x.replace('"', '')
    x = x.replace("'", '')
    return x

def get_relay_ir_filename(network_key):
    return f"{NETWORK_INFO_FOLDER}/{clean_name(network_key)}.relay.pkl"

def get_task_info_filename(network_key, target):
    network_task_key = (network_key,) + (str(target.kind),)
    return f"{NETWORK_INFO_FOLDER}/{clean_name(network_task_key)}.task.pkl"

def get_target_signature(target):
    # Include architecture/model in key to avoid cross-arch record mixing
    # (e.g. sm_75 states reused on sm_89 runtime).
    kind = str(target.kind)
    attrs = []
    for k in ("arch", "model"):
        v = str(target.attrs.get(k, "")).strip()
        if v:
            attrs.append(f"{k}={v}")
    return kind if not attrs else f"{kind}|{'|'.join(attrs)}"


def get_to_measure_filename(task, target=None):
    target = target or task.target
    task_key = (task.workload_key, get_target_signature(target))
    return f"{TO_MEASURE_PROGRAM_FOLDER}/{clean_name(task_key)}.json"

def get_to_measure_filename_legacy(task):
    task_key = (task.workload_key, str(task.target.kind))
    return f"{TO_MEASURE_PROGRAM_FOLDER}/{clean_name(task_key)}.json"


def get_to_measure_filename_compat(task, target=None):
    # Prefer the new signature, fallback to historical kind-only key.
    import os

    new_file = get_to_measure_filename(task, target)
    if os.path.exists(new_file):
        return new_file

    old_file = get_to_measure_filename_legacy(task)
    if os.path.exists(old_file):
        return old_file
    return new_file


def get_measure_record_filename(task, target=None):
    target = target or task.target
    task_key = (task.workload_key, get_target_signature(target))
    return f"{MEASURE_RECORD_FOLDER}/{target.model}/{clean_name(task_key)}.json"


def get_measure_record_filename_legacy(task, target=None):
    target = target or task.target
    task_key = (task.workload_key, str(target.kind))
    return f"{MEASURE_RECORD_FOLDER}/{target.model}/{clean_name(task_key)}.json"


def get_measure_record_filename_compat(task, target=None):
    # Prefer the new signature, fallback to historical kind-only key.
    import os

    new_file = get_measure_record_filename(task, target)
    if os.path.exists(new_file):
        return new_file

    old_file = get_measure_record_filename_legacy(task, target)
    if os.path.exists(old_file):
        return old_file
    return new_file

def get_all_tasks_path():
    candidates = []
    env_override = os.environ.get("TVM_ALL_TASKS_PATH", "").strip()
    if env_override:
        candidates.append(Path(env_override))

    candidates.extend(
        [
            Path(NETWORK_INFO_FOLDER) / "all_tasks.pkl",
            Path(NETWORK_INFO_FOLDER) / "network_info" / "all_tasks.pkl",
            Path("dataset_gpu_old/network_info/all_tasks.pkl"),
            Path("dataset_cpu/network_info/all_tasks.pkl"),
        ]
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Unable to locate all_tasks.pkl; checked: %s"
        % ", ".join(str(path) for path in candidates)
    )


def load_and_register_tasks():
    tasks = pickle.load(open(get_all_tasks_path(), "rb"))

    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors)

    return tasks


####################################
##### Other Utilities
####################################

def dtype2torch(x):
    import torch

    return {
        'float32': torch.float32
    }[x]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
