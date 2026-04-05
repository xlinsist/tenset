#!/usr/bin/env bash
set -euo pipefail

# One-shot parallel measurement runner with retry + integrity checks.
# Default target scenario: 2000 tasks x 200 schedules on 8 GPUs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${ROOT_DIR}/scripts"

START_IDX=0
END_IDX=2000
SIZE=200
BATCH_SIZE=32
GPU_LIST="0,1,2,3,4,5,6,7"
TARGET="cuda -arch=sm_89 -model=rtx4090"
MODEL="rtx4090"
MAX_RETRY=3
GENERATE_PROGRAMS=1
TRIM_TO_SIZE=1
BUILD_TIMEOUT=120
RUN_TIMEOUT=20
BUILD_N_PARALLEL=1
REPEAT=1
NUMBER=1
LOG_DIR="${SCRIPTS_DIR}/run_logs/parallel_$(date +%Y%m%d_%H%M%S)"

usage() {
  cat <<'EOF'
Usage:
  run_parallel_measure.sh [options]

Options:
  --start-idx N         Start task index (default: 0)
  --end-idx N           End task index, exclusive (default: 2000)
  --size N              Schedules per task (default: 200)
  --batch-size N        Measurement batch size (default: 32)
  --gpus LIST           GPU list, comma separated (default: 0,1,2,3,4,5,6,7)
  --target STR          TVM target string (default: "cuda -arch=sm_89 -model=rtx4090")
  --model STR           Target model folder name under measure_records (default: rtx4090)
  --max-retry N         Retry rounds for incomplete tasks (default: 3)
  --build-timeout N     Build timeout seconds for measure_programs (default: 120)
  --run-timeout N       Run timeout seconds for measure_programs (default: 20)
  --build-n-parallel N  Build parallelism for measure_programs (default: 1)
  --repeat N            Runner repeat for measure_programs (default: 1)
  --number N            Runner number for measure_programs (default: 1)
  --no-generate         Skip dump_programs stage
  --no-trim             Do not trim logs to --size lines when > --size
  --log-dir DIR         Log output directory
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-idx) START_IDX="$2"; shift 2 ;;
    --end-idx) END_IDX="$2"; shift 2 ;;
    --size) SIZE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --max-retry) MAX_RETRY="$2"; shift 2 ;;
    --build-timeout) BUILD_TIMEOUT="$2"; shift 2 ;;
    --run-timeout) RUN_TIMEOUT="$2"; shift 2 ;;
    --build-n-parallel) BUILD_N_PARALLEL="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --number) NUMBER="$2"; shift 2 ;;
    --no-generate) GENERATE_PROGRAMS=0; shift 1 ;;
    --no-trim) TRIM_TO_SIZE=0; shift 1 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ "$END_IDX" -le "$START_IDX" ]]; then
  echo "Invalid range: --end-idx must be > --start-idx"
  exit 1
fi

# Normalize log dir before changing cwd.
if [[ "${LOG_DIR}" != /* ]]; then
  LOG_DIR="${ROOT_DIR}/${LOG_DIR}"
fi

mkdir -p "$LOG_DIR"
cd "$SCRIPTS_DIR"

export PATH="/usr/local/cuda/bin:${PATH}"
export PYTHONPATH="../python"
export TVM_LIBRARY_PATH="../build"

CUDA_ARCH="$(echo "${TARGET}" | sed -n 's/.*-arch=\([^[:space:]]*\).*/\1/p')"
if [[ -n "${CUDA_ARCH}" ]]; then
  export TVM_CUDA_TARGET_ARCH="${CUDA_ARCH}"
  echo "[INFO] TVM_CUDA_TARGET_ARCH=${TVM_CUDA_TARGET_ARCH}"
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPU_COUNT="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
else
  DETECTED_GPU_COUNT=0
fi
if [[ "${DETECTED_GPU_COUNT}" -gt 0 ]]; then
  FILTERED=()
  for g in "${GPUS[@]}"; do
    if [[ "$g" -lt "${DETECTED_GPU_COUNT}" ]]; then
      FILTERED+=("$g")
    fi
  done
  if [[ "${#FILTERED[@]}" -lt "${#GPUS[@]}" ]]; then
    echo "[INFO] clamp gpus from ${GPU_LIST} to $(IFS=,; echo "${FILTERED[*]}") (detected=${DETECTED_GPU_COUNT})"
  fi
  GPUS=("${FILTERED[@]}")
fi
NUM_GPUS="${#GPUS[@]}"
if [[ "$NUM_GPUS" -eq 0 ]]; then
  echo "No GPU provided via --gpus"
  exit 1
fi

echo "[INFO] log_dir=${LOG_DIR}"
echo "[INFO] range=[${START_IDX}, ${END_IDX}) tasks=$((END_IDX-START_IDX)) size=${SIZE} gpus=${GPU_LIST}"
echo "[INFO] target=${TARGET}"
echo "[INFO] build_timeout=${BUILD_TIMEOUT} run_timeout=${RUN_TIMEOUT} build_n_parallel=${BUILD_N_PARALLEL} repeat=${REPEAT} number=${NUMBER}"

if [[ "$GENERATE_PROGRAMS" -eq 1 ]]; then
  echo "[INFO] Stage A: dump_programs"
  START_TS="$(date +%s)"
  (python3 dump_programs.py --start-idx "$START_IDX" --end-idx "$END_IDX" --size "$SIZE" \
      --target "${TARGET}" \
      > "${LOG_DIR}/dump_programs.log" 2>&1) || true
  END_TS="$(date +%s)"
  echo "[INFO] dump_programs seconds=$((END_TS-START_TS))"
fi

run_sharded_round() {
  local round_id="$1"
  echo "[INFO] Stage B Round ${round_id}: sharded measure"
  local st ed
  st="$(date +%s)"
  for rank in "${!GPUS[@]}"; do
    local gpu="${GPUS[$rank]}"
    local sidx=$((START_IDX + rank))
    if [[ "$sidx" -ge "$END_IDX" ]]; then
      continue
    fi
    (
      CUDA_VISIBLE_DEVICES="${gpu}" \
      python3 measure_programs.py \
        --target "${TARGET}" \
        --isolate-per-input \
        --start-idx "${sidx}" \
        --step-idx "${NUM_GPUS}" \
        --end-idx "${END_IDX}" \
        --batch-size "${BATCH_SIZE}" \
        --build-timeout "${BUILD_TIMEOUT}" \
        --run-timeout "${RUN_TIMEOUT}" \
        --build-n-parallel "${BUILD_N_PARALLEL}" \
        --repeat "${REPEAT}" \
        --number "${NUMBER}" \
        > "${LOG_DIR}/measure_round${round_id}_gpu${gpu}.log" 2>&1 || true
    ) &
  done
  wait
  ed="$(date +%s)"
  echo "[INFO] round=${round_id} sharded_seconds=$((ed-st))"
}

collect_status() {
  local out_file="$1"
  python3 - <<'PY' > "${out_file}"
import os
import pickle
import sys
import tvm  # Needed for unpickling SearchTask

sys.path.insert(0, ".")
from common import clean_name
from common import get_measure_record_filename, get_measure_record_filename_legacy

start_idx = int(os.environ["RUN_START_IDX"])
end_idx = int(os.environ["RUN_END_IDX"])
size = int(os.environ["RUN_SIZE"])
model = os.environ["RUN_MODEL"]
target_str = os.environ["RUN_TARGET"]
target = tvm.target.Target(target_str)

tasks = pickle.load(open("dataset/network_info/all_tasks.pkl", "rb"))

for i in range(start_idx, end_idx):
    task = tasks[i]
    line_count = 0
    rel = ""
    for cand in [
        get_measure_record_filename(task, target),
        get_measure_record_filename_legacy(task, target),
        get_measure_record_filename_legacy(task, task.target),
    ]:
        if os.path.exists(cand):
            with open(cand, "r") as f:
                line_count = sum(1 for _ in f)
            rel = cand
            break
    if not rel:
        rel = get_measure_record_filename(task, target)
    ok = 1 if line_count >= size else 0
    print(f"{i}\t{rel}\t{line_count}\t{ok}")

sys.stdout.flush()
os._exit(0)
PY
}

trim_if_needed() {
  if [[ "$TRIM_TO_SIZE" -eq 0 ]]; then
    return
  fi
  local status_file="$1"
  while IFS=$'\t' read -r idx path line_count ok; do
    if [[ "$line_count" -gt "$SIZE" ]]; then
      tail -n "$SIZE" "$path" > "${path}.tmp"
      mv "${path}.tmp" "$path"
    fi
  done < "$status_file"
}

collect_missing_indices() {
  local status_file="$1"
  local missing_file="$2"
  awk -F '\t' -v need="${SIZE}" '$3 < need {print $1}' "$status_file" > "$missing_file"
}

retry_missing_once() {
  local round_id="$1"
  local missing_file="$2"
  local missing_count
  missing_count="$(wc -l < "$missing_file")"
  if [[ "$missing_count" -eq 0 ]]; then
    echo "[INFO] retry round ${round_id}: no missing task"
    return
  fi

  echo "[INFO] retry round ${round_id}: missing tasks=${missing_count}"
  local st ed
  st="$(date +%s)"

  for rank in "${!GPUS[@]}"; do
    local gpu="${GPUS[$rank]}"
    (
      while IFS= read -r idx; do
        if [[ -z "$idx" ]]; then
          continue
        fi
        if [[ $((idx % NUM_GPUS)) -ne "$rank" ]]; then
          continue
        fi
        CUDA_VISIBLE_DEVICES="${gpu}" \
        python3 measure_programs.py \
          --target "${TARGET}" \
          --isolate-per-input \
          --start-idx "${idx}" \
          --end-idx "$((idx + 1))" \
          --batch-size "${BATCH_SIZE}" \
          --build-timeout "${BUILD_TIMEOUT}" \
          --run-timeout "${RUN_TIMEOUT}" \
          --build-n-parallel "${BUILD_N_PARALLEL}" \
          --repeat "${REPEAT}" \
          --number "${NUMBER}" \
          >> "${LOG_DIR}/retry_round${round_id}_gpu${gpu}.log" 2>&1 || true
      done < "$missing_file"
    ) &
  done
  wait

  ed="$(date +%s)"
  echo "[INFO] retry_round=${round_id} seconds=$((ed-st))"
}

export RUN_START_IDX="$START_IDX"
export RUN_END_IDX="$END_IDX"
export RUN_SIZE="$SIZE"
export RUN_MODEL="$MODEL"
export RUN_TARGET="$TARGET"

TOTAL_ST="$(date +%s)"

run_sharded_round 1

status_file="${LOG_DIR}/status_round1.tsv"
missing_file="${LOG_DIR}/missing_round1.txt"
collect_status "$status_file"
trim_if_needed "$status_file"
collect_missing_indices "$status_file" "$missing_file"

for r in $(seq 1 "$MAX_RETRY"); do
  missing_count="$(wc -l < "$missing_file")"
  if [[ "$missing_count" -eq 0 ]]; then
    break
  fi
  retry_missing_once "$r" "$missing_file"
  status_file="${LOG_DIR}/status_retry${r}.tsv"
  missing_file="${LOG_DIR}/missing_retry${r}.txt"
  collect_status "$status_file"
  trim_if_needed "$status_file"
  collect_missing_indices "$status_file" "$missing_file"
done

final_status="${LOG_DIR}/status_final.tsv"
collect_status "$final_status"
trim_if_needed "$final_status"

export FINAL_STATUS_FILE="$final_status"
python3 - <<'PY' > "${LOG_DIR}/summary.txt"
import os
import statistics

status_path = os.environ["FINAL_STATUS_FILE"]
size = int(os.environ["RUN_SIZE"])

lines = [x.strip().split("\t") for x in open(status_path, "r") if x.strip()]
counts = [int(x[2]) for x in lines]
ok = sum(1 for c in counts if c >= size)
total = len(counts)
missing = total - ok

print(f"total_tasks={total}")
print(f"ok_tasks={ok}")
print(f"missing_tasks={missing}")
print(f"min_lines={min(counts) if counts else 0}")
print(f"max_lines={max(counts) if counts else 0}")
print(f"avg_lines={statistics.mean(counts) if counts else 0:.2f}")
PY

TOTAL_ED="$(date +%s)"
echo "total_seconds=$((TOTAL_ED-TOTAL_ST))" >> "${LOG_DIR}/summary.txt"

echo "[INFO] ===== FINAL SUMMARY ====="
cat "${LOG_DIR}/summary.txt"
echo "[INFO] log_dir=${LOG_DIR}"
