#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPAT_DIR="${SCRIPT_DIR}/tools/vllm_compat"

# vLLM 0.7.x still expects `transformers` 4.x special-token helpers.
export PYTHONPATH="${COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# Disable vLLM usage telemetry by default in this legacy env. The bundled
# cpuinfo/hub stack is incompatible with its background usage reporter.
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"

count_visible_gpus() {
  local cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-}"

  if [[ -z "${cuda_visible_devices}" ]]; then
    echo 1
    return
  fi

  IFS=',' read -r -a gpu_ids <<< "${cuda_visible_devices}"
  echo "${#gpu_ids[@]}"
}

version_ge() {
  local lhs="${1}"
  local rhs="${2}"
  [[ "$(printf '%s\n%s\n' "${rhs}" "${lhs}" | sort -V | tail -n1)" == "${lhs}" ]]
}

detect_vllm_version() {
  local version=""

  version="$(python - <<'PY' 2>/dev/null
try:
    import vllm
    print(vllm.__version__)
except Exception:
    pass
PY
)"

  if [[ -z "${version}" ]] && command -v vllm >/dev/null 2>&1; then
    version="$(vllm --version 2>/dev/null | awk 'NF {print $NF; exit}')"
  fi

  if [[ -z "${version}" ]]; then
    version="unknown"
  fi

  printf '%s\n' "${version}"
}

TP_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-$(count_visible_gpus)}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.72}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2}"
PORT="${VLLM_PORT:-8080}"
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
VLLM_VERSION="$(detect_vllm_version)"

EXTRA_ARGS=()

# Default to eager mode for this eval-oriented server to avoid extra CUDA graph
# allocations on already-tight 44 GB cards. Override with VLLM_ENFORCE_EAGER=0.
if [[ "${VLLM_ENFORCE_EAGER:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi

if [[ "${MODEL}" == Qwen/Qwen3* ]] \
  && [[ "${VLLM_SKIP_MODEL_VERSION_CHECK:-0}" != "1" ]] \
  && { [[ "${VLLM_VERSION}" == "unknown" ]] || ! version_ge "${VLLM_VERSION}" "0.8.4"; }; then
  cat >&2 <<EOF
run_vllm_server.sh: ${MODEL} is not a practical target for vLLM ${VLLM_VERSION}.
This vLLM version falls back to the generic Transformers backend for Qwen3/Qwen3-MoE,
which constructs the model directly on GPU and can OOM during startup even with a small
max_model_len.

Use one of these instead:
  1. Upgrade to vLLM >= 0.8.4 for native Qwen3 support.
  2. Switch to a model supported by this legacy env, for example:
     VLLM_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ ./run_vllm_server.sh
  3. If you intentionally want to bypass this guard, set:
     VLLM_SKIP_MODEL_VERSION_CHECK=1
EOF
  exit 1
fi

VLLM_LOGGING_LEVEL=DEBUG \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve "${MODEL}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --no-enable-prefix-caching \
  --port "${PORT}" \
  "${EXTRA_ARGS[@]}"

# VLLM_LOGGING_LEVEL=DEBUG PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 --gpu-memory-utilization 0.8 --max-num-seqs 4 --no-enable-prefix-caching --port 8080
# VLLM_LOGGING_LEVEL=DEBUG PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ --gpu-memory-utilization 0.8 --max-num-seqs 4 --no-enable-prefix-caching --port 8080
# VLLM_LOGGING_LEVEL=DEBUG PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ --gpu-memory-utilization 0.8 --max-num-seqs 4 --no-enable-prefix-caching --port 8081
