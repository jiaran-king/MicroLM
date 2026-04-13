#!/usr/bin/env bash
# serve_vllm.sh — Start vLLM OpenAI-compatible server for Qwen 1.5B InstructIE LoRA.
#
# Usage:
#   bash scripts/serve_vllm.sh                    # defaults (port 8000, GPU auto)
#   bash scripts/serve_vllm.sh --port 8001        # custom port
#   bash scripts/serve_vllm.sh --cpu              # force CPU (slow, for testing)
#
# Prerequisites:
#   pip install vllm
#   Run export_final_model.py first to create outputs/qwen_lora_merged_final/
#
# The server exposes an OpenAI-compatible API at http://localhost:<PORT>/v1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────
MODEL_PATH="${PROJECT_ROOT}/outputs/qwen_lora_merged_final"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
TP_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
DTYPE="${VLLM_DYPE:-auto}"
USE_CPU=0

# ── Parse args ───────────────────────────────────────────────────────────
for arg in "$@"; do
    case $arg in
        --port=*)
            PORT="${arg#*=}"
            ;;
        --host=*)
            HOST="${arg#*=}"
            ;;
        --cpu)
            USE_CPU=1
            ;;
        --tp=*)
            TP_SIZE="${arg#*=}"
            ;;
        --max-model-len=*|--max-model-length=*)
            MAX_MODEL_LEN="${arg#*=}"
            ;;
        --help|-h)
            echo "Usage: $0 [--port=PORT] [--host=HOST] [--cpu] [--tp=N] [--max-model-len=N]"
            exit 0
            ;;
    esac
done

# ── Validate model path ──────────────────────────────────────────────────
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Merged model not found at: $MODEL_PATH"
    echo "Run first: python scripts/export_final_model.py"
    exit 1
fi

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "[ERROR] config.json not found in: $MODEL_PATH"
    echo "The merged model may be incomplete. Re-run export_final_model.py"
    exit 1
fi

echo "============================================"
echo "  vLLM Server — Qwen 1.5B InstructIE LoRA"
echo "============================================"
echo "  Model:      $MODEL_PATH"
echo "  Host:Port:  ${HOST}:${PORT}"
echo "  TP size:    ${TP_SIZE}"
echo "  Max len:    ${MAX_MODEL_LEN}"
echo "  API docs:   http://localhost:${PORT}/docs"
echo "============================================"

# ── Build command ─────────────────────────────────────────────────────────
VLLM_CMD=(
    vllm serve
    "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --tensor-parallel-size "${TP_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --dtype "${DTYPE}"
    # Chat template is already in the merged model's tokenizer_config.json
)

if [ "$USE_CPU" -eq 1 ]; then
    VLLM_CMD+=(--device cpu --enforce-eager)
fi

exec "${VLLM_CMD[@]}"
