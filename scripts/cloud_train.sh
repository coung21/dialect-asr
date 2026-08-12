#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly UV_INSTALL_URL="https://astral.sh/uv/install.sh"

dry_run=false
wandb_online=true
for argument in "$@"; do
    case "${argument}" in
        --dry-run) dry_run=true ;;
        --offline-wandb|--disable-wandb) wandb_online=false ;;
    esac
done

run_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${dry_run}" == false ]]; then
        "$@"
    fi
}

install_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi
    if ! command -v curl >/dev/null 2>&1; then
        echo "ERROR: curl is required to install uv." >&2
        exit 127
    fi

    echo "uv is not installed; installing it from ${UV_INSTALL_URL}"
    if [[ "${dry_run}" == true ]]; then
        echo "+ curl -LsSf ${UV_INSTALL_URL} | sh"
        return
    fi
    curl -LsSf "${UV_INSTALL_URL}" | sh

    # The standalone uv installer normally writes one of these environment files.
    if [[ -f "${HOME}/.local/bin/env" ]]; then
        # shellcheck disable=SC1091
        source "${HOME}/.local/bin/env"
    elif [[ -f "${HOME}/.cargo/env" ]]; then
        # shellcheck disable=SC1091
        source "${HOME}/.cargo/env"
    else
        export PATH="${HOME}/.local/bin:${PATH}"
    fi
    command -v uv >/dev/null 2>&1 || {
        echo "ERROR: uv was installed but is not available on PATH." >&2
        exit 127
    }
}

cd "${PROJECT_ROOT}"
install_uv

# Keep reusable dependency/model caches on the mounted project volume by default.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_ROOT}/.cache/uv}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}"

run_command uv sync

if [[ "${dry_run}" == false ]]; then
    uv run python - <<'PY'
import os
import sys

import torch

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        memory_gib = properties.total_memory / 1024**3
        print(f"GPU {index}: {properties.name} ({memory_gib:.1f} GiB)")
elif os.environ.get("CLOUD_ALLOW_CPU") != "1":
    raise SystemExit(
        "ERROR: No CUDA GPU detected. Set CLOUD_ALLOW_CPU=1 only if CPU training is intentional."
    )
PY
fi

if [[ "${wandb_online}" == true ]]; then
    if [[ "${dry_run}" == true ]]; then
        echo "+ uv run wandb login --verify"
    else
        echo "Verifying W&B authentication..."
        uv run wandb login --verify
    fi
fi

# run_experiment handles dataset download, tests, train, one final W&B artifact,
# and evaluation. Sync is skipped because this bootstrap already completed it.
if [[ "${dry_run}" == true ]]; then
    "${SCRIPT_DIR}/run_experiment.sh" --skip-sync "$@"
else
    run_command "${SCRIPT_DIR}/run_experiment.sh" --skip-sync "$@"
fi
