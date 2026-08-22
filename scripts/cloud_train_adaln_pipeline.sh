#!/usr/bin/env bash
#
# Full cloud pipeline for the AdaLN dialect-conditioned PhoWhisper experiment:
#   uv sync -> download ViMD -> train DID (ECAPA-TDNN, pushed to W&B as an
#   artifact) -> train ASR "adaln" with that DID checkpoint frozen -> evaluate
#   (the ASR checkpoint is pushed to W&B automatically, same as every other
#   experiment: trainer/default.yaml already sets wandb_log_model=end).
#
# This only orchestrates existing tools; it does not duplicate their logic:
#   - scripts/download_vimd.sh   for the dataset
#   - scripts/train_did.py       for the DID branch (already wandb.log_artifact()s
#                                 its checkpoint at the end of training)
#   - scripts/run_experiment.sh  for the ASR "adaln" train + eval (already
#                                 uploads the final checkpoint to W&B via
#                                 trainer.wandb_log_model=end)

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly UV_INSTALL_URL="https://astral.sh/uv/install.sh"

experiment="adaln"
did_output_dir="outputs/did-ecapa-tdnn"
did_wandb_project="dialect-asr"
did_wandb_run_name="did-ecapa-tdnn"
skip_did=false
did_checkpoint_path=""
dry_run=false
wandb_mode="online"
declare -a did_args=()
declare -a train_overrides=()
declare -a eval_overrides=()
declare -a run_experiment_flags=()

usage() {
    cat <<'EOF'
Cloud pipeline: sync -> download ViMD -> train DID -> train ASR "adaln" -> evaluate.

Usage:
  scripts/cloud_train_adaln_pipeline.sh [options]

DID stage:
  --did-output-dir PATH        Where train_did.py saves its checkpoint
                                (default: outputs/did-ecapa-tdnn).
  --did-wandb-project NAME     W&B project for the DID run (default: dialect-asr).
  --did-wandb-run-name NAME    W&B run name for the DID run (default: did-ecapa-tdnn);
                                also the base of the artifact name "<name>-did-model".
  --did-arg ARG                Extra argv forwarded as-is to train_did.py
                                (repeatable), e.g. --did-arg --epochs --did-arg 30.
  --skip-did                   Skip DID training; requires --did-checkpoint-path.
  --did-checkpoint-path PATH   Use this checkpoint instead of the one just trained
                                (local path or "wandb-artifact:entity/project/name:version").
                                Implied by --skip-did if not given explicitly there.

ASR "adaln" stage (forwarded to scripts/run_experiment.sh):
  --experiment NAME             Experiment identifier (default: adaln); must have
                                a matching configs/experiment/<NAME>.yaml.
  --train-override KEY=VALUE    Repeatable; forwarded to the train command.
  --eval-override KEY=VALUE     Repeatable; forwarded to the eval command(s).
  --skip-eval                   Train without the final eval command.
  --eval-split SPLIT            test, validation, or all (default: test).

Shared:
  --offline-wandb              Log W&B locally without network (both stages).
  --disable-wandb               Disable W&B for both stages.
  --dry-run                     Print commands without executing them.
  -h, --help                    Show this help.

Example:
  scripts/cloud_train_adaln_pipeline.sh \
    --did-arg --epochs --did-arg 30 \
    --eval-override data.max_test_samples=500
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment)
            [[ $# -ge 2 ]] || die "--experiment requires a value."
            experiment="$2"
            shift 2
            ;;
        --did-output-dir)
            [[ $# -ge 2 ]] || die "--did-output-dir requires a value."
            did_output_dir="$2"
            shift 2
            ;;
        --did-wandb-project)
            [[ $# -ge 2 ]] || die "--did-wandb-project requires a value."
            did_wandb_project="$2"
            shift 2
            ;;
        --did-wandb-run-name)
            [[ $# -ge 2 ]] || die "--did-wandb-run-name requires a value."
            did_wandb_run_name="$2"
            shift 2
            ;;
        --did-arg)
            [[ $# -ge 2 ]] || die "--did-arg requires a value."
            did_args+=("$2")
            shift 2
            ;;
        --skip-did)
            skip_did=true
            shift
            ;;
        --did-checkpoint-path)
            [[ $# -ge 2 ]] || die "--did-checkpoint-path requires a value."
            did_checkpoint_path="$2"
            shift 2
            ;;
        --train-override)
            [[ $# -ge 2 ]] || die "--train-override requires KEY=VALUE."
            train_overrides+=("$2")
            shift 2
            ;;
        --eval-override)
            [[ $# -ge 2 ]] || die "--eval-override requires KEY=VALUE."
            eval_overrides+=("$2")
            shift 2
            ;;
        --skip-eval)
            run_experiment_flags+=("--skip-eval")
            shift
            ;;
        --eval-split)
            [[ $# -ge 2 ]] || die "--eval-split requires test, validation, or all."
            run_experiment_flags+=("--eval-split" "$2")
            shift 2
            ;;
        --offline-wandb)
            wandb_mode="offline"
            shift
            ;;
        --disable-wandb)
            wandb_mode="disabled"
            shift
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *) die "Unknown argument: $1" ;;
    esac
done

if [[ "${skip_did}" == true && -z "${did_checkpoint_path}" ]]; then
    die "--skip-did requires --did-checkpoint-path."
fi

if [[ "${did_output_dir}" != /* ]]; then
    did_output_dir="${PROJECT_ROOT}/${did_output_dir}"
fi

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

declare -a wandb_env=()
case "${wandb_mode}" in
    online)
        if [[ "${dry_run}" == true ]]; then
            echo "+ uv run wandb login --verify"
        else
            echo "Verifying W&B authentication..."
            uv run wandb login --verify
        fi
        ;;
    offline) wandb_env+=("WANDB_MODE=offline") ;;
    disabled) wandb_env+=("WANDB_MODE=disabled") ;;
esac

run_command "${SCRIPT_DIR}/download_vimd.sh"

if [[ "${skip_did}" == false ]]; then
    did_command=(
        uv run python scripts/train_did.py
        --output-dir "${did_output_dir}"
        --wandb-project "${did_wandb_project}"
        --wandb-run-name "${did_wandb_run_name}"
    )
    case "${wandb_mode}" in
        online) did_command+=(--wandb-mode online) ;;
        offline) did_command+=(--wandb-mode offline) ;;
        disabled) did_command+=(--wandb-mode disabled) ;;
    esac
    did_command+=("${did_args[@]}")
    run_command "${did_command[@]}"

    did_checkpoint_path="${did_output_dir}/final_model.pt"
    if [[ "${dry_run}" == false && ! -f "${did_checkpoint_path}" ]]; then
        die "DID training did not produce ${did_checkpoint_path}."
    fi
fi

echo "Using DID checkpoint: ${did_checkpoint_path}"

asr_command=(
    "${SCRIPT_DIR}/run_experiment.sh"
    --skip-sync --skip-dataset
    --experiment "${experiment}"
    --train-override "model.did_checkpoint_path=${did_checkpoint_path}"
)
case "${wandb_mode}" in
    offline) asr_command+=(--offline-wandb) ;;
    disabled) asr_command+=(--disable-wandb) ;;
esac
for override in "${train_overrides[@]}"; do
    asr_command+=(--train-override "${override}")
done
for override in "${eval_overrides[@]}"; do
    asr_command+=(--eval-override "${override}")
done
asr_command+=("${run_experiment_flags[@]}")

# When dry_run, invoke run_experiment.sh directly with its own --dry-run so it
# prints its internal train/eval command breakdown instead of collapsing to a
# single opaque line.
if [[ "${dry_run}" == true ]]; then
    asr_command+=(--dry-run)
    "${asr_command[@]}"
else
    run_command "${asr_command[@]}"
fi
