#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

experiment=""
output_dir=""
resume_checkpoint=""
eval_checkpoint=""
eval_split="test"
skip_sync=false
skip_dataset=false
skip_tests=false
skip_train=false
skip_eval=false
dry_run=false
wandb_mode="online"
declare -a train_overrides=()
declare -a eval_overrides=()

usage() {
    cat <<'EOF'
Run the complete dialect-ASR pipeline: sync, dataset, tests, train and eval.

Usage:
  scripts/run_experiment.sh --experiment NAME [options]

Required:
  --experiment NAME              Experiment/run identifier, e.g. baseline.

Paths and checkpoints:
  --output-dir PATH              Default: outputs/NAME.
  --resume PATH                  Resume the training Trainer checkpoint.
  --eval-checkpoint PATH_OR_REPO Evaluate this checkpoint instead of final/.
  --eval-split SPLIT             test, validation, or all (default: test).

Experiment-specific Hydra parameters (repeatable):
  --train-override KEY=VALUE     Applied only to the train command.
  --eval-override KEY=VALUE      Applied only to the eval command(s).

Pipeline controls:
  --skip-sync                    Skip uv sync.
  --skip-dataset                 Skip ViMD download/validation.
  --skip-tests                   Skip pytest.
  --skip-train                   Evaluate an existing checkpoint only.
  --skip-eval                    Train without the final eval command.
  --offline-wandb                Log W&B locally without network.
  --disable-wandb                Disable W&B for both train and eval.
  --dry-run                      Print commands without executing them.
  -h, --help                     Show this help.

Examples:
  # Baseline on one NVIDIA GPU.
  scripts/run_experiment.sh --experiment baseline \
    --train-override trainer.fp16=true

  # Small smoke experiment without W&B.
  scripts/run_experiment.sh --experiment smoke --disable-wandb \
    --train-override data.max_train_samples=16 \
    --train-override data.max_validation_samples=8 \
    --train-override trainer.num_train_epochs=1 \
    --eval-override data.max_test_samples=8

  # Future fusion experiment with separate train/eval overrides.
  scripts/run_experiment.sh --experiment film \
    --train-override model=film \
    --train-override fusion.dropout=0.1 \
    --eval-override data.max_test_samples=500
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

require_override() {
    local value="$1"
    [[ "${value}" == *=* ]] || die "Hydra override must use KEY=VALUE: ${value}"
    case "${value}" in
        trainer.output_dir=*|final_model_dir=*)
            die "Use --output-dir instead of overriding ${value%%=*}."
            ;;
        trainer.wandb_log_model=*)
            die "W&B artifact policy is fixed: train=end and eval=false."
            ;;
    esac
}

run_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${dry_run}" == false ]]; then
        "$@"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment)
            [[ $# -ge 2 ]] || die "--experiment requires a value."
            experiment="$2"
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || die "--output-dir requires a value."
            output_dir="$2"
            shift 2
            ;;
        --resume)
            [[ $# -ge 2 ]] || die "--resume requires a checkpoint path."
            resume_checkpoint="$2"
            shift 2
            ;;
        --eval-checkpoint)
            [[ $# -ge 2 ]] || die "--eval-checkpoint requires a path or repo ID."
            eval_checkpoint="$2"
            shift 2
            ;;
        --eval-split)
            [[ $# -ge 2 ]] || die "--eval-split requires test, validation, or all."
            eval_split="$2"
            shift 2
            ;;
        --train-override)
            [[ $# -ge 2 ]] || die "--train-override requires KEY=VALUE."
            require_override "$2"
            train_overrides+=("$2")
            shift 2
            ;;
        --eval-override)
            [[ $# -ge 2 ]] || die "--eval-override requires KEY=VALUE."
            require_override "$2"
            eval_overrides+=("$2")
            shift 2
            ;;
        --skip-sync) skip_sync=true; shift ;;
        --skip-dataset) skip_dataset=true; shift ;;
        --skip-tests) skip_tests=true; shift ;;
        --skip-train) skip_train=true; shift ;;
        --skip-eval) skip_eval=true; shift ;;
        --offline-wandb) wandb_mode="offline"; shift ;;
        --disable-wandb) wandb_mode="disabled"; shift ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help)
            usage
            exit 0
            ;;
        *) die "Unknown argument: $1" ;;
    esac
done

[[ -n "${experiment}" ]] || die "--experiment is required."
[[ "${experiment}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "Experiment name may contain only letters, numbers, dot, underscore, dash."
[[ "${eval_split}" == "test" || "${eval_split}" == "validation" || "${eval_split}" == "all" ]] \
    || die "--eval-split must be test, validation, or all."
[[ ! ("${skip_train}" == true && "${skip_eval}" == true) ]] \
    || die "Both train and eval are disabled; there is nothing to run."

if [[ -z "${output_dir}" ]]; then
    output_dir="outputs/${experiment}"
fi
if [[ "${output_dir}" != /* ]]; then
    output_dir="${PROJECT_ROOT}/${output_dir}"
fi
readonly output_dir
readonly final_model_dir="${output_dir}/final"

declare -a wandb_overrides=()
case "${wandb_mode}" in
    online) ;;
    offline) wandb_overrides+=("trainer.wandb_mode=offline") ;;
    disabled)
        wandb_overrides+=("trainer.report_to=none" "trainer.wandb_mode=disabled")
        ;;
esac

cd "${PROJECT_ROOT}"

echo "Experiment: ${experiment}"
echo "Output:     ${output_dir}"

if [[ "${skip_sync}" == false ]]; then
    run_command uv sync
fi
if [[ "${skip_dataset}" == false ]]; then
    run_command "${SCRIPT_DIR}/download_vimd.sh"
fi
if [[ "${skip_tests}" == false ]]; then
    run_command uv run pytest -q
fi

if [[ "${skip_train}" == false ]]; then
    train_command=(
        uv run python run.py
        "mode=train"
        "evaluate_after_train=false"
        "trainer.output_dir=${output_dir}"
        "trainer.run_name=${experiment}-train"
        "trainer.wandb_group=${experiment}"
        "trainer.wandb_tags=[vimd,${experiment},train]"
        "trainer.wandb_log_model=end"
    )
    if [[ -n "${resume_checkpoint}" ]]; then
        train_command+=("checkpoint=${resume_checkpoint}")
    fi
    train_command+=("${wandb_overrides[@]}" "${train_overrides[@]}")
    run_command "${train_command[@]}"
fi

if [[ "${skip_eval}" == true ]]; then
    exit 0
fi

if [[ -z "${eval_checkpoint}" ]]; then
    eval_checkpoint="${final_model_dir}"
fi
if [[ "${dry_run}" == false && "${eval_checkpoint}" == "${final_model_dir}" && ! -d "${eval_checkpoint}" ]]; then
    die "Final checkpoint not found: ${eval_checkpoint}. Train first or use --eval-checkpoint."
fi

if [[ "${eval_split}" == "all" ]]; then
    eval_splits=(validation test)
else
    eval_splits=("${eval_split}")
fi

for split in "${eval_splits[@]}"; do
    eval_command=(
        uv run python run.py
        "mode=eval"
        "split=${split}"
        "checkpoint=${eval_checkpoint}"
        "trainer.output_dir=${output_dir}"
        "trainer.run_name=${experiment}-eval-${split}"
        "trainer.wandb_group=${experiment}"
        "trainer.wandb_tags=[vimd,${experiment},eval,${split}]"
        "trainer.wandb_log_model=false"
    )
    eval_command+=("${wandb_overrides[@]}" "${eval_overrides[@]}")
    run_command "${eval_command[@]}"
done
