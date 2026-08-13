#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

readonly REPO_ID="${VIMD_REPO_ID:-nguyendv02/ViMD_Dataset}"
# Pin the snapshot already used by this project for reproducible experiments.
readonly REVISION="${VIMD_REVISION:-3a5b30157034e7eadd5c75fae1a820c6f9383398}"
readonly DESTINATION="${VIMD_DEST:-${PROJECT_ROOT}/data/ViMD_Dataset}"
readonly MAX_WORKERS="${VIMD_MAX_WORKERS:-4}"
readonly DOWNLOAD_ATTEMPTS="${VIMD_DOWNLOAD_ATTEMPTS:-5}"
readonly XET_ATTEMPTS="${VIMD_XET_ATTEMPTS:-2}"
readonly RETRY_DELAY_SECONDS="${VIMD_RETRY_DELAY_SECONDS:-15}"

readonly EXPECTED_TRAIN_SHARDS=103
readonly EXPECTED_VALID_SHARDS=13
readonly EXPECTED_TEST_SHARDS=14

usage() {
    cat <<'EOF'
Download the pinned ViMD dataset snapshot from Hugging Face.

Usage:
  scripts/download_vimd.sh
  scripts/download_vimd.sh --check-only

Environment overrides:
  VIMD_DEST         Local destination directory.
  VIMD_REVISION     Hugging Face branch/tag/commit (default: pinned commit).
  VIMD_MAX_WORKERS  Concurrent downloads (default: 4).
  VIMD_DOWNLOAD_ATTEMPTS
                    Total resumable attempts (default: 5).
  VIMD_XET_ATTEMPTS Attempts through Xet before HTTP fallback (default: 2).
  VIMD_RETRY_DELAY_SECONDS
                    Delay between attempts (default: 15).
  VIMD_REPO_ID      Dataset repository ID.
  HF_TOKEN          Hugging Face token, only when authentication is required.
  HF_HUB_DISABLE_XET
                    Set to 1 to use HTTP immediately instead of Xet.

The download is resumable: retries and reruns reuse completed local files.
EOF
}

count_shards() {
    local pattern="$1"
    find "${DESTINATION}/data" -maxdepth 1 -type f -name "${pattern}" 2>/dev/null \
        | wc -l \
        | tr -d '[:space:]'
}

validate_dataset() {
    local train_count valid_count test_count empty_count

    if [[ ! -f "${DESTINATION}/README.md" ]]; then
        echo "ERROR: Missing ${DESTINATION}/README.md" >&2
        return 1
    fi
    if [[ ! -d "${DESTINATION}/data" ]]; then
        echo "ERROR: Missing ${DESTINATION}/data" >&2
        return 1
    fi

    train_count="$(count_shards 'train-*.parquet')"
    valid_count="$(count_shards 'valid-*.parquet')"
    test_count="$(count_shards 'test-*.parquet')"
    empty_count="$(find "${DESTINATION}/data" -maxdepth 1 -type f -name '*.parquet' -size 0 | wc -l | tr -d '[:space:]')"

    echo "ViMD shard validation:"
    echo "  train: ${train_count}/${EXPECTED_TRAIN_SHARDS}"
    echo "  valid: ${valid_count}/${EXPECTED_VALID_SHARDS}"
    echo "  test:  ${test_count}/${EXPECTED_TEST_SHARDS}"

    if [[ "${train_count}" -ne "${EXPECTED_TRAIN_SHARDS}" \
        || "${valid_count}" -ne "${EXPECTED_VALID_SHARDS}" \
        || "${test_count}" -ne "${EXPECTED_TEST_SHARDS}" ]]; then
        echo "ERROR: ViMD download is incomplete. Rerun this script to resume." >&2
        return 1
    fi
    if [[ "${empty_count}" -ne 0 ]]; then
        echo "ERROR: Found ${empty_count} empty parquet file(s)." >&2
        return 1
    fi

    echo "OK: ViMD is complete at ${DESTINATION}"
}

check_only=false
case "${1:-}" in
    "") ;;
    --check-only) check_only=true ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "ERROR: Unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
esac

if [[ "${check_only}" == true ]]; then
    validate_dataset
    exit $?
fi

if ! [[ "${MAX_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: VIMD_MAX_WORKERS must be a positive integer." >&2
    exit 2
fi
if ! [[ "${DOWNLOAD_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: VIMD_DOWNLOAD_ATTEMPTS must be a positive integer." >&2
    exit 2
fi
if ! [[ "${XET_ATTEMPTS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: VIMD_XET_ATTEMPTS must be a non-negative integer." >&2
    exit 2
fi
if ! [[ "${RETRY_DELAY_SECONDS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: VIMD_RETRY_DELAY_SECONDS must be a non-negative integer." >&2
    exit 2
fi

cd "${PROJECT_ROOT}"
mkdir -p "${DESTINATION}"

if command -v uv >/dev/null 2>&1; then
    downloader=(uv run hf)
elif command -v hf >/dev/null 2>&1; then
    downloader=(hf)
else
    echo "ERROR: Neither 'uv' nor the Hugging Face 'hf' CLI is available." >&2
    echo "Install project dependencies first with: uv sync" >&2
    exit 127
fi

echo "Repository:  ${REPO_ID}"
echo "Revision:    ${REVISION}"
echo "Destination: ${DESTINATION}"
echo "Expected size is approximately 56 GB. Existing files will be reused."

download_dataset() {
    "${downloader[@]}" download "${REPO_ID}" \
        --repo-type dataset \
        --revision "${REVISION}" \
        --include "README.md" \
        --include "data/*.parquet" \
        --local-dir "${DESTINATION}" \
        --max-workers "${MAX_WORKERS}"
}

download_complete=false
for ((attempt = 1; attempt <= DOWNLOAD_ATTEMPTS; attempt++)); do
    if [[ "${HF_HUB_DISABLE_XET:-0}" == "1" || "${attempt}" -gt "${XET_ATTEMPTS}" ]]; then
        export HF_HUB_DISABLE_XET=1
        transfer_backend="HTTP (Xet disabled)"
    else
        transfer_backend="Xet"
    fi

    echo "Download attempt ${attempt}/${DOWNLOAD_ATTEMPTS} via ${transfer_backend}."
    if download_dataset; then
        download_complete=true
        break
    fi

    if [[ "${attempt}" -lt "${DOWNLOAD_ATTEMPTS}" ]]; then
        echo "WARNING: Download attempt failed; completed files are preserved." >&2
        echo "Retrying in ${RETRY_DELAY_SECONDS}s..." >&2
        sleep "${RETRY_DELAY_SECONDS}"
    fi
done

if [[ "${download_complete}" != true ]]; then
    echo "ERROR: ViMD download failed after ${DOWNLOAD_ATTEMPTS} attempts." >&2
    echo "Rerun the same command to resume the remaining files." >&2
    exit 1
fi

validate_dataset
