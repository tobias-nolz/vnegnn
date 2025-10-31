#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/data}"
SCRIPTS="$PROJECT_ROOT/scripts"

# Make glob patterns that don't match expand to nothing (avoid literal pattern)
shopt -s nullglob

# --- Defaults ---------------------------------------------------------------
# Detect number of CPU cores in a portable way (fallback to 1)
_detect_nproc() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v getconf >/dev/null 2>&1; then
        getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu 2>/dev/null || echo 1
    else
        echo 1
    fi
}

JOBS="$(_detect_nproc)"
DEVICE=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -j, --jobs NUM         number of threads/cores to use (default: detected)
  -d, --device DEVICE    device id/name to pass to embedding script (optional)
      --data-root PATH   override data root (default: $DATA_ROOT)
  -h, --help             show this help message

Notes:
- The script passes -j/--jobs to both Python preprocessing scripts.
- The --device option is forwarded to the embedding script only.
EOF
}

# --- Parse arguments -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs)
            if [[ -n "${2:-}" && ${2:0:1} != "-" ]]; then
                JOBS="$2"
                shift 2
            else
                echo "Error: --jobs requires a numeric argument" >&2
                exit 1
            fi
            ;;
        -d|--device)
            if [[ -n "${2:-}" && ${2:0:1} != "-" ]]; then
                DEVICE="$2"
                shift 2
            else
                echo "Error: --device requires an argument" >&2
                exit 1
            fi
            ;;
        --data-root)
            if [[ -n "${2:-}" && ${2:0:1} != "-" ]]; then
                DATA_ROOT="$2"
                shift 2
            else
                echo "Error: --data-root requires a path argument" >&2
                exit 1
            fi
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# --- Main loop: iterate every "raw" subfolder under each dataset -----------
# Pattern: $DATA_ROOT/<dataset>/raw/
for raw_folder in "$DATA_ROOT"/*/raw/; do
    if [ -d "$raw_folder" ]; then
        echo "Processing raw folder: $raw_folder"

        # Build embedding command and optionally include device
        embed_cmd=(python3 "$SCRIPTS/generate_esm_embeddings.py"
                   -p "$raw_folder"
                   -j "$JOBS")
        if [ -n "$DEVICE" ]; then
            embed_cmd+=(--device "$DEVICE")
        fi

        "${embed_cmd[@]}"

        # Run binding info extraction (threads forwarded)
        extract_cmd=(python3 "$SCRIPTS/extract_binding_info.py"
                     -p "$raw_folder"
                     -j "$JOBS"
                     -t 4.0
                     -b processes)
        "${extract_cmd[@]}"
    fi
done

echo "All raw folders processed."

