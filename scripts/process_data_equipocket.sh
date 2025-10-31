#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
DATA_ROOT="$PROJECT_ROOT/data/data"
SCRIPTS="$PROJECT_ROOT/scripts"

# Process all folders in data/equipocket
for folder in "$DATA_ROOT"/*; do
    if [ -d "$folder" ]; then
        echo "Processing folder: $folder"
        python3 "$SCRIPTS/protein_features.py" -i "$folder" -j "$(nproc)"
        python3 "$SCRIPTS/extract_binding_atoms.py" -p "$folder" -j "$(nproc)" -t 4.0 -b processes
    fi
done

echo "All folders processed."
