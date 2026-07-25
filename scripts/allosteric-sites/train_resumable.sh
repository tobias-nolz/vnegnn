#!/usr/bin/env bash
# Resume-on-crash wrapper for the joint VN-EGNN run.
#
# ModelCheckpoint writes `last.ckpt` into a per-run timestamped dir
# (${paths.output_dir}/checkpoints), so a crash otherwise means restarting from epoch 0.
# This relaunches from the newest `last.ckpt` after a non-zero exit, up to MAX_RETRIES.
# Lightning restores epoch/optimizer/callback state from the checkpoint; a fresh W&B run
# is started each attempt (logging fragments across attempts, training does not).
#
# Usage:
#   scripts/allosteric-sites/train_resumable.sh [extra hydra overrides...]
set -u

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MAX_RETRIES="${MAX_RETRIES:-5}"
RUNS_DIR="logs/train/runs"
PY="${PY:-python}"

latest_last_ckpt() {
    ls -t "${RUNS_DIR}"/*/checkpoints/last.ckpt 2>/dev/null | head -1
}

attempt=0
ckpt_arg=""
while [ "${attempt}" -le "${MAX_RETRIES}" ]; do
    if [ "${attempt}" -eq 0 ]; then
        echo ">>> Launching joint training (attempt 0, fresh)"
    else
        ckpt="$(latest_last_ckpt)"
        if [ -z "${ckpt}" ]; then
            echo ">>> No last.ckpt found to resume from; aborting."
            exit 1
        fi
        echo ">>> Resuming from ${ckpt} (attempt ${attempt}/${MAX_RETRIES})"
        ckpt_arg="ckpt_path=${ckpt}"
    fi

    ${PY} src/train.py experiment=vnegnn_joint ${ckpt_arg} "$@"
    status=$?

    if [ "${status}" -eq 0 ]; then
        echo ">>> Training finished cleanly (exit 0)."
        exit 0
    fi

    echo ">>> Training exited with status ${status}."
    attempt=$((attempt + 1))
done

echo ">>> Exhausted ${MAX_RETRIES} retries; giving up."
exit 1
