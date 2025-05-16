#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
OUTPUT_PATH="${PROJECT_ROOT}/tmp/cycle-based-predictions"

mkdir -p "${OUTPUT_PATH}"

poetry run beep-head-training-and-evaluation \
    --train_data_path "${PROJECT_ROOT}/tmp/beep-embedded-data/train.csv" \
    --val_data_path "${PROJECT_ROOT}/tmp/beep-embedded-data/validation.csv" \
    --test_data_path "${PROJECT_ROOT}/tmp/beep-embedded-data/test.csv" \
    --output_path "${OUTPUT_PATH}" \
    --target_name "target_cycle_life" \
    --cycle_cutoff 100 \
    --cv_type "kfold" \
    --n_splits 5 \
    --random_state 42 \
    --avg_cycles False \
    --run_diagnostics True \
    --cycle_window 5