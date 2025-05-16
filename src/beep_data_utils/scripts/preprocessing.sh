#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

export DATA_HANDLER_DEVICE_DESIGN_DATA_COLUMNS='{"device":["battery_chemistry"], "unit":[]}'
export DATA_HANDLER_CATEGORICAL_DEVICE_DESIGN_DATA_COLUMNS='{"device":["battery_chemistry"]}'

DATA_DIR="${PROJECT_ROOT}/src/beep_data_utils/resources"

# Define input paths
BASE_DATA_PATH="${DATA_DIR}/beep_dataset.csv"
SECONDARY_DATA_PATH_TEST="${DATA_DIR}/test_secondary_beep_dataset_small.csv"
SECONDARY_DATA_PATH="${DATA_DIR}/train_secondary_beep_dataset_small.csv"
# Add more datasets here

OUT_OF_DOMAIN_PATH="${PROJECT_ROOT}/src/beep_data_utils/resources/out_of_domain.txt"
OUTPUT_PATH="${PROJECT_ROOT}/src/tmp/beep"

mkdir -p "${OUTPUT_PATH}"

poetry run beep-dataset-generator \
    "${BASE_DATA_PATH}" "${SECONDARY_DATA_PATH}" "${SECONDARY_DATA_PATH_TEST}"\
    --sampling_frequencies "cycle,cycle,cycle" \
    --target_variables "cycle_life"\
    --uniformed_dataset_path "${OUTPUT_PATH}/uniformed_dataset.csv" \
    --pretraining_output_path "${OUTPUT_PATH}/pretraining-data" \
    --drop_cycle_zero \
    --mask_nan_values \
    --cycle_based_averaging \
    --unpaired_dataset_output_path "${OUTPUT_PATH}/unpaired.json"