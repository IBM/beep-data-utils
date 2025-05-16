# BEEP dataset preparation and training scripts

This repository contains a python package that implements utilities related to data preparation and training for the BEEP (Battery Evaluation and Early Prediction) dataset(s). 

Dependencies are managed with poetry:

````
poetry install
````

## Workflow Overview

The process involves three key stages:

1.  **Dataset Generation (Optional but Recommended):** Convert raw BEEP pickle files into a unified CSV format suitable for the preprocessor.
2.  **Dataset Splitting & Preprocessing:** Take the unified CSV (or multiple input CSVs) and generate the necessary train, validation, and test splits, applying preprocessing steps like cycle-based averaging and NaN masking.
3.  **Training & Evaluation:** Train a prediction model (e.g., a Random Forest Regressor) using raw values (or embeddings) to predict a specific target, such as End-of-Life (cycle life). Note that embedding generation techniques are not implemented in this repo.

## Detailed Steps

### 1. Dataset Generation (Pickle to CSV)

This step converts the raw data into a single, large CSV file.

**Prerequisite:** You need to obtain the original BEEP pickle files. Follow the instructions provided by the [original data providers](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation) to generate these files (e.g., `batch1.pkl`, `batch2.pkl`, etc.).

**Command:**

```bash
# Ensure you have poetry installed and are in the project root

poetry run beep-dataset-creation \
    --pickle-data-path "/path/to/pickle/data" \
    --uniformed-dataset-path "output/beep_unified_dataset.csv" \
    --chunk-size 1000000 # Adjust based on memory
```
### 2. Dataset Splitting & Preprocessing
This step takes one or more input CSV datasets, performs preprocessing (like cycle-based averaging), splits them into train/validation/test sets based on device IDs, and prepares them for the training.

**Example Command (using multiple input files and cycle averaging):**
```bash
#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)" # Adjust depth if needed

# Define paths to your input CSV datasets
# Example using BEEP main, secondary, NCA/NCM
DATA_DIR="/path/to/your/prepared/csv/data" # CHANGE THIS
BASE_DATA_PATH="${DATA_DIR}/beep_main_ds_quartiles.csv"
SECONDARY_DATA_PATH_TEST="${DATA_DIR}/test_secondary_beep_dataset_quartile_charge_policy.csv"
THIRD_DATA_PATH="${DATA_DIR}/nca_ncm_cycle_life.csv"
# FOURTH_DATA_PATH="${DATA_DIR}/" # Add more datasets

OUTPUT_PATH="${PROJECT_ROOT}/tmp/MY_BEEP_EXPERIMENT/run1" # CHANGE THIS

# Tells the preprocessor which columns contain device-level metadata
export DIGITAL_TWIN_SETTINGS_DATA_HANDLER_DEVICE_DESIGN_DATA_COLUMNS='{"device":["battery_chemistry"]}'
# Tells the preprocessor which of the above are categorical
export DIGITAL_TWIN_SETTINGS_DATA_HANDLER_CATEGORICAL_DEVICE_DESIGN_DATA_COLUMNS='{"device":["battery_chemistry"]}'

mkdir -p "${OUTPUT_PATH}"

poetry run beep-dataset-generator \
     "${BASE_DATA_PATH}" "${THIRD_DATA_PATH}" "${SECONDARY_DATA_PATH_TEST}" \
    --sampling_frequencies "cycle,cycle,cycle" \
    --target_variables "cycle_life" \
    --uniformed_dataset_path "${OUTPUT_PATH}/uniformed_dataset.csv" \
    --pretraining_output_path "${OUTPUT_PATH}/pretraining-data" \
    --drop_cycle_zero \
    --mask_nan_values \
    --unpaired_dataset_output_path "${OUTPUT_PATH}/unpaired.csv" \
    --cycle_based_averaging # Use cycle-based aggregation

echo "Preprocessing complete. Output located in ${OUTPUT_PATH}"

```
CLI Options:

- testing_data_paths (Positional Arguments): Paths to the input CSV files.
- sampling_frequencies: How to handle time-series data for each input file (e.g., cycle for cycle-based aggregation, 10s for time-based resampling, none to skip). Must match the number of input files if specified.
- target_variables: Comma-separated list of column names to be treated as targets (e.g., cycle_life).
- pretraining_output_path: Directory where the final train.csv, validation.csv, test.csv, and dataset_metadata.json will be saved for the FM.
- seed: Random seed for splitting.
- drop_cycle_zero: Flag to remove data from cycle 0 (as its often 0).
- mask_nan_values: Flag to replace NaNs with a special mask token.
- cycle_based_averaging: Flag to perform cycle-level aggregation (calculating mean, min, max, etc. per cycle). If used, --sampling_frequencies should likely be set to cycle or none.
- unpaired_dataset_output_path: Optional path to save data suitable for unpaired pre-training tasks.
- training_data_fraction: Proportion of devices used for training (default: 0.8). The rest is split between validation and test (respecting out_of_domain_devices_path).
- sampling_frequency: Overall sampling frequency for the generated instances.

### 3. Head Training & Evaluation
Train a downstream model (e.g., Random Forest) on the generated embeddings to predict the target variable (e.g., cycle life).

**Command:**
```bash
#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)" # Adjust depth

# --- Configuration ---
# Path to the directory containing the embedding CSVs from Step 4
BASEPATH="${PROJECT_ROOT}/tmp/MY_BEEP_EXPERIMENT/embeddings" # CHANGE THIS
# Output directory for head model results
OUTPUT_PATH="${BASEPATH}/cycle_life_head/run1" # CHANGE THIS

# Target variable name (must exist in the embedding CSVs)
TARGET_NAME="target_cycle_life"
CYCLE_CUTOFF=100

mkdir -p "${OUTPUT_PATH}"

poetry run beep-head-training-and-evaluation \
    --train_data_path "${BASEPATH}/train_embeddings.csv" \
    --val_data_path "${BASEPATH}/validation_embeddings.csv" \
    --test_data_path "${BASEPATH}/test_embeddings.csv" \
    --output_path "${OUTPUT_PATH}" \
    --target_name "${TARGET_NAME}" \
    --cycle_cutoff ${CYCLE_CUTOFF} \
    --cv_type "kfold" \
    --n_splits 5 \
    --random_state 42 \
    # --avg_cycles False \ # Option to average embeddings over windows, default False
    # --cycle_window 10 \ # Used if avg_cycles=True
    --run_diagnostics False # Set to True to run data distribution checks

```
