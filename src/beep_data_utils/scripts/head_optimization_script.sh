#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
BASE_OUTPUT_PATH="${PROJECT_ROOT}/tmp/beep-predictions"

# Array of cycle windows to try
CYCLE_WINDOWS=(10 20 50 100)

# Array of models to try
MODELS=("random_forest" "ebm")

# Array of cross-validation splits to try
CV_SPLITS=(3 5 10 15 20)

# Parameter grids
RF_PARAMS='{
    "random_forest": {
        "n_estimators": [50, 100, 200, 400, 800],
        "max_depth": [10, 20, 30, 50, 80, 100],
        "min_samples_split": [2, 5, 10],
        "n_jobs": [-1]
    }
}'

ELASTIC_NET_PARAMS='{
    "elastic_net": {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "max_iter": [2000]
    }
}'

RIDGE_PARAMS='{
    "ridge": {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
        "max_iter": [2000]
    }
}'

EBM_PARAMS='{
    "ebm": {
        "max_rounds": [100, 200, 500, 1000],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_samples_leaf": [2, 5, 10, 20],
        "max_bins": [256]
    }
}'

# Function to run training for a specific configuration
run_training() {
    local cycle_window=$1
    local model=$2
    local params=$3
    local n_splits=$4
    local output_dir="${BASE_OUTPUT_PATH}/window_${cycle_window}/splits_${n_splits}/${model}"

    mkdir -p "${output_dir}"

    echo "Running training for:"
    echo "  Cycle window: ${cycle_window}"
    echo "  CV splits: ${n_splits}"
    echo "  Model: ${model}"
    echo "  Output directory: ${output_dir}"

    poetry run beep-head-training-and-evaluation \
        --train_data_path "/embeddings/train.csv" \
        --val_data_path "/embeddings/validation.csv" \
        --test_data_path "/embeddings/test.csv" \
        --output_path "${output_dir}" \
        --target_name "target_cycle_life" \
        --cycle_cutoff 100 \
        --cv_type "kfold" \
        --n_splits "${n_splits}" \
        --model_parameters_grid "${params}" \
        --dataset_metadata_path "${PROJECT_ROOT}//dataset_metadata.json" \
        --random_state 42 \
        --avg_cycles True \
        --model_name "${model}" \
        --cycle_window "${cycle_window}"
}

# Run all combinations
for cycle_window in "${CYCLE_WINDOWS[@]}"; do
    for n_splits in "${CV_SPLITS[@]}"; do
        for model in "${MODELS[@]}"; do
            case ${model} in
                "random_forest")
                    run_training ${cycle_window} ${model} "${RF_PARAMS}" ${n_splits}
                    ;;
                "elastic_net")
                    run_training ${cycle_window} ${model} "${ELASTIC_NET_PARAMS}" ${n_splits}
                    ;;
                "ridge")
                    run_training ${cycle_window} ${model} "${RIDGE_PARAMS}" ${n_splits}
                    ;;
                "ebm")
                    run_training ${cycle_window} ${model} "${EBM_PARAMS}" ${n_splits}
                    ;;
            esac
        done
    done
done

# Create Python script for analysis
cat > "${BASE_OUTPUT_PATH}/analyze_results.py" << 'EOF'
import pandas as pd
import glob
import os
import numpy as np

def find_best_model():
    base_path = "/src/tmp/beep-predictions"
    results = []

    print(f"Searching for results in: {base_path}")

    # Walk through all results directories
    window_dirs = glob.glob(f"{base_path}/window_*")

    for window_dir in window_dirs:
        window_size = int(os.path.basename(window_dir).split('_')[1])

        # Get splits directories
        splits_dirs = glob.glob(f"{window_dir}/splits_*")
        for splits_dir in splits_dirs:
            n_splits = int(os.path.basename(splits_dir).split('_')[1])

            # Look in each model type directory
            model_dirs = glob.glob(f"{splits_dir}/*/")

            for model_dir in model_dirs:
                model_name = os.path.basename(os.path.dirname(model_dir))

                # Look in the nested model directory for results.json
                nested_model_dir = glob.glob(f"{model_dir}/{model_name}/")
                if nested_model_dir:
                    results_path = os.path.join(nested_model_dir[0], "results.json")

                    try:
                        if os.path.exists(results_path):
                            results_df = pd.read_json(results_path, orient='index').T
                            results_df['window_size'] = window_size
                            results_df['n_splits'] = n_splits
                            results_df['model'] = model_name
                            results.append(results_df)
                            print(f"Loaded results: {model_name}, window={window_size}, splits={n_splits}")
                    except Exception as e:
                        print(f"Error loading {results_path}: {str(e)}")
                        continue

    if not results:
        print("No results found!")
        return

    # Combine all results
    all_results = pd.concat(results, ignore_index=True)

    # Find best model based on validation RMSE
    best_idx = all_results['test_rmse'].idxmin()
    best_model = all_results.iloc[best_idx]

    print("\n=== Best Model Configuration (Based on Validation RMSE) ===")
    print(f"Model: {best_model['model']}")
    print(f"Cycle Window: {best_model['window_size']}")
    print(f"CV Splits: {best_model['n_splits']}")
    print("\nMetrics:")
    print(f"Validation RMSE: {best_model['val_rmse']:.4f}")
    print(f"Test RMSE: {best_model['test_rmse']:.4f}")
    print(f"Test R²: {best_model['test_r2']:.4f}")

    # Show top 5 models
    print("\n=== Top 5 Models (by Validation RMSE) ===")
    top_5 = all_results.nsmallest(5, 'val_rmse')
    for idx, model in top_5.iterrows():
        print(f"\nRank {idx + 1}:")
        print(f"Model: {model['model']}")
        print(f"Window Size: {model['window_size']}")
        print(f"CV Splits: {model['n_splits']}")
        print(f"Validation RMSE: {model['val_rmse']:.4f}")
        print(f"Test RMSE: {model['test_rmse']:.4f}")
        print(f"Test R²: {model['test_r2']:.4f}")

    # Save detailed results
    output_file = os.path.join(base_path, "all_model_results.csv")
    all_results.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

    # Additional analysis by different groupings
    print("\n=== Analysis by Cross-validation Splits ===")
    splits_analysis = all_results.groupby('n_splits').agg({
        'val_rmse': ['mean', 'std', 'min'],
        'test_rmse': ['mean', 'std', 'min']
    }).round(4)
    print(splits_analysis)

    print("\n=== Analysis by Window Size ===")
    window_analysis = all_results.groupby('window_size').agg({
        'val_rmse': ['mean', 'std', 'min'],
        'test_rmse': ['mean', 'std', 'min']
    }).round(4)
    print(window_analysis)

    print("\n=== Analysis by Model Type ===")
    model_analysis = all_results.groupby('model').agg({
        'val_rmse': ['mean', 'std', 'min'],
        'test_rmse': ['mean', 'std', 'min']
    }).round(4)
    print(model_analysis)

    # Analyze statistical significance of differences between models
    print("\n=== Statistical Analysis of Model Differences ===")
    from scipy import stats

    models = all_results['model'].unique()
    print("RMSE comparison between models (p-values):")
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1_rmse = all_results[all_results['model'] == models[i]]['val_rmse']
            model2_rmse = all_results[all_results['model'] == models[j]]['val_rmse']
            t_stat, p_val = stats.ttest_ind(model1_rmse, model2_rmse)
            print(f"{models[i]} vs {models[j]}: p = {p_val:.4f}")

if __name__ == "__main__":
    find_best_model()
EOF

# Run analysis
python3 "${BASE_OUTPUT_PATH}/analyze_results.py"