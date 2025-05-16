import shutil
import tempfile

import importlib_resources
import pandas as pd
from beep_data_utils.data.preprocessor import BeepDataPreprocessor


def test_read_experiments():
    with importlib_resources.as_file(
        importlib_resources.files("beep_data_utils") / "resources/beep_dataset.csv"
    ) as base_data_path:
        preprocessor = BeepDataPreprocessor(
            testing_data_paths=[base_data_path],
            sampling_frequencies={str(base_data_path): None},
            target_variables=["voltage", "cycle_life"],
        )

        assert len(preprocessor.experiments) > 0
        experiment = preprocessor.experiments[0]
        op_conds = experiment.operating_conditions
        assert len(op_conds) > 0
        assert "voltage" not in op_conds.columns
        assert "target_voltage" in op_conds.columns
        assert "current" in op_conds.columns
        assert "target_cycle_life" in op_conds.columns
        assert "temperature" in op_conds.columns


def test_experiment_to_instances():
    with importlib_resources.as_file(
        importlib_resources.files("beep_data_utils") / "resources/beep_dataset.csv"
    ) as base_data_path:
        preprocessor = BeepDataPreprocessor(
            testing_data_paths=[base_data_path],
            sampling_frequencies={str(base_data_path): None},
            target_variables=["voltage", "cycle_life"],
        )

        experiment = preprocessor.experiments[0]
        instances = preprocessor.experiment_to_instances(
            experiment=experiment,
            padded_unit_tokens_length=5,
            padding_token=0,
            mask_nan_values=False,
        )

        # Check basic structure
        assert isinstance(instances, pd.DataFrame)
        assert len(instances) > 0
        expected_columns = [
            "cycle_number",
            "current",
            "charge_capacity",
            "discharge_capacity",
            "temperature",
            "time",
            "discharge_dQdV",
            "token_0",
            "target_voltage",
            "target_cycle_life",
        ]
        for col in expected_columns:
            assert col in instances.columns


def test_generate_dataset(tmp_path):
    with importlib_resources.as_file(
        importlib_resources.files("beep_data_utils") / "resources/beep_dataset.csv"
    ) as base_data_path:
        preprocessor = BeepDataPreprocessor(
            testing_data_paths=[base_data_path],
            sampling_frequencies={str(base_data_path): "60s"},
            target_variables=["voltage", "cycle_life"],
        )

        output_path = tmp_path / "output.csv"
        _, _ = preprocessor.generate_dataset(
            output_path=output_path,
            padding_token=0,
            mask_nan_values=False,
        )
        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert len(df) > 0

        essential_columns = [
            "cycle_number",
            "current",
            "charge_capacity",
            "discharge_capacity",
            "temperature",
            "time",
            "target_voltage",
            "target_cycle_life",
        ]
        for col in essential_columns:
            assert col in df.columns


def test_sampling_frequency():
    """Test that sampling frequency reduces data points appropriately."""
    with importlib_resources.as_file(
        importlib_resources.files("beep_data_utils") / "resources/beep_dataset.csv"
    ) as base_data_path:
        # First get baseline without sampling
        base_preprocessor = BeepDataPreprocessor(
            testing_data_paths=[base_data_path],
            sampling_frequencies={str(base_data_path): None},
            target_variables=["voltage", "cycle_life"],
        )
        experiment = base_preprocessor.experiments[0]
        baseline_instances = base_preprocessor.experiment_to_instances(
            experiment=experiment,
            padded_unit_tokens_length=5,
            padding_token=0,
        )

        sampling_frequencies = ["0.1S", "10S"]
        previous_length = len(baseline_instances)

        for freq in sampling_frequencies:
            sampled_preprocessor = BeepDataPreprocessor(
                testing_data_paths=[base_data_path],
                sampling_frequencies={str(base_data_path): freq},
                target_variables=["voltage", "cycle_life"],
            )
            experiment = sampled_preprocessor.experiments[0]
            sampled_instances = sampled_preprocessor.experiment_to_instances(
                experiment=experiment,
                padded_unit_tokens_length=5,
                padding_token=0,
            )

            assert len(sampled_instances) < previous_length

            essential_columns = [
                "cycle_number",
                "current",
                "charge_capacity",
                "discharge_capacity",
                "temperature",
                "time",
                "target_voltage",
                "target_cycle_life",
            ]
            for col in essential_columns:
                assert col in sampled_instances.columns
                assert not sampled_instances[col].isna().all()

            assert (sampled_instances["time"].diff()[1:] >= 0).all()

            previous_length = len(sampled_instances)


def test_sampling_in_generate_dataset(tmp_path):
    """Test sampling through the complete generate_dataset pipeline."""
    with importlib_resources.as_file(
        importlib_resources.files("beep_data_utils") / "resources/beep_dataset.csv"
    ) as base_data_path:
        # First generate without sampling
        base_preprocessor = BeepDataPreprocessor(
            testing_data_paths=[base_data_path],
            sampling_frequencies={str(base_data_path): None},
            target_variables=["voltage", "cycle_life"],
        )

        base_output_path = tmp_path / "base_output.csv"
        sampled_output_path = tmp_path / "sampled_output.csv"

        base_preprocessor.generate_dataset(
            output_path=base_output_path,
            padding_token=0,
        )

        sampled_preprocessor = BeepDataPreprocessor(
            testing_data_paths=[base_data_path],
            sampling_frequencies={str(base_data_path): "60S"},
            target_variables=["voltage", "cycle_life"],
        )
        sampled_preprocessor.generate_dataset(
            output_path=sampled_output_path,
            padding_token=0,
        )

        base_df = pd.read_csv(base_output_path)
        sampled_df = pd.read_csv(sampled_output_path)

        assert len(sampled_df) < len(base_df)

        essential_columns = [
            "cycle_number",
            "current",
            "charge_capacity",
            "discharge_capacity",
            "temperature",
            "time",
            "target_voltage",
            "target_cycle_life",
        ]
        for col in essential_columns:
            assert col in sampled_df.columns
            assert not sampled_df[col].isna().all()


def test_multiple_datasets():
    """Test handling of multiple datasets with different sampling frequencies."""
    with importlib_resources.as_file(
        importlib_resources.files("beep_data_utils") / "resources/beep_dataset.csv"
    ) as base_data_path:
        with tempfile.NamedTemporaryFile() as temp_file:
            shutil.copy(base_data_path, temp_file.name)
            preprocessor = BeepDataPreprocessor(
                testing_data_paths=[base_data_path, temp_file.name],
                sampling_frequencies={
                    str(base_data_path): "0.01S",
                    str(temp_file.name): "10S",
                },
                target_variables=["cycle_life"],
            )

        assert len(preprocessor.experiments) > 0

        instances_1s = preprocessor.experiment_to_instances(
            experiment=preprocessor.experiments[0],
            padded_unit_tokens_length=5,
            padding_token=0,
        )

        instances_10s = preprocessor.experiment_to_instances(
            experiment=preprocessor.experiments[-1],
            padded_unit_tokens_length=5,
            padding_token=0,
        )

        assert len(instances_1s) > len(instances_10s)

        for instances in [instances_1s, instances_10s]:
            assert "target_cycle_life" in instances.columns
