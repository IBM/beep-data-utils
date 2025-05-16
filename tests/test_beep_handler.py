import importlib_resources
import numpy as np
from beep_data_utils.data.handler import BeepDataHandler


def test_read_experiments_drop_cycle_zero():
    with importlib_resources.as_file(
        importlib_resources.files("beep_data_utils") / "resources/beep_dataset.csv"
    ) as base_data_path:
        handler = BeepDataHandler()
        experiment_set = handler.read_experiments(
            testing_data_path=base_data_path,
            target_variables=["voltage", "cycle_life"],
            drop_cycle_zero=True,
        )

        for exp in experiment_set:
            cycle_numbers = exp.operating_conditions["cycle_number"].unique()
            assert 0 not in cycle_numbers
            assert all(cycle_numbers > 0)


def test_operating_conditions_structure():
    """Test structure of operating conditions data."""
    with importlib_resources.as_file(
        importlib_resources.files("beep_data_utils") / "resources/beep_dataset.csv"
    ) as base_data_path:
        handler = BeepDataHandler()
        experiment_set = handler.read_experiments(base_data_path, target_variables=["cycle_life"])

        required_columns = [
            "cycle_number",
            "current",
            "voltage",
            "charge_capacity",
            "discharge_capacity",
            "temperature",
            "time",
            "discharge_dQdV",
            "target_cycle_life",
        ]

        for exp in experiment_set:
            op_conds = exp.operating_conditions
            assert all(col in op_conds.columns for col in required_columns)


def test_experiment_set_attributes():
    with importlib_resources.as_file(
        importlib_resources.files("beep_data_utils") / "resources/beep_dataset.csv"
    ) as base_data_path:
        handler = BeepDataHandler()
        experiment_set = handler.read_experiments(
            base_data_path,
            target_variables=["voltage", "cycle_life"],
        )
        _ = experiment_set


def test_sampling_frequency_handling():
    """Test that sampling frequency is properly applied at the handler level."""
    with importlib_resources.as_file(
        importlib_resources.files("beep_data_utils") / "resources/beep_dataset.csv"
    ) as base_data_path:
        handler = BeepDataHandler()

        # Get baseline without sampling
        base_experiment_set = handler.read_experiments(
            testing_data_path=base_data_path,
            target_variables=["voltage", "cycle_life"],
            sampling_frequency=None,
        )

        sampled_experiment_set = handler.read_experiments(
            testing_data_path=base_data_path,
            target_variables=["voltage", "cycle_life"],
            sampling_frequency="60S",
        )

        # Compare numbers
        for base_exp, sampled_exp in zip(base_experiment_set, sampled_experiment_set):
            base_times = base_exp.operating_conditions["time"]
            sampled_times = sampled_exp.operating_conditions["time"]

            assert len(sampled_times) < len(base_times)

            assert (sampled_times.diff()[1:] > 0).all()

            essential_cols = [
                "current",
                "temperature",
                "target_voltage",
                "target_cycle_life",
            ]
            for col in essential_cols:
                assert not sampled_exp.operating_conditions[col].isna().all()


def test_charge_policy_quartiles():
    """Test that charge policy quartiles are present in the operating conditions."""
    with importlib_resources.as_file(
        importlib_resources.files("beep_data_utils") / "resources/beep_dataset.csv"
    ) as base_data_path:
        handler = BeepDataHandler()
        experiment_set = handler.read_experiments(
            testing_data_path=base_data_path, target_variables=["cycle_life"]
        )

        for experiment in experiment_set:
            op_conds = experiment.operating_conditions
            quartile_columns = [
                col for col in op_conds.columns if col.startswith("charge_policy_Q")
            ]

            # Verify Quartiles Q1-Q4 are present
            expected_quartiles = [f"charge_policy_Q{i}" for i in range(1, 5)]
            assert all(q in quartile_columns and q != np.nan for q in expected_quartiles)
