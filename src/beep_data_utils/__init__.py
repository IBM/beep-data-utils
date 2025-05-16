"""Package initialization."""

import importlib_metadata

__version__ = importlib_metadata.distribution(
    "beep_data_utils"
).version  # managed by poetry_bumpversion
