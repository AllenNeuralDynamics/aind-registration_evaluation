"""
File that defines the constants used
in the package
"""

from pathlib import Path
from typing import Union

import dask.array as da
import numpy as np

# IO types
PathLike = Union[str, Path]
ArrayLike = Union[da.Array, np.ndarray]
