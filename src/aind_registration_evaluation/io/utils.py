"""
Utility functions for image readers
"""

from typing import List, Optional

import numpy as np

from aind_registration_evaluation._shared.types import ArrayLike


def add_leading_dim(data: ArrayLike):
    """
    Adds a new dimension to existing data.
    Parameters
    ------------------------
    arr: ArrayLike
        Dask/numpy array that contains image data.

    Returns
    ------------------------
    ArrayLike:
        Padded dask/numpy array.
    """

    return data[None, ...]


def extract_data(
    arr: ArrayLike, last_dimensions: Optional[int] = None
) -> ArrayLike:
    """
    Extracts n dimensional data (numpy array or dask array)
    given expanded dimensions.
    e.g., (1, 1, 1, 1600, 2000) -> (1600, 2000)
    e.g., (1, 1600, 2000) -> (1600, 2000)
    e.g., (1, 1, 2, 1600, 2000) -> (2, 1600, 2000)

    Parameters
    ------------------------
    arr: ArrayLike
        Numpy or dask array with image data. It is assumed
        that the last dimensions of the array contain
        the information about the image.

    last_dimensions: Optional[int]
        If given, it selects the number of dimensions given
        stating from the end
        of the array
        e.g., arr=(1, 1, 1600, 2000) last_dimensions=3 -> (1, 1600, 2000)
        e.g., arr=(1, 1, 1600, 2000) last_dimensions=1 -> (2000)

    Raises
    ------------------------
    ValueError:
        Whenever the last dimensions value is higher
        than the array dimensions.

    Returns
    ------------------------
    ArrayLike:
        Reshaped array with the selected indices.
    """

    if last_dimensions is not None:
        if last_dimensions > arr.ndim:
            raise ValueError(
                "Last dimensions should be lower than array dimensions"
            )

    else:
        last_dimensions = len(arr.shape) - arr.shape.count(1)

    dynamic_indices = [slice(None)] * arr.ndim

    for idx in range(arr.ndim - last_dimensions):
        dynamic_indices[idx] = 0

    return arr[tuple(dynamic_indices)]


def create_sample_data_2D(
    delta_x: Optional[int] = 3, delta_y: Optional[int] = 3
) -> List:
    """
    Function to create 2D dummy data.

    Parameters
    ------------------------

    delta_x: Optional[int]
        Translation over the x axis in the sample 2D data

    delta_y: Optional[int]
        Translation over the y axis in the sample 2D data

    Returns
    ------------------------
    List:
        List where the first two positions correspond to
        image 1 and 2 respectively and the last position
        correspond to the transformation matrix.
    """

    def value_func_3d(x, y):
        """values for xy grid"""
        return 1 * x + 1 * y

    x = np.linspace(0, 499, 500)
    y = np.linspace(0, 499, 500)

    points = (x, y)
    image = value_func_3d(*np.meshgrid(*points, indexing="ij"))
    transform = np.matrix([[1, 0, delta_y], [0, 1, delta_x], [0, 0, 1]])

    return [image, image, transform]


def create_sample_data_3D(
    delta_x: Optional[int] = 3,
    delta_y: Optional[int] = 3,
    delta_z: Optional[int] = 3,
) -> List:
    """
    Function to create 3D dummy data.

    Parameters
    ------------------------

    delta_x: Optional[int]
        Translation over the x axis in the sample 3D data

    delta_y: Optional[int]
        Translation over the y axis in the sample 3D data

    delta_z: Optional[int]
        Translation over the z axis in the sample 3D data

    Returns
    ------------------------
    List:
        List where the first two positions correspond to
        image 1 and 2 respectively and the last position
        correspond to the transformation matrix.
    """

    def value_func_3d(x, y, z):
        """values for xyz grid"""
        return 1 * x + 1 * y - z

    x = np.linspace(0, 499, 500)
    y = np.linspace(0, 499, 500)
    z = np.linspace(0, 199, 200)
    points = (x, y, z)
    image = value_func_3d(*np.meshgrid(*points, indexing="ij"))
    transform = np.matrix(
        [
            [1, 0, 0, delta_y],
            [0, 1, 0, delta_x],
            [0, 0, 1, delta_z],
            [0, 0, 0, 1],
        ]
    )

    return [image, image, transform]
