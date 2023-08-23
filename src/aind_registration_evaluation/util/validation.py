"""
Image validation functions
"""

import numpy as np

from aind_registration_evaluation._shared.types import ArrayLike


def validate_image_transform(
    image_1: ArrayLike, image_2: ArrayLike, transform_matrix: np.matrix
):
    """
    Validates the shape of the images as well as the
    transformation matrix

    Parameters
    -----------
    image_1: ArrayLike
        Provided image 1

    image_2: ArrayLike
        Provided image 2

    transform_matrix: np.matrix
        Transformation matrix that relates both images
        to the same image coordinate system
    """
    image_1_len = len(image_1.shape)
    image_2_len = len(image_2.shape)

    if (
        image_1_len == image_2_len
        and image_2_len != transform_matrix.shape[0] - 1
        and image_2_len + 1 != transform_matrix.shape[1]
    ):
        raise ValueError(
            f"""
            Transformation matrix with shape {transform_matrix.shape}
            does not match image dimensions {image_1_len}
            """
        )
