"""
Module to check image intersections
"""

from typing import Tuple

import numpy as np

from aind_registration_evaluation._shared.types import ArrayLike


def check_image_intersection_2D(
    bounds_1: ArrayLike, bounds_2: ArrayLike
) -> bool:
    """
    Checks if the provided boundaries share an
    area of intersection

    Parameters
    ------------------------

    bounds_1: ArrayLike
        Array with the position of the boundaries
        within image 1 in order X, Y.

    bounds_2: ArrayLike
        Array with the position of the boundaries
        within image 2 in order X, Y.

    Returns
    ------------------------
    Boolean:
        True if there is an intersection, False otherwise.
    """

    # X0 == X1 or Y0 == Y1 for each image then it is no image
    # For boundaries in each image
    for bound in [bounds_1, bounds_2]:
        if bound[0][0] == bound[1][0] or bound[0][1] == bound[1][1]:
            return False

    # If images are on sides of each other in X
    if bounds_1[0][0] >= bounds_2[1][0] or bounds_2[0][0] >= bounds_1[1][0]:
        return False

    # If images are on top of each other in Y
    if bounds_1[0][1] >= bounds_2[1][1] or bounds_2[0][1] >= bounds_1[1][1]:
        return False

    return True


def check_image_intersection_3D(
    bounds_1: ArrayLike, bounds_2: ArrayLike
) -> bool:
    """
    Checks if the provided boundaries share an area of intersection

    Parameters
    ------------------------

    bounds_1: np.ndarray
        Array with the position of the boundaries
        within image 1 in order X, Y, Z.

    bounds_2: np.ndarray
        Array with the position of the boundaries
        within image 2 in order X, Y, Z.

    Returns
    ------------------------
    Boolean:
        True if there is an intersection, False otherwise.
    """
    # X0 == X1 or Y0 == Y1 or Z0 == Z1 for each image then it is no image
    # For boundaries in each image
    for bound in [bounds_1, bounds_2]:
        if (
            bound[0][0] == bound[1][0]
            or bound[0][1] == bound[1][1]
            or bound[0][2] == bound[1][2]
        ):
            return False

    # TODO check offset by combined axis

    return True


def calculate_bounds(
    image_1_shape: Tuple, image_2_shape: Tuple, transform: np.ndarray
) -> Tuple:
    """
    Calculate bounds of coverage for two images and a transform
    where image1 is in its own coordinate system and image 2 is mapped
    to image 1's coords with the transform

    Parameters
    ------------------------
    image_1: Tuple
        First image which will be used as default
        in the coordinate system

    image_2: Tuple
        Second image which will be used to map it's
        position to a common coordinate system based on image_1

    transform: np.ndarray
        Transformation matrix applied over the two images

    Returns
    ------------------------
    Tuple:
        Tuple with the calculated boundaries.
        For 2D images the boundary axis order is: Y X
        For 3D images the boundary axis order is: Z Y X

    """

    dimensions_zeros = np.zeros(len(image_1_shape), dtype=np.int8)

    # First boundary
    bound_1 = np.array([dimensions_zeros, list(image_1_shape)])

    # Minimum point
    pt_min = np.matrix(np.append(dimensions_zeros, 1)).transpose()

    # Maximum point
    pt_max = np.matrix(np.array(list(image_2_shape) + [1])).transpose()

    # Getting coordinates for second image into image_1 coordinate system
    coord_1 = np.squeeze(
        transform * pt_min,
    ).tolist()[
        0
    ][:-1]

    coord_2 = np.squeeze(transform * pt_max).tolist()[0][:-1]

    bound_2 = np.array([coord_1, coord_2])

    return bound_1, bound_2
