"""
Module to sample points in image
intersection areas
"""

from typing import List, Optional, Tuple

import numpy as np
from natsort import natsorted

from aind_registration_evaluation._shared.types import ArrayLike
from aind_registration_evaluation.util.intersection import (
    check_image_intersection_2D, check_image_intersection_3D)


def two_mult_closest(nums: List[int], target=int) -> List[int]:
    """
    Finds the closest two numbers from a list of
    numbers that multiply to a target

    Parameters
    ----------
    nums: List[int]
        List of nums from where we
        will find the two closests

    target: int
        Multiplicative target

    Returns
    ----------
    List[int]
        List contains a list of
        the two numbers that were found
    """
    nums = sorted(nums)
    nLimit = len(nums)

    if nLimit == 2:
        return [nums[0], nums[1]]

    elif nLimit <= 1:
        raise ValueError(
            f"Please, check the divisors to multiply. Received: {nums}"
        )

    left = nLimit // 2
    right = nLimit // 2

    while left >= 0 and right <= nLimit - 1:
        mult = nums[left] * nums[right]
        if mult == target:
            return [nums[left], nums[right]]

        elif mult < target:
            right += 1
        else:
            left -= 1

    return [nums[left], nums[right]]


def three_mult_closest(nums: List[int], target: int) -> List[int]:
    """
    Finds the closest three numbers from
    a list of numbers that multiply to
    a target

    Parameters
    ----------
    nums: List[int]
        List of nums from where we
        will find the two closests

    target: int
        Multiplicative target

    Returns
    ----------
    List[int]
        List contains a list of
        the two numbers that were found
    """

    nums = sorted(nums)
    n = len(nums)
    closest = 9999
    closest_selected = {target: []}
    selected_nums = []

    if n <= 3:
        return nums

    middle_pos = n // 2
    idx_iter = 0

    while middle_pos >= 0 and middle_pos <= n - 1 and idx_iter < n:
        lower_pos = middle_pos
        higher_pos = middle_pos

        while lower_pos >= 0 and higher_pos <= n - 1:
            mult = nums[lower_pos] * nums[middle_pos] * nums[higher_pos]
            selected_nums = [
                nums[lower_pos],
                nums[middle_pos],
                nums[higher_pos],
            ]

            if abs(closest - target) > abs(mult - target) and mult != target:
                closest = mult
                closest_selected[mult] = selected_nums.copy()

            if mult < target:
                higher_pos += 1

            elif mult > target:
                lower_pos -= 1

            else:
                return selected_nums

        if mult > target:
            middle_pos -= 1

        else:
            middle_pos += 1

        idx_iter += 1

    indices = natsorted(closest_selected)

    return closest_selected[indices[0]]


def get_multiplicatives(num_points: int, mode="2d") -> Tuple[int]:
    """
    Gets middle multiplicative divisors of a set of points.
    Helper function used to build the grid of points.

    Parameters
    ------------------------

    num_points: int
        Number of points that will be sampled in the
        intersection image as a grid.

    mode: str
        Mode to return number of multiplicatives
        that sum up to num_points value.
        Default: "2d". Possible options ["2d", "3d"]

    Raises
    ------------------------
    ValueError:
        In situations where num_points is negative or zero

    NotImplementedError:
        In siuations where the mode is not available

    Returns
    ------------------------
    Tuple:
        Middle multiplicative divisors of the set of
        points to be displayed as a grid.
    """
    mode = mode.casefold()

    if num_points <= 0:
        raise ValueError("Please, check the number of points.")

    if mode not in ["2d", "3d"]:
        raise NotImplementedError(f"Mode {mode} has not been implemented")

    divs = []

    for i in range(1, num_points):
        if num_points % i == 0 and i != 1:
            divs.append(i)

    if mode == "2d":
        new_n_points = two_mult_closest(nums=divs, target=num_points)

    else:
        new_n_points = three_mult_closest(nums=divs, target=num_points)

    closest_points = np.prod(np.array(new_n_points))

    if closest_points != num_points:
        print(f"Setting new number of points to {closest_points}")

    return new_n_points


def sample_nd_grid_points(
    bounds_1: List[int], bounds_2: List[int], numpoints: int
) -> List[List[int]]:
    """
    Samples 2 or 3 dimensional points with grid
    from within  the intersection of the two
    provided images using the transformation matrix

    Parameters
    ----------
    bounds_1: List[int]
        List containing the boundaries where image 1
        starts and ends in the shared coordinate system
        that connects both images

    bounds_2: List[int]
        List containing the boundaries where image 2
        starts and ends in the shared coordinate system
        that connects both images

    numpoints: int
        Number of points that will be sampled in the
        intersection area

    Returns
    ----------
    List[List[int]]
        List of points per dimension in the intersection area.
        The order is Y X points for 2D images and Z Y X for
        3D.
    """
    dims_sample_points = None

    # Number of dimensions on the image
    n_dims = len(bounds_1[0])

    o_min = []
    o_max = []

    # Getting left corner point and right corner point
    # in the intersection area
    for dim_idx in range(n_dims):
        o_min_val = max(bounds_1[0][dim_idx], bounds_2[0][dim_idx])
        o_max_val = min(bounds_1[1][dim_idx], bounds_2[1][dim_idx])

        o_min.append(o_min_val)
        o_max.append(o_max_val)

    # multiplicatives_mode = "2d" if n_dims == 2 else "3d"
    # matrix_vals = get_multiplicatives(numpoints, mode=multiplicatives_mode)

    num_points_axis = np.ceil(numpoints ** (1 / n_dims)).astype(np.uint8)
    matrix_vals = [num_points_axis for i in range(n_dims)]

    if np.prod(matrix_vals) != numpoints:
        # TODO Send message with logger
        print(
            f"Rounding to {num_points_axis} to keep all axis with same points"
        )

    # For 2D Y X area in intersection
    # For 3D Z Y X area in intersection
    # e.g., distanceZ = Zmax - Zmin
    inter_areas = {}
    for dim_idx in range(n_dims):
        inter_areas[dim_idx] = abs(o_max[dim_idx] - o_min[dim_idx])

    # Sorting the areas by axis
    # inter_areas_sorted = [
    #     list(val)
    #     for val in sorted(
    #         inter_areas.items(), key=lambda axis: axis[1], reverse=False
    #     )
    # ]

    # Asigning incremental points to incremental intersection areas
    # for idx_dim in range(n_dims):
    #     inter_areas_sorted[idx_dim][1] = matrix_vals[idx_dim]

    # inter_areas_sorted = dict(inter_areas_sorted)

    nd_linespaces = [
        np.linspace(
            start=o_min[idx],
            stop=o_max[idx],
            num=matrix_vals[
                idx
            ],  # inter_areas_sorted[idx],  # It's a List[int] [axis, n_points]
            dtype=int,
        )
        for idx in range(n_dims)
    ]

    dims_sample_points = [
        array.flatten()
        for array in np.meshgrid(
            *nd_linespaces,
            indexing="ij",
        )
    ]

    return dims_sample_points


def sample_points_in_overlap(
    bounds_1: ArrayLike,
    bounds_2: ArrayLike,
    numpoints: int,
    sample_type: Optional[str] = "random",
) -> np.ndarray:
    """
    samples points in the overlap regions and returns a list of points

    sampling types : random, grid, feature extracted

    Parameters
    ------------------------

    bounds_1: ArrayLike
        Image 1 calculated boundaries in each dimension,
        each position could be (x, y) or (x, y, z)
        depending on the image dimensionality.

    bounds_2: ArrayLike
        Image 2 calculated boundaries in each dimension,
        each position could be (x, y) or (x, y, z)
        depending on the image dimensionality.

    numpoints: int
        Number of sample points that will be used to create the grid

    sample_type: str
        random, grid or feature_extracted.

    Returns
    ------------------------
    np.ndarray:
        Array with the overlaped sample points.
    """
    sample_type = sample_type.lower()
    numpoints = int(numpoints)
    dims_sample_points = []
    n_dims = len(bounds_1[0])

    if sample_type not in ["random", "grid", "feature_extracted"]:
        raise NotImplementedError(
            f"{sample_type} sample type has not been implemented."
        )

    if numpoints < 0:
        raise ValueError(
            "Error in the number of points, it must be a positive integer."
        )

    if n_dims == 2 and not check_image_intersection_2D(bounds_1, bounds_2):
        raise ValueError(
            """2D Images do not intersect. Please,
            check the transformation matrix."""
        )

    elif n_dims == 3 and not check_image_intersection_3D(bounds_1, bounds_2):
        raise ValueError(
            """3D Images do not intersect. Please,
            check the transformation matrix."""
        )

    if sample_type == "random":
        for dim_idx in range(n_dims):
            o_min = max(bounds_1[0][dim_idx], bounds_2[0][dim_idx])
            o_max = min(bounds_1[1][dim_idx], bounds_2[1][dim_idx]) - 1

            random_choice = np.random.choice(range(o_min, o_max), numpoints)

            dims_sample_points.append(random_choice)

    elif sample_type == "grid":
        if n_dims == 2:
            dims_sample_points = sample_nd_grid_points(
                bounds_1=bounds_1, bounds_2=bounds_2, numpoints=numpoints
            )

        elif n_dims == 3:
            dims_sample_points = sample_nd_grid_points(
                bounds_1=bounds_1,
                bounds_2=bounds_2,
                numpoints=numpoints,
            )

        else:
            raise ValueError(
                "Sampling points in image dimensions higher than 3"
                "has not been developed yet"
            )

    dims_sample_points = np.array(dims_sample_points).transpose()
    return dims_sample_points


def prune_points_to_fit_window(
    image_shape: Tuple, points: np.array, window_size: int
) -> np.array:
    """
    Checks if generated points can be used for metric
    evaluation in the specified window size
    given a set of points and an image shape.

    Parameters
    ------------------------
    image_shape: Tuple
        Image shape

    points: np.array
        Sample points in an overlap region given two
        images using a transformation matrix

    window_size: int
        Window size applied over each axis

    Returns
    ------------------------
    np.array:
        Array with the points that fit the window
        size inside image shape.

    """
    n_dims = len(image_shape)
    zero_coord = np.zeros((n_dims,), dtype=np.uint8)

    def check_window_size(nested_point: np.array) -> bool:
        """
        Map function applied over a numpy array
        to check the window size to a point.

        Parameters
        ------------------------
        nested_point: np.array
            Individual point from the points array.

        Returns
        ------------------------
        bool:
            True if the (point (x, y) +- window size)
            is inside intersection image shape,
            False otherwise.
        """
        positive_modified_point = nested_point + window_size
        negative_modified_point = nested_point - window_size

        positive_point_window_inside = np.less_equal(
            positive_modified_point, image_shape
        )
        negative_point_window_inside = np.greater_equal(
            negative_modified_point, zero_coord
        )

        if np.all(positive_point_window_inside) and np.all(
            negative_point_window_inside
        ):
            return True

        return False

    selected_indices = np.array(list(map(check_window_size, points)))

    return points[selected_indices]
