"""
Module to check image intersections
"""

from typing import List, Tuple

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


# TODO Change to adaptive non-max supression based on
# response threshold not overlap_threshold
def kd_non_max_suppression(
    nd_boxes: List[List[int]], overlap_threshold: float, order_axis: str = "y"
):
    """
    Applies n-dimensional non-maxima
    suppression to a set of bounding
    boxes/cubes using a relative
    overlap threshold.

    Parameters
    -----------
    boxes: List[List[int]]
        List of boxes with where the
        form depends on its dimensionality.
        If it's a 2D image, boxes will be
        [y_start, x_start, y_end, x_end] and
        3D [z_start, y_start, x_start, z_end,
        y_end, x_end]. Start refers to the
        top-left corner and end to top-bottom
        corner.

    overlap_threshold: float
        Refers to the relative overlap threshold
        between two bounding boxes or cubes.

    order_axis: str
        Axis used to order boxes/cubes.
        Default: 'y'

    Returns
    ----------
    List[int]
        Set of indices after removing overlapped
        boxes/cubes
    """
    n_nd_boxes = len(nd_boxes)
    order_axis = order_axis.lower()

    if n_nd_boxes <= 1:
        return nd_boxes

    if nd_boxes.dtype.kind == "i":
        nd_boxes = nd_boxes.astype(np.float32)

    if order_axis not in ["z", "y", "x"]:
        raise NotImplementedError("This dimension is not available")

    pruned_indices = []
    dimensionality = len(nd_boxes[0]) // 2

    if dimensionality == 2 and order_axis == "z":
        raise ValueError(
            "Providing ordering in Z axis, but only YX available."
        )

    order_axis_pos = {"z": -3, "y": -2, "x": -1}

    # grab the coordinates of the bounding boxes/cubes
    coords = []
    area_list = []
    # Order is [Y X Y X] or [Z Y X Z Y X]
    for idx_dim in range(0, dimensionality * 2, 2):
        coords.append(nd_boxes[:, idx_dim])
        coords.append(nd_boxes[:, idx_dim + 1])

    for idx_dim in range(dimensionality):
        # [axis]2 - [axis]1 + 1 where [axis] = [Z, Y, X] for 3D
        area_list.append(
            nd_boxes[:, idx_dim + dimensionality] - nd_boxes[:, idx_dim] + 1
        )

    # compute the area of the boxes/cubes and sort
    # by the bottom-right order axis
    if dimensionality == 2:
        area = area_list[0] * area_list[1]
    else:
        area = area_list[0] * area_list[1] * area_list[2]

    idxs = np.argsort(coords[order_axis_pos[order_axis]])

    axis = list(range(dimensionality))

    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pruned_indices.append(i)
        box_cube_area = []
        for ax in axis:
            # e.g., y_end, y_start
            axis_end = np.minimum(
                coords[ax + dimensionality][i],  # accessing end corner
                coords[ax + dimensionality][idxs[:last]],
            )

            axis_start = np.maximum(
                coords[ax][i],  # accessing start corner
                coords[ax][idxs[:last]],
            )

            box_cube_area.append(np.maximum(0, axis_end - axis_start + 1))

        if dimensionality == 2:
            box_cube_area = box_cube_area[0] * box_cube_area[1]
        else:
            box_cube_area = (
                box_cube_area[0] * box_cube_area[1] * box_cube_area[1]
            )

        # compute the ratio of overlap
        overlap = box_cube_area / area[idxs[:last]]

        # delete all boxes with higher overlap
        # than the threshold
        idxs = np.delete(
            idxs,
            np.concatenate(([last], np.where(overlap > overlap_threshold)[0])),
        )
    # return only remaining bboxs/cubes
    return np.array(pruned_indices, dtype=int)


def generate_corner(val, window_size, mode, img_shape, axis):
    possible_val = val + window_size

    if mode == "summation":
        if possible_val > img_shape[axis] - 1:
            possible_val = img_shape[axis] - 1

    elif mode == "subtraction":
        possible_val = val - window_size
        if possible_val < 0:
            possible_val = 0

    else:
        raise ValueError("Mode not available")

    return possible_val


def kd_compute_bboxs_cubes(
    points: List[int], window_size: int, img_shape: Tuple[int]
):
    n_dims = len(img_shape)
    img_geometries = []

    if n_dims == 2:
        img_geometries = np.array(
            [
                np.array(
                    [
                        # Start top left corner
                        generate_corner(
                            y_val,
                            window_size,
                            "subtraction",
                            img_shape,
                            axis=-2,
                        ),
                        generate_corner(
                            x_val,
                            window_size,
                            "subtraction",
                            img_shape,
                            axis=-1,
                        ),
                        # End bottom right corner
                        generate_corner(
                            y_val, window_size, "summation", img_shape, axis=-2
                        ),
                        generate_corner(
                            x_val, window_size, "summation", img_shape, axis=-1
                        ),
                    ]
                )
                for y_val, x_val in points
            ]
        )

    elif n_dims == 3:
        img_geometries = np.array(
            [
                np.array(
                    [
                        # Start top left corner
                        generate_corner(
                            z_val,
                            window_size,
                            "subtraction",
                            img_shape,
                            axis=-3,
                        ),
                        generate_corner(
                            y_val,
                            window_size,
                            "subtraction",
                            img_shape,
                            axis=-2,
                        ),
                        generate_corner(
                            x_val,
                            window_size,
                            "subtraction",
                            img_shape,
                            axis=-1,
                        ),
                        # End bottom right corner
                        generate_corner(
                            z_val, window_size, "summation", img_shape, axis=-3
                        ),
                        generate_corner(
                            y_val, window_size, "summation", img_shape, axis=-2
                        ),
                        generate_corner(
                            x_val, window_size, "summation", img_shape, axis=-1
                        ),
                    ]
                )
                for z_val, y_val, x_val in points
            ]
        )

    return img_geometries


def generate_overlap_slices(
    shapes: List[Tuple], orientation: str, overlap_ratio: float
) -> tuple:
    """
    Generate the slices of the overlap region between two
    images. Images could be 2D or 3D.

    Parameters
    ------------
    shapes: List[tuple]
        List of shapes where position 0 represents
        image 1 and position 1 image 2.

    orientation: str
        Overlap orientation. It could be ["z", "y", "x"]

    overlap_ratio: float
        Overlap percentage between the images in the given
        orientation.

    Raises
    ------------
    NotImplementedError:
        If the image is not 2D or 3D

    Returns
    ------------
    tuple:
        Tuple with the slices of the overlap region
        and the offset.
    """
    image_1_shape = shapes[0]
    image_2_shape = shapes[1]
    ndims = len(image_1_shape)

    if ndims not in [2, 3]:
        raise NotImplementedError("Only 2D or 3D images available")

    # Getting overlapping area for images
    overlap_area_1 = (np.array(image_1_shape) * overlap_ratio).astype(int)
    overlap_area_2 = (np.array(image_2_shape) * overlap_ratio).astype(int)

    iterate_reverse_axis = list(range(-ndims, 0, 1))

    if orientation == "x":
        curr_axis = -1
    elif orientation == "y":
        curr_axis = -2
    elif orientation == "z":
        curr_axis = -3
    else:
        raise NotImplementedError("Only ZYX orientations accepted")

    if orientation == "z" and ndims != 3:
        raise ValueError("Please, provide a 3D array for z orientation")

    offset_img_1 = image_1_shape[curr_axis] - overlap_area_1[curr_axis]

    # Order must be ZYX
    slices_1 = []
    slices_2 = []

    for ax in iterate_reverse_axis:
        if ax == curr_axis:
            slices_1.append(slice(offset_img_1, image_1_shape[ax]))
            slices_2.append(slice(0, overlap_area_2[ax]))
        else:
            slices_1.append(slice(0, image_1_shape[ax]))
            slices_2.append(slice(0, image_2_shape[ax]))

    return tuple(slices_1), tuple(slices_2), offset_img_1
