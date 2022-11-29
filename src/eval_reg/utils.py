# import this
from typing import List, Optional, Tuple, Union

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ArrayLike = Union[da.Array, np.array]


def get_multiplicatives_2D(num_points):
    divs = []

    for i in range(1, num_points):
        if num_points % i == 0:
            divs.append(i)

    len_div = len(divs)

    middle = len_div // 2
    return divs[middle], divs[middle + 1]


def check_image_intersection_2D(
    bounds_1: np.ndarray, bounds_2: np.ndarray
) -> bool:

    # X0 == X1 or Y0 == Y1 for each image then it is no image
    # For boundaries in each image
    for bound in [bounds_1, bounds_2]:
        if bound[0][0] == bound[1][0] or bound[0][1] == bound[1][1]:
            return False

    # If images are on sides of each other in X
    if bounds_1[0][0] > bounds_2[1][0] or bounds_2[0][0] > bounds_1[1][0]:
        return False

    # If images are on top of each other in Y
    if bounds_1[0][1] > bounds_2[1][1] or bounds_2[0][1] > bounds_1[1][1]:
        return False

    return True


def check_image_intersection_3D(
    bounds_1: np.ndarray, bounds_2: np.ndarray
) -> bool:
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


def sample_points_in_overlap(
    bounds_1: np.ndarray,
    bounds_2: np.ndarray,
    numpoints: int,
    sample_type: Optional[str] = "random",
) -> np.ndarray:
    """
    samples points in the overlap regions and returns a list of points

    sampling types : random, grid, feature extracted

    Parameters
    ------------------------

    bounds_1: np.ndarray
        Image 1 calculated boundaries in each dimension, each position could be (x, y) or (x, y, z)
        depending on the image dimensionality.

    bounds_2: np.ndarray
        Image 2 calculated boundaries in each dimension, each position could be (x, y) or (x, y, z)
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

    if sample_type not in ["random", "grid", "feature_extracted"]:
        raise NotImplementedError(
            f"{sample_type} sample type has not been implemented."
        )

    if numpoints < 0:
        raise ValueError(
            "Error in the number of points, it must be a positive integer."
        )

    if len(bounds_1) == 2 and not check_image_intersection_2D(
        bounds_1, bounds_2
    ):
        raise ValueError(
            "2D Images do not intersect. Please, check the transformation matrix."
        )

    elif len(bounds_1) == 3 and not check_image_intersection_3D(
        bounds_1, bounds_2
    ):
        raise ValueError(
            "3D Images do not intersect. Please, check the transformation matrix."
        )

    n_dims = len(bounds_1[0])
    dims_sample_points = []

    if sample_type == "random":

        for dim_idx in range(n_dims):

            o_min = max(bounds_1[0][dim_idx], bounds_2[0][dim_idx])

            o_max = min(bounds_1[1][dim_idx], bounds_2[1][dim_idx])

            dims_sample_points.append(
                np.random.choice(range(o_min, o_max), numpoints)
            )

    elif sample_type == "grid":
        # Only for 2D so far
        o_min = []
        o_max = []

        for dim_idx in range(n_dims):

            o_min.append(max(bounds_1[0][dim_idx], bounds_2[0][dim_idx]))
            o_max.append(min(bounds_1[1][dim_idx], bounds_2[1][dim_idx]))

        y_space = abs(o_min[1] - o_min[0])
        x_space = abs(o_max[1] - o_max[0])

        matrix_vals = get_multiplicatives_2D(numpoints)
        x_points_distance = None
        y_points_distance = None

        if y_space > x_space:
            y_points_distance = matrix_vals[1]
            x_points_distance = matrix_vals[0]

        else:
            y_points_distance = matrix_vals[0]
            x_points_distance = matrix_vals[1]

        dims_sample_points = [
            array.flatten()
            for array in np.meshgrid(
                np.linspace(
                    o_min[0], o_max[0], y_points_distance, dtype=int
                ),  # For Y
                np.linspace(
                    o_min[1], o_max[1], x_points_distance, dtype=int
                ),  # For X
                indexing="ij",
            )
        ]

    dims_sample_points = np.array(dims_sample_points).transpose()
    return dims_sample_points


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
        First image which will be used as default in the coordinate system

    image_2: Tuple
        Second image which will be used to map it's position to a common coordinate system
        based on image_1

    transform: np.ndarray
        Transformation matrix applied over the two images

    Returns
    ------------------------
    Tuple:
        Tuple with the calculated boundaries.

    """

    dimensions_zeros = np.zeros(len(image_1_shape), dtype=np.int8)

    # First boundary
    bound_1 = np.array([dimensions_zeros, list(image_1_shape)])

    # Minimum point
    pt_min = np.matrix(np.append(dimensions_zeros, 1)).transpose()

    # Maximum point
    pt_max = np.matrix(np.array(list(image_2_shape) + [1])).transpose()

    # Getting coordinates for second image into image_1 coordinate system
    coord_1 = np.squeeze(transform * pt_min,).tolist()[
        0
    ][:-1]

    coord_2 = np.squeeze(transform * pt_max).tolist()[0][:-1]

    bound_2 = np.array([coord_1, coord_2])

    return bound_1, bound_2


def prune_points_to_fit_window(
    image_shape: Tuple, points: np.array, window_size: int
) -> np.array:
    """
    Checks if generated points can be used for metric evaluation in the specified window size
    given a set of points and an image shape.

    Parameters
    ------------------------
    image_shape: Tuple
        Image shape

    points: np.array
        Sample points in an overlap region given two images using a transformation matrix

    window_size: int
        Window size applied over each axis

    Returns
    ------------------------
    np.array:
        Array with the points that fit the window size inside image shape.

    """

    def check_window_size(nested_point: np.array) -> bool:
        """
        Map function applied over a numpy array

        Parameters
        ------------------------
        nested_point: np.array
            Individual point from the points array.

        Returns
        ------------------------
        bool:
            True if the (point (x, y) + window size) is inside image shape,
            False otherwise.
        """
        modified_point = nested_point + window_size
        point_window_inside = np.less_equal(modified_point, image_shape)
        unique_val = np.unique(point_window_inside)

        if len(unique_val) == 1 and unique_val[0]:
            return True

        return False

    selected_indices = np.array(list(map(check_window_size, points)))

    return points[selected_indices]


def extract_data(
    arr: ArrayLike, last_dimensions: Optional[int] = None
) -> ArrayLike:
    """
    Extracts n dimensional data (numpy array or dask array) given expanded dimensions.
    e.g., (1, 1, 1, 1600, 2000) -> (1600, 2000)
    e.g., (1, 1600, 2000) -> (1600, 2000)
    e.g., (1, 1, 2, 1600, 2000) -> (2, 1600, 2000)

    Parameters
    ------------------------
    arr: ArrayLike
        Numpy or dask array with image data. It is assumed that the last dimensions
        of the array contain the information about the image.

    last_dimensions: Optional[int]
        If given, it selects the number of dimensions given stating from the end
        of the array
        e.g., arr=(1, 1, 1600, 2000) last_dimensions=3 -> (1, 1600, 2000)
        e.g., arr=(1, 1, 1600, 2000) last_dimensions=1 -> (2000)

    Raises
    ------------------------
    ValueError:
        Whenever the last dimensions value is higher than the array dimensions.

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


def visualize_images(
    image_1_data: ArrayLike,
    image_2_data: ArrayLike,
    bounds: List[np.ndarray],
    pruned_points: ArrayLike,
    selected_pruned_points: ArrayLike,
) -> None:
    def adjust_axis_dims(bounds: List):
        # TODO
        pass

    if image_1_data.ndim != image_2_data.ndim:
        raise ValueError("Images should have the same shape")

    if image_1_data.ndim > 3:
        raise ValueError("Only 2D/3D images are supported")

    bounds_1 = bounds[0]
    bounds_2 = bounds[1]
    # from scipy.ndimage import rotate

    if image_1_data.ndim == 2:
        # plot directly the images and grid
        print(bounds)

        size_x = max(bounds_1[1][0], bounds_2[1][0])
        size_y = max(bounds_1[1][1], bounds_2[1][1])

        adjusted_img_1 = np.ones((size_x, size_y)) * 255
        adjusted_img_2 = np.ones((size_x, size_y)) * 255

        adjusted_img_1[: bounds_1[1][0], : bounds_1[1][1]] = image_1_data
        adjusted_img_2[bounds_2[0][0] :, bounds_2[0][1] :] = image_2_data

        fig, ax = plt.subplots()

        y_points = [point[0] for point in pruned_points]
        x_points = [point[1] for point in pruned_points]

        selected_y_points = [point[0] for point in selected_pruned_points]
        selected_x_points = [point[1] for point in selected_pruned_points]

        plt.scatter(x=x_points, y=y_points, c="r", s=10)
        plt.scatter(x=selected_x_points, y=selected_y_points, c="b", s=10)

        rectangle_image_1 = Rectangle(
            xy=(bounds_1[0][1], bounds_1[0][0]),
            width=bounds_1[1][1],
            height=bounds_1[1][0],
            linewidth=1,
            edgecolor="#FF2D00",
            facecolor="none",
            linestyle=":",
        )

        rectangle_image_2 = Rectangle(
            xy=(bounds_2[0][1], bounds_2[0][0]),
            width=bounds_2[1][1],
            height=bounds_2[1][0],
            linewidth=1,
            edgecolor="#13FF00",
            facecolor="none",
            linestyle="--",
        )

        alpha = 0.7
        ax.imshow(adjusted_img_1, alpha=alpha)  # , cmap='bone')
        ax.imshow(adjusted_img_2, alpha=alpha)  # , cmap='bone')

        ax.add_patch(rectangle_image_1)
        ax.add_patch(rectangle_image_2)

        plt.show()

    else:
        # 3D image, find a way to render 3D images with 3D grid
        pass
