"""
Module for utility functions
"""
from typing import List, Optional, Tuple, Union

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from natsort import natsorted

ArrayLike = Union[da.Array, np.array]


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


def check_image_intersection_2D(
    bounds_1: np.ndarray, bounds_2: np.ndarray
) -> bool:
    """
    Checks if the provided boundaries share an
    area of intersection

    Parameters
    ------------------------

    bounds_1: np.ndarray
        Array with the position of the boundaries
        within image 1 in order X, Y.

    bounds_2: np.ndarray
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
    bounds_1: np.ndarray, bounds_2: np.ndarray
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
        Image 1 calculated boundaries in each dimension,
        each position could be (x, y) or (x, y, z)
        depending on the image dimensionality.

    bounds_2: np.ndarray
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
                "Sampling points in image dimensions higher than 3 has not been developed yet"
            )

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


class SliceTracker:
    """
    Slice Tracker class to visualize 3D images
    """

    def __init__(
        self,
        fig_axes,
        image_1_data,
        image_2_data,
        points,
        selected_ponts,
        # rectangles,
        vmin=0,
        vmax=200,
        alpha=0.7,
        color="bone",
    ) -> None:
        """
        Class constructor
        """
        self.axes = fig_axes
        self.image_1_data = image_1_data
        self.image_2_data = image_2_data
        self.vmin = vmin
        self.vmax = vmax
        self.alpha = alpha
        self.color = color
        self.points = points
        self.selected_points = selected_ponts
        self.points_1 = self.points_2 = None
        self.plot_image_1 = self.plot_image_2 = None

        # Setting up message
        self.axes.set_title("Scroll to move to slices")

        self.__slices, self.__rows, self.__cols = image_1_data.shape

        # Starting visualization in the middle of the 3D block
        self.idx = self.__slices // 2
        self.get_current_slice()
        self.update_slice()

    def search_points_in_slice(self):
        """
        Searches for points in a slice
        """

        def search_points(points: List):
            """
            Helper function to search points
            in a slice

            Parameters
            -----------
            points: List[List[int]]
                Points in 3D space where
                the 3 lists are in order
                Z Y X
            """

            points_x = []
            points_y = []

            for z_point_pos in range(len(points[0])):
                if self.idx == points[0][z_point_pos]:
                    points_y.append(points[1][z_point_pos])
                    points_x.append(points[2][z_point_pos])

            return points_x, points_y

        points_x, points_y = search_points(self.points)
        sel_points_x, sel_points_y = search_points(self.selected_points)

        return [points_y, points_x], [sel_points_y, sel_points_x]

    def on_scroll(self, event):
        """
        Event to scroll in the image volume
        """
        if event.button == "up":
            self.idx = (self.idx + 1) % self.__slices
        else:
            self.idx = (self.idx - 1) % self.__slices
        self.update_slice()

    def update_slice(self):
        """
        Updates the current slice based on
        the scroll
        """

        self.update_points()
        self.get_current_slice()

        self.axes.set_ylabel("Slice %s" % self.idx)
        self.axes.figure.canvas.draw()

    def update_points(self):
        """
        Updates the points in a slice based on
        the scroll
        """
        pts, sl_pts = self.search_points_in_slice()

        if self.points_1 and self.points_2:
            self.points_1.remove()
            self.points_2.remove()

        self.points_1 = self.axes.scatter(y=pts[0], x=pts[1], c="r", s=20)
        self.points_2 = self.axes.scatter(
            y=sl_pts[0], x=sl_pts[1], c="#5DD9A7", s=20
        )

    def get_current_slice(self):
        """
        Gets the current slice and shows it
        in the renderer
        """
        if self.plot_image_1 and self.plot_image_2:
            self.plot_image_1.remove()
            self.plot_image_2.remove()

        self.plot_image_1 = self.axes.imshow(
            self.image_1_data[self.idx, :, :],
            alpha=self.alpha,
            cmap=self.color,
            vmin=self.vmin,
            vmax=self.vmax,
        )

        self.plot_image_2 = self.axes.imshow(
            self.image_2_data[self.idx, :, :],
            alpha=self.alpha,
            cmap=self.color,
            vmin=self.vmin,
            vmax=self.vmax,
        )


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


def visualize_images(
    image_1_data: ArrayLike,
    image_2_data: ArrayLike,
    bounds: List[np.ndarray],
    pruned_points: ArrayLike,
    selected_pruned_points: ArrayLike,
    transform: np.matrix,
    metric_name: str,
) -> None:
    """
    Function that plots an image to help visualize the
    intersection area, sampled points and images.

    Parameters
    ------------------------

    image_1_data: ArrayLike
        Array with the data of the image 1.

    image_2_data: ArrayLike
        Array with the data of the image 1.

    bounds: List[np.ndarray]
        List with the boundaries for the intersection area of the images.

    pruned_points: ArrayLike
        Points inside the intersection area that fit based on a window size.

    selected_pruned_points: ArrayLike
        Points inside the intersection area that were used for the metric.

    transform: np.matrix
        Transformation matrix applied to the images

    metric_name: str
        Metric name for the set of points
    """

    def generate_new_boundaries():
        lower_bound_1, upper_bound_1 = None, None
        lower_bound_2, upper_bound_2 = None, None

        if ty >= 0:
            lower_bound_1 = [bounds_1[0][0]]
            upper_bound_1 = [bounds_1[1][0]]

            lower_bound_2 = [bounds_2[0][0]]
            upper_bound_2 = [bounds_2[1][0]]

        else:
            lower_bound_1 = [abs(ty)]
            upper_bound_1 = [bounds_2[1][0] + abs(ty) * 2]

            lower_bound_2 = [bounds_1[0][0]]
            upper_bound_2 = [bounds_1[1][0]]

        if tx >= 0:
            lower_bound_1.append(bounds_1[0][1])
            upper_bound_1.append(bounds_1[1][1])

            lower_bound_2.append(bounds_2[0][1])
            upper_bound_2.append(bounds_2[1][1])

        else:
            lower_bound_1 = [bounds_2[0][1]]
            upper_bound_1 = [bounds_2[1][1]]

            lower_bound_2 = [bounds_1[0][1]]
            upper_bound_2 = [bounds_1[1][1]]

        img_1_bounds = [lower_bound_1, upper_bound_1]
        img_2_bounds = [lower_bound_2, upper_bound_2]

        return img_1_bounds, img_2_bounds

    if image_1_data.ndim != image_2_data.ndim:
        raise ValueError("Images should have the same shape")

    if image_1_data.ndim > 3:
        raise ValueError("Only 2D/3D images are supported")

    bounds_1 = bounds[0]
    bounds_2 = bounds[1]
    # from scipy.ndimage import rotate

    if image_1_data.ndim == 2:
        # plot directly the images and grid

        # Getting max boundaries in X and Y
        ty = transform[0, -1]
        tx = transform[1, -1]

        size_y = image_1_data.shape[0] + abs(ty)
        size_x = image_1_data.shape[1] + abs(tx)

        # size_x = max(bounds_1[1][0], bounds_2[1][0])
        # size_y = max(bounds_1[1][1], bounds_2[1][1])

        img_1_bounds, img_2_bounds = generate_new_boundaries()

        # Image within same coordinate system
        adjusted_img_1 = np.ones((size_y, size_x)) * 255
        adjusted_img_2 = np.ones((size_y, size_x)) * 255

        # flake8: noqa: E203
        # Getting data from the images to common coordinate image
        points_orig_img = (
            np.array(
                [np.arange(0, image_1_data.shape[0], step=1, dtype=np.intp)]
            ),
            np.array(
                [np.arange(0, image_1_data.shape[1], step=1, dtype=np.intp)]
            ),
        )

        points_img_1 = (
            np.array(
                [
                    np.arange(
                        img_1_bounds[0][0],
                        img_1_bounds[1][0],
                        step=1,
                        dtype=np.intp,
                    )
                ]
            ),
            np.array(
                [
                    np.arange(
                        img_1_bounds[0][1],
                        img_1_bounds[1][1],
                        step=1,
                        dtype=np.intp,
                    )
                ]
            ),
        )

        points_img_2 = (
            np.array(
                [
                    np.arange(
                        img_2_bounds[0][0],
                        img_2_bounds[1][0],
                        step=1,
                        dtype=np.intp,
                    )
                ]
            ),
            np.array(
                [
                    np.arange(
                        img_2_bounds[0][1],
                        img_2_bounds[1][1],
                        step=1,
                        dtype=np.intp,
                    )
                ]
            ),
        )

        xv_img_1, yv_img_1 = np.meshgrid(*points_img_1, indexing="ij")
        xv_img_2, yv_img_2 = np.meshgrid(*points_img_2, indexing="ij")
        xv_orig, yv_orig = np.meshgrid(*points_orig_img, indexing="ij")

        # vindex takes some time when processing with dask
        adjusted_img_1[xv_img_1, yv_img_1] = (
            image_1_data[xv_orig, yv_orig]
            if isinstance(image_1_data, np.ndarray)
            else image_1_data.vindex[xv_orig, yv_orig]
        )
        adjusted_img_2[xv_img_2, yv_img_2] = (
            image_2_data[xv_orig, yv_orig]
            if isinstance(image_2_data, np.ndarray)
            else image_2_data.vindex[xv_orig, yv_orig]
        )

        fig, ax = plt.subplots()

        fig.suptitle(f"Metric: {metric_name}", fontsize=16)

        # Setting X,Y positions of points within the grid
        y_points = [point[0] for point in pruned_points]
        x_points = [point[1] for point in pruned_points]

        # Setting X,Y positions of points within the
        # grid that were used to metric estimation
        selected_y_points = [point[0] for point in selected_pruned_points]
        selected_x_points = [point[1] for point in selected_pruned_points]

        # Scattering points within image in common coordinate system
        plt.scatter(x=x_points, y=y_points, c="r", s=10)
        plt.scatter(x=selected_x_points, y=selected_y_points, c="g", s=10)

        # Rectangles to divide images
        rectangle_image_1 = Rectangle(
            xy=(img_1_bounds[0][1], img_1_bounds[0][0]),
            width=img_1_bounds[1][1],
            height=img_1_bounds[1][0],
            linewidth=1,
            edgecolor="#FF2D00",
            facecolor="none",
            linestyle=":",
        )

        rectangle_image_2 = Rectangle(
            xy=(img_2_bounds[0][1], img_2_bounds[0][0]),
            width=img_1_bounds[1][1],
            height=img_1_bounds[1][0],
            linewidth=1,
            edgecolor="#13FF00",
            facecolor="none",
            linestyle="--",
        )

        # Alpha for overlaying
        alpha = 0.7
        ax.imshow(adjusted_img_1, alpha=alpha)  # , cmap='bone')
        ax.imshow(adjusted_img_2, alpha=alpha)  # , cmap='bone')

        ax.add_patch(rectangle_image_1)
        ax.add_patch(rectangle_image_2)

        plt.show()

    else:
        # 3D image, find a way to render 3D images with 3D grid

        size_x = max(bounds_1[1][0], bounds_2[1][0])
        size_y = max(bounds_1[1][1], bounds_2[1][1])
        size_z = max(bounds_1[1][2], bounds_2[1][2])

        # Image within same coordinate system
        adjusted_img_1 = np.ones((size_x, size_y, size_z)) * 255
        adjusted_img_2 = np.ones((size_x, size_y, size_z)) * 255

        # Getting data from the images to common coordinate image
        adjusted_img_1[
            : bounds_1[1][0], : bounds_1[1][1], : bounds_1[1][2]
        ] = image_1_data
        adjusted_img_2[
            bounds_2[0][0] :, bounds_2[0][1] :, bounds_2[0][2] :
        ] = image_2_data

        # Setting Z, Y, X positions of points within the grid
        z_points = [point[0] for point in pruned_points]
        y_points = [point[1] for point in pruned_points]
        x_points = [point[2] for point in pruned_points]

        # Setting Z,Y,X positions of points within the grid
        # that were used to metric estimation
        selected_z_points = [point[0] for point in selected_pruned_points]
        selected_y_points = [point[1] for point in selected_pruned_points]
        selected_x_points = [point[2] for point in selected_pruned_points]

        points = [z_points, y_points, x_points]
        selected_points = [
            selected_z_points,
            selected_y_points,
            selected_x_points,
        ]

        # visualize_image_3D(combined_volume, points, selected_points)

        # Rectangles to divide images
        rectangle_image_1 = Rectangle(
            xy=(bounds_1[0][2], bounds_1[0][1]),
            width=bounds_1[1][2],
            height=bounds_1[1][1],
            linewidth=1,
            edgecolor="#FF2D00",
            facecolor="none",
            linestyle=":",
        )

        rectangle_image_2 = Rectangle(
            xy=(bounds_2[0][2], bounds_2[0][1]),
            width=bounds_1[1][2],
            height=bounds_1[1][1],
            linewidth=1,
            edgecolor="#13FF00",
            facecolor="none",
            linestyle="--",
        )

        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.suptitle(f"Metric: {metric_name}", fontsize=16)

        tracker = SliceTracker(
            fig_axes=ax,
            image_1_data=adjusted_img_1,
            image_2_data=adjusted_img_2,
            points=points,
            selected_ponts=selected_points,
            # rectangles=[rectangle_image_1, rectangle_image_2]
        )

        ax.add_patch(rectangle_image_1)
        ax.add_patch(rectangle_image_2)
        fig.canvas.mpl_connect("scroll_event", tracker.on_scroll)
        fig.tight_layout()
        plt.show()
