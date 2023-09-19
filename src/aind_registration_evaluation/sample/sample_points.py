"""
Module to sample points in image
intersection areas
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_laplace, maximum_filter
from skimage.feature import peak_local_max

from aind_registration_evaluation._shared.types import ArrayLike
from aind_registration_evaluation.util.intersection import (
    check_image_intersection_2D, check_image_intersection_3D,
    kd_compute_bboxs_cubes, kd_non_max_suppression)

from .utils import kd_pad_fft_buterworth


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


def kd_max_min_local_points(
    image: ArrayLike,
    filter_size: int,
    pad_width: Optional[int] = 0,
    max_relative_threshold: Optional[float] = 0.2,
    min_relative_threshold: Optional[float] = 0.2,
    n_keypoints: Optional[int] = 100,
):
    """
    Computes the local maximas and minimas
    in an image using the filter size as
    minimum distance between patches of data.

    Paramters
    ----------
    image: ArrayLike
        N-Dimensional image data

    filter_size: int
        Size that is taken at every element
        of the data to compute the maximum
        filtering. This integer is applied
        over each axis.

    pad_width: Optional[int]
        Padding width for the useful for
        the non-linear filtering.
        Default: 0

    max_relative_threshold: Optional[float]
        Minimum relative threshold to get the
        intensity of peaks for the local maxima
        values in the data. This is calculated
        as max(image) * max_relative_threshold.
        Default: 0.2

    min_relative_threshold: Optional[float]
        Minimum relative threshold to get the
        intensity of peaks for the local minima
        values in the data. This is calculated
        as max(image) * max_relative_threshold.
        Default: 0.2

    n_keypoints: Optional[int]
        Number of keypoints to sample. This
        only applied if we get too many
        keypoints from the current parameters.
        Default: 100

    Returns
    ----------
    Tuple[np.array, np.array, np.array]
        Tuple with the local maxima coordinates,
        local minima coordinates and filtered
        image.
    """
    image_max_filter = maximum_filter(image, size=filter_size)

    if pad_width > 0:
        pad_img_shape = image_max_filter.shape
        slices = tuple(
            [
                slice(pad_width, pad_shape - pad_width)
                for pad_shape in pad_img_shape
            ]
        )

        image_max_filter = image_max_filter[slices]

    # Getting local max points
    maxima_coordinates = peak_local_max(
        image=image_max_filter,
        min_distance=filter_size,
        threshold_rel=max_relative_threshold,
        num_peaks=n_keypoints,
    )

    # Getting local minima points
    minima_coordinates = peak_local_max(
        image=-image_max_filter,
        min_distance=filter_size,
        threshold_rel=min_relative_threshold,
        num_peaks=n_keypoints,
    )

    return maxima_coordinates, minima_coordinates, image_max_filter


def kd_max_energy_points(
    image: ArrayLike,
    sigma: int,
    filter_size: int,
    pad_width: Optional[int] = 0,
    max_relative_threshold: Optional[float] = 0.2,
    n_keypoints: Optional[int] = 100,
):
    """
    Gets the points where there are
    local maximas patching the data
    with a filter_size*2 from the
    energy image computed with
    a Laplacian of Gaussian.

    Paramters
    ----------
    image: ArrayLike
        N-Dimensional image data

    sigma: int
        Sigma used in the gaussian.
        Greater the gaussian, the more
        the smoothing.

    filter_size: int
        Size that is taken at every element
        of the data to compute the maximum
        filtering. This integer is applied
        over each axis.

    pad_width: Optional[int]
        Padding width for the useful for
        the non-linear filtering.
        Default: 0

    max_relative_threshold: Optional[float]
        Minimum relative threshold to get the
        intensity of peaks for the local maxima
        values in the data. This is calculated
        as max(image) * max_relative_threshold.
        Default: 0.2

    n_keypoints: Optional[int]
        Number of keypoints to sample. This
        only applied if we get too many
        keypoints from the current parameters.
        Default: 100

    Returns
    ----------
    Tuple[np.array, np.array]
        Tuple with the local maxima coordinates
        and filtered image.
    """
    image_gaussian_laplaced = gaussian_laplace(
        input=image, sigma=sigma, mode="reflect"
    )

    if pad_width > 0:
        pad_img_shape = image_gaussian_laplaced.shape
        slices = tuple(
            [slice(pad_width, shape - pad_width) for shape in pad_img_shape]
        )

        image_gaussian_laplaced = image_gaussian_laplaced[slices]

    # Getting local max points
    max_energy_points = peak_local_max(
        image=image_gaussian_laplaced,
        min_distance=filter_size,
        threshold_rel=max_relative_threshold,
        num_peaks=n_keypoints,
    )

    return np.array(max_energy_points, dtype=int), image_gaussian_laplaced


def kd_fft_keypoints(
    image: ArrayLike,
    filter_size: int,
    pad_width: Optional[int] = 0,
    max_relative_threshold: Optional[float] = 0.2,
    min_relative_threshold: Optional[float] = 0.05,
    overlap_threshold: Optional[float] = 0.3,
    n_keypoints: Optional[int] = 100,
):
    """
    Gets keypoints using the following
    approach:
    1. Computes the FFT and then buterworth filter
    in the signal.
    2. Gets local maximas and minimas from the image.
    3. Removes points in the same area using the
    non-maxima suppression algorithm.

    Parameters
    -----------
    image: ArrayLike
        N-Dimensional image data

    filter_size: int
        Size that is taken at every element
        of the data to compute the maximum
        filtering. This integer is applied
        over each axis.

    pad_width: Optional[int]
        Padding width for the useful for
        the non-linear filtering.
        Default: 0

    max_relative_threshold: Optional[float]
        Minimum relative threshold to get the
        intensity of peaks for the local maxima
        values in the data. This is calculated
        as max(image) * max_relative_threshold.
        Default: 0.2

    overlap_threshold: Optional[float]
        Overlap threshold betweend the
        bounding boxes computed around
        each corner using the filter_size
        value.
        Default: 0.3

    n_keypoints: Optional[int]
        Number of keypoints to sample. This
        only applied if we get too many
        keypoints from the current parameters.
        Default: 100

    Returns
    ---------
    Tuple[np.array, np.array]
        Tuple with the local minimas and
        maximas in the first position and
        the filtered image in the second.
    """
    inversed_fft_image = kd_pad_fft_buterworth(image, pad_width=pad_width)

    max_points, min_points, response_img = kd_max_min_local_points(
        inversed_fft_image,
        filter_size=filter_size,
        pad_width=pad_width,
        max_relative_threshold=max_relative_threshold,
        min_relative_threshold=min_relative_threshold,
        n_keypoints=n_keypoints,
    )

    # Computing bboxs based on window size = pad_width
    max_bboxs = kd_compute_bboxs_cubes(
        max_points, pad_width, response_img.shape
    )
    min_bboxs = kd_compute_bboxs_cubes(
        min_points, pad_width, response_img.shape
    )

    # Non-max suppression with created bboxs
    idxs_pruned_max_bboxs = kd_non_max_suppression(
        max_bboxs, overlap_threshold
    )
    idxs_pruned_min_bboxs = kd_non_max_suppression(
        min_bboxs, overlap_threshold
    )

    pruned_max_points = max_points.copy()
    pruned_min_points = min_points.copy()

    # Getting non-max prunned points
    if len(idxs_pruned_max_bboxs):
        pruned_max_points = max_points[idxs_pruned_max_bboxs]

    if len(idxs_pruned_min_bboxs):
        pruned_min_points = min_points[idxs_pruned_min_bboxs]

    return (
        np.concatenate((pruned_max_points, pruned_min_points), axis=0),
        response_img,
    )


def kd_fft_energy_keypoints(
    image: np.ndarray,
    filter_size: int,
    pad_width: Optional[int] = 0,
    sigma: Optional[int] = 9,
    max_relative_threshold: Optional[float] = 0.2,
    overlap_threshold: Optional[float] = 0.3,
    n_keypoints: Optional[int] = 100,
) -> Tuple[np.array]:
    """
    Computes the k-dimensional fft energy-based
    keypoint feature selection. This approach
    is organized as follows:

    1. Use padding. For now, it's only zero padding but we could apply other types of padding.
    2. Fast-Fourier Transform for the given image.
    3. Butterworth filter in frecuency domain.
    4. Inverse Fast-Fourier Transform to the FFT'd signal.
    5. Get the energy map using laplacian of gaussian.
    6. Unpad image and get local maximas and minimas from the images.
    7. Compute bounding boxes based on a window_size equal to padding size
    7. Use non-maxima suppression to prune identified points (created bboxs) in local maximas and minimas for both images.

    Parameters
    -----------
    image: np.ndarray
        2D or 3D image used to get
        the keypoints

    pad_width: Optional[int]
        Pad width applied on the image
        to avoid non-linear filtering
        artifacts.
        Default: 0

    sigma: Optional[int]
        Gaussian sigma for the laplacian
        of gaussian
        Default: 9

    max_relative_threshold: Optional[float]
        Relative threshold for the image signal
        to avoid sampling in areas where we have
        no signal.
        f = max(image) * max_relative_threshold
        Default: 0.2

    overlap_threshold: Optional[float]
        Relative overlap threshold for the
        bounding boxes. This is used in the
        kd non-maxima supression.
        Default: 0.3

    n_keypoints: Optional[int]
        Number of keypoints to sample.
        If the number of keypoints is lower than
        the ones identified, we will return the
        best keypoints based on the intensity
        value.
        Default: 100

    Returns
    -----------
    Tuple[np.array]
        Tuple with the identified point locations
        in cartesian space and the response image
        after applying filtering
    """
    inversed_fft_image = kd_pad_fft_buterworth(image, pad_width=pad_width)

    energy_points, response_img = kd_max_energy_points(
        image=inversed_fft_image,
        sigma=sigma,
        filter_size=filter_size,
        pad_width=pad_width,
        max_relative_threshold=max_relative_threshold,
        n_keypoints=n_keypoints,
    )

    # Computing bboxs based on window size = pad_width
    max_bboxs = kd_compute_bboxs_cubes(
        energy_points, pad_width, response_img.shape
    )

    # Non-max suppression with created bboxs
    idxs_pruned_energy_bboxs = kd_non_max_suppression(
        max_bboxs, overlap_threshold
    )

    pruned_max_points = energy_points.copy()

    # Getting non-max prunned points
    if len(idxs_pruned_energy_bboxs) and isinstance(
        idxs_pruned_energy_bboxs[0], int
    ):
        pruned_max_points = energy_points[idxs_pruned_energy_bboxs]

    return pruned_max_points, response_img


def kd_compute_keypoints_hog(
    image_gradient_magnitude: np.array,
    image_gradient_orientation: List[np.array],
    keypoint: np.array,
    n_dims: int,
    window_size: Optional[int] = 16,
    bins: Optional[List] = [8, 4],
) -> Tuple[np.array]:
    """
    Computes the k-dimensional histogram of
    gradients for a set of keypoints located
    in an image

    Parameters
    -----------
    image_gradient_magnitude: np.array
        Image gradient magnitude computed
        from an image

    image_gradient_orientation: List[np.array]
        List containing the gradient orientation
        of the image. This list will contain only
        phi if it's a 2D image and phi and theta
        if it's 3D. Theta for a 3D image refers to
        the polar angle and phi as the
        azimuthal angle (in 2D angle between
        a point from the origin (0,0)
        and point (x,y) ).

    keypoint: np.array
        Keypoint location in cartesian space

    n_dims: int
        Number of image dimensions

    window_size: Optional[int]
        Window size around the point. We follow
        many papers in the selection of the
        default value.
        Default: 16

    bins: Optional[List]
        Number of bins applied over each image
        dimension. If it's 2D, we will pick only
        the first axis.
        Default: [8, 4]

    Raises
    ----------
    ValueError:
        If the number of dimensions is 1 or lower
        or greater than 3. This approach is only
        for 2D and 3D images.

    Returns
    ----------
    """

    if n_dims <= 1 or n_dims > 3:
        raise ValueError("Provide a correct number of dimensions")

    # Converting keypoints to integers if necessary
    if keypoint.dtype.kind != "i":
        keypoint = keypoint.astype(np.uint32)

    # Get window_size region around the keypoint
    slices = []
    for axis_val in keypoint:
        slices.append(
            slice(
                (axis_val - window_size + 1) // 2,
                (axis_val + window_size + 1) // 2,
            )
        )
    slices = tuple(slices)

    # Getting smoothed magnitude and orientation for a keypoint region
    region_magnitude = image_gradient_magnitude[slices]

    region_orientation_yx = image_gradient_orientation[0][slices]
    region_orientation_z_yx = None

    if n_dims == 3:
        region_orientation_z_yx = image_gradient_orientation[0][slices]
        region_orientation_yx = image_gradient_orientation[1][slices]

    cell_size = int(region_magnitude.shape[0] // 4)
    cell_size_per_dimension = [cell_size] * n_dims

    if n_dims == 2:
        # Order ZYX
        cell_size_per_dimension = [1] + cell_size_per_dimension

    # Iterating each quadrant and computing hogs
    # TODO optimize
    # Quadrants are organized as follow
    # [0, 1]
    # [2, 3]

    # n_quadrants = (n_dims - 1) * 4
    # hogs_per_quadrant = {i: {} for i in range(n_quadrants)}
    weighted_gradient_hist = []

    for z_idx in range(cell_size_per_dimension[0]):
        for y_idx in range(cell_size_per_dimension[1]):
            for x_idx in range(cell_size_per_dimension[2]):
                if n_dims == 2:
                    quadrant = (
                        slice(
                            cell_size * y_idx, cell_size * y_idx + cell_size
                        ),
                        slice(
                            cell_size * x_idx, cell_size * x_idx + cell_size
                        ),
                    )
                else:
                    quadrant = (
                        slice(
                            cell_size * z_idx, cell_size * z_idx + cell_size
                        ),
                        slice(
                            cell_size * y_idx, cell_size * y_idx + cell_size
                        ),
                        slice(
                            cell_size * x_idx, cell_size * x_idx + cell_size
                        ),
                    )

                cell_magnitude = region_magnitude[quadrant]
                cell_orientation_yx = region_orientation_yx[quadrant]

                if n_dims == 2:
                    _range = (-180, 180)

                    hist, hist_bins = np.histogram(
                        cell_orientation_yx,
                        bins=bins[0],
                        range=_range,
                        weights=cell_magnitude,
                        density=False,
                    )
                    # Recomputing angles since histogram returns bins + 1
                    # hist_bins = (hist_bins[:-1] + hist_bins[1:]) / 2.
                    weighted_gradient_hist.append(hist)

                else:
                    cell_orientation_z_xy = region_orientation_z_yx[quadrant]
                    _range = ((-180, 180), (-90, 90))

                    (
                        hist,
                        hist_bins_polar,
                        hist_bins_elevation,
                    ) = np.histogram2d(
                        cell_orientation_z_xy.flatten(),
                        cell_orientation_yx.flatten(),
                        bins=bins,
                        range=_range,
                        weights=cell_magnitude.flatten(),
                        density=False,
                    )

                    # Recomputing angles since histogram returns bins + 1
                    hist_bins_polar = (
                        hist_bins_polar[:-1] + hist_bins_polar[1:]
                    ) / 2.0
                    hist_bins_elevation = (
                        hist_bins_elevation[:-1] + hist_bins_elevation[1:]
                    ) / 2.0
                    weighted_gradient_hist.append(hist)

    feature_vector = np.array(weighted_gradient_hist)
    feature_vector = np.expand_dims(feature_vector.flatten(), axis=1)
    feature_vector = feature_vector / (
        np.sqrt(np.sum(feature_vector**2)) + 1e-8
    )

    # Normalizing feat vect
    # feature_vector = feature_vector / np.sqrt(np.sum(np.power(feature_vector, 2)))#/= np.linalg.norm(feature_vector)#
    # feature_vector = np.sqrt(feature_vector)

    return {"keypoint": keypoint, "feature_vector": feature_vector}
