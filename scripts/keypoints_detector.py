import functools
from typing import List, Tuple

import numpy as np
import tifffile as tif
import zarr
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_laplace, maximum_filter
from skimage.feature import peak_local_max

"""

hist_sorted = np.argsort(hist)
                    hist_bins = (hist_bins[:-1] + hist_bins[1:]) / 2.0
                    hist1, bins1 = hist[hist_sorted][::-1], hist_bins[hist_sorted][::-1]
                    print(hist1.shape, hist.shape, hist1, "SEP:", hist)
                    
                    
                    
                    f, ax = plt.subplots(1,1)
                    plot_quiver(ax, hist, hist_bins, scale=1200)
                    plt.show()
                    plt.imshow(cell_magnitude)
                    plt.show()
                    exit()

"""


def pol2cart(rho, phi):
    x = rho * np.cos(np.deg2rad(phi))
    y = rho * np.sin(np.deg2rad(phi))
    print("x,y: ", x, y)
    return x, y


def plot_quiver(ax, x, y, scale=1, n=None, color=None):
    if n is None:
        n = len(x)
    # x, y = pol2cart(hist, bins)
    if color is not None:
        ax.quiver(
            np.zeros(n),
            np.zeros(n),
            x[:n],
            y[:n],
            units="xy",
            scale=scale,
            color=color,
        )
    else:
        ax.quiver(
            np.zeros(n), np.zeros(n), x[:n], y[:n], units="xy", scale=scale
        )


def _get_nd_butterworth_filter(
    shape,
    factor,
    order,
    high_pass,
    real,
    dtype=np.float64,
    squared_butterworth=True,
):
    """Create a N-dimensional Butterworth mask for an FFT

    Parameters
    ----------
    shape : tuple of int
        Shape of the n-dimensional FFT and mask.
    factor : float
        Fraction of mask dimensions where the cutoff should be.
    order : float
        Controls the slope in the cutoff region.
    high_pass : bool
        Whether the filter is high pass (low frequencies attenuated) or
        low pass (high frequencies are attenuated).
    real : bool
        Whether the FFT is of a real (True) or complex (False) image
    squared_butterworth : bool, optional
        When True, the square of the Butterworth filter is used.

    Returns
    -------
    wfilt : ndarray
        The FFT mask.

    """
    ranges = []
    for i, d in enumerate(shape):
        # start and stop ensures center of mask aligns with center of FFT
        axis = np.arange(-(d - 1) // 2, (d - 1) // 2 + 1) / (d * factor)
        ranges.append(np.fft.ifftshift(axis**2))
    # for real image FFT, halve the last axis
    if real:
        limit = d // 2 + 1
        ranges[-1] = ranges[-1][:limit]
    # q2 = squared Euclidean distance grid
    q2 = functools.reduce(
        np.add, np.meshgrid(*ranges, indexing="ij", sparse=True)
    )
    q2 = q2.astype(dtype)
    q2 = np.power(q2, order)
    wfilt = 1 / (1 + q2)
    if high_pass:
        wfilt = 1 - wfilt
    if not squared_butterworth:
        np.sqrt(wfilt, out=wfilt)
    return wfilt


def get_inversed_fft(fft_image):
    is_real = np.isrealobj(fft_image)

    if is_real:
        return np.fft.irfftn(fft_image).real, is_real

    return np.fft.ifftn(fft_image).real, is_real


def get_fft(image):
    return np.fft.fftn(image)


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


def kd_pad_fft_buterworth(image, pad_width=0):
    if pad_width > 0:
        image = np.pad(array=image, mode="constant", pad_width=pad_width)

    img_fft = get_fft(image)
    is_real = np.isrealobj(img_fft)

    butter_filter = _get_nd_butterworth_filter(
        img_fft.shape, 0.005, 2.0, True, is_real, np.float16, True
    )

    inversed_img_fft, is_real_inversed = get_inversed_fft(
        butter_filter * img_fft
    )
    return inversed_img_fft


def kd_max_min_local_points(
    image,
    maximum_filter_size,
    pad_width=0,
    max_relative_threshold=0.2,
    min_relative_threshold=0.2,
):
    image_max_filter = maximum_filter(image, size=maximum_filter_size)

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
        min_distance=maximum_filter_size * 2,
        threshold_rel=max_relative_threshold,
    )

    # Getting local minima points
    minima_coordinates = peak_local_max(
        image=-image_max_filter,
        min_distance=maximum_filter_size * 2,
        threshold_rel=min_relative_threshold,
    )

    return maxima_coordinates, minima_coordinates, image_max_filter


def kd_max_energy_points(
    image,
    sigma,
    maximum_filter_size,
    pad_width,
    max_relative_threshold=0.2,
    n_keypoints=100,
):
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
        min_distance=maximum_filter_size * 2,
        threshold_rel=max_relative_threshold,
        num_peaks=n_keypoints,
    )

    return np.array(max_energy_points, dtype=int), image_gaussian_laplaced


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


def kd_fft_keypoints(
    image,
    pad_width=0,
    max_relative_threshold=0.2,
    min_relative_threshold=0.05,
    overlap_threshold=0.3,
):
    inversed_fft_image = kd_pad_fft_buterworth(image, pad_width=pad_width)

    maximum_filter_size = pad_width // 4
    max_points, min_points, response_img = kd_max_min_local_points(
        inversed_fft_image,
        maximum_filter_size=maximum_filter_size,
        pad_width=pad_width,
        max_relative_threshold=max_relative_threshold,
        min_relative_threshold=min_relative_threshold,
    )

    print("Identified min max points: ", len(max_points), len(min_points))

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

    print(
        "Idxs pruned max min: ", idxs_pruned_max_bboxs, idxs_pruned_min_bboxs
    )

    pruned_max_points = max_points.copy()
    pruned_min_points = min_points.copy()

    # Getting non-max prunned points
    if len(idxs_pruned_max_bboxs):
        pruned_max_points = max_points[idxs_pruned_max_bboxs]

    if len(idxs_pruned_min_bboxs):
        pruned_min_points = min_points[idxs_pruned_min_bboxs]

    print(
        "Identified min max points after suppression: ",
        len(pruned_max_points),
        len(pruned_min_points),
    )

    return pruned_max_points, pruned_min_points, response_img


def kd_fft_energy_keypoints(
    image,
    pad_width=0,
    sigma=9,
    max_relative_threshold=0.2,
    overlap_threshold=0.3,
    n_keypoints=100,
):
    inversed_fft_image = kd_pad_fft_buterworth(image, pad_width=pad_width)
    maximum_filter_size = pad_width // 4

    energy_points, response_img = kd_max_energy_points(
        image=inversed_fft_image,
        sigma=sigma,
        maximum_filter_size=maximum_filter_size,
        pad_width=pad_width,
        max_relative_threshold=max_relative_threshold,
        n_keypoints=n_keypoints,
    )

    print("Identified min max points: ", len(energy_points))

    # Computing bboxs based on window size = pad_width
    max_bboxs = kd_compute_bboxs_cubes(
        energy_points, pad_width, response_img.shape
    )

    # Non-max suppression with created bboxs
    idxs_pruned_energy_bboxs = kd_non_max_suppression(
        max_bboxs, overlap_threshold
    )

    print("Idxs pruned max min: ", idxs_pruned_energy_bboxs)
    pruned_max_points = energy_points.copy()

    # Getting non-max prunned points
    if len(idxs_pruned_energy_bboxs):
        pruned_max_points = energy_points[idxs_pruned_energy_bboxs]

    print(
        "Identified energy points after suppression: ", len(pruned_max_points)
    )

    return pruned_max_points, response_img


def derivate_image_axis(image, axis: List[int], n_lvls=1):
    from scipy.ndimage import sobel

    if len(axis) == 0:
        ValueError("Please, provide a list with the axis")

    derivatives = []

    for ax in axis:
        derivatives.append(
            # np.diff(image, n=n_lvls, axis=ax)
            sobel(image, ax)
        )

    return derivatives


def kd_gradient_magnitudes_and_orientations(derivated_images: List[np.array]):
    """
    Computes the gradient magnitudes and orientations
    for a 2D or 3D image.

    Parameters
    ------------
    derivated_images: List[np.array]
        List with the derivated images in
        [Dy, Dx] order for a 2D image or
        [Dz, Dy, Dx] order for a 3D image

    Returns
    ------------
    Tuple
        Tuple with the gradient magnitude
        (radius in spherical coordinates),
        theta for a 3D image referring to
        the polar angle and phi as the
        azimuthal angle (in 2D angle between
        a point from the origin (0,0)
        and point (x,y) )
    """
    n_dims = len(derivated_images)

    if n_dims <= 1 and n_dims > 3:
        raise ValueError("Only 2D or 3D images are allowed")

    powered_derivatives = np.power(derivated_images, 2)
    print("Powered derivatives: ", powered_derivatives.shape)
    sum_powered_derivatives = np.sum(powered_derivatives, axis=0)
    # also called radius in spherical coordinates
    gradient_magnitude = np.sqrt(sum_powered_derivatives)
    print("Gradient magnitude: ", gradient_magnitude.shape)
    phi = None
    theta = None

    if n_dims == 2:
        # Angle between origin and point x,y in a circle
        phi = np.arctan2(derivated_images[0], derivated_images[1])
        phi = np.rad2deg(phi)  # Azimuth angle

    else:
        # Angle between origin and point x,y,z in a sphere
        phi = np.arctan2(derivated_images[1], derivated_images[2])  # Angle XY
        phi = np.rad2deg(phi)  # Azimuth angle

        theta = np.arccos(
            derivated_images[0] / gradient_magnitude
        )  # Angle Z - plane XY
        theta = np.rad2deg(theta)  # Elevation

    return gradient_magnitude, phi, theta


def kd_compute_keypoints_hog_2(
    image_gradient_magnitude: np.array,
    image_gradient_orientation: List[np.array],
    keypoint: np.array,
    n_dims: int,
    window_size: int = 16,
    bins=[4, 8],
) -> Tuple[np.array]:
    if n_dims <= 1 or n_dims > 3:
        raise ValueError("Provide a correct number of dimensions")

    # Converting keypoints to integers if necessary
    if keypoint.dtype.kind != "i":
        keypoint = keypoint.astype(np.uint32)

    # Get window_size region around the keypoint
    slices = []
    for axis_val in keypoint:
        slices.append(
            slice(axis_val - window_size // 2, axis_val + window_size // 2)
        )
    slices = tuple(slices)

    # Getting smoothed magnitude and orientation for a keypoint region
    region_magnitude = image_gradient_magnitude[slices]

    region_orientation_yx = image_gradient_orientation[0][slices]
    region_orientation_z_yx = None

    if n_dims == 3:
        region_orientation_z_yx = image_gradient_orientation[0][slices]
        region_orientation_yx = image_gradient_orientation[1][slices]

    weighted_gradient_histograms = []
    weighted_gradient_bins = []

    cell_size = int(region_magnitude.shape[0] // 4)
    cell_size_per_dimension = [cell_size] * n_dims

    if n_dims == 2:
        # Order ZYX
        cell_size_per_dimension = [1] + cell_size_per_dimension

    def get_quadrant(matrix_size, row, col):
        mid = matrix_size // 2

        if col < mid:
            if row < mid:
                return 0
            else:
                return 1
        else:
            if row < mid:
                return 2
            else:
                return 3

    # Iterating each quadrant and computing hogs
    # TODO optimize
    # Quadrants are organized as follow
    # [0, 1]
    # [2, 3]
    hogs_per_quadrant = {
        0: {},
        1: {},
        2: {},
        3: {},
    }

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

                hist_response = None
                if n_dims == 2:
                    _range = (-180, 180)
                    # _range = (-np.pi, np.pi)

                    hist, hist_bins = np.histogram(
                        cell_orientation_yx,
                        bins=bins[0],
                        range=_range,
                        weights=cell_magnitude,
                        density=False,
                    )
                    # Recomputing angles since histogram returns bins + 1
                    hist_bins = (hist_bins[:-1] + hist_bins[1:]) / 2.0

                    hist_response = [hist, hist_bins]

                else:
                    cell_orientation_z_xy = region_orientation_z_yx[quadrant]
                    hist_data = [
                        cell_orientation_z_xy.flatten(),
                        cell_orientation_yx.flatten(),
                    ]
                    # hist, _ = np.histogramdd(
                    #     hist_data,
                    #     bins=bins,
                    #     range=((0, 360), (0, 180)),#(-np.pi, np.pi), (-np.pi//2, np.pi//2)),
                    #     weights=cell_magnitude.flatten()
                    # )
                    _range = ((-180, 180), (-90, 90))
                    # _range = ((-np.pi, np.pi), (-np.pi/4, np.pi/4))

                    print(_range)
                    hist, _, _ = np.histogram2d(
                        cell_orientation_z_xy.flatten(),
                        cell_orientation_yx.flatten(),
                        bins=bins,
                        range=_range,
                        weights=cell_magnitude.flatten(),
                        density=False,
                    )

                hogs_per_quadrant[get_quadrant(cell_size, x_idx, y_idx)][
                    (y_idx, x_idx)
                ] = {
                    "wgh": hist_response[0],
                    "wgb": hist_response[1],
                    "img": cell_magnitude,
                }

                weighted_gradient_histograms.append(hist_response[0])
                weighted_gradient_bins.append(hist_response[1])

    # wgh = np.array(wgh).reshape((128, 1))
    # wgh = wgh / np.sqrt(np.sum(wgh * wgh))
    # feature_vector = np.sqrt(wgh)
    dominant_orientation = None

    for ys in range(cell_size):
        for xs in range(cell_size):
            print(
                f"pos ({ys}, {xs}) belongs to quadrant {(get_quadrant(cell_size, xs, ys))}"
            )

    print("HoGs per quadrant: ", hogs_per_quadrant[0])

    if n_dims == 2:
        # 2D space
        for we in weighted_gradient_histograms:
            print(we)
        weighted_gradient_histograms = np.array(weighted_gradient_histograms)

        sum_of_w_g_h = np.sum(weighted_gradient_histograms, axis=1)
        print(sum_of_w_g_h, weighted_gradient_histograms.shape)

        arg_max_idx = np.argmax(sum_of_w_g_h)

        print(
            "SUM: ",
            sum_of_w_g_h[arg_max_idx],
            weighted_gradient_histograms[arg_max_idx],
            arg_max_idx,
        )

        dominant_orientation = [arg_max_idx * (360 / bins[0])]
        print("Dominant orientation: ", dominant_orientation)
        print(f"Argmax: {arg_max_idx}")

        check = (
            np.min(region_magnitude),
            np.max(region_magnitude),
        )

        # for q, items in hogs_per_quadrant.items():
        #     for pos, item in items.items():
        #         wgh, wgb, img = item.values()
        #         f, ax = plt.subplots(1,2)
        #         ax[0].set_title("FOV")
        #         ax[0].imshow(region_magnitude, cmap="gray", vmin=check[0], vmax=check[1])
        #         ax[1].set_title(str(f"quadrant {q} -> {pos}"))
        #         ax[1].imshow(img, cmap="gray", vmin=check[0], vmax=check[1])
        #         plt.show()

        x = weighted_gradient_histograms[arg_max_idx] * np.cos(
            np.deg2rad(weighted_gradient_bins[arg_max_idx])
        )

        sum_vector_x = np.sum(x)
        x = np.append(x, sum_vector_x)
        y = weighted_gradient_histograms[arg_max_idx] * np.sin(
            np.deg2rad(weighted_gradient_bins[arg_max_idx])
        )

        sum_vector_y = np.sum(y)
        y = np.append(y, sum_vector_y)

        print("X: ", x)
        print("Y: ", y)

        # x = [40, 38, 36, 34, 32, 30, 28, 26] * np.cos(
        #     np.deg2rad(weighted_gradient_bins[arg_max_idx])
        # )

        # y = [40, 38, 36, 34, 32, 30, 28, 26] * np.sin(
        #     np.deg2rad(weighted_gradient_bins[arg_max_idx])
        # )

        print(
            "Dominant orientation: ",
            weighted_gradient_histograms[arg_max_idx],
            weighted_gradient_bins[arg_max_idx],
            arg_max_idx,
        )
        print(
            "Resultant vector angle: ",
            np.rad2deg(np.arctan2(np.rad2deg(y[-1]), np.rad2deg(x[-1]))),
        )

        f, ax = plt.subplots(1, 2)
        ax[0].imshow(
            hogs_per_quadrant[3][(3, 3)]["img"]
        )  # , vmin=check[0], vmax=check[1])
        plot_quiver(
            ax=ax[1],
            x=x,  # hist=weighted_gradient_histograms[arg_max_idx],
            y=y,  # bins=weighted_gradient_bins[arg_max_idx],
            scale=1200,
            color=[
                "red",
                "yellow",
                "blue",
                "gray",
                "purple",
                "green",
                "orange",
                "cyan",
            ],
        )
        plt.show()

        exit()

        angle_rad = np.radians(img_1_ori["dominant_orientation"][0])
        length = 20
        x2 = x + length * np.cos(angle_rad)
        y2 = y + length * np.sin(angle_rad)
        ax.plot([x, x2], [y, y2], "r-")

    else:
        # 3D space
        max_bin_idx = np.unravel_index(np.argmax(hist), hist.shape)
        print(max_bin_idx)
        azimuth_bin_center = (max_bin_idx[0] + 0.5) * (360 / bins[0])
        elevation_bin_center = (max_bin_idx[1] + 0.5) * (180 / bins[1]) - 90
        dominant_orientation = [azimuth_bin_center, elevation_bin_center]

        # weighted_gradient_histogram = np.array(weighted_gradient_histogram)
        # sum_of_w_g_h_z_yx = np.sum(weighted_gradient_histogram, axis=1)
        # sum_of_w_g_h_yx = np.sum(weighted_gradient_histogram, axis=2)

        # arg_max_z_yx = np.argmax(sum_of_w_g_h_z_yx)
        # arg_max_yx = np.argmax(sum_of_w_g_h_yx)

        # dominant_orientation = [
        #     arg_max_z_yx * (360 / bins[0]),
        #     arg_max_yx * (180 / bins[1])
        # ]

    return {"keypoint": keypoint, "dominant_orientation": dominant_orientation}


def get_quadrant_2d(matrix_size, row, col):
    mid = matrix_size // 2

    if col < mid:
        if row < mid:
            return 0
        else:
            return 1
    else:
        if row < mid:
            return 2
        else:
            return 3


def get_quadrant_3d(matrix_size, depth, row, col):
    mid = matrix_size // 2

    if col < mid:
        if row < mid:
            if depth < mid:
                return 0
            else:
                return 1
        else:
            if depth < mid:
                return 2
            else:
                return 3
    else:
        if row < mid:
            if depth < mid:
                return 4
            else:
                return 5
        else:
            if depth < mid:
                return 6
            else:
                return 7


def kd_compute_keypoints_hog(
    image_gradient_magnitude: np.array,
    image_gradient_orientation: List[np.array],
    keypoint: np.array,
    n_dims: int,
    window_size: int = 16,
    bins=[4, 8],
) -> Tuple[np.array]:
    if n_dims <= 1 or n_dims > 3:
        raise ValueError("Provide a correct number of dimensions")

    # Converting keypoints to integers if necessary
    if keypoint.dtype.kind != "i":
        keypoint = keypoint.astype(np.uint32)

    # Get window_size region around the keypoint
    slices = []
    for axis_val in keypoint:
        slices.append(
            slice(axis_val - window_size // 2, axis_val + window_size // 2)
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

                    hist, hist_bins = np.histogram2d(
                        cell_orientation_z_xy.flatten(),
                        cell_orientation_yx.flatten(),
                        bins=bins,
                        range=_range,
                        weights=cell_magnitude.flatten(),
                        density=False,
                    )

                    # Recomputing angles since histogram returns bins + 1
                    hist_bins = (hist_bins[:-1] + hist_bins[1:]) / 2.0
                    weighted_gradient_hist.append(hist)

                    # hogs_per_quadrant[get_quadrant_3d(cell_size, z_idx, y_idx, x_idx)][
                    #     (z_idx, y_idx, x_idx)
                    # ] = {
                    #     "wgh": hist,
                    #     "wgb": hist_bins,
                    #     "magnitude": cell_magnitude
                    # }

                    # hogs_per_quadrant[
                    # get_quadrant_3d(cell_size, z_idx, y_idx, x_idx)
                    # ][(z_idx, y_idx, x_idx)] = {
                    #     "wgh": hist_response[0],
                    #     "wgb": hist_response[1],
                    #     "magnitude": cell_magnitude
                    # }

                    # sum_wgh = np.sum(hist)

                    # if "max" not in hogs_per_quadrant[q] or hogs_per_quadrant[q]["max"] < sum_wgh:
                    #     hogs_per_quadrant[q]["max"] = sum_wgh

    if n_dims == 2:
        # 2D space
        feature_vector = np.array(weighted_gradient_hist)
        feature_vector = np.expand_dims(feature_vector.flatten(), axis=1)
        feature_vector = feature_vector / (
            np.sqrt(np.sum(feature_vector**2)) + 1e-8
        )

        # Normalizing feat vect
        # feature_vector = feature_vector / np.sqrt(np.sum(np.power(feature_vector, 2)))#/= np.linalg.norm(feature_vector)#
        # feature_vector = np.sqrt(feature_vector)

    else:
        # 3D space
        pass

    return {"keypoint": keypoint, "feature_vector": feature_vector}


def test_fft_max_min_keypoints(img_1, img_2):
    pad_width = np.min(img_1.shape) // 4
    (
        max_img_1_keypoints,
        min_img_1_keypoints,
        filtered_img_1,
    ) = kd_fft_keypoints(image=img_1, pad_width=pad_width)
    (
        max_img_2_keypoints,
        min_img_2_keypoints,
        filtered_img_2,
    ) = kd_fft_keypoints(image=img_2, pad_width=pad_width)

    # comparison img1 filters
    f, axarr = plt.subplots(1, 2)
    f.suptitle("Image 1", fontsize=20)
    axarr[0].set_title("Original")
    axarr[0].imshow(img_1)
    axarr[0].plot(
        max_img_1_keypoints[:, 1], max_img_1_keypoints[:, 0], "r."
    )  # max points
    axarr[0].plot(
        min_img_1_keypoints[:, 1], min_img_1_keypoints[:, 0], "b."
    )  # min points

    axarr[1].set_title("Max Filtered")
    axarr[1].imshow(filtered_img_1)
    axarr[1].plot(
        max_img_1_keypoints[:, 1], max_img_1_keypoints[:, 0], "r."
    )  # max points
    axarr[1].plot(
        min_img_1_keypoints[:, 1], min_img_1_keypoints[:, 0], "b."
    )  # max points

    plt.tight_layout()
    plt.show()

    # comparison img2 filters

    f, axarr = plt.subplots(1, 2)

    f.suptitle("Image 2", fontsize=20)
    axarr[0].set_title("Original")
    axarr[0].imshow(img_2)
    axarr[0].plot(
        max_img_2_keypoints[:, 1], max_img_2_keypoints[:, 0], "r."
    )  # max points
    axarr[0].plot(
        min_img_2_keypoints[:, 1], min_img_2_keypoints[:, 0], "b."
    )  # min points

    axarr[1].set_title("Max Filtered")
    axarr[1].imshow(filtered_img_2)
    axarr[1].plot(
        max_img_2_keypoints[:, 1], max_img_2_keypoints[:, 0], "r."
    )  # max points
    axarr[1].plot(
        min_img_2_keypoints[:, 1], min_img_2_keypoints[:, 0], "b."
    )  # max points
    plt.tight_layout()
    plt.show()

    # comparison img1 - img2

    f, axarr = plt.subplots(1, 2)

    f.suptitle("Images", fontsize=20)
    axarr[0].set_title("Img 1")
    axarr[0].imshow(img_1)
    axarr[0].plot(
        max_img_1_keypoints[:, 1], max_img_1_keypoints[:, 0], "r."
    )  # max points
    axarr[0].plot(
        min_img_1_keypoints[:, 1], min_img_1_keypoints[:, 0], "b."
    )  # min points

    axarr[1].set_title("Img 2")
    axarr[1].imshow(img_2)
    axarr[1].plot(
        max_img_2_keypoints[:, 1], max_img_2_keypoints[:, 0], "r."
    )  # max points
    axarr[1].plot(
        min_img_2_keypoints[:, 1], min_img_2_keypoints[:, 0], "b."
    )  # max points
    plt.tight_layout()
    plt.show()

    # comparison filtered1 - filtered2

    f, axarr = plt.subplots(1, 2)

    f.suptitle("FFT-Butterworth maximum filter", fontsize=20)
    axarr[0].set_title("Img 1")
    axarr[0].imshow(filtered_img_1)
    axarr[0].plot(
        max_img_1_keypoints[:, 1], max_img_1_keypoints[:, 0], "r."
    )  # max points
    axarr[0].plot(
        min_img_1_keypoints[:, 1], min_img_1_keypoints[:, 0], "b."
    )  # min points

    axarr[1].set_title("Img 2")
    axarr[1].imshow(filtered_img_2)
    axarr[1].plot(
        max_img_2_keypoints[:, 1], max_img_2_keypoints[:, 0], "r."
    )  # max points
    axarr[1].plot(
        min_img_2_keypoints[:, 1], min_img_2_keypoints[:, 0], "b."
    )  # max points
    plt.tight_layout()
    plt.show()


def compute_feature_space_distances(
    features_1: Tuple[np.array],
    features_2: Tuple[np.array],
    feature_axis: int = 0,
    cart_axis: int = 1,
) -> np.array:
    # Features have the feature vector and cartesian location of the point
    # in cartesian space.

    # Feature distances -> save all differences left points -> right points
    feature_distances = np.empty(
        (features_1[feature_axis].shape[0], features_2[feature_axis].shape[0])
    )

    for feat_idx in range(features_1[feature_axis].shape[0]):
        feat_dif = (
            features_2[feature_axis] - features_1[feature_axis][feat_idx]
        )
        feat_dif = np.power(feat_dif, 2).sum(axis=1).flatten()
        loc_dif = np.sqrt(
            np.sum(
                np.power(
                    features_2[cart_axis] - features_1[cart_axis][feat_idx], 2
                ),
                axis=-1,
            )
        )

        # Difference in feature space and cartesian location
        distance = feat_dif + loc_dif

        feature_distances[feat_idx] = distance

    return feature_distances


def generate_key_features_per_img2d(img_2d, n_keypoints):
    pad_width = np.min(img_2d.shape) // 6
    img_2d_keypoints_energy, img_response = kd_fft_energy_keypoints(
        image=img_2d,
        pad_width=pad_width,
        n_keypoints=n_keypoints,
    )

    dy_val, dx_val = derivate_image_axis(
        gaussian_filter(img_2d, sigma=8), [0, 1]
    )

    # img_2d_dy = np.zeros(img_2d.shape, dtype=img_2d.dtype)
    # img_2d_dx = np.zeros(img_2d.shape, dtype=img_2d.dtype)

    # dy_val = np.sqrt(dy_val)
    # dx_val = np.sqrt(dx_val)

    # f, ax = plt.subplots(1,2)

    # ax[0].imshow(dy_val, cmap="gray")#vmin=0, vmax=0.2)
    # ax[1].imshow(dx_val, cmap="gray") #vmin=0, vmax=0.2)
    # plt.show()

    # img_2d_dy[:-1, :] = np.float32(dy_val)
    # img_2d_dx[:, :-1] = np.float32(dx_val)

    (
        gradient_magnitude,
        gradient_orientation,
        gradient_orientation_polar,
    ) = kd_gradient_magnitudes_and_orientations(
        derivated_images=[dy_val, dx_val]  # [img_2d_dy, img_2d_dx]
    )

    img_keypoints_features = [
        kd_compute_keypoints_hog(
            image_gradient_magnitude=gradient_magnitude,
            image_gradient_orientation=[gradient_orientation],
            keypoint=keypoint,
            n_dims=2,
            window_size=16,
            bins=[8],
        )
        for keypoint in img_2d_keypoints_energy
    ]

    keypoints = []
    features = []
    for key_feat in img_keypoints_features:
        # print(f"Keypoint {key_feat['keypoint']} feat shape: {key_feat['feature_vector'].shape}")
        keypoints.append(key_feat["keypoint"])
        features.append(key_feat["feature_vector"])

    return {
        "keypoints": np.array(keypoints),
        "features": np.array(features),
        "response_img": img_response,
    }


def get_pairs_from_distances(distances: np.array, delete_points=True):
    # Distances is an array NxM
    pairs = {}
    right_assigned_points = {}

    for left_key_idx in range(distances.shape[0]):
        right_min_idx = distances[left_key_idx].argmin()
        right_min_distance = distances[left_key_idx][right_min_idx]

        old_assignment = (
            right_assigned_points.get(right_min_idx) if delete_points else None
        )

        if (
            old_assignment is not None
            and old_assignment["distance"] > right_min_distance
        ):
            old_left_point = old_assignment["point_idx"]
            del pairs[old_left_point]

            pairs[left_key_idx] = right_min_idx
            right_assigned_points[right_min_idx] = {
                "point_idx": left_key_idx,
                "distance": right_min_distance,
            }

        elif old_assignment is None:
            pairs[left_key_idx] = right_min_idx
            right_assigned_points[right_min_idx] = {
                "point_idx": left_key_idx,
                "distance": right_min_distance,
            }

    return pairs


def plot_matches(
    ax,
    image1,
    image2,
    keypoints1,
    keypoints2,
    matches,
    keypoints_color="k",
    matches_color=None,
    only_matches=False,
    alignment="horizontal",
):
    """Plot matched features.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matches and image are drawn in this ax.
    image1 : (N, M [, 3]) array
        First grayscale or color image.
    image2 : (N, M [, 3]) array
        Second grayscale or color image.
    keypoints1 : (K1, 2) array
        First keypoint coordinates as ``(row, col)``.
    keypoints2 : (K2, 2) array
        Second keypoint coordinates as ``(row, col)``.
    matches : (Q, 2) array
        Indices of corresponding matches in first and second set of
        descriptors, where ``matches[:, 0]`` denote the indices in the first
        and ``matches[:, 1]`` the indices in the second set of descriptors.
    keypoints_color : matplotlib color, optional
        Color for keypoint locations.
    matches_color : matplotlib color, optional
        Color for lines which connect keypoint matches. By default the
        color is chosen randomly.
    only_matches : bool, optional
        Whether to only plot matches and not plot the keypoint locations.
    alignment : {'horizontal', 'vertical'}, optional
        Whether to show images side by side, ``'horizontal'``, or one above
        the other, ``'vertical'``.

    """
    image1 = image1.astype(np.float16)
    image2 = image2.astype(np.float16)

    new_shape1 = list(image1.shape)
    new_shape2 = list(image2.shape)

    if image1.shape[0] < image2.shape[0]:
        new_shape1[0] = image2.shape[0]
    elif image1.shape[0] > image2.shape[0]:
        new_shape2[0] = image1.shape[0]

    if image1.shape[1] < image2.shape[1]:
        new_shape1[1] = image2.shape[1]
    elif image1.shape[1] > image2.shape[1]:
        new_shape2[1] = image1.shape[1]

    if new_shape1 != image1.shape:
        new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1[: image1.shape[0], : image1.shape[1]] = image1
        image1 = new_image1

    if new_shape2 != image2.shape:
        new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2[: image2.shape[0], : image2.shape[1]] = image2
        image2 = new_image2

    offset = np.array(image1.shape)
    if alignment == "horizontal":
        image = np.concatenate([image1, image2], axis=1)
        offset[0] = 0
    elif alignment == "vertical":
        image = np.concatenate([image1, image2], axis=0)
        offset[1] = 0
    else:
        mesg = (
            f"plot_matches accepts either 'horizontal' or 'vertical' for "
            f"alignment, but '{alignment}' was given. See "
            f"https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.plot_matches "  # noqa
            f"for details."
        )
        raise ValueError(mesg)

    if not only_matches:
        ax.scatter(
            keypoints1[:, 1],
            keypoints1[:, 0],
            facecolors="none",
            edgecolors=keypoints_color,
        )
        ax.scatter(
            keypoints2[:, 1] + offset[1],
            keypoints2[:, 0] + offset[0],
            facecolors="none",
            edgecolors=keypoints_color,
        )

    ax.imshow(image)  # , cmap='gray')
    ax.axis((0, image1.shape[1] + offset[1], image1.shape[0] + offset[0], 0))

    rng = np.random.default_rng()

    for i in range(matches.shape[0]):
        idx1 = matches[i, 0]
        idx2 = matches[i, 1]

        if matches_color is None:
            color = rng.random(3)
        else:
            color = matches_color

        ax.plot(
            (keypoints1[idx1, 1], keypoints2[idx2, 1] + offset[1]),
            (keypoints1[idx1, 0], keypoints2[idx2, 0] + offset[0]),
            "-",
            color=color,
        )

        ax.annotate(
            str(idx1),
            (keypoints1[idx1, 1], keypoints1[idx1, 0]),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
            color=color,
        )

        ax.annotate(
            str(idx2),
            (keypoints2[idx2, 1] + offset[1], keypoints2[idx2, 0] + offset[0]),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
            color=color,
        )


def test_fft_energy_keypoints(img_1, img_2):
    n_keypoints = 200
    img_1_dict = generate_key_features_per_img2d(
        img_1, n_keypoints=n_keypoints
    )
    img_2_dict = generate_key_features_per_img2d(
        img_2, n_keypoints=n_keypoints
    )

    feature_vector_img_1 = (img_1_dict["features"], img_1_dict["keypoints"])
    feature_vector_img_2 = (img_2_dict["features"], img_2_dict["keypoints"])

    distances = compute_feature_space_distances(
        feature_vector_img_1, feature_vector_img_2
    )

    point_matches = get_pairs_from_distances(
        distances=distances, delete_points=True
    )

    print(
        f"N keypoints img_1: {img_1_dict['keypoints'].shape} img_2: {img_2_dict['keypoints'].shape}"
    )

    # Showing only points
    # comparison img1 filters
    print("\n Keypoint confidence img 1")
    for key_idx in range(len(img_1_dict["keypoints"])):
        print(
            f"Confidence for point {key_idx} is: ",
            img_1_dict["response_img"][
                img_1_dict["keypoints"][key_idx][0],
                img_1_dict["keypoints"][key_idx][1],
            ],
        )

    print("\n Keypoint confidence img 2")
    for key_idx in range(len(img_2_dict["keypoints"])):
        print(
            f"Confidence for point {key_idx} is: ",
            img_2_dict["response_img"][
                img_2_dict["keypoints"][key_idx][0],
                img_2_dict["keypoints"][key_idx][1],
            ],
        )

    f, axarr = plt.subplots(1, 2)
    f.suptitle("Image 1", fontsize=20)
    axarr[0].set_title("Original")
    axarr[0].imshow(img_1)
    axarr[0].plot(
        img_1_dict["keypoints"][:, 1], img_1_dict["keypoints"][:, 0], "r."
    )  # max points

    axarr[1].set_title("FFT Gauss Laplaced Filtered")
    axarr[1].imshow(img_1_dict["response_img"])
    axarr[1].plot(
        img_1_dict["keypoints"][:, 1], img_1_dict["keypoints"][:, 0], "r."
    )  # max points

    plt.tight_layout()
    plt.show()
    # comparison img2 filters

    f, axarr = plt.subplots(1, 2)

    f.suptitle("Image 2", fontsize=20)
    axarr[0].set_title("Original")
    axarr[0].imshow(img_2)
    axarr[0].plot(
        img_2_dict["keypoints"][:, 1], img_2_dict["keypoints"][:, 0], "r."
    )  # max points

    axarr[1].set_title("FFT Gauss Laplaced Filtered")
    axarr[1].imshow(img_2_dict["response_img"])
    axarr[1].plot(
        img_2_dict["keypoints"][:, 1], img_2_dict["keypoints"][:, 0], "r."
    )  # max points
    plt.tight_layout()
    plt.show()

    # comparison img1 - img2
    f, axarr = plt.subplots(1, 2)

    f.suptitle("FFT-Butterworth Gaussian Laplaced Filtered", fontsize=20)
    axarr[0].set_title("Image 1")
    axarr[0].imshow(img_1_dict["response_img"])
    axarr[0].plot(
        img_1_dict["keypoints"][:, 1], img_1_dict["keypoints"][:, 0], "r."
    )  # max points

    axarr[1].set_title("Image 2")
    axarr[1].imshow(img_2_dict["response_img"])
    axarr[1].plot(
        img_2_dict["keypoints"][:, 1], img_2_dict["keypoints"][:, 0], "r."
    )  # max points
    plt.tight_layout()
    plt.show()

    f, axarr = plt.subplots(1, 2)

    f.suptitle("Points in images", fontsize=20)
    axarr[0].set_title("Img 1")
    axarr[0].imshow(img_1)
    axarr[0].plot(
        img_1_dict["keypoints"][:, 1], img_1_dict["keypoints"][:, 0], "r."
    )  # max points

    axarr[1].set_title("Img 2")
    axarr[1].imshow(img_2)
    axarr[1].plot(
        img_2_dict["keypoints"][:, 1], img_2_dict["keypoints"][:, 0], "r."
    )  # max points
    plt.tight_layout()
    plt.show()

    f, axarr = plt.subplots(1, 1, figsize=(10, 5))
    f.suptitle("Matched points", fontsize=20)
    # Set titles and labels
    axarr.set_xlabel("X")
    axarr.set_ylabel("Y")

    idxs1, idxs2 = list(point_matches.keys()), list(point_matches.values())
    matches = np.column_stack((idxs1, idxs2))

    plot_matches(
        ax=axarr,
        image1=img_1,
        image2=img_2,
        keypoints1=img_1_dict["keypoints"],
        keypoints2=img_2_dict["keypoints"],
        keypoints_color="red",
        matches=matches,
        matches_color="red",
        only_matches=False,
    )

    plt.tight_layout()

    # Show the plot
    plt.show()

    f, axarr = plt.subplots(1, 1, figsize=(10, 5))
    f.suptitle("Matched points", fontsize=20)
    # Set titles and labels
    axarr.set_xlabel("X")
    axarr.set_ylabel("Y")

    idxs1, idxs2 = list(point_matches.keys()), list(point_matches.values())
    matches = np.column_stack((idxs1, idxs2))

    plot_matches(
        ax=axarr,
        image1=img_1_dict["response_img"],
        image2=img_2_dict["response_img"],
        keypoints1=img_1_dict["keypoints"],
        keypoints2=img_2_dict["keypoints"],
        # keypoints_color='red',
        matches=matches,
        matches_color="red",
        only_matches=False,
    )

    plt.tight_layout()

    # Show the plot
    plt.show()


def test_img_3d_orientations(img_3d):
    pad_width_3d = np.min(img_3d.shape) // 4
    # Getting keypoints
    img_3d_keypoints_energy, response_image = kd_fft_energy_keypoints(
        image=img_3d,
        pad_width=pad_width_3d,
    )

    f, ax = plt.subplots(1, 2)
    f.suptitle("Original image vs response image")

    slide = 0
    ax[0].imshow(img_3d[slide, :, :])
    ax[1].imshow(response_image[slide, :, :])
    plt.show()

    # Getting derivatives in each axis
    dz_val, dy_val, dx_val = derivate_image_axis(img_3d, [0, 1, 2])

    # Getting image derivatives
    filtered_img_3d_dz = np.zeros(img_3d.shape, dtype=img_3d.dtype)
    filtered_img_3d_dy = np.zeros(img_3d.shape, dtype=img_3d.dtype)
    filtered_img_3d_dx = np.zeros(img_3d.shape, dtype=img_3d.dtype)

    filtered_img_3d_dz[:-1, :, :] = np.float32(dz_val)
    filtered_img_3d_dy[:, :-1, :] = np.float32(dy_val)
    filtered_img_3d_dx[:, :, :-1] = np.float32(dx_val)

    # Getting gradient magnitude, and orientations in YX plane and polar coordinates
    (
        gradient_magnitude_3d,
        gradient_orientation_yx_3d,
        gradient_orientation_z_yx_3d,
    ) = kd_gradient_magnitudes_and_orientations(
        derivated_images=[
            filtered_img_3d_dz,
            filtered_img_3d_dy,
            filtered_img_3d_dx,
        ]
    )

    img_3d_keypoint_energy_orientations = []
    for keypoint in img_3d_keypoints_energy:
        print(f"Computing hog for keypoint {keypoint}")
        img_3d_res = kd_compute_keypoints_hog(
            image_gradient_magnitude=gradient_magnitude_3d,
            image_gradient_orientation=[
                gradient_orientation_z_yx_3d,
                gradient_orientation_yx_3d,
            ],
            keypoint=keypoint,
            n_dims=3,
            window_size=16,
            bins=[8, 4],
        )
        img_3d_keypoint_energy_orientations.append(img_3d_res)

    for img_3d_ori in img_3d_keypoint_energy_orientations:
        print("orientations: ", img_3d_ori)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Set plot labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Dominant 3D Point Orientation")

    # Plot the dominant orientation as a line
    for keypoints_ori_dict in img_3d_keypoint_energy_orientations:
        keypoint = keypoints_ori_dict["keypoint"]
        orientation = keypoints_ori_dict["dominant_orientation"]
        x, y, z = keypoint[2], keypoint[1], keypoint[0]

        ax.scatter(zs=z, ys=y, xs=x, c="blue", marker="o", s=50)

        angle_rad = np.radians(orientation)
        length = 15
        z2 = z + length * np.cos(angle_rad[0])
        x2 = x + length * np.sin(angle_rad[1])
        y2 = y + length * np.cos(angle_rad[1])

        ax.plot(zs=[z, z2], ys=[y, y2], xs=[x, x2], c="r", linestyle="-")

        ax.plot(x, y, z, color="red", linewidth=2)

    # Show the 3D plot
    plt.show()


def main():
    BASE_PATH = "/Users/camilo.laiton/Documents/images/"

    img_1_path = BASE_PATH + "Ex_488_Em_525_468770_468770_830620_012820.zarr/0"
    img_2_path = BASE_PATH + "Ex_488_Em_525_501170_501170_830620_012820.zarr/0"

    img_3D_path = BASE_PATH + "block_10.tif"

    img_1 = zarr.open(img_1_path, "r")[0, 0, 0, :, 1800:]
    img_2 = zarr.open(img_2_path, "r")[0, 0, 0, :, :200]
    img_3d = tif.imread(img_3D_path)[120:184, 200:456, 200:456]
    print("3D image shape: ", img_3d.shape)

    # test_fft_max_min_keypoints(img_1, img_2)
    # test_fft_energy_keypoints(img_1, img_2)
    # test_img_2d_orientations(img_3d[10, :, :])# (img_1)#
    # test_img_3d_orientations(img_3d)
    test_fft_energy_keypoints(img_1, img_2)


if __name__ == "__main__":
    main()
