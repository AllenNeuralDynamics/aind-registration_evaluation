"""
Sampling utility functions
"""

import functools
from typing import List, Optional, Tuple

import numpy as np
from natsort import natsorted
from scipy.ndimage import sobel

from aind_registration_evaluation._shared.types import ArrayLike


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


def kd_butterworth_filter(
    shape: Tuple[int],
    factor: float,
    order: float,
    high_pass: bool,
    real: bool,
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


def get_inversed_fft(fft_image: ArrayLike):
    """
    Converts an image that is in frequency
    space (FFTd) to the spatial domain.

    Parameters
    -----------
    fft_image: ArrayLike
        Image in the frequency domain
        that was computed using a Fast-Fourier
        Transform.

    Returns
    -----------
    ArrayLike:
        Image in the spatial domain.
    """
    is_real = np.isrealobj(fft_image)

    if is_real:
        return np.fft.irfftn(fft_image).real, is_real

    return np.fft.ifftn(fft_image).real, is_real


def get_fft(image: ArrayLike):
    """
    Computes the N-dimensional Discrete
    FFT in an image array.

    Returns
    ---------
    ArrayLike:
        FFTd image
    """
    return np.fft.fftn(image)


def kd_pad_fft_buterworth(image: ArrayLike, pad_width: Optional[int] = 0):
    """
    Computes the buterworth filter in the
    FFTd image.

    Parameters
    -----------
    image: ArrayLike
        N-Dimensional image data

    pad_width: Optional[int]
        Padding for the image

    Returns
    -----------
    Filtered image in the spatial
    domain
    """
    if pad_width > 0:
        image = np.pad(array=image, mode="constant", pad_width=pad_width)

    img_fft = get_fft(image)
    is_real = np.isrealobj(img_fft)

    butter_filter = kd_butterworth_filter(
        img_fft.shape, 0.005, 2.0, True, is_real, np.float16, True
    )

    inversed_img_fft, is_real_inversed = get_inversed_fft(
        butter_filter * img_fft
    )
    return inversed_img_fft


def derivate_image_axis(image: ArrayLike, axis: List[int]):  # , n_lvls=1):
    """
    Compute image derivatives

    Parameters
    -----------
    image: ArrayLike
        Image to apply nd derivatives

    axis: List[int]
        Axis where we want to derive
    """
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
    sum_powered_derivatives = np.sum(powered_derivatives, axis=0)
    # also called radius in spherical coordinates
    gradient_magnitude = np.sqrt(sum_powered_derivatives)
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
