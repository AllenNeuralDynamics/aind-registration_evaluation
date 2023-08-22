"""
    Module to calculate Metrics between features
    using in-memory methods
"""

import math
from typing import Optional, Tuple, Type

import cv2
import dask.array as da
import numpy as np
import sklearn.metrics as sk_metrics
from phasepack import phasecong
from skimage import metrics

from aind_registration_evaluation._shared.types import ArrayLike
from aind_registration_evaluation.io import ImageReader

from ._metric import ImageMetrics


class SmallImageMetrics(ImageMetrics):
    """
    Class for calculating a metric on small images
    """

    def __init__(
        self,
        image_1: ImageReader,
        image_2: ImageReader,
        metric_type: str,
        window_size: int,
    ):
        """
        Class constructor of small image metrics.

        Parameters
        ------------------------
        image_1: ArrayLike
            Data of image 1

        image_2: ArrayLike
            Data of image 2

        metric_type: str
            Acronym of the metric that will be computed

        window_size: int
            Window size (horizontal and vertical) of the
            patch extracted from the images based
            on a point located in the same coordinate system

            |---- (window_size * 2) + 1 ----|
            |                               |
            |                               |
            |                               |
            |              .                |
            |                               |
            |                               |
            |                               |
            |-------------------------------|

        compute_dask: Boolean
            Boolean that indicates if the dask graph will be computed
            in the case of a large image. Default True, False returns
            the dask jobs and they have to be executed outside.

        dtype: Type
            Dtype used to compute the metrics.

        """
        super().__init__(image_1, image_2, metric_type, window_size)

    def get_patches(
        self, windowed_points: np.ndarray, transform: np.matrix
    ) -> Tuple[np.ndarray]:
        """
        Method to get patches from the images using numpy.

        Parameters
        ------------------------
        windowed_points: np.array
            Points that are inside of the intersection
            area. These do not go out from the area
            using the window size.

        transform: np.matrix
            Transformation matrix that will be
            applied to get the patches

        Returns
        ------------------------
        Tuple[ArrayLike]
            Tuple with the patches.

        """

        point_1_windowed = windowed_points[0]
        point_2_windowed = windowed_points[1]

        # image_2_shape = self.image_2.shape
        len_dims = len(point_1_windowed)

        patch_1 = None
        patch_2 = None

        # Range of values in interval for each axis
        # dims = tuple(
        #     [
        #         np.linspace(
        #           start=0,
        #           stop=image_2_shape[idx_dim],
        #           num=image_2_shape[idx_dim]
        #         )
        #         for idx_dim in range(len(image_2_shape))
        #     ]
        # )

        if len_dims == 2:
            patch_1 = self.image_1[point_1_windowed[0], point_1_windowed[1]]
            patch_2 = self.image_2[point_2_windowed[0], point_2_windowed[1]]

        elif len_dims == 3:
            patch_1 = self.image_1[
                point_1_windowed[0], point_1_windowed[1], point_1_windowed[2]
            ]
            patch_2 = self.image_2[
                point_2_windowed[0], point_2_windowed[1], point_2_windowed[2]
            ]
        else:
            raise NotImplementedError("Only 2D or 3D dimensions are accepted")

        # try:
        #     patch_2 = scipy.interpolate.interpn(
        #         dims, self.image_2, point_2_windowed.transpose()
        #     )

        # except ValueError:
        #     return None, None

        return patch_1, patch_2

    def mean_squared_error(
        self, patch_1: np.ndarray, patch_2: np.ndarray
    ) -> float:
        """
        Method to compute the mean squared error metric using numpy.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1
            and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2
            and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the mean squeared error.
        """
        return sk_metrics.mean_squared_error(patch_1, patch_2)

    def structural_similarity_index(
        self, patch_1: np.ndarray, patch_2: np.ndarray
    ) -> float:
        """
        Method to compute the structural similarity
        index error metric using skimage.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1
            and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2
            and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the structural similarity index error.
        """
        patch_1 = self.recover_image_dimensionality(patch_1)
        patch_2 = self.recover_image_dimensionality(patch_2)

        return metrics.structural_similarity(patch_1, patch_2)

    def mean_absolute_error(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Method to compute the mean absolue error
        metric using numpy.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1
            and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2
            and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the mean absolute error.
        """
        return sk_metrics.mean_absolute_error(patch_1, patch_2)

    def r2_score(
        self, patch_1: da.core.Array, patch_2: da.core.Array
    ) -> float:
        """
        Method to compute the r2 score error
        metric using sklearn.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1
            and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2
            and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the r2 score error.
        """
        return sk_metrics.r2_score(patch_1, patch_2)

    def max_error(self, patch_1: ArrayLike, patch_2: ArrayLike) -> float:
        """
        Method to compute the max error metric with sklearn.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1
            and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2
            and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the max error.
        """
        return sk_metrics.max_error(patch_1, patch_2)

    def normalized_cross_correlation_traditional(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Method to compute the normalized cross
        correlation error metric using numpy.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1
            and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2
            and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the normalized
            cross correlation error.
        """

        if patch_1.ndim != 1:
            patch_1 = patch_1.flatten()

        if patch_2.ndim != 1:
            patch_2 = patch_2.flatten()

        if patch_1.shape != patch_2.shape:
            raise ValueError("Images must have the same shape")

        # Centering values after calculating mean
        centered_patch_1 = patch_1 - np.mean(patch_1, dtype=self.dtype)
        centered_patch_2 = patch_2 - np.mean(patch_2, dtype=self.dtype)

        numerator = np.transpose(centered_patch_1).dot(centered_patch_2)

        # denominator
        squared_centered_patch_1 = np.sqrt(
            np.transpose(centered_patch_1).dot(centered_patch_1),
            dtype=self.dtype,
        )
        squared_centered_patch_2 = np.sqrt(
            np.transpose(centered_patch_2).dot(centered_patch_2),
            dtype=self.dtype,
        )

        # Multiplicating squared centered patches
        denominator = squared_centered_patch_1 * squared_centered_patch_2

        return numerator / denominator

    def normalized_cross_correlation(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Method to compute the normalized cross correlation error
        metric based on ITK snap implementation using numpy.
        See detailed description in
        https://itk.org/Doxygen/html/classitk_1_1CorrelationImageToImageMetricv4.html

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1
            and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2
            and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the normalized
            cross correlation error.
        """

        if patch_1.ndim != 1:
            patch_1 = patch_1.flatten()

        if patch_2.ndim != 1:
            patch_2 = patch_2.flatten()

        if patch_1.shape != patch_2.shape:
            raise ValueError("Images must have the same shape")

        mean_patch_1 = np.mean(patch_1, dtype=self.dtype)
        mean_patch_2 = np.mean(patch_2, dtype=self.dtype)

        # Centering values after calculating mean
        centered_patch_1 = patch_1 - mean_patch_1
        centered_patch_2 = patch_2 - mean_patch_2

        numerator = np.inner(centered_patch_1, centered_patch_2) ** 2

        # Calculating 2-norm over centered patches - None means 2-norm
        norm_patch_1 = np.linalg.norm(centered_patch_1, ord=None) ** 2
        norm_patch_2 = np.linalg.norm(centered_patch_2, ord=None) ** 2

        # Multiplicating norms
        denominator = norm_patch_1 * norm_patch_2

        return -(numerator / denominator)

    def mutual_information(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Method to compute the mutual information error metric using numpy.
        Note: Check the used dtype to reach a higher precision in the metric

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1
            and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2
            and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the mutual information error.
        """
        joint_histogram, _, _ = np.histogram2d(patch_1, patch_2)
        pxy = joint_histogram / np.sum(joint_histogram, dtype=self.dtype)
        py = np.sum(pxy, axis=0, dtype=self.dtype)
        px = np.sum(pxy, axis=1, dtype=self.dtype)

        px_py = px[:, None] * py[None, :]
        non_zero_pxy_pos = pxy > 0

        mutual_information = np.sum(
            pxy[non_zero_pxy_pos]
            * np.log(
                pxy[non_zero_pxy_pos] / px_py[non_zero_pxy_pos],
                dtype=self.dtype,
            ),
            dtype=self.dtype,
        )

        return mutual_information

    def normalized_mutual_information(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Method to compute the mutual information error metric using numpy.
        Note: Check the used dtype to reach a higher precision in the metric

        See: Normalised Mutual Information of: A normalized entropy
        measure of 3-D medical image alignment,
        Studholme,  jhill & jhawkes (1998).

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1
            and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2
            and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the mutual information error.
        """

        joint_histogram, _, _ = np.histogram2d(patch_1, patch_2)

        # Compute marginal histograms
        joint_histogram = joint_histogram + self.eps
        joint_histogram = joint_histogram / np.sum(
            joint_histogram, dtype=self.dtype
        )

        py = np.sum(joint_histogram, axis=0, dtype=self.dtype)
        px = np.sum(joint_histogram, axis=1, dtype=self.dtype)

        numerator = np.sum(py * np.log(px), dtype=self.dtype) + np.sum(
            px * np.log(px), dtype=self.dtype
        )
        denominator = (
            np.sum(joint_histogram * np.log(joint_histogram), dtype=self.dtype)
            - 1
        )

        mutual_information = numerator / denominator

        return mutual_information

    def information_theoretic_similarity(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Method to compute the information theoretic similarity
        error metric with numpy as cv2. It is assumed that the e
        term refers to euler number from the paper.
        Based on https://github.com/up42/image-similarity-measures package


        Mohammed Abdulameer Aljanabi, Zahir M. Hussain, Noor Abd
        Alrazak Shnain & Song Feng Lu (2019) Design of a hybrid measure
        for image similarity: a statistical, algebraic,
        and information-theoretic approach, European Journal of Remote
        Sensing, 52:sup4, 2-15, DOI: 10.1080/22797254.2019.1628617

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1
            and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2
            and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the information
            theoretic similarity error.
        """

        def _ehs(x: np.ndarray, y: np.ndarray):
            """
            Entropy-Histogram Similarity measure
            """
            H = (np.histogram2d(x.flatten(), y.flatten()))[0] + math.e
            return -np.sum(np.nan_to_num(H * np.log2(H)))

        def _edge_c(x: np.ndarray, y: np.ndarray):
            """
            Edge correlation coefficient based on Canny detector
            """

            # Adding thresholding to better find edges
            # high_thresh, thresh_im = cv2.threshold(
            # x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            # )
            # lowThresh = 0.5*high_thresh

            balance = 1.0  # 0.0625

            lowThresh = 100
            high_thresh = 255
            g = cv2.Canny(
                (x * balance).astype(np.uint8), lowThresh, high_thresh
            )
            h = cv2.Canny(
                (y * balance).astype(np.uint8), lowThresh, high_thresh
            )

            g0 = np.mean(g)
            h0 = np.mean(h)

            numerator = np.sum((g - g0) * (h - h0))

            # Denominator is close to 0 when there are
            # no features that has edges in the image
            denominator = np.sqrt(
                np.sum(np.square(g - g0)) * np.sum(np.square(h - h0))
            )

            return numerator / denominator

        A = 0.3
        B = 0.5
        C = 0.7

        ehs_val = _ehs(patch_1, patch_2)
        canny_val = _edge_c(patch_1, patch_2)

        numerator = canny_val * ehs_val * (A + B) + math.e
        denominator = (
            A * canny_val * ehs_val
            + B * ehs_val
            + C * metrics.structural_similarity(patch_1, patch_2)
            + math.e
        )

        return np.nan_to_num(numerator / denominator)

    def peak_signal_to_noise_ratio(
        self, patch_1: ArrayLike, patch_2: ArrayLike, img_max_val: float = 255
    ) -> float:
        """
        Method to compute a the peak signal to noise ratio
        error metric using numpy.

        See: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1
            and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2
            and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the peak signal
            to noise ratio error.
        """
        mse = self.mean_squared_error(patch_1, patch_2)
        psnr = 20 * np.log10(
            img_max_val / np.sqrt(mse, dtype=self.dtype), dtype=self.dtype
        )

        return psnr

    def feature_similarity_index_metric(
        self,
        patch_1: ArrayLike,
        patch_2: ArrayLike,
        T1: float = 0.85,
        T2: float = 160,
    ) -> float:
        """
        Method to compute a the feature similarity index metric
        error metric using numpy and cv2.

        See: L. Zhang, L. Zhang, X. Mou and D. Zhang, "FSIM: A Feature
        Similarity Index for Image Quality Assessment," in IEEE Transactions
        on Image Processing, vol. 20, no. 8, pp. 2378-2386, Aug. 2011,
        doi: 10.1109/TIP.2011.2109730.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1
            and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2
            and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the feature similarity
            index metric error.
        """

        def numerical_gradient_magnitude(
            patched_image: ArrayLike,
            image_depth: Type = cv2.CV_16U,
            method: str = "scharr",
        ):
            """
            Computes the gradient magnitude for an image

            Parameters
            -------------
            patched_image: ArrayLike
                Patch of an image

            image_depth: Type
                Image depth. Default: 16 bit

            Method: str
                Method to compute the gradient magnitude
                Default: scharr
            """

            if method == "sobel":
                gradient_magnitude_function = cv2.Sobel

            elif method == "scharr":
                gradient_magnitude_function = cv2.Scharr

            else:
                raise NotImplementedError(
                    "Accepted gradient magnitude methods are scharr or sobel"
                )

            x_gradient = gradient_magnitude_function(
                patched_image, image_depth, 1, 0
            )
            y_gradient = gradient_magnitude_function(
                patched_image, image_depth, 0, 1
            )

            return np.sqrt(x_gradient**2 + y_gradient**2, dtype=self.dtype)

        def numerical_similarity_measure(
            patch_1: ArrayLike, patch_2: ArrayLike, C: float
        ) -> float:
            """
            Computes the numerical similarity measure

            Parameters
            --------------
            patch_1: ArrayLike
                Patch from image 1

            patch_2: ArrayLike
                Patch from image 2

            C: float
                Constant value

            Returns
            --------------
            float
                Computed value for the similarity
                measure
            """

            numerator = 2 * patch_1 * patch_2 + C
            denominator = patch_1**2 + patch_2**2 + C

            return numerator / denominator

        # Importance of Phase Congruence and Gradient Magnitud features
        alpha = 1
        beta = 1

        patch_1 = self.recover_image_dimensionality(patch_1)
        patch_2 = self.recover_image_dimensionality(patch_2)

        # Calculating phase congruency
        # Adding the list of size 6 in 4th position
        # which corresponds to the phase congruency

        pc_pos = 4
        phase_congruence_patch_1 = phasecong(
            patch_1, nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978
        )[pc_pos]

        phase_congruence_patch_2 = phasecong(
            patch_2, nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978
        )[pc_pos]

        # Sum phase congruencies per orientation
        phase_congruence_patch_1_sum = np.sum(
            phase_congruence_patch_1, axis=0, dtype=self.dtype
        )
        phase_congruence_patch_2_sum = np.sum(
            phase_congruence_patch_2, axis=0, dtype=self.dtype
        )

        # Getting edges by grandient magnitude
        # Using Scharr by default as suggested by authors
        gradient_magnitude_patch_1 = numerical_gradient_magnitude(patch_1)
        gradient_magnitude_patch_2 = numerical_gradient_magnitude(patch_2)

        S_phase_congruency = numerical_similarity_measure(
            phase_congruence_patch_1_sum, phase_congruence_patch_2_sum, T1
        )
        S_gradient_magnitude = numerical_similarity_measure(
            gradient_magnitude_patch_1, gradient_magnitude_patch_2, T2
        )

        # Following formula described in the paper
        S_l = (S_phase_congruency**alpha) * (S_gradient_magnitude**beta)

        numerator = np.sum(
            S_l
            * np.maximum(
                phase_congruence_patch_1_sum, phase_congruence_patch_2_sum
            )
        )

        # Adding eps to avoid Zero-Division
        denominator = (
            np.sum(
                np.maximum(
                    phase_congruence_patch_1_sum, phase_congruence_patch_2_sum
                )
            )
            + self.eps
        )
        fsim = numerator / denominator
        return fsim


# TODO integrate feature space metric into metric's classes
def compute_feature_space_distances(
    features_1: Tuple[np.array],
    features_2: Tuple[np.array],
    feature_axis: Optional[int] = 0,
    cartesian_axis: Optional[int] = 1,
    feature_weight: Optional[float] = 0.2,
) -> np.array:
    """
    Compute feature space distance metric
    between two set of features extracted
    from points in images.

    Parameters
    -----------
    features_1: Tuple[np.array]
        Tuple that contains two arrays,
        one is the feature vector that
        represents that point in space
        while the second array represents
        the location in cartesian space
        of that point for the image 1

    features_2: Tuple[np.array]
        Tuple that contains two arrays,
        one is the feature vector that
        represents that point in space
        while the second array represents
        the location in cartesian space
        of that point for the image 2

    feature_axis: Optional[int]
        Axis in the tuple where the
        feature vectors are stored.
        Default: 0

    cartesian_axis: Optional[int]
        Axis in the tuple where the
        cartesian locations are stored.
        Default: 1

    feature_weight: Optional[float]
        Weight used in the metric for
        the feature vector. The cartesian
        weight is calculated using
        (1 - feature_weight)
        Default: 0.2

    Raises
    -----------
    ValueError:
        - If feature_axis or cartesian_axis
        parameters are not in axis 0 or 1.
        - If feature_weight is not in the
        range 0.0 - 1.0.


    Returns
    -----------
    np.narray:
        Keypoint distances of each point
        from image 1 to image 2

    """

    if feature_weight < 0 or feature_weight > 1:
        raise ValueError("Feature weight range is from 0.0 to 1.0")

    if cartesian_axis not in [0, 1]:
        raise ValueError("Provide a valid axis for cartesian_axis")

    if feature_axis not in [0, 1]:
        raise ValueError("Provide a valid axis for cartesian_axis")

    # Features have the feature vector and cartesian location of the point
    # in cartesian space.
    cartesian_weight = 1 - feature_weight

    # Feature distances -> save all differences left points -> right points
    keypoint_distances_shape = (
        features_1[feature_axis].shape[0],
        features_2[feature_axis].shape[0],
    )

    feat_distances = np.array([], dtype=np.float32)
    loc_distances = np.array([], dtype=np.float32)

    for feat_idx in range(features_1[feature_axis].shape[0]):
        feat_dif = (
            features_2[feature_axis] - features_1[feature_axis][feat_idx]
        )
        feat_dif = np.power(feat_dif, 2).sum(axis=1).flatten()
        loc_dif = np.sqrt(
            np.sum(
                np.power(
                    features_2[cartesian_axis]
                    - features_1[cartesian_axis][feat_idx],
                    2,
                ),
                axis=-1,
            )
        )

        feat_distances = np.append(feat_distances, feat_dif)
        loc_distances = np.append(loc_distances, loc_dif)

    # Normalization
    feat_distances = feat_distances / feat_distances[feat_distances.argmax()]
    loc_distances = loc_distances / loc_distances[loc_distances.argmax()]

    feat_distances = feat_distances.reshape(keypoint_distances_shape)
    loc_distances = loc_distances.reshape(keypoint_distances_shape)

    keypoint_distances = (feature_weight * feat_distances) + (
        cartesian_weight * loc_distances
    )
    return keypoint_distances


def get_pairs_from_distances(
    distances: np.array,
    delete_points: Optional[bool] = True,
    metric_threshold: Optional[float] = 0.2,
) -> dict:
    """
    Get point pairs based on best distance

    Parameters
    -----------
    distances: np.array
        Array of NxM where N is the number
        of keypoints identified for image 1
        and M the keypoints for image 2

    delete_points: Optional[bool]
        Boolean that indicates if only
        one match point is returned. In other
        words, if it's true returns 1-1 match,
        N-1 match otherwise.
        Default: True

    Returns
    -----------
    Dict
        Dictionary with keys pointing to the
        keypoint indice on the left image and
        value pointing to the keypoint indice
        of the right image
    """

    pairs = {}
    right_assigned_points = {}

    for left_key_idx in range(distances.shape[0]):
        right_min_idx = distances[left_key_idx].argmin()
        right_min_distance = distances[left_key_idx][right_min_idx]

        if right_min_distance < metric_threshold:
            old_assignment = (
                right_assigned_points.get(right_min_idx)
                if delete_points
                else None
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

    print(right_assigned_points)
    return pairs
