"""
    Module to calculate Metrics between features
    using dask
"""

import warnings
from typing import Tuple, Type, Union

import cv2
import dask.array as da
import numpy as np
from phasepack import phasecong
from skimage import metrics

from aind_registration_evaluation.io import ImageReader

from ._metric import ImageMetrics

warnings.filterwarnings("ignore")
ArrayLike = Union[da.core.Array, np.ndarray]


# We're working with dask for large images
class LargeImageMetrics(ImageMetrics):
    """
    Class for calculating a metric on large images
    """

    def __init__(
        self,
        image_1: ImageReader,
        image_2: ImageReader,
        metric_type: str,
        window_size: int,
    ):
        """
        Class constructor of large image metrics.

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
    ) -> Tuple[ArrayLike]:
        """
        Method to get patches from the images using dask.

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

        # dims = list(
        #     [
        #         da.from_array(
        #             np.linspace(
        #                 0, image_2_shape[idx_dim], image_2_shape[idx_dim]
        #             )
        #         )
        #         for idx_dim in range(len(image_2_shape))
        #     ]
        # )

        if len_dims == 2:
            patch_1 = self.image_1.vindex[
                point_1_windowed[0], point_1_windowed[1]
            ]
            patch_2 = self.image_2.vindex[
                point_2_windowed[0], point_2_windowed[1]
            ]

        elif len_dims == 3:
            patch_1 = self.image_1.vindex[
                point_1_windowed[0], point_1_windowed[1], point_1_windowed[2]
            ]
            patch_2 = self.image_2.vindex[
                point_2_windowed[0], point_2_windowed[1], point_2_windowed[2]
            ]
        else:
            raise NotImplementedError("Only 2D or 3D dimensions are accepted")

        # Send patch without computing # TODO Return a dask array
        # patch_2 = delayed(scipy.interpolate.interpn)(
        #     dims, self.image_2, point_2_windowed.transpose()
        # )

        # patch_2 = da.from_delayed(
        #     patch_2, shape=patch_1.shape, dtype=patch_1.dtype
        # )

        return patch_1, patch_2

    def mean_squared_error(
        self, patch_1: da.core.Array, patch_2: da.core.Array
    ) -> float:
        """
        Method to compute the mean squared
        error metric using dask.

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
        error = da.map_blocks(
            lambda a, b: (a - b) ** 2, patch_1, patch_2, dtype=self.dtype
        )

        value_error = None
        try:
            value_error = error.mean(axis=0)

            if self.compute_dask:
                value_error = value_error.compute()

        except ValueError:
            value_error = None

        return value_error

    def mean_absolute_error(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Method to compute the mean absolue
        error metric using dask.

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
        error = da.map_blocks(
            lambda a, b: da.abs(a - b), patch_1, patch_2, dtype=self.dtype
        )

        value_error = None
        try:
            value_error = da.average(error, weights=None, axis=0)

            if self.compute_dask:
                value_error = value_error.compute()

        except ValueError:
            value_error = None

        return value_error

    def structural_similarity_index(
        self, patch_1: da.core.Array, patch_2: da.core.Array
    ) -> float:
        """
        Method to compute the structural similarity
        index error metric using dask.

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
            Float with the value of the structural
            similarity index error.
        """

        value_error = None

        try:
            value_error = da.map_blocks(
                lambda a, b: metrics.structural_similarity(
                    a,
                    b,
                    # Activate these parameters to match
                    # original matlab paper , check data type
                    # gaussian_weights=True,
                    # sigma=1.5,
                    # use_sample_covariance=False,
                    # data_range=255
                ),
                patch_1,
                patch_2,
                dtype=self.dtype,
            )

            if self.compute_dask:
                value_error = value_error.compute()

        except ValueError:
            value_error = None

        return value_error

    def r2_score(
        self, patch_1: da.core.Array, patch_2: da.core.Array
    ) -> float:
        """
        Method to compute the r2 score
        error metric using dask.

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
        value_error = None

        if len(patch_1.shape) == 1:
            patch_1 = patch_1.reshape((-1, 1))
            patch_2 = patch_2.reshape((-1, 1))

        weight = 1.0

        # TODO find a way in how num and den is computed in one run
        numerator = (
            (weight * (patch_1 - patch_2) ** 2)
            .sum(axis=0, dtype=self.dtype)
            .compute()
        )
        denominator = (
            (weight * (patch_1 - patch_2.mean(axis=0)) ** 2)
            .sum(axis=0, dtype=self.dtype)
            .compute()
        )

        nonzero_denominator = denominator != 0
        nonzero_numerator = numerator != 0
        valid_score = nonzero_denominator & nonzero_numerator

        r2_scores = da.ones([patch_1.shape[1]])
        with np.errstate(all="ignore"):
            r2_scores[valid_score] = 1 - (
                numerator[valid_score] / denominator[valid_score]
            )
            r2_scores[nonzero_numerator & ~nonzero_denominator] = 0.0

        value_error = r2_scores.mean(axis=0)
        if self.compute_dask:
            value_error = value_error.compute()

        return value_error

    def max_error(self, patch_1: ArrayLike, patch_2: ArrayLike) -> float:
        """
        Method to compute the max error metric with dask.

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
        value_error = None

        try:
            value_error = da.map_blocks(
                lambda a, b: abs(a - b), patch_1, patch_2, dtype=self.dtype
            )

            value_error = da.max(value_error)

            if self.compute_dask:
                value_error = value_error.compute()

        except ValueError:
            value_error = None

        return value_error

    def normalized_cross_correlation(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Method to compute the normalized cross correlation
        error metric based on ITK snap implementation using dask.
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
        value_error = None

        try:
            if patch_1.ndim != 1:
                patch_1 = patch_1.flatten()

            if patch_2.ndim != 1:
                patch_2 = patch_2.flatten()

            mean_patch_1 = da.mean(patch_1, dtype=self.dtype)
            mean_patch_2 = da.mean(patch_2, dtype=self.dtype)

            # Centering values after calculating mean
            centered_patch_1 = da.map_blocks(
                lambda a: a - mean_patch_1, patch_1, dtype=self.dtype
            )
            centered_patch_2 = da.map_blocks(
                lambda a: a - mean_patch_2, patch_2, dtype=self.dtype
            )

            numerator = da.dot(centered_patch_1, centered_patch_2) ** 2

            # Calculating 2-norm over centered patches - None means 2-norm
            norm_patch_1 = da.linalg.norm(centered_patch_1, ord=None) ** 2
            norm_patch_2 = da.linalg.norm(centered_patch_2, ord=None) ** 2

            # Multiplicating norms
            denominator = norm_patch_1 * norm_patch_2

            value_error = -(numerator / denominator)

            if self.compute_dask:
                value_error = value_error.compute()

        except ValueError:
            value_error = None

        return value_error

    def mutual_information(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Method to compute the mutual information
        error metric using dask.

        Note: Limitation with dask mutual information:
        it is computationally expensive since we have
        to go 3 times per patch of data to calculate
        the joint histogram. One for min, one for max
        and one for hist.

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

        value_error = None

        try:
            range_bin_patch_1 = [
                da.min(patch_1).compute(),
                da.max(patch_1).compute(),
            ]
            range_bin_patch_2 = [
                da.min(patch_2).compute(),
                da.max(patch_2).compute(),
            ]

            range_bins = [range_bin_patch_1, range_bin_patch_2]

            joint_histogram, _, _ = da.histogram2d(
                patch_1, patch_2, bins=(10, 10), range=range_bins
            )

            pxy = joint_histogram / da.sum(joint_histogram, dtype=self.dtype)
            py = da.sum(pxy, axis=0, dtype=self.dtype)
            px = da.sum(pxy, axis=1, dtype=self.dtype)

            px_py = px[:, None] * py[None, :]
            non_zero_pxy_pos = pxy > 0

            value_error = da.sum(
                pxy[non_zero_pxy_pos]
                * da.log(
                    pxy[non_zero_pxy_pos] / px_py[non_zero_pxy_pos],
                    dtype=self.dtype,
                ),
                dtype=self.dtype,
            )

            if self.compute_dask:
                value_error = value_error.compute()

        except ValueError:
            value_error = None

        return value_error

    def normalized_mutual_information(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Method to compute the mutual information error metric using dask.

        Note 1: Check the used dtype to reach a higher precision in the metric

        Note 2: Limitation with dask normalized mutual information:
        it is computationally expensive since we have to go 3 times
        per patch of data to calculate the joint histogram.
        One for min, one for max and one for hist.

        See: Normalised Mutual Information of: A normalized entropy
        measure of 3-D medical image alignment, Studholme,  jhill &
        jhawkes (1998).

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

        value_error = None

        try:
            # Normalised Mutual Information of: A normalized entropy
            # measure of 3-D medical image alignment, Studholme,
            # jhill & jhawkes (1998).

            range_bin_patch_1 = [
                da.min(patch_1).compute(),
                da.max(patch_1).compute(),
            ]
            range_bin_patch_2 = [
                da.min(patch_2).compute(),
                da.max(patch_2).compute(),
            ]

            range_bins = [range_bin_patch_1, range_bin_patch_2]

            joint_histogram, _, _ = da.histogram2d(
                patch_1, patch_2, bins=(10, 10), range=range_bins
            )
            joint_histogram += self.eps

            joint_histogram = joint_histogram / da.sum(
                joint_histogram, dtype=self.dtype
            )

            py = da.sum(joint_histogram, axis=0, dtype=self.dtype)
            px = da.sum(joint_histogram, axis=1, dtype=self.dtype)

            numerator = da.sum(py * da.log(px), dtype=self.dtype) + da.sum(
                px * da.log(px), dtype=self.dtype
            )
            denominator = (
                da.sum(
                    joint_histogram * da.log(joint_histogram), dtype=self.dtype
                )
                - 1
            )

            value_error = numerator / denominator

            if self.compute_dask:
                value_error = value_error.compute()

        except ValueError:
            value_error = None

        return value_error

    def information_theoretic_similarity(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Method to compute the information theoretic
        similarity error metric with dask.

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
        pass

    def peak_signal_to_noise_ratio(
        self, patch_1: ArrayLike, patch_2: ArrayLike, img_max_val: float = 255
    ) -> float:
        """
        Method to compute a the peak signal to noise
        ratio error metric using dask.

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

        psnr = None

        if mse:
            psnr = 20 * da.log10(
                img_max_val / da.sqrt(mse, dtype=self.dtype), dtype=self.dtype
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
        Method to compute a the feature similarity index
        metric error metric using dask.

        See: L. Zhang, L. Zhang, X. Mou and D. Zhang, "FSIM: A
        Feature Similarity Index for Image Quality Assessment,"
        in IEEE Transactions on Image Processing, vol. 20, no. 8,
        pp. 2378-2386, Aug. 2011, doi: 10.1109/TIP.2011.2109730.

        Note: Currently not computed using dask since we need
        to convert phasecong to dask compatible if Sharmi agrees

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
            Float with the value of the feature
            similarity index metric error.
        """

        raise NotImplementedError(
            """Feature similarity index metric
            has not been yet implemented for large images"""
        )

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

            x_gradient = da.map_blocks(
                lambda a: gradient_magnitude_function(a, image_depth, 1, 0),
                patched_image,
                dtype=self.dtype,
            )
            y_gradient = da.map_blocks(
                lambda a: gradient_magnitude_function(a, image_depth, 0, 1),
                patched_image,
                dtype=self.dtype,
            )

            return da.sqrt(x_gradient**2 + y_gradient**2, dtype=self.dtype)

        def numerical_similarity_measure(
            patch_1: ArrayLike, patch_2: ArrayLike, C: float
        ):
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

        # TODO Convert phasecong dask compatible if Sharmi agrees
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

        numerator = da.sum(
            S_l
            * da.maximum(
                phase_congruence_patch_1_sum, phase_congruence_patch_2_sum
            )
        )

        # Adding eps to avoid Zero-Division
        denominator = (
            da.sum(
                da.maximum(
                    phase_congruence_patch_1_sum, phase_congruence_patch_2_sum
                )
            )
            + self.eps
        )

        print(numerator.compute(), denominator.compute())

        fsim = numerator / denominator

        try:
            fsim = fsim.compute()

        except ValueError as err:
            print(err)
            fsim = None

        return fsim
