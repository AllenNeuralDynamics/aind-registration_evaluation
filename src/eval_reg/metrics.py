""" Modules to calculate Metrics between features
"""

import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Type, Union

import cv2
import dask.array as da
import numpy as np
import scipy
import sklearn.metrics as sk_metrics
from dask import delayed
from io_utils import ImageReader
from phasepack import phasecong
from skimage import metrics

warnings.filterwarnings("ignore")
ArrayLike = Union[da.core.Array, np.ndarray]


class ImageMetrics(ABC):
    def __init__(
        self,
        image_1: ArrayLike,
        image_2: ArrayLike,
        metric_type: str,
        window_size: int,
        compute_dask: Optional[bool] = True,
        dtype: Optional[Type] = np.float64,
    ) -> None:
        """
        Class constructor.

        Parameters
        ------------------------
        image_1: ArrayLike
            Data of image 1

        image_2: ArrayLike
            Data of image 2

        metric_type: str
            Acronym of the metric that will be computed

        window_size: int
            Window size (horizontal and vertical) of the patch extracted from the images based
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
        self.__image_1 = image_1
        self.__image_2 = image_2
        self.__metric_type = metric_type
        self.__window_size = window_size
        self.__compute_dask = compute_dask
        self.__dtype = dtype

        # eps
        self.__eps = np.finfo(self.dtype).eps  # math.e

        self.__metrics_acronyms = {
            "SSD": self.mean_squared_error,
            "SSIM": self.structural_similarity_index,
            "MAE": self.mean_absolute_error,
            "R2": self.r2_score,
            "MAX_ERR": self.max_error,
            "NCC": self.normalized_cross_correlation,
            "MI": self.mutual_information,
            "NMI": self.normalized_mutual_information,
            "ISSM": self.information_theoretic_similarity,
            "PSNR": self.peak_signal_to_noise_ratio,
            "FSIM": self.feature_similarity_index_metric,
        }

        assert (
            self.__metric_type in self.__metrics_acronyms
        ), "This metric has not been implemented yet"

    @property
    def image_1(self) -> ArrayLike:
        """
        Getter of image 1. Returns the image data.

        Returns
        ------------------------
        ArrayLike
            Image data

        """
        return self.__image_1

    @image_1.setter
    def image_1(self, new_image_1: ArrayLike) -> None:
        """
        Setter of image 1.

        Parameters
        ------------------------
        new_image_1: ArrayLike
            New data of image 1

        """
        self.__image_1 = new_image_1

    @property
    def image_2(self) -> ArrayLike:
        """
        Getter of image 2. Returns the image data.

        Returns
        ------------------------
        ArrayLike
            Image data

        """
        return self.__image_2

    @image_2.setter
    def image_2(self, new_image_2: ArrayLike) -> None:
        """
        Setter of image 2.

        Parameters
        ------------------------
        new_image_2: ArrayLike
            New data of image 2

        """
        self.__image_2 = new_image_2

    @property
    def metric_type(self) -> str:
        """
        Getter of the metric.

        Returns
        ------------------------
        str
            String with the acronym of the used metric

        """
        return self.__metric_type

    @metric_type.setter
    def metric_type(self, new_metric_type: str) -> None:
        """
        Setter of the metric.

        Parameters
        ------------------------
        new_metric_type: str
            String with the acronym of the used metric

        """
        self.__metric_type = new_metric_type

    @property
    def window_size(self) -> int:
        """
        Getter of the window size.

        Returns
        ------------------------
        int
            Integer that represents the window size used over each axis

        """
        return self.__window_size

    @window_size.setter
    def window_size(self, new_window_size: int) -> None:
        """
        Setter of the window size.

        Parameters
        ------------------------
        new_window_size: int
            Integer that represents the window size used over each axis

        """
        self.__window_size = new_window_size

    @property
    def eps(self) -> np.float:
        """
        Getter of the difference between 1.0 and the next smallest representable float larger than 1.0.
        See: https://numpy.org/doc/stable/reference/generated/numpy.finfo.html

        Returns
        ------------------------
        np.float
            Integer that represents the window size used over each axis

        """
        return self.__eps

    @eps.setter
    def eps(self, new_dtype: Type) -> None:
        """
        Setter of the next smallest representable float larger than 1.0.
        See: https://numpy.org/doc/stable/reference/generated/numpy.finfo.html

        Parameters
        ------------------------
        new_dtype: Type
            Dtype that will be used to calculate eps.

        """
        self.__eps = np.finfo(new_dtype).eps

    @property
    def compute_dask(self) -> bool:
        """
        Getter of compute dask parameter which is a boolean that dictates if dask graph will be
        computed in-place to calculate the metric.

        Returns
        ------------------------
        bool
            Compute dask attribute

        """
        return self.__compute_dask

    @compute_dask.setter
    def compute_dask(self, new_compute_value: bool) -> None:
        """
        Setter of compute dask parameter which is a boolean that dictates if dask graph will be
        computed in-place to calculate the metric.

        Parameters
        ------------------------
        new_compute_value: bool
            Boolean that dictates if dask graph will be computed in-place to calculate the metric.
            Set to False to get the dask jobs in order to compute them outside which would give
            computational advantages depending on the dask config and machine(s).

        """
        self.__compute_dask = new_compute_value

    @property
    def dtype(self) -> Type:
        """
        Getter of the dtype attribute.

        Returns
        ------------------------
        Type
            Dtype used for computation.
        """
        return self.__dtype

    @dtype.setter
    def dtype(self, new_dtype: Type) -> None:
        """
        Setter of the dtype attribute.

        Parameters
        ------------------------
        new_dtype: Type
            Dtype used for computation.

        """
        self.__dtype = new_dtype

    @abstractmethod
    def get_patches(
        self, windowed_points: np.array, transform: np.matrix
    ) -> Any:
        """
        Abstract method to get patches from the images.

        Parameters
        ------------------------
        windowed_points: np.array
            Points that are inside of the intersection area. These do not go out
            from the area using the window size.

        transform: np.matrix
            Transformation matrix that will be applied to get the patches

        Returns
        ------------------------
        Any
            Patches from the images.

        """
        pass

    def compute_metric_for_patch(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Abstract method to compute a metric based on two patches from the images.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the computed metric.
        """
        met_value = None

        if self.__metric_type in self.__metrics_acronyms:
            met_value = self.__metrics_acronyms[self.__metric_type](
                patch_1, patch_2
            )

        return met_value

    @abstractmethod
    def mean_squared_error(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Abstract method to compute the mean squared error metric.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the mean squeared error.
        """
        pass

    @abstractmethod
    def mean_absolute_error(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Abstract method to compute the mean absolue error metric.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the mean absolute error.
        """
        pass

    @abstractmethod
    def structural_similarity_index(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Abstract method to compute the structural similarity index error metric.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the structural similarity index error.
        """
        pass

    @abstractmethod
    def r2_score(self, patch_1: ArrayLike, patch_2: ArrayLike) -> float:
        """
        Abstract method to compute the r2 score error metric.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the r2 score error.
        """
        pass

    @abstractmethod
    def max_error(self, patch_1: ArrayLike, patch_2: ArrayLike) -> float:
        """
        Abstract method to compute the max error error metric.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the max error.
        """
        pass

    @abstractmethod
    def normalized_cross_correlation(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Abstract method to compute the normalized cross correlation error metric.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the normalized cross correlation error.
        """
        pass

    @abstractmethod
    def mutual_information(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Abstract method to compute the mutual information error metric.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the mutual information error.
        """
        pass

    @abstractmethod
    def normalized_mutual_information(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Abstract method to compute the normalized mutual information error metric.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the normalized mutual information error.
        """
        pass

    @abstractmethod
    def information_theoretic_similarity(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Abstract method to compute the information theoretic similarity error metric.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the information theoretic similarity error.
        """
        pass

    @abstractmethod
    def peak_signal_to_noise_ratio(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Abstract method to compute a the peak signal to noise ratio error metric.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the peak signal to noise ratio error.
        """
        pass

    @abstractmethod
    def feature_similarity_index_metric(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Abstract method to compute a the feature similarity index metric error metric.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the feature similarity index metric error.
        """
        pass

    def calculate_metrics(
        self, point: np.array, transform: np.matrix
    ) -> float:
        """
        Method to calculate the metric over a point and using a transformation matrix.

        Parameters
        ------------------------
        point: np.array
            2D/3D point in the image common coordinate system.

        transform: np.matrix
            Transformation matrix that will be applied in the intersection area.

        Returns
        ------------------------
        float
            Float with the value of the computed metric.
        """

        image_2_shape = self.__image_2.shape

        if self.window_size == 0:
            pass

        else:
            # XY or XYZ
            points_per_dim = [
                np.expand_dims(
                    np.linspace(
                        point[idx_dim] - self.window_size,
                        point[idx_dim] + self.window_size,
                        2 * self.window_size + 1,
                    ),
                    axis=0,
                )
                for idx_dim in range(len(image_2_shape))
            ]

            # Flattened points for get patches and with extra dimension for meshgrid
            # points_per_dim_flattened = tuple(
            #     [np.squeeze(pt_flattened).flatten() for point_per_dim in points_per_dim]
            # )
            points_per_dim = tuple(points_per_dim)

            grid_per_dim = np.meshgrid(*points_per_dim, indexing="ij")
            grid_per_dim = [grid_dim.flatten() for grid_dim in grid_per_dim]

            point_1_windowed = np.vstack(grid_per_dim)
            homogenous_pts = np.matrix(
                np.vstack([point_1_windowed, np.ones(grid_per_dim[0].shape)])
            )

            point_2_windowed = np.array(
                np.linalg.inv(transform) * homogenous_pts
            )[: len(image_2_shape), :]

            patch_1, patch_2 = self.get_patches(
                [
                    point_1_windowed.astype(np.int32),
                    point_2_windowed.astype(np.int32),
                ],
                transform,
            )

            if isinstance(patch_1, type(None)):
                return None

            patch_2 = patch_2.astype(patch_1.dtype)

            return self.compute_metric_for_patch(patch_1, patch_2)

    def recover_image_dimensionality(
        self, image_array: ArrayLike
    ) -> ArrayLike:
        """
        Method to recover the image dimensionality from a flattened array.

        Parameters
        ------------------------
        image_array: ArrayLike
            Array with the image data.

        Returns
        ------------------------
        ArrayLike
            Reshaped array based on the window size and dimensionality.
        """

        # Image patch size in 2D
        reshape_size = (self.window_size * 2) + 1

        if len(self.image_1.shape) == 2:
            image_array = image_array.reshape(reshape_size, reshape_size)
        elif len(self.image_2.shape) == 3:
            image_array = image_array.reshape(
                reshape_size, reshape_size, reshape_size
            )

        else:
            raise ValueError(
                "Images with more than 3 dimensions are not accepted."
            )

        return image_array


# We're working with dask for large images
class LargeImageMetrics(ImageMetrics):
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
            Window size (horizontal and vertical) of the patch extracted from the images based
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
            Points that are inside of the intersection area. These do not go out
            from the area using the window size.

        transform: np.matrix
            Transformation matrix that will be applied to get the patches

        Returns
        ------------------------
        Tuple[ArrayLike]
            Tuple with the patches.

        """

        point_1_windowed = windowed_points[0]
        point_2_windowed = windowed_points[1]

        image_2_shape = self.image_2.shape
        len_dims = len(point_1_windowed)

        patch_1 = None
        patch_2 = None

        dims = list(
            [
                da.from_array(
                    np.linspace(
                        0, image_2_shape[idx_dim], image_2_shape[idx_dim]
                    )
                )
                for idx_dim in range(len(image_2_shape))
            ]
        )

        if len_dims == 2:
            patch_1 = self.image_1.vindex[
                point_1_windowed[0], point_1_windowed[1]
            ]
        elif len_dims == 3:
            patch_1 = self.image_1.vindex[
                point_1_windowed[0], point_1_windowed[1], point_1_windowed[2]
            ]
        else:
            raise NotImplementedError("Only 2D or 3D dimensions are accepted")

        # Send patch without computing # TODO Return a dask array
        patch_2 = delayed(scipy.interpolate.interpn)(
            dims, self.image_2, point_2_windowed.transpose()
        )

        patch_2 = da.from_delayed(
            patch_2, shape=patch_1.shape, dtype=patch_1.dtype
        )

        return patch_1, patch_2

    def mean_squared_error(
        self, patch_1: da.core.Array, patch_2: da.core.Array
    ) -> float:
        """
        Method to compute the mean squared error metric using dask.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

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
            value_error = error.mean()

            if self.compute_dask:
                value_error = value_error.compute()

        except ValueError:
            value_error = None

        return value_error

    def mean_absolute_error(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Method to compute the mean absolue error metric using dask.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

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
        Method to compute the structural similarity index error metric using dask.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the structural similarity index error.
        """

        value_error = None

        try:

            value_error = da.map_blocks(
                lambda a, b: metrics.structural_similarity(
                    a,
                    b,
                    # Activate these parameters to match original matlab paper , check data type
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
        Method to compute the r2 score error metric using dask.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

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

        try:

            patch_1_mean = patch_1.mean()

            numerator = da.map_blocks(
                lambda a, b: weight * (a - b) ** 2,
                patch_1,
                patch_2,
                dtype=self.dtype,
            ).sum(axis=0, dtype=self.dtype)[0]

            denominator = da.map_blocks(
                lambda a: weight * (a - patch_1_mean) ** 2,
                patch_1,
                dtype=self.dtype,
            ).sum(axis=0, dtype=self.dtype)[0]

            numerator = numerator.compute()
            denominator = denominator.compute()

            nonzero_denominator = denominator != 0
            nonzero_numerator = numerator != 0

            # Non-zero numerator and Non-zero denominator: use the formula
            if nonzero_denominator & nonzero_numerator:

                value_error = 1 - (numerator / denominator)

            # Non-zero Numerator and Zero Denominator: set values to 0.0 so it does not go to Inf
            if nonzero_numerator & ~nonzero_denominator:
                value_error = 0.0

            value_error = value_error

            if self.compute_dask:
                value_error = value_error.compute()

        except ValueError:
            value_error = None

        return value_error

    def max_error(self, patch_1: ArrayLike, patch_2: ArrayLike) -> float:
        """
        Method to compute the max error metric with dask.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

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
        Method to compute the normalized cross correlation error metric based on ITK snap implementation using dask.
        See detailed description in https://itk.org/Doxygen/html/classitk_1_1CorrelationImageToImageMetricv4.html

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the normalized cross correlation error.
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
        Method to compute the mutual information error metric using dask.

        Note: Limitation with dask mutual information: it is computationally expensive
        since we have to go 3 times per patch of data to calculate the joint histogram.
        One for min, one for max and one for hist.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

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

        Note 2: Limitation with dask normalized mutual information: it is computationally expensive
        since we have to go 3 times per patch of data to calculate the joint histogram.
        One for min, one for max and one for hist.

        See: Normalised Mutual Information of: A normalized entropy measure of 3-D medical image alignment,
        Studholme,  jhill & jhawkes (1998).

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the mutual information error.
        """

        value_error = None

        try:
            # Normalised Mutual Information of: A normalized entropy measure of 3-D medical image alignment, Studholme,  jhill & jhawkes (1998).

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
        Method to compute the information theoretic similarity error metric with dask.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the information theoretic similarity error.
        """
        pass

    def peak_signal_to_noise_ratio(
        self, patch_1: ArrayLike, patch_2: ArrayLike, img_max_val: float = 255
    ) -> float:
        """
        Method to compute a the peak signal to noise ratio error metric using dask.

        See: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the peak signal to noise ratio error.
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
        Method to compute a the feature similarity index metric error metric using dask.

        See: L. Zhang, L. Zhang, X. Mou and D. Zhang, "FSIM: A Feature Similarity Index for Image Quality Assessment,"
        in IEEE Transactions on Image Processing, vol. 20, no. 8, pp. 2378-2386, Aug. 2011, doi: 10.1109/TIP.2011.2109730.

        Note: Currently not computed using dask since we need to convert phasecong
        to dask compatible if Sharmi agrees

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the feature similarity index metric error.
        """

        raise NotImplementedError(
            "Feature similarity index metric has not been yet implemented for large images"
        )

        def numerical_gradient_magnitude(
            patched_image: ArrayLike,
            image_depth: Type = cv2.CV_16U,
            method: str = "scharr",
        ):

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

            numerator = 2 * patch_1 * patch_2 + C
            denominator = patch_1**2 + patch_2**2 + C

            return numerator / denominator

        # Importance of Phase Congruence and Gradient Magnitud features
        alpha = 1
        beta = 1

        patch_1 = self.recover_image_dimensionality(patch_1)
        patch_2 = self.recover_image_dimensionality(patch_2)

        # Calculating phase congruency
        # Adding the list of size 6 in 4th position which corresponds to the phase congruency

        # TODO Convert phasecong dask compatible if Sharmi agrees
        # pc_pos = 4
        # phase_congruence_patch_1 = phasecong(
        #     patch_1, nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978
        # )[pc_pos]

        # phase_congruence_patch_2 = phasecong(
        #     patch_2, nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978
        # )[pc_pos]

        # Sum phase congruencies per orientation
        phase_congruence_patch_1_sum = np.sum(
            phase_congruence_patch_1, axis=0, dtype=self.dtype
        )
        phase_congruence_patch_2_sum = np.sum(
            phase_congruence_patch_2, axis=0, dtype=self.dtype
        )

        # Getting edges by grandient magnitude - Using Scharr by default as suggested by authors
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


class SmallImageMetrics(ImageMetrics):
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
            Window size (horizontal and vertical) of the patch extracted from the images based
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

        point_1_windowed = windowed_points[0]
        point_2_windowed = windowed_points[1]

        image_2_shape = self.image_2.shape
        len_dims = len(point_1_windowed)

        patch_1 = None
        patch_2 = None

        # Range of values in interval for each axis
        dims = tuple(
            [
                np.linspace(0, image_2_shape[idx_dim], image_2_shape[idx_dim])
                for idx_dim in range(len(image_2_shape))
            ]
        )

        if len_dims == 2:
            patch_1 = self.image_1[point_1_windowed[0], point_1_windowed[1]]

        elif len_dims == 3:
            patch_1 = self.image_1[
                point_1_windowed[0], point_1_windowed[1], point_1_windowed[2]
            ]
        else:
            raise NotImplementedError("Only 2D or 3D dimensions are accepted")

        try:
            patch_2 = scipy.interpolate.interpn(
                dims, self.image_2, point_2_windowed.transpose()
            )

        except ValueError:
            return None, None

        return patch_1, patch_2

    def mean_squared_error(
        self, patch_1: np.ndarray, patch_2: np.ndarray
    ) -> float:
        """
        Method to compute the mean squared error metric using numpy.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the mean squeared error.
        """
        substact = np.subtract(patch_1, patch_2, dtype=self.dtype)
        squared_mean = np.square(substact).mean()
        return squared_mean

    def structural_similarity_index(
        self, patch_1: np.ndarray, patch_2: np.ndarray
    ) -> float:
        """
        Method to compute the structural similarity index error metric using skimage.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

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
        Method to compute the mean absolue error metric using numpy.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

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
        Method to compute the r2 score error metric using sklearn.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

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
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

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
        Method to compute the normalized cross correlation error metric using numpy.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the normalized cross correlation error.
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
        Method to compute the normalized cross correlation error metric based on ITK snap implementation using numpy.
        See detailed description in https://itk.org/Doxygen/html/classitk_1_1CorrelationImageToImageMetricv4.html

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the normalized cross correlation error.
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
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

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

        See: Normalised Mutual Information of: A normalized entropy measure of 3-D medical image alignment,
        Studholme,  jhill & jhawkes (1998).

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

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
        Method to compute the information theoretic similarity error metric with numpy as cv2.
        It is assumed that the e term refers to euler number from the paper.
        Based on https://github.com/up42/image-similarity-measures package


        Mohammed Abdulameer Aljanabi, Zahir M. Hussain, Noor Abd Alrazak Shnain & Song Feng Lu (2019)
        Design of a hybrid measure for image similarity: a statistical, algebraic, and information-theoretic approach,
        European Journal of Remote Sensing, 52:sup4, 2-15, DOI: 10.1080/22797254.2019.1628617

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the information theoretic similarity error.
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
            # high_thresh, thresh_im = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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

            # Denominator is close to 0 when there are no features that has edges in the image
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
        Method to compute a the peak signal to noise ratio error metric using numpy.

        See: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the peak signal to noise ratio error.
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
        Method to compute a the feature similarity index metric error metric using numpy and cv2.

        See: L. Zhang, L. Zhang, X. Mou and D. Zhang, "FSIM: A Feature Similarity Index for Image Quality Assessment,"
        in IEEE Transactions on Image Processing, vol. 20, no. 8, pp. 2378-2386, Aug. 2011, doi: 10.1109/TIP.2011.2109730.

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1 and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2 and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the feature similarity index metric error.
        """

        def numerical_gradient_magnitude(
            patched_image: ArrayLike,
            image_depth: Type = cv2.CV_16U,
            method: str = "scharr",
        ):

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
        ):

            numerator = 2 * patch_1 * patch_2 + C
            denominator = patch_1**2 + patch_2**2 + C

            return numerator / denominator

        # Importance of Phase Congruence and Gradient Magnitud features
        alpha = 1
        beta = 1

        patch_1 = self.recover_image_dimensionality(patch_1)
        patch_2 = self.recover_image_dimensionality(patch_2)

        # Calculating phase congruency
        # Adding the list of size 6 in 4th position which corresponds to the phase congruency

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

        # Getting edges by grandient magnitude - Using Scharr by default as suggested by authors
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


class ImageMetricsFactory:
    def __init__(self):
        """
        Class constructor of image metrics factory.
        """
        self.__array_type = [da.core.Array, np.ndarray]

        self.factory = {
            da.core.Array: LargeImageMetrics,
            np.ndarray: SmallImageMetrics,
        }

    @property
    def array_type(self) -> List:
        """
        Getter to return the array type.

        Returns
        ------------------------
        List
            List thar contains the accepted dtypes for the images

        """
        return self.__array_type

    def create(
        self,
        image_1: ArrayLike,
        image_2: ArrayLike,
        metric_type: str,
        window_size: int,
    ) -> ImageMetrics:
        """
        Method to create the image reader depending on the type of image:
        - large: Metrics will be computed using dask.
        - small: Metrics will be computed using numpy, sklearn, scipy or cv2 where it's needed.

        Parameters
        ------------------------
        image_1: ArrayLike
            2D/3D image data.

        image_2: ArrayLike
            2D/3D image data.

        Returns
        ------------------------
        ImageMetrics
            Class that contains the image metrics to be computed depending on the type of image.
            - large: LargeImageMetrics class
            - small: SmallImageMetrics class
        """
        image_type = type(image_1)

        if image_type not in self.__array_type:
            raise NotImplementedError(
                f"Image array type {image_type} not supported"
            )

        return self.factory[image_type](
            image_1, image_2, metric_type, window_size
        )
