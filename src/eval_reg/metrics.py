""" Modules to calculate Metrics between features
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Type, Union

import dask.array as da
import numpy as np
import scipy
import sklearn.metrics as sk_metrics
from dask import delayed
from io_utils import ImageReader
from skimage import metrics

ArrayLike = Union[da.core.Array, np.ndarray]


class ImageMetrics(ABC):
    def __init__(
        self,
        image_1: ImageReader,
        image_2: ImageReader,
        metric_type: str,
        compute_dask: Optional[bool] = True,
        dtype: Optional[Type] = np.float64,
    ):
        self.__image_1 = image_1
        self.__image_2 = image_2
        self.__metric_type = metric_type
        self.__compute_dask = compute_dask
        self.__dtype = dtype
        self.__metrics_acronyms = {
            "SSD": self.mean_squared_error,
            "SSIM": self.structural_similarity_index,
            "MAE": self.mean_absolute_error,
            "R2": self.r2_score,
            "MAX_ERR": self.max_error,
            "NCC": self.normalized_cross_correlation,
            "MI": self.mutual_information,
            "NMI": self.normalized_mutual_information,
        }

        assert (
            self.__metric_type in self.__metrics_acronyms
        ), "This metric has not been implemented yet"

    @property
    def image_1(self) -> ImageReader:
        return self.__image_1

    @image_1.setter
    def image_1(self, new_image_1: ImageReader) -> None:
        self.__image_1 = new_image_1

    @property
    def image_2(self) -> ImageReader:
        return self.__image_2

    @image_2.setter
    def image_2(self, new_image_2: ImageReader) -> None:
        self.__image_2 = new_image_2

    @property
    def metric_type(self) -> str:
        return self.__metric_type

    @metric_type.setter
    def metric_type(self, new_metric_type: str) -> None:
        self.__metric_type = new_metric_type

    @property
    def compute_dask(self) -> str:
        return self.__compute_dask

    @compute_dask.setter
    def compute_dask(self, new_compute_value: bool) -> None:
        self.__compute_dask = new_compute_value

    @property
    def dtype(self) -> Type:
        return self.__dtype

    @dtype.setter
    def compute(self, new_dtype: Type) -> None:
        self.__dtype = new_dtype

    @abstractmethod
    def get_patches(
        self, windowed_points: np.array, transform: np.matrix
    ) -> Any:
        pass

    def compute_metric_for_patch(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
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
        pass

    @abstractmethod
    def mean_absolute_error(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        pass

    @abstractmethod
    def structural_similarity_index(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        pass

    @abstractmethod
    def r2_score(self, patch_1: ArrayLike, patch_2: ArrayLike) -> float:
        pass

    @abstractmethod
    def max_error(self, patch_1: ArrayLike, patch_2: ArrayLike) -> float:
        pass

    @abstractmethod
    def normalized_cross_correlation(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        pass

    @abstractmethod
    def mutual_information(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        pass

    def calculate_metrics(
        self, point: np.array, transform: np.matrix, window_size: int
    ) -> float:
        image_2_shape = self.__image_2.shape

        if window_size == 0:
            pass

        else:
            # XY or XYZ
            points_per_dim = [
                np.expand_dims(
                    np.linspace(
                        point[idx_dim] - window_size,
                        point[idx_dim] + window_size,
                        2 * window_size + 1,
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
            # print(grid_per_dim[0], len(grid_per_dim), point)

            point_1_windowed = np.vstack(grid_per_dim)
            homogenous_pts = np.matrix(
                np.vstack([point_1_windowed, np.ones(grid_per_dim[0].shape)])
            )

            point_2_windowed = (np.linalg.inv(transform) * homogenous_pts)[
                : len(image_2_shape), :
            ]
            patch_1, patch_2 = self.get_patches(
                [
                    point_1_windowed.astype(np.int32),
                    point_2_windowed.astype(np.int32),
                ],
                transform,
            )

            if isinstance(patch_1, type(None)):
                return None

            return self.compute_metric_for_patch(patch_1, patch_2)


# We're working with dask for large images
class LargeImageMetrics(ImageMetrics):
    def __init__(
        self, image_1: ImageReader, image_2: ImageReader, metric_type: str
    ):
        super().__init__(image_1, image_2, metric_type)

    def get_patches(
        self, windowed_points: np.ndarray, transform: np.matrix
    ) -> Tuple[delayed]:

        point_1_windowed = windowed_points[0]
        point_2_windowed = windowed_points[1]

        image_2_shape = self.image_2.shape
        len_dims = len(point_1_windowed)

        patch_1 = None
        patch_2 = None

        dims = tuple(
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

        return patch_1, patch_2

    def mean_squared_error(
        self, patch_1: da.core.Array, patch_2: da.core.Array
    ) -> float:
        error = da.map_blocks(
            lambda a, b: (a - b) ** 2, patch_1, patch_2, dtype=self.dtype
        )
        # error.visualize()
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
        error = da.map_blocks(
            lambda a, b: abs(a - b), patch_1, patch_2, dtype=self.dtype
        )
        # error.visualize()
        value_error = None
        try:
            value_error = error.mean()

            if self.compute_dask:
                value_error = value_error.compute()

        except ValueError:
            value_error = None

        return value_error

    def structural_similarity_index(
        self, patch_1: da.core.Array, patch_2: da.core.Array
    ) -> float:

        value_error = None

        try:
            patch_2 = da.from_delayed(
                patch_2, shape=patch_1.shape, dtype=patch_1.dtype
            )
            value_error = delayed(
                metrics.structural_similarity(patch_1, patch_2)
            )

            if self.compute_dask:
                value_error = value_error.compute()

        except ValueError:
            value_error = None

        return value_error

    def r2_score(
        self, patch_1: da.core.Array, patch_2: da.core.Array
    ) -> float:
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
        "See detailed description in https://itk.org/Doxygen/html/classitk_1_1CorrelationImageToImageMetricv4.html"
        value_error = None

        try:
            patch_2 = da.from_delayed(
                patch_2, shape=patch_1.shape, dtype=patch_1.dtype
            )
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
        # Limitation with dask mutual information, computationally expensive
        # since we have to go 3 times per patch of data to calculate the joint histogram
        # One for min, one for max and one for hist
        value_error = None

        try:

            patch_2 = da.from_delayed(
                patch_2, shape=patch_1.shape, dtype=patch_1.dtype
            )

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
        # Limitation with dask mutual information, computationally expensive
        # since we have to go 3 times per patch of data to calculate the joint histogram
        # One for min, one for max and one for hist
        value_error = None

        try:
            # Normalised Mutual Information of: A normalized entropy measure of 3-D medical image alignment, Studholme,  jhill & jhawkes (1998).
            eps = np.finfo(self.dtype).eps

            patch_2 = da.from_delayed(
                patch_2, shape=patch_1.shape, dtype=patch_1.dtype
            )

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
            joint_histogram += eps

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


class SmallImageMetrics(ImageMetrics):
    def __init__(
        self, image_1: ImageReader, image_2: ImageReader, metric_type: str
    ):
        super().__init__(image_1, image_2, metric_type)

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
        return metrics.mean_squared_error(patch_1, patch_2)

    def structural_similarity_index(
        self, patch_1: np.ndarray, patch_2: np.ndarray
    ) -> float:
        return metrics.structural_similarity(patch_1, patch_2)

    def mean_absolute_error(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        return sk_metrics.mean_absolute_error(patch_1, patch_2)

    def r2_score(
        self, patch_1: da.core.Array, patch_2: da.core.Array
    ) -> float:
        return sk_metrics.r2_score(patch_1, patch_2)

    def max_error(self, patch_1: ArrayLike, patch_2: ArrayLike) -> float:
        return sk_metrics.max_error(patch_1, patch_2)

    def normalized_cross_correlation_traditional(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:

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
        norm_patch_1 = np.sqrt(
            np.transpose(centered_patch_1).dot(centered_patch_1),
            dtype=self.dtype,
        )
        norm_patch_2 = np.sqrt(
            np.transpose(centered_patch_2).dot(centered_patch_2),
            dtype=self.dtype,
        )

        # Multiplicating norms
        denominator = norm_patch_1 * norm_patch_2

        return numerator / denominator

    def normalized_cross_correlation(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        "See detailed description in https://itk.org/Doxygen/html/classitk_1_1CorrelationImageToImageMetricv4.html"

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

        # float16 by default, for a higher precision it could be increased
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

        # Normalised Mutual Information of: A normalized entropy measure of 3-D medical image alignment, Studholme,  jhill & jhawkes (1998).
        eps = np.finfo(self.dtype).eps
        joint_histogram, _, _ = np.histogram2d(patch_1, patch_2)

        # Compute marginal histograms
        joint_histogram = joint_histogram + eps
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


class ImageMetricsFactory:
    def __init__(self):
        self.__array_type = [da.core.Array, np.ndarray]

        self.factory = {
            da.core.Array: LargeImageMetrics,
            np.ndarray: SmallImageMetrics,
        }

    @property
    def array_type(self) -> List:
        return self.__array_type

    def create(
        self, image_1: ImageReader, image_2: ImageReader, metric_type: str
    ) -> ImageMetrics:
        image_type = type(image_1)

        if image_type not in self.__array_type:
            raise NotImplementedError(
                f"Image array type {image_type} not supported"
            )

        return self.factory[image_type](image_1, image_2, metric_type)
