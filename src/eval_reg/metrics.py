""" Modules to calculate Metrics between features
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

import dask.array as da
import numpy as np
import scipy
from dask import delayed
from io_utils import ImageReader
from skimage import metrics
import sklearn.metrics as sk_metrics

ArrayLike = Union[da.core.Array, np.ndarray]


class ImageMetrics(ABC):
    def __init__(
        self, image_1: ImageReader, image_2: ImageReader, metric_type: str
    ):
        self.__image_1 = image_1
        self.__image_2 = image_2
        self.__metric_type = metric_type
        self.__metrics_acronyms = {
            "SSD": self.mean_squared_error,
            "SSIM": self.structural_similarity_index,
            "MAE": self.mean_absolute_error,
            "R2": self.r2_score,
        }

        assert (
            self.__metric_type in self.__metrics_acronyms
        ), "Metrics has not been implemented yet"

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
        self, image_1: ArrayLike, image_2: ArrayLike
    ) -> float:
        pass

    @abstractmethod
    def r2_score(self, image_1: ArrayLike, image_2: ArrayLike) -> float:
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
        error = da.map_blocks(lambda a, b: (a - b) ** 2, patch_1, patch_2)
        # error.visualize()
        value_error = None
        try:
            value_error = error.mean().compute()
        except ValueError:
            value_error = None

        return value_error

    def mean_absolute_error(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        error = da.map_blocks(lambda a, b: abs(a - b), patch_1, patch_2)
        # error.visualize()
        value_error = None
        try:
            value_error = error.mean().compute()
        except ValueError:
            value_error = None

        return value_error

    def structural_similarity_index(
        self, patch_1: da.core.Array, patch_2: da.core.Array
    ) -> float:

        value_error = None

        try:
            patch_1 = patch_1.compute()
            patch_2 = patch_2.compute()

            value_error = metrics.structural_similarity(patch_1, patch_2)

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
                lambda a, b: weight * (a - b) ** 2, patch_1, patch_2
            ).sum(axis=0, dtype="f8")[0]
            denominator = da.map_blocks(
                lambda a: weight * (a - patch_1_mean) ** 2, patch_1
            ).sum(axis=0, dtype="f8")[0]

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
