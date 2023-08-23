""" Module to calculate Metrics between features
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Optional, Type, Union

import dask.array as da
import numpy as np

warnings.filterwarnings("ignore")
ArrayLike = Union[da.core.Array, np.ndarray]


class ImageMetrics(ABC):
    """
    Abstract class to calculate a metric in
    a pair of images
    """

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
            Window size (horizontal and vertical) of the patch
            extracted from the images based on a point located
            in the same coordinate system

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
            "ssd": self.mean_squared_error,
            "ssim": self.structural_similarity_index,
            "mae": self.mean_absolute_error,
            "r2": self.r2_score,
            "max_err": self.max_error,
            "ncc": self.normalized_cross_correlation,
            "mi": self.mutual_information,
            "nmi": self.normalized_mutual_information,
            "issm": self.information_theoretic_similarity,
            "psnr": self.peak_signal_to_noise_ratio,
            "fsim": self.feature_similarity_index_metric,
        }

        assert (
            self.__metric_type in self.__metrics_acronyms
        ), f"This metric has not been implemented yet {self.__metric_type}"

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
    def eps(self) -> np.float64:
        """
        Getter of the difference between 1.0 and the next
        smallest representable float larger than 1.0.
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
        Getter of compute dask parameter which is a
        boolean that dictates if dask graph will be
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
        Setter of compute dask parameter which is a boolean
        that dictates if dask graph will be computed
        in-place to calculate the metric.

        Parameters
        ------------------------
        new_compute_value: bool
            Boolean that dictates if dask graph will be
            computed in-place to calculate the metric.
            Set to False to get the dask jobs in order
            to compute them outside which would give
            computational advantages depending on the
            dask config and machine(s).

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
            Points that are inside of the intersection area.
            These do not go out from the area using
            the window size.

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
        Abstract method to compute a metric based
        on two patches from the images.

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
        pass

    @abstractmethod
    def structural_similarity_index(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Abstract method to compute the structural
        similarity index error metric.

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
        pass

    @abstractmethod
    def r2_score(self, patch_1: ArrayLike, patch_2: ArrayLike) -> float:
        """
        Abstract method to compute the r2 score error metric.

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
        pass

    @abstractmethod
    def max_error(self, patch_1: ArrayLike, patch_2: ArrayLike) -> float:
        """
        Abstract method to compute the max error error metric.

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
        pass

    @abstractmethod
    def normalized_cross_correlation(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Abstract method to compute the normalized
        cross correlation error metric.

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
        pass

    @abstractmethod
    def mutual_information(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Abstract method to compute the mutual
        information error metric.

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
            Float with the value of the mutual
            information error.
        """
        pass

    @abstractmethod
    def normalized_mutual_information(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Abstract method to compute the normalized
        mutual information error metric.

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
            mutual information error.
        """
        pass

    @abstractmethod
    def information_theoretic_similarity(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Abstract method to compute the information
        theoretic similarity error metric.

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

    @abstractmethod
    def peak_signal_to_noise_ratio(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Abstract method to compute a the peak
        signal to noise ratio error metric.

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
            Float with the value of the peak
            signal to noise ratio error.
        """
        pass

    @abstractmethod
    def feature_similarity_index_metric(
        self, patch_1: ArrayLike, patch_2: ArrayLike
    ) -> float:
        """
        Abstract method to compute a the feature
        similarity index metric error metric.

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
        pass

    def calculate_metric(self, point: np.array, transform: np.matrix) -> float:
        """
        Method to calculate the metric over a point
        and using a transformation matrix.

        Parameters
        ------------------------
        point: np.array
            2D/3D point in the image common coordinate system.

        transform: np.matrix
            Transformation matrix that will be
            applied in the intersection area.

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
                        start=point[idx_dim] - self.window_size,
                        stop=point[idx_dim] + self.window_size,
                        num=2 * self.window_size + 1,
                    ),
                    axis=0,
                )
                for idx_dim in range(len(image_2_shape))
            ]

            # Flattened points for get patches and
            # with extra dimension for meshgrid
            # points_per_dim_flattened = tuple(
            #     [np.squeeze(pt_flattened).flatten()
            #       for point_per_dim in points_per_dim
            # ]
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
        Method to recover the image dimensionality
        from a flattened array.

        Parameters
        ------------------------
        image_array: ArrayLike
            Array with the image data.

        Returns
        ------------------------
        ArrayLike
            Reshaped array based on the window size
            and dimensionality.
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
