"""
    Image metric's factory
"""
from typing import List

import dask.array as da
import numpy as np

from aind_registration_evaluation._shared.types import ArrayLike

from ._metric import ImageMetrics
from .large_scale import LargeImageMetrics
from .small_scale import SmallImageMetrics


class ImageMetricsFactory:
    """
    Class image metrics factory
    """

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
        - small: Metrics will be computed using numpy,
          sklearn, scipy or cv2 where it's needed.

        Parameters
        ------------------------
        image_1: ArrayLike
            2D/3D image data.

        image_2: ArrayLike
            2D/3D image data.

        Returns
        ------------------------
        ImageMetrics
            Class that contains the image metrics to be computed
            depending on the type of image.
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
