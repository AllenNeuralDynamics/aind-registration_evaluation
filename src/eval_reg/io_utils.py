from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import zarr

# IO types
PathLike = Union[str, Path]


class ImageReader(ABC):
    def __init__(self, data_path: PathLike) -> None:
        """
        Class constructor of image reader.

        Parameters
        ------------------------
        data_path: PathLike
            Path where the image is located

        """

        self.__data_path = Path(data_path)
        super().__init__()

    @abstractmethod
    def as_dask_array(self, chunk_size: Optional[Any] = None) -> da.Array:
        """
        Abstract method to return the image as a dask array.

        Parameters
        ------------------------
        chunk_size: Optional[Any]
            If provided, the image will be rechunked to the desired
            chunksize

        Returns
        ------------------------
        da.Array
            Dask array with the image

        """
        pass

    @abstractmethod
    def as_numpy_array(self) -> np.ndarray:
        """
        Abstract method to return the image as a numpy array.

        Returns
        ------------------------
        np.ndarray
            Numpy array with the image

        """
        pass

    @abstractproperty
    def shape(self) -> Tuple:
        """
        Abstract method to return the shape of the image.

        Returns
        ------------------------
        Tuple
            Tuple with the shape of the image

        """
        pass

    @abstractproperty
    def chunks(self) -> Tuple:
        """
        Abstract method to return the chunks of the image if it's possible.

        Returns
        ------------------------
        Tuple
            Tuple with the chunks of the image

        """
        pass

    @property
    def data_path(self) -> PathLike:
        """
        Getter to return the path where the image is located.

        Returns
        ------------------------
        PathLike
            Path of the image

        """
        return self.__data_path

    @data_path.setter
    def data_path(self, new_data_path: PathLike) -> None:
        """
        Setter of the path attribute where the image is located.

        Parameters
        ------------------------
        new_data_path: PathLike
            New path of the image

        """
        self.__data_path = Path(new_data_path)


class OMEZarrReader(ImageReader):
    def __init__(
        self, data_path: PathLike, multiscale: Optional[int] = 0
    ) -> None:
        """
        Class constructor of image OMEZarr reader.

        Parameters
        ------------------------
        data_path: PathLike
            Path where the image is located

        multiscale: Optional[int]
            Desired multiscale to read from the image. Default: 0 which is
            supposed to be the highest resolution

        """
        super().__init__(Path(data_path).joinpath(str(multiscale)))

    def as_dask_array(self, chunk_size: Optional[Any] = None) -> da.array:
        """
        Method to return the image as a dask array.

        Parameters
        ------------------------
        chunk_size: Optional[Any]
            If provided, the image will be rechunked to the desired
            chunksize

        Returns
        ------------------------
        da.Array
            Dask array with the image

        """
        image = da.from_zarr(self.data_path)

        if chunk_size:
            image = image.rechunk(chunks=chunk_size)

        return image

    def as_numpy_array(self):
        """
        Method to return the image as a numpy array.

        Returns
        ------------------------
        np.ndarray
            Numpy array with the image

        """
        return zarr.open(self.data_path, "r")[:]

    @property
    def shape(self):
        """
        Method to return the shape of the image.

        Returns
        ------------------------
        Tuple
            Tuple with the shape of the image

        """
        return zarr.open(self.data_path, "r").shape

    @property
    def chunks(self):
        """
        Method to return the chunks of the image.

        Returns
        ------------------------
        Tuple
            Tuple with the chunks of the image

        """
        return zarr.open(self.data_path, "r").chunks


class ImageReaderFactory:
    def __init__(self):
        """
        Class to create the image reader factory.
        """
        self.__extensions = [".zarr"]  # , ".npy"]
        self.factory = {".zarr": OMEZarrReader}

    @property
    def extensions(self) -> List:
        """
        Method to return the allowed format extensions of the images.

        Returns
        ------------------------
        List
            List with the allowed image format extensions

        """
        return self.__extensions

    def create(self, data_path: PathLike) -> ImageReader:
        """
        Method to create the image reader based on the format.

        Returns
        ------------------------
        List
            List with the allowed image format extensions

        """

        data_path = Path(data_path)
        ext = data_path.suffix

        if ext not in self.__extensions:
            raise NotImplementedError(f"File type {ext} not supported")

        return self.factory[ext](data_path)


def create_sample_data_2D(
    delta_x: Optional[int] = 3, delta_y: Optional[int] = 3
) -> List:
    """
    Function to create 2D dummy data.

    Parameters
    ------------------------

    delta_x: Optional[int]
        Translation over the x axis in the sample 2D data

    delta_y: Optional[int]
        Translation over the y axis in the sample 2D data

    Returns
    ------------------------
    List:
        List where the first two positions correspond to image 1 and 2 respectively and
        the last position correspond to the transformation matrix.
    """

    def value_func_3d(x, y):
        return 1 * x + 1 * y

    x = np.linspace(0, 499, 500)
    y = np.linspace(0, 499, 500)

    points = (x, y)
    image = value_func_3d(*np.meshgrid(*points, indexing="ij"))
    transform = np.matrix([[1, 0, delta_x], [0, 1, delta_y], [0, 0, 1]])

    return [image, image, transform]


def create_sample_data_3D(
    delta_x: Optional[int] = 3,
    delta_y: Optional[int] = 3,
    delta_z: Optional[int] = 3,
) -> List:
    """
    Function to create 3D dummy data.

    Parameters
    ------------------------

    delta_x: Optional[int]
        Translation over the x axis in the sample 3D data

    delta_y: Optional[int]
        Translation over the y axis in the sample 3D data

    delta_z: Optional[int]
        Translation over the z axis in the sample 3D data

    Returns
    ------------------------
    List:
        List where the first two positions correspond to image 1 and 2 respectively and
        the last position correspond to the transformation matrix.
    """

    def value_func_3d(x, y, z):
        return 1 * x + 1 * y - z

    x = np.linspace(0, 499, 500)
    y = np.linspace(0, 499, 500)
    z = np.linspace(0, 199, 200)
    points = (x, y, z)
    image = value_func_3d(*np.meshgrid(*points, indexing="ij"))
    transform = np.matrix(
        [
            [1, 0, 0, delta_x],
            [0, 1, 0, delta_y],
            [0, 0, 1, delta_z],
            [0, 0, 0, 1],
        ]
    )

    return [image, image, transform]


def get_data(
    path_image_1: PathLike, path_image_2: PathLike, data_type: str
) -> List:
    """
    Function that gets data depending the datatype.

    Parameters
    ------------------------

    path_image_1: PathLike
        Path to image 1.

    path_image_2: PathLike
        Path to image 2.

    data_type: str
        If we are creating sample data for testing purposes or processing real data

    Returns
    ------------------------
    List:
        List where the first two positions correspond to image 1 and 2 respectively and
        the last position correspond to the transformation matrix.
    """
    if data_type == "dummy_2D":
        return create_sample_data_2D()

    elif data_type == "dummy_3D":
        return create_sample_data_3D()

    else:
        # real data leaving the choice to the reader factory in how to load the image
        loaded_img_1 = ImageReaderFactory().create(path_image_1)
        loaded_img_2 = ImageReaderFactory().create(path_image_2)

        # TODO Get transformation matrix from terastitcher
        # transform = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        transform = np.matrix([[1, 0, 0], [0, 1, 1800], [0, 0, 1]])

        return [loaded_img_1, loaded_img_2, transform]
