from abc import ABC, abstractmethod, abstractproperty
from typing import Union, Optional, Any, Tuple, List
from pathlib import Path
import dask.array as da
import numpy as np
import zarr

# IO types 
PathLike = Union[str, Path]

class ImageReader(ABC):
    def __init__(self, data_path:PathLike):
        self.__data_path = Path(data_path)
        super().__init__()
        
    @abstractmethod
    def as_dask_array(self, chunk_size:Optional[Any]=None) -> da.Array:
        pass
    
    @abstractmethod
    def as_numpy_array(self) -> np.ndarray:
        pass
    
    @abstractproperty
    def shape(self) -> Tuple:
        pass
    
    @abstractproperty
    def chunks(self) -> Tuple:
        pass
    
    @property
    def data_path(self) -> PathLike:
        return self.__data_path
    
    @data_path.setter
    def data_path(self, new_data_path:PathLike) -> None:
        self.__data_path = Path(new_data_path)
        
class OMEZarrReader(ImageReader):
    def __init__(self, data_path:PathLike, multiscale:Optional[int]=0):
        super().__init__(Path(data_path).joinpath(str(multiscale)))
    
    def as_dask_array(self, chunk_size:Optional[Any]=None) -> da.array:
        image = da.from_zarr(self.data_path)
        
        if chunk_size:
            image = image.rechunk(chunks=chunk_size)
        
        return image
    
    def as_numpy_array(self):
        return zarr.open(self.data_path, "r")[:]
    
    @property
    def shape(self):
        return zarr.open(self.data_path, "r").shape
    
    @property
    def chunks(self):
        return zarr.open(self.data_path, "r").chunks

class ImageReaderFactory:
    def __init__(self):
        self.__extensions = [".zarr"]#, ".npy"]
        
        self.factory = {
            ".zarr" : OMEZarrReader
        }

    @property
    def extensions(self):
        return self.__extensions

    def create(self, data_path:PathLike) -> ImageReader:
        
        data_path = Path(data_path)
        ext = data_path.suffix
        
        if ext not in self.__extensions:
            raise NotImplementedError(f"File type {ext} not supported")
        
        return self.factory[ext](data_path)

def create_sample_data_2D(
    delta_x:Optional[int]=3, 
    delta_y:Optional[int]=3
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
    I = value_func_3d(*np.meshgrid(*points, indexing='ij'))
    transform = np.matrix([
        [1,0, delta_x],
        [0,1,delta_y],
        [0,0,1]
    ])
    
    return [I, I, transform]

def create_sample_data_3D(
    delta_x:Optional[int]=3, 
    delta_y:Optional[int]=3, 
    delta_z:Optional[int]=3
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
    I = value_func_3d(*np.meshgrid(*points, indexing='ij'))
    transform = np.matrix([
        [1, 0, 0, delta_x],
        [0, 1, 0, delta_y],
        [0, 0, 1, delta_z],
        [0, 0, 0, 1]
    ])
    
    return [I,I, transform]

def get_data(
    path_image_1:PathLike, 
    path_image_2:PathLike,
    data_type:str
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
    if data_type == 'dummy_2D':
        return create_sample_data_2D()
    
    elif data_type == 'dummy_3D':
        return create_sample_data_3D()
    
    else:
        # real data leaving the choice to the reader factory in how to load the image
        loaded_img_1 = ImageReaderFactory().create(path_image_1)
        loaded_img_2 = ImageReaderFactory().create(path_image_2)
        
        # TODO Get transformation matrix from terastitcher
        # transform = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        transform = np.matrix([
            [1,0,50],
            [0,1,50],
            [0,0,1]
        ])
        
        return [loaded_img_1, loaded_img_2, transform]

def main():
    ome_zarr_tile = 'C:/Users/camilo.laiton/Documents/registration_evaluation/src/eval_reg/images/Ex_488_Em_525_468770_468770_830620_012820.zarr'
    tile = ImageReaderFactory().create(ome_zarr_tile)
    print(f"Tile shape: {tile.shape} Tile chunks: {tile.shape}")
    tile_dask = tile.as_dask_array()
    tile_numpy = tile.as_numpy_array()
    
    print(tile_dask, tile_numpy, type(tile_dask), type(tile_numpy))
    
if __name__ == "__main__":
    main()