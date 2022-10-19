""" Modules to calculate Metrics between features
"""

from abc import ABC, abstractmethod, abstractproperty
from skimage import metrics
import numpy as np
import scipy
from dask import delayed
import utils
import dask.array as da
from io_utils import ImageReader
from typing import List
import xarray as xr
from typing import List, Any, Tuple, Union

ArrayLike = Union[da.core.Array, np.ndarray]

class ImageMetrics(ABC):
    def __init__(self, image_1:ImageReader, image_2:ImageReader, metric_type:str):
        self.__image_1 = image_1
        self.__image_2 = image_2
        self.__metric_type = metric_type
    
    @property
    def image_1(self) -> ImageReader:
        return self.__image_1
    
    @image_1.setter
    def image_1(self, new_image_1:ImageReader) -> None:
        self.__image_1 = new_image_1
    
    @property
    def image_2(self) -> ImageReader:
        return self.__image_2
    
    @image_2.setter
    def image_2(self, new_image_2:ImageReader) -> None:
        self.__image_2 = new_image_2
    
    @property
    def metric_type(self) -> str:
        return self.__metric_type
    
    @metric_type.setter
    def metric_type(self, new_metric_type:str) -> None:
        self.__metric_type = new_metric_type
    
    @abstractmethod
    def get_patches(self, windowed_points:np.array, transform:np.matrix) -> Any:
        pass
    
    def compute_metric_for_patch(self, patch_1:ArrayLike, patch_2:ArrayLike) -> float:
        met_value = None
        
        if self.__metric_type == "SSD":
            met_value = self.mean_squared_error(patch_1, patch_2)
        
        elif self.__metric_type == "SSIM":
            met_value = self.structural_similarity_index(patch_1, patch_2)
        
        return met_value
    
    @abstractmethod
    def mean_squared_error(self, patch_1:ArrayLike, patch_2:ArrayLike) -> float:
        pass

    @abstractmethod
    def structural_similarity_index(self, image_1:ArrayLike, image_2:ArrayLike) -> float:
        pass
    
    def calculate_metrics(
        self, 
        point:np.array,
        transform:np.matrix,
        window_size:int
    ) -> float:
        image_1_shape = self.__image_1.shape
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
                        2*window_size+1
                    ),
                    axis=0
                ) for idx_dim in range(len(image_2_shape))
            ]
            
            # Flattened points for get patches and with extra dimension for meshgrid
            # points_per_dim_flattened = tuple(
            #     [np.squeeze(pt_flattened).flatten() for point_per_dim in points_per_dim]
            # )
            points_per_dim = tuple(points_per_dim)
            
            grid_per_dim = np.meshgrid(*points_per_dim, indexing='ij')
            grid_per_dim = [grid_dim.flatten() for grid_dim in grid_per_dim]
            # print(grid_per_dim[0], len(grid_per_dim), point)
            
            point_1_windowed = np.vstack(grid_per_dim)
            homogenous_pts = np.matrix(
                np.vstack(
                    [
                        point_1_windowed,
                        np.ones(grid_per_dim[0].shape)
                    ]
                )
            )
            
            point_2_windowed = (np.linalg.inv(transform)*homogenous_pts)[:len(image_2_shape),:]
            patch_1, patch_2 = self.get_patches(
                [point_1_windowed.astype(np.int32), point_2_windowed.astype(np.int32)],
                transform
            )
            
            if type(patch_1) == type(None):
                return None

            return self.compute_metric_for_patch(patch_1, patch_2)

# We're working with dask for large images
class LargeImageMetrics(ImageMetrics):
    def __init__(self, image_1:ImageReader, image_2:ImageReader, metric_type:str):
        super().__init__(image_1, image_2, metric_type)
        
    def get_patches(self, windowed_points:np.ndarray, transform:np.matrix) -> Tuple[delayed]:
        
        point_1_windowed = windowed_points[0]
        point_2_windowed = windowed_points[1]
        
        image_2_shape = self.image_2.shape
        len_dims = len(point_1_windowed)
        
        patch_1 = None
        patch_2 = None
        
        dims = tuple([
            da.from_array(np.linspace(
                0, 
                image_2_shape[idx_dim], 
                image_2_shape[idx_dim]
            )) for idx_dim in range(len(image_2_shape))
        ])
        
        if len_dims == 2:
            patch_1 = self.image_1.vindex[point_1_windowed[0], point_1_windowed[1]]
        elif len_dims == 3:
            patch_1 = self.image_1.vindex[point_1_windowed[0], point_1_windowed[1], point_1_windowed[2]]
        else:
            raise NotImplementedError("Only 2D or 3D dimensions are accepted")

        # Send patch without computing
        patch_2 = delayed(scipy.interpolate.interpn)(
            dims, 
            self.image_2,
            point_2_windowed.transpose()
        )
        
        return patch_1, patch_2
    
    def mean_squared_error(self, patch_1:da.core.Array, patch_2:da.core.Array) -> float:
        error = da.map_blocks(lambda a, b: (a - b)**2, patch_1, patch_2)
        # error.visualize()
        value_error = None
        try:
            value_error = error.mean().compute()
        except ValueError:
            value_error = None
            
        return value_error

    def structural_similarity_index(self, patch_1:da.core.Array, patch_2:da.core.Array) -> float:
        
        value_error = None
        
        try:
            patch_1 = patch_1.compute()
            patch_2 = patch_2.compute()
            
            value_error = metrics.structural_similarity(patch_1, patch_2)

        except ValueError:
            value_error = None

        return value_error
    
class SmallImageMetrics(ImageMetrics):
    def __init__(self, image_1:ImageReader, image_2:ImageReader, metric_type:str):
        super().__init__(image_1, image_2, metric_type)
        
    def get_patches(self, windowed_points:np.ndarray, transform:np.matrix) -> Tuple[np.ndarray]:
        
        point_1_windowed = windowed_points[0]
        point_2_windowed = windowed_points[1]
        
        image_2_shape = self.image_2.shape
        len_dims = len(point_1_windowed)
        
        patch_1 = None
        patch_2 = None
        
        # Range of values in interval for each axis
        dims = tuple([
            np.linspace(
                0, 
                image_2_shape[idx_dim], 
                image_2_shape[idx_dim]
            ) for idx_dim in range(len(image_2_shape))
        ])
        
        if len_dims == 2:
            patch_1 = self.image_1[point_1_windowed[0], point_1_windowed[1]]
        elif len_dims == 3:
            patch_1 = self.image_1[point_1_windowed[0], point_1_windowed[1], point_1_windowed[2]]
        else:
            raise NotImplementedError("Only 2D or 3D dimensions are accepted")

        try:
            patch_2 = scipy.interpolate.interpn(
                dims, 
                self.image_2,
                point_2_windowed.transpose()
            )
        except ValueError:
            return None, None
        
        return patch_1, patch_2
    
    def mean_squared_error(self, patch_1:np.ndarray, patch_2:np.ndarray) -> float:
        return metrics.mean_squared_error(patch_1, patch_2)

    def structural_similarity_index(self, patch_1:np.ndarray, patch_2:np.ndarray) -> float:
        return metrics.structural_similarity(patch_1, patch_2)

class ImageMetricsFactory:
    def __init__(self):
        self.__array_type = [da.core.Array, np.ndarray]
        
        self.factory = {
            da.core.Array: LargeImageMetrics,
            np.ndarray: SmallImageMetrics
        }

    @property
    def array_type(self) -> List:
        return self.__array_type
    
    def create(
        self, 
        image_1:ImageReader, 
        image_2:ImageReader, 
        metric_type:str
    ) -> ImageMetrics:
        image_type = type(image_1)
        
        if image_type not in self.__array_type:
            raise NotImplementedError(f"Image array type {image_type} not supported")

        return self.factory[image_type](image_1, image_2, metric_type)

def calculate_metrics(pt1, data, args):
    """Given a pair of points, the images and the transform, 
        Calculate metrics for window size. If window_size = 0, calculate just for one point"""
        
    if args['data_type'] == 'large':
        I1 = np.squeeze(data[0][:,args['channel'], :,:,:])
        I2 = np.squeeze(data[1][:,args['channel'], :,:,:])
    else:
        I1 = data[0]
        I2 = data[1]
    transform = data[2]

    X = np.linspace(0, I2.shape[0], I2.shape[0])  
    Y = np.linspace(0, I2.shape[1], I2.shape[1])  
    Z = np.linspace(0, I2.shape[2], I2.shape[2])   
    
    if args['window_size'] == 0:
        pt2 = (np.linalg.inv(transform)*np.matrix(list(pt1)+[1]).transpose())[:3]
        pt2 = np.squeeze(np.asarray(pt2))
        Patch1,Patch2 = get_patches(I1,I2, np.expand(pt1,axis=0),np.expand_dims(pt2,axis=0),X,Y,Z, args)
        
    else:
        #print("need to work on this")
        pt1_X = np.expand_dims(np.linspace(pt1[0]-args['window_size'], pt1[0]+args['window_size'], 2*args['window_size']+1)  , axis=0)
        pt1_Y = np.expand_dims(np.linspace(pt1[1]-args['window_size'], pt1[1]+args['window_size'], 2*args['window_size']+1)  , axis=0)
        pt1_Z = np.expand_dims(np.linspace(pt1[2]-args['window_size'], pt1[2]+args['window_size'], 2*args['window_size']+1)  , axis=0)
        points = (pt1_X, pt1_Y, pt1_Z)
        X,Y,Z = np.meshgrid(*points, indexing='ij')
        
        pt1_win = np.vstack ([X.flatten(), Y.flatten(), Z.flatten()])
        homogenous_pts = np.matrix(np.vstack([pt1_win, np.ones(X.flatten().shape)]))
        pt2_win = (np.linalg.inv(transform)*homogenous_pts)[:3,:]
        Patch1,Patch2 = get_patches(data, pt1_win,pt2_win,np.squeeze(pt1_X), np.squeeze(pt1_Y), np.squeeze(pt1_Z), args)
        
    if Patch1 is not None:
        met = compute_metric_for_patch(Patch1,Patch2,args['metric'])
        return met
    else:
        return None