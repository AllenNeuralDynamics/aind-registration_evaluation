import this
import numpy as np
import dask.array as da
from io_utils import ImageReader
from typing import Tuple, List, Optional, Union

ArrayLike = Union[da.Array, np.array]

def sample_points_in_overlap(
    bounds_1:np.ndarray, 
    bounds_2:np.ndarray,
    numpoints:int,
    sample_type:Optional[str]='random'
) -> np.ndarray:
    """
    samples points in the overlap regions and returns a list of points
    
    sampling types : random, grid, feature extracted
    
    Parameters
    ------------------------
    
    bounds_1: np.ndarray
        Image 1 calculated boundaries in each dimension, each position could be (x, y) or (x, y, z)
        depending on the image dimensionality.
        
    bounds_2: np.ndarray
        Image 2 calculated boundaries in each dimension, each position could be (x, y) or (x, y, z)
        depending on the image dimensionality.
    
    numpoints: int
        Number of sample points that will be used to create the grid
    
    sample_type: str
        random, grid or feature_extracted.
    
    Returns
    ------------------------
    np.ndarray:
        Array with the overlaped sample points.
    """
    sample_type = sample_type.lower()
    numpoints = int(numpoints)
    
    if sample_type not in ['random', 'grid', 'feature_extracted']:
        raise NotImplementedError(f"{sample_type} sample type has not been implemented.")
    
    if numpoints < 0:
        raise ValueError("Error in the number of points, it must be a positive integer.")
    
    #check if there is intersection
    #######NEED TO DO!!!!

    #if there is, then : 
    
    n_dims = len(bounds_1[0])
    dims_sample_points = []
    
    if sample_type == "random":
        
        for dim_idx in range(n_dims):
            
            o_min = max(
                bounds_1[0][dim_idx], bounds_2[0][dim_idx]
            )
            
            o_max = min(
                bounds_1[1][dim_idx], bounds_2[1][dim_idx]
            )

            dims_sample_points.append(
                np.random.choice( range(o_min, o_max), numpoints )
            )
            
    dims_sample_points = np.array(dims_sample_points).transpose()
    return dims_sample_points

def calculate_bounds(
    image_1_shape:Tuple, 
    image_2_shape:Tuple,
    transform:np.ndarray
) -> Tuple:
    
    """
    Calculate bounds of coverage for two images and a transform
    where image1 is in its own coordinate system and image 2 is mapped
    to image 1's coords with the transform
    
    Parameters
    ------------------------
    image_1: Tuple
        First image which will be used as default in the coordinate system
    
    image_2: Tuple
        Second image which will be used to map it's position to a common coordinate system
        based on image_1
        
    transform: np.ndarray
        Transformation matrix applied over the two images
    
    Returns
    ------------------------
    Tuple:
        Tuple with the calculated boundaries.
    
    """

    dimensions_zeros = np.zeros(len(image_1_shape), dtype=np.int8)
    
    # First boundary
    bound_1 = np.array(
        [
            dimensions_zeros,
            list(image_1_shape)
        ]
    )
    
    # Minimum point
    pt_min = np.matrix(
        np.append(dimensions_zeros, 1)
    ).transpose()
    
    # Maximum point
    pt_max = np.matrix(
        np.array( list(image_2_shape) +[1] )
    ).transpose()
    
    # Getting coordinates for second image into image_1 coordinate system
    coord_1 = np.squeeze(
        transform*pt_min,
    ).tolist()[0][:-1]
    
    coord_2 = np.squeeze(
        transform*pt_max
    ).tolist()[0][:-1]
        
    bound_2 = np.array([coord_1, coord_2])

    return bound_1, bound_2
    
def prune_points_to_fit_window(
    image_shape:Tuple,
    points:np.array,
    window_size:int
) -> np.array:
    """
    Checks if generated points can be used for metric evaluation in the specified window size
    given a set of points and an image shape.
    
    Parameters
    ------------------------
    image_shape: Tuple
        Image shape
    
    points: np.array
        Sample points in an overlap region given two images using a transformation matrix
        
    window_size: int
        Window size applied over each axis
    
    Returns
    ------------------------
    np.array:
        Array with the points that fit the window size inside image shape.
    
    """
    
    def check_window_size(nested_point:np.array)-> bool:
        """
        Map function applied over a numpy array
        
        Parameters
        ------------------------
        nested_point: np.array
            Individual point from the points array.
        
        Returns
        ------------------------
        bool:
            True if the (point (x, y) + window size) is inside image shape,
            False otherwise.
        """
        modified_point = nested_point + window_size
        point_window_inside = np.less(modified_point, image_shape)
        unique_val = np.unique(point_window_inside)
        
        if len(unique_val) == 1 and unique_val[0] == True:
            return True
        
        return False
    
    selected_indices = np.array(
        list(
            map(
                check_window_size, points
            )
        )
    )
    
    return points[selected_indices]

def extract_data(arr:ArrayLike, last_dimensions:Optional[int]=None) -> ArrayLike:
    
    if last_dimensions != None:
    
        if last_dimensions > arr.ndim:
            raise ValueError(
                "Last dimensions should be lower than array dimensions"
            )
    
    else:
        last_dimensions = len(arr.shape) - arr.shape.count(1)
    
    dynamic_indices = [slice(None)]*arr.ndim
    
    for idx in range(arr.ndim - last_dimensions):
        dynamic_indices[idx] = 0
 
    return arr[tuple(dynamic_indices)]

def affine_transform_dask(
        input,
        matrix,
        offset=0.0,
        output_shape=None,
        output_chunks=None,
        **kwargs
):
    """
    Wraps `ndimage.affine_transformation` for dask arrays.
    For every output chunk, only the slice containing the
    relevant part of the input is passed on to
    `ndimage.affine_transformation`.
    To do:
      - optionally use cupyx.scipy.ndimage.affine_transform
    API wraps `ndimage.affine_transformation`, except for `output_chunks`.
    :param input: N-D numpy or dask array
    :param matrix:
    :param offset:
    :param output_shape:
    :param output_chunks:
    :param kwargs:
    :return: dask array
    """

    def resample_chunk(chunk, matrix, offset, kwargs, block_info=None):

        N = chunk.ndim
        input_shape = input.shape
        chunk_shape = chunk.shape

        chunk_offset = [i[0] for i in block_info[0]['array-location']]
        # print('chunk_offset', chunk_offset)

        chunk_edges = np.array([i for i in np.ndindex(tuple([2] * N))])\
                      * np.array(chunk_shape) + np.array(chunk_offset)
        rel_input_edges = np.dot(matrix, chunk_edges.T).T + offset

        # print('rel_input_edges', rel_input_edges) # ok
        # print('chunk_edges', chunk_edges) # ok

        rel_input_i = np.min(rel_input_edges, 0)
        rel_input_f = np.max(rel_input_edges, 0)

        # not sure yet how many additional pixels to include
        # (depends on interp order?)
        for dim, upper in zip(range(N), input_shape):
            rel_input_i[dim] = np.clip(rel_input_i[dim] - 2, 0, upper)
            rel_input_f[dim] = np.clip(rel_input_f[dim] + 2, 0, upper)

        rel_input_i = rel_input_i.astype(np.int64)
        rel_input_f = rel_input_f.astype(np.int64)

        # print('min max input', rel_input_i, rel_input_f)

        rel_input_slice = tuple([slice(int(rel_input_i[dim]),
                                       int(rel_input_f[dim]))
                                 for dim in range(N)])

        rel_input = input[rel_input_slice]

        # print('rel_input_slice', rel_input_slice)

        # modify offset to point into cropped input
        # y = Mx + o
        # coordinate substitution:
        # y' = y - y0(min_coord_px)
        # x' = x - x0(chunk_offset)
        # then
        # y' = Mx' + o + Mx0 - y0
        # M' = M
        # o' = o + Mx0 - y0

        offset_prime = offset + np.dot(matrix, chunk_offset) - rel_input_i

        chunk = ndimage.affine_transform(rel_input,
                                         matrix,
                                         offset_prime,
                                         output_shape=chunk_shape,
                                         **kwargs)

        return chunk

    if output_shape is None: output_shape = input.shape

    transformed = da.zeros(output_shape,
                           dtype=input.dtype,
                           chunks=output_chunks)

    transformed = transformed.map_blocks(resample_chunk,
                                         dtype=input.dtype,
                                         matrix=matrix,
                                         offset=offset,
                                         kwargs=kwargs,
                                         )

    return transformed

