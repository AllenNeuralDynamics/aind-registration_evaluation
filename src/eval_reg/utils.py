import this
import numpy as np
import dask.array as da

def sample_points_in_overlap(bounds1, bounds2, args ):
        """samples points in the overlap regions 
           and returns a list of points
           sampling types : random, grid, feature extracted
        """
        
        #check if there is intersection
        #######NEED TO DO!!!!

        #if there is, then : 
        if args['type'] == "random":
            o_min = [np.max([bounds1[0][0], bounds2[0][0]]),
                        np.max([bounds1[0][1], bounds2[0][1]]),
                            np.max([bounds1[0][2], bounds2[0][2]])]

            o_max = [np.min([bounds1[1][0], bounds2[1][0]]),
                        np.min([bounds1[1][1], bounds2[1][1]]),
                            np.min([bounds1[1][2], bounds2[1][2]])]

            print(range(o_min[0], o_max[0]))
            x = list(np.random.choice( range(o_min[0], o_max[0]), args['numpoints'] ))
            y = list(np.random.choice(range(o_min[1], o_max[1]), args['numpoints']))
            z = list(np.random.choice(range(o_min[2], o_max[2]), args['numpoints']))
        
        return list(np.array([x,y,z]).transpose())

    

def calculate_bounds(data, args):
    """
    Calculate bounds of coverage for two images and a transform
    where image1 is in its own coordinate system and image 2 is mapped
    to image 1's coords with the transform
    """

    image1_shape, image2_shape = get_image_shapes(data, args)
    transform = data[2]
    b1 = [[0,0,0],list(image1_shape)]
    pt_min = np.matrix([0,0,0,1]).transpose()
    pt_max = np.matrix(list(image2_shape)+[1]).transpose()
    b2 = [np.squeeze(transform*pt_min).tolist()[0][:3],
            np.squeeze(transform*pt_max).tolist()[0][:3]]  
            
    return b1,b2 

def prune_points_to_fit_window(pts, window_size, data, args):

    image_shape, image2_shape = get_image_shapes(data, args)

    newpts = []
    
    for p in pts:
        if (p[0]+window_size > image_shape[0]) | (p[1]+window_size > image_shape[1]) |(p[2]+window_size > image_shape[2]) :
            print("skip point")
        else:
            newpts.append(p)

    return newpts

def get_image_shapes(data, args):
    if args['datatype'] == 'large':
        return np.squeeze(data[0][:,args['channel'], :,:,:]).shape, np.squeeze(data[1][:,args['channel'], :,:,:]).shape

    else:
        return data[0].shape, data[1].shape


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

