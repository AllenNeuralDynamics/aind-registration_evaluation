import numpy as np
from dask.array import from_zarr


def get_data(args):
    """
    Function to get input data based on user input.
    """
    if args['datatype'] == 'dummy':
        return create_sample_data()
    elif args['datatype'] == 'large':
        I1,I2 = read_large_images(args)
        transform = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        return I1,I2,transform
    else:
        return 0,0,0

def create_sample_data():
    """
    Function to create dummy data.
    """
    def value_func_3d(x, y, z):
        return 1 * x + 1 * y - z

    x = np.linspace(0, 499, 500)
    y = np.linspace(0, 499, 500)
    z = np.linspace(0, 199, 200)
    points = (x, y, z)
    I = value_func_3d(*np.meshgrid(*points, indexing='ij'))
    transform = np.matrix([[1,0,0,3],[0,1,0,3],[0,0,1,3],[0,0,0,1]])
    
    return I,I, transform

def read_large_images(args):
    '''Read large images'''
    I1 = from_zarr(args['image1'])
    I2 = from_zarr(args['image2'])
    #I2 = 0
    return I1,I2