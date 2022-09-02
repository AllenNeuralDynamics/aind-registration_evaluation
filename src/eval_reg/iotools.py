import numpy as np

def get_data(args):
    """
    Function to get input data based on user input.
    """
    if args['datatype'] == 'dummy':
        return create_sample_data()
    else:
        return 0,0,0

def create_sample_data():
    """
    Function to create dummy data.
    """
    def value_func_3d(x, y, z):
        return 1 * x + 1 * y - z

    x = np.linspace(0, 9, 10)
    y = np.linspace(0, 9, 10)
    z = np.linspace(0, 9, 10)
    points = (x, y, z)
    I = value_func_3d(*np.meshgrid(*points, indexing='ij'))
    transform = np.matrix([[1,0,0,3],[0,1,0,3],[0,0,1,3],[0,0,0,1]])
    return I,I, transform