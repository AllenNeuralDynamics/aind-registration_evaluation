import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
import tifffile as tiff

def euler_to_rotation_matrix(angles):
    '''
        Converts Euler angles to a 3x3 rotation matrix.

        - The function computes the rotation matrix using the given Euler angles in degrees.
        - The angles are applied in the order of Z-Y-X, which is the standard convention for Euler angles.

        Parameters
        ------------------------
        angles: List or array-like
            A list or array of three angles [rx, ry, rz] in degrees, where:
                - rx: Rotation angle around the x-axis.
                - ry: Rotation angle around the y-axis.
                - rz: Rotation angle around the z-axis.

        Returns
        ------------------------
        numpy.ndarray
            A 3x3 rotation matrix representing the combined rotation defined by the Euler angles.
    '''
    rx, ry, rz = np.deg2rad(angles)
    cos_rx, sin_rx = np.cos(rx), np.sin(rx)
    cos_ry, sin_ry = np.cos(ry), np.sin(ry)
    cos_rz, sin_rz = np.cos(rz), np.sin(rz)

    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, cos_rx, -sin_rx],
                                  [0, sin_rx, cos_rx]])

    rotation_matrix_y = np.array([[cos_ry, 0, sin_ry],
                                  [0, 1, 0],
                                  [-sin_ry, 0, cos_ry]])

    rotation_matrix_z = np.array([[cos_rz, -sin_rz, 0],
                                  [sin_rz, cos_rz, 0],
                                  [0, 0, 1]])

    rotation_matrix = np.dot(np.dot(rotation_matrix_z, rotation_matrix_y), rotation_matrix_x)
    return rotation_matrix

def create_transformation_matrices(image_shape ):
    '''
        Generates a list of transformation matrices for image augmentation.

        - The function creates 10 transformation matrices for 3D images.
        - Each matrix includes a combination of rotation, translation, scaling, and shearing transformations.
        - The transformations are centered around the middle of the image.

        Parameters
        ------------------------
        image_shape: Tuple or list of 3 integers
            Shape of the image in the form (z, y, x).

        Returns
        ------------------------
        List of 4x4 numpy.ndarray
            A list of 4x4 transformation matrices. Each matrix represents a combination of rotation, translation,
            scaling, and shearing applied to the image.
    '''
    matrices = []
    center = np.array(image_shape) / 2

    for i in range(10):
        # Random parameters for transformations
        angles = np.random.uniform(-180, 180, size=3)
        rotation_matrix = euler_to_rotation_matrix(angles)

        translation = np.random.uniform(-40, 40, size=3)  
        scale_factors = np.random.uniform(1, 1, size=3) 
        shear_factors = np.random.uniform(-0.3, 0.3, size=3)  

        # Create scale matrix
        scale_matrix = np.diag(np.append(scale_factors, 1))

        # Create rotation matrix
        rot_matrix = np.eye(4)
        rot_matrix[:3, :3] = rotation_matrix @ scale_matrix[:3, :3]

        # Create shear matrix (for 3D: X-Y, X-Z, and Y-Z shearing)
        shear_matrix = np.eye(4)
        shear_matrix[0, 1] = shear_factors[0]  # Shear in X-Y plane
        shear_matrix[0, 2] = shear_factors[1]  # Shear in X-Z plane
        shear_matrix[1, 2] = shear_factors[2]  # Shear in Y-Z plane

        # Create translation matrix
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = translation

        # Translation to Origin
        translation_to_origin = np.eye(4)
        translation_to_origin[:3, 3] = -center

        # Translation Back
        translation_back = np.eye(4)
        translation_back[:3, 3] = center

        # Combine all transformations
        centered_transformation = translation_back @ shear_matrix @ rot_matrix @ translation_to_origin
        final_transformation = centered_transformation @ translation_matrix

        matrices.append(final_transformation.tolist())

    return matrices