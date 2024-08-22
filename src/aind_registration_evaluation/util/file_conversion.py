import tifffile
import z5py
import numpy as np
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage import transform
from skimage.exposure import rescale_intensity


def prepareFiles(confocal, warpedCortical, mask_threshold, downsampling_factor):
    """
    Crops and downsamples N5 files to usable TIFF files and stores them in scratch.

    Parameters
    ------------------------
    confocal : str
        Filepath of the confocal image.
    warpedCortical : str
        Filepath of the cortical image.
    mask_threshold : int
        Intensity threshold for masking the image during cropping.
    downsampling_factor : int
        Factor by which to downsample the image, expressed as 2 ** downsampling_factor.

    Returns
    ------------------------
    None
    """
    if not os.path.isfile('/scratch/confocal_s0.tif'):
        n5_to_tiff(confocal, '/scratch/confocal_s0.tif', 's0')
    if not os.path.isfile('/scratch/cortical_s0.tif'):
        n5_to_tiff_warped(warpedCortical, '/scratch/cortical_s0.tif')
    if not (os.path.isfile('/scratch/confocal_s0_cropped.tif') or os.path.isfile('/scratch/cortical_s0_cropped.tif')):
        apply_mask_and_crop('/scratch/cortical_s0.tif', '/scratch/cortical_s0_cropped.tif', '/scratch/confocal_s0.tif', '/scratch/confocal_s0_cropped.tif', mask_threshold)
    if not os.path.isfile('/scratch/confocal_s0_cropped_downsampled.tif'):
        downsample_and_int16('/scratch/confocal_s0_cropped.tif', '/scratch/confocal_s0_cropped_downsampled.tif', downsampling_factor)
    if not os.path.isfile('/scratch/cortical_s0_cropped_downsampled.tif'):
        downsample_and_int16('/scratch/cortical_s0_cropped.tif', '/scratch/cortical_s0_cropped_downsampled.tif', downsampling_factor)


def tiff_to_n5(tiff_filename, n5_filename, downsampling_factors):
    """
    Converts a TIFF file to N5 format with specified resolution levels.

    Parameters
    ------------------------
    tiff_filename : str
        Filepath of the input TIFF image.
    n5_filename : str
        Filepath for the output N5 file.
    downsampling_factors : list of int
        List of downsampling factors for each resolution level.

    Returns
    ------------------------
    None
    """
    image_volume = tifffile.imread(tiff_filename)
    shape = image_volume.shape
    f = z5py.File(n5_filename)

    if "c0" not in f:
        group = f.create_group("c0")
    else:
        group = f["c0"]

    for i, factor in enumerate(downsampling_factors):
        if "s" + str(i) in group:
            del group["s" + str(i)]
        ds_data = image_volume[::factor, ::factor, ::factor]
        downsampled_shape = ds_data.shape
        ds = group.create_dataset('s' + str(i), shape=downsampled_shape, dtype=image_volume.dtype, chunks=(128, 128, 128), compression="gzip")
        ds[:] = ds_data
        ds.attrs["downsamplingFactors"] = tuple([factor, factor, factor])

    f.close()


def n5_to_tiff(n5_filename, output_path, scale_level):
    """
    Converts an N5 image to TIFF format at a specified resolution level.

    Parameters
    ------------------------
    n5_filename : str
        Filepath of the input N5 image.
    output_path : str
        Filepath for the output TIFF image.
    scale_level : str
        Resolution level in the N5 file to extract.

    Returns
    ------------------------
    None
    """
    f = z5py.File(n5_filename)
    image = np.array(f["c0"][scale_level])
    tifffile.imwrite(output_path, image)


def n5_to_tiff_warped(n5_filename, output_path):
    """
    Converts a warped N5 image to TIFF format at full resolution.

    Parameters
    ------------------------
    n5_filename : str
        Filepath of the input N5 image.
    output_path : str
        Filepath for the output TIFF image.

    Returns
    ------------------------
    None
    """
    f = z5py.File(n5_filename)
    image = np.array(f["c0"])
    tifffile.imwrite(output_path, image)


def apply_mask(inFilepath, outFilepath, threshold, newValue):
    """
    Applies a mask to an image by setting pixels below a threshold to a specified value.

    Parameters
    ------------------------
    inFilepath : str
        Filepath of the input TIFF image.
    outFilepath : str
        Filepath for the output masked image.
    threshold : int
        Intensity threshold below which pixels will be set to newValue.
    newValue : int
        Value to assign to pixels below the threshold.

    Returns
    ------------------------
    None
    """
    data = tifffile.imread(inFilepath)
    mask = data <= threshold
    data[mask] = newValue
    tifffile.imwrite(outFilepath, data)


def apply_mask_and_crop(inFilepath_cortical, outFilepath_cortical, inFilepath_confocal, outFilepath_confocal, threshold):
    """
    Applies a mask to the cortical image and crops both the cortical and confocal images based on the overlapping regions.

    Parameters
    ------------------------
    inFilepath_cortical : str
        Filepath of the warped cortical image.
    outFilepath_cortical : str
        Filepath for the output cropped cortical image.
    inFilepath_confocal : str
        Filepath of the confocal image (reference).
    outFilepath_confocal : str
        Filepath for the output cropped confocal image.
    threshold : int
        Intensity threshold for masking and cropping non-overlapping areas.

    Returns
    ------------------------
    None
    """
    data_cortical = tifffile.imread(inFilepath_cortical)
    data_confocal = tifffile.imread(inFilepath_confocal)
    data_cortical = data_cortical.astype(np.int16)
    data_confocal = data_confocal.astype(np.int16)
    mask = data_cortical <= threshold
    data_cortical[mask] = 0

    nonzeros_cortical = np.nonzero(data_cortical)
    x_min_cortical, x_max_cortical = nonzeros_cortical[0].min(), nonzeros_cortical[0].max()
    y_min_cortical, y_max_cortical = nonzeros_cortical[1].min(), nonzeros_cortical[1].max()
    z_min_cortical, z_max_cortical = nonzeros_cortical[2].min(), nonzeros_cortical[2].max()

    x_min = x_min_cortical if x_min_cortical < data_confocal.shape[0] - 1 else data_confocal.shape[0] - 1
    x_max = min(x_max_cortical, data_confocal.shape[0] - 1)
    y_min = y_min_cortical if y_min_cortical < data_confocal.shape[1] - 1 else data_confocal.shape[1] - 1
    y_max = min(y_max_cortical, data_confocal.shape[1] - 1)
    z_min = z_min_cortical if z_min_cortical < data_confocal.shape[2] - 1 else data_confocal.shape[2] - 1
    z_max = min(z_max_cortical, data_confocal.shape[2] - 1)

    cropped_cortical = data_cortical[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    cropped_confocal = data_confocal[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] 

    tifffile.imwrite(outFilepath_confocal, cropped_confocal)
    tifffile.imwrite(outFilepath_cortical, cropped_cortical)


def downsample_and_int16(tiff_path, output_path, downscale_factor):
    """
    Downsamples a 3D image stack to a specified resolution and converts it to int16 format.

    Parameters
    ------------------------
    tiff_path : str
        Filepath of the input TIFF image.
    output_path : str
        Filepath for the output downsampled image.
    downscale_factor : int
        Factor by which to downsample the image, expressed as 2 ** downscale_factor.

    Returns
    ------------------------
    None
    """
    img_stack = tifffile.imread(tiff_path)
    new_shape = (img_stack.shape[0] // 2**downscale_factor, img_stack.shape[1] // 2**downscale_factor, img_stack.shape[2] // 2**downscale_factor)
    downsampled_img_stack = transform.resize(img_stack, new_shape, order=3, preserve_range=True).astype(np.int16)
    tifffile.imwrite(output_path, downsampled_img_stack)


def normalize_image(inFilepath, outFilepath):
    """
    Normalizes the pixel intensities of an image to span the full range from 0 to 255.

    Parameters
    ------------------------
    inFilepath : str
        Filepath of the input TIFF image.
    outFilepath : str
        Filepath for the output normalized image.

    Returns
    ------------------------
    None
    """
    image = tifffile.imread(inFilepath)
    normalized_image = rescale_intensity(image, in_range='image', out_range=(0, 255)).astype(np.uint8)
    tifffile.imwrite(outFilepath, normalized_image)


def resample_image(image, current_resolution, desired_resolution):
    """
    Resample a 3D image to the desired resolution.

    Parameters
    ------------------------
    image : 3D numpy array
        The input image with shape (z, y, x).
    current_resolution : tuple of float
        Current resolutions for each dimension (z_res, y_res, x_res).
    desired_resolution : tuple of float
        Desired resolutions for each dimension (z_res, y_res, x_res).

    Returns
    ------------------------
    3D numpy array
        The resampled image.
    """
    current_z_res, current_y_res, current_x_res = current_resolution
    desired_z_res, desired_y_res, desired_x_res = desired_resolution

    # Calculate the zoom factors for each dimension
    zoom_factors = (
        current_z_res / desired_z_res,
        current_y_res / desired_y_res,
        current_x_res / desired_x_res
    )

    # Resample the image using the zoom factors
    resampled_image = zoom(image, zoom_factors, order=1)  # Using linear interpolation (order=1)

    return resampled_image
