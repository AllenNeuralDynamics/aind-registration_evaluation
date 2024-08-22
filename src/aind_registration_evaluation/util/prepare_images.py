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
    '''
    crops and downsamples n5 files to usable tiff file and stores it in scratch

    Parameters
    ------------------------
    confocal: filepath of the confocal image
    warpedCortical: filepath of the cortical image
    mask_treshold: mask to crop the picture
    downsampling_factor: 2**downsampling factor to change resolution

    Returns
    ------------------------
    none

    '''
    if not os.path.isfile('/scratch/confocal_s0.tif'):
        n5_to_tiff(confocal, '/scratch/confocal_s0.tif', 's0')
    if not os.path.isfile('/scratch/cortical_s0.tif'):
        n5_to_tiff_warped(warpedCortical, '/scratch/cortical_s0.tif')
    if not (os.path.isfile('/scratch/confocal_s0_cropped.tif') or os.path.isfile('/scratch/cortical_s0_cropped.tif')):
        apply_mask_and_crop('/scratch/cortical_s0.tif','/scratch/cortical_s0_cropped.tif', '/scratch/confocal_s0.tif', '/scratch/confocal_s0_cropped.tif', mask_threshold)
    if not (os.path.isfile('/scratch/confocal_s0_cropped_downsampled.tif')):
        downsample_and_int16('/scratch/confocal_s0_cropped.tif','/scratch/confocal_s0_cropped_downsampled.tif', downsampling_factor)
    if not (os.path.isfile('/scratch/cortical_s0_cropped_downsampled.tif')):
        downsample_and_int16('/scratch/cortical_s0_cropped.tif','/scratch/cortical_s0_cropped_downsampled.tif', downsampling_factor)


def tiff_to_n5(tiff_filename, n5_filename, downsampling_factors):
    '''
    converts tiff to n5 with the specified resolution

    Parameters
    ------------------------
    tiff_filename: input file path (string)
    n5_filename: out file path (string)
    downsampling_factor: 2**downsampling factor to change resolution

    Returns
    ------------------------
    none

    '''
    image_volume = tifffile.imread(tiff_filename)
    shape = image_volume.shape
    # Create an .n5 dataset
    f = z5py.File(n5_filename)
    # Create a group for the multiscale data
    if "c0" not in f:
        group = f.create_group("c0")
    else:
        group = f["c0"]

    # Create datasets for each scale level
    for i, factor in enumerate(downsampling_factors):
        # Calculate downsampled shape
        if "s"+str(i) in group:
            del group["s"+str(i)] 
        ds_data = image_volume[::factor, ::factor, ::factor]
        downsampled_shape = ds_data.shape
        ds = group.create_dataset('s'+str(i), shape=downsampled_shape,dtype=image_volume.dtype, chunks=(128, 128, 128), compression="gzip")
        ds[:] = ds_data
        # Optionally, store additional metadata
        ds.attrs["downsamplingFactors"] = tuple([factor,factor,factor])  # Example: store resolution

    # Close the .n5 file
    f.close()


def n5_to_tiff(n5_filename, output_path, scale_level):
    '''
    converts n5 to to tif with the specified resolution

    Parameters
    ------------------------
    tiff_filename: out file path (string)
    n5_filename: input file path (string)
    downsampling_factor: pyramid level of resolution

    Returns
    ------------------------
    none

    '''
    f = z5py.File(n5_filename)
    image = np.array(f["c0"][scale_level])
    tifffile.imwrite(output_path, image)

def n5_to_tiff_warped( n5_filename, output_path):
    '''
    converts the warped n5 to tiff (comes without resolution scale) in full resolution

    Parameters
    ------------------------
    tiff_filename: out file path (string)
    n5_filename: input file path (string)

    Returns
    ------------------------
    none

    '''
    f = z5py.File(n5_filename)
    image = np.array(f["c0"])
    tifffile.imwrite(output_path, image)

def apply_mask(inFilepath, outFilepath, threshhold, newValue):
    '''
    applies a mask on the picture

    Parameters
    ------------------------
    inFilepath: filepath of tifffile (string)
    outFilepath: filepath for desired filelocation of masked image
    threshold: intensity threshold under which the pixel will be set to newValue
    newValue: the newValue of pixels lower than threshold

    Returns
    ------------------------
    none

    '''
    data = tifffile.imread(inFilepath)
    mask = data <= threshhold
    data[mask] = newValue
    tifffile.imwrite(outFilepath, data)


def apply_mask_and_crop(inFilepath_cortical, outFilepath_cortical, inFilepath_confocal, outFilepath_confocal, threshold):
    '''
    applies a mask on the picture and crops the image of the nonoverlapping parts of the registered images

    Parameters
    ------------------------
    inFilepath_cortical: inFilepath of the warped image
    outFilepath_cortical: filepath for desired filelocation of cropped cortical image
    inFilepath_confocal: inFilepath of the target image
    outFilepath_cortical: filepath for desired filelocation of cropped confocal image
    threshold: intensity threshold under which the pixel will interpreted as not overlapping

    Returns
    ------------------------
    none

    '''
    data_cortical = tifffile.imread(inFilepath_cortical)
    data_confocal = tifffile.imread(inFilepath_confocal)
    data_cortical = data_cortical.astype(np.int16)
    data_confocal = data_confocal.astype(np.int16)
    mask = data_cortical <= threshold
    data_cortical[mask] = 0

    nonzeros_cortical = np.nonzero(data_cortical)
    # x_min, x_max = nonzeros[0].min(), nonzeros[0].max()
    # y_min, y_max = nonzeros[1].min(), nonzeros[1].max()
    # z_min, z_max = nonzeros[2].min(), nonzeros[2].max()

    x_min_cortical, x_max_cortical = nonzeros_cortical[0].min(), nonzeros_cortical[0].max()
    y_min_cortical, y_max_cortical = nonzeros_cortical[1].min(), nonzeros_cortical[1].max()
    z_min_cortical, z_max_cortical = nonzeros_cortical[2].min(), nonzeros_cortical[2].max()

    # Ensure the cortical bounds do not exceed the confocal dimensions
    x_min= x_min_cortical if x_min_cortical < data_confocal.shape[0] - 1 else data_confocal.shape[0] - 1
    x_max = min(x_max_cortical, data_confocal.shape[0] - 1)
    y_min= y_min_cortical if y_min_cortical < data_confocal.shape[1] - 1 else data_confocal.shape[1] - 1
    y_max = min(y_max_cortical, data_confocal.shape[1] - 1)
    z_min= z_min_cortical if z_min_cortical < data_confocal.shape[2] - 1 else data_confocal.shape[2] - 1
    z_max = min(z_max_cortical, data_confocal.shape[2] - 1)

    cropped_cortical = data_cortical[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    cropped_confocal = data_confocal[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] 

    tifffile.imwrite(outFilepath_confocal, cropped_confocal)
    tifffile.imwrite(outFilepath_cortical, cropped_cortical)

def downsample_and_int16(tiff_path, output_path, downscale_factor):
    '''
    Downsamples the images to a specified resolution.

    Parameters
    ------------------------
    tiff_path: input path of TIFF image
    output_path: desired file location of downsampled image
    downscale_factor: 2**downscale_factor specifies the pyramid level resolution

    Returns
    ------------------------
    None
    '''

    img_stack = tifffile.imread(tiff_path)

    # Calculate new dimensions for exactly halving the current dimensions
    new_shape = (img_stack.shape[0] // 2**downscale_factor, img_stack.shape[1] // 2**downscale_factor, img_stack.shape[2] // 2**downscale_factor)

    # Resize the entire image stack using skimage
    downsampled_img_stack = transform.resize(img_stack, new_shape, order=3, preserve_range=True).astype(np.int16)

    # Save the downsampled image stack
    tifffile.imwrite(output_path, downsampled_img_stack)

def normalize_image(inFilepath, outFilepath):
    '''
    normalizes the image

    Parameters
    ------------------------
    inFilePath: location of the image
    outFilepth: location of the normalized image

    Returns
    ------------------------
    none
    '''
    image = tifffile.imread(inFilepath)
    normalized_image = rescale_intensity(image, in_range='image', out_range=(-32768, 32767)).astype(np.int16)
    tifffile.imwrite(outFilepath, normalized_image)
