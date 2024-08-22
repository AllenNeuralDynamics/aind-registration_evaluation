import numpy as np
from skimage import filters, morphology, measure
from scipy.spatial import KDTree
import tifffile as tiff
import random

def get_props(image, binaryThreshold):
    '''
    	- Applies Gaussian blur to smooth the image
		- Creates a binary mask on the image to extract the bright areas. Binary threshold should be adjusted for different images
		- Opens the image to identify single cells more distinct
		- Removes small object that could be rest noise of bright areas in the image which are not a cell
        - Labels the image and extracts the region properties

        Parameters
        ------------------------
        image: 3d numpy array

        binaryThreshold: threshold to create a binary image based on intensity (int)

        Returns
        ------------------------
        regionprops list (see https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)
    '''
    blurred_image = filters.gaussian(image, sigma=1)
    binary_image = blurred_image > binaryThreshold
    opened_image = morphology.opening(binary_image, morphology.ball(1))
    largeObjects = morphology.remove_small_objects(opened_image, 500)
    labels = morphology.label(largeObjects)
    regions = measure.regionprops(labels)
    return regions

def get_centroid_distances(props1, props2, distanceThreshold):
    '''
    - Calculates distances of the centroids of each cell in one image to each cell in the other image.
    - A distance threshold is applied to only store the centroids that are close to each other, indicating that these identify the same cells in registered images.
    - Centroids from img1 that all match to the same centroid in img2 but with longer distance are removed because each centroid should have a distinct match (the closest) with a centroid in the other image.

    Parameters
    ------------------------
    props1, props2: regionprops list from both images (see https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)

    distanceThreshold: pixel count for allowed distance of centroids to be seen as a match

    Returns
    ------------------------
    Dictionary of shape {distance: {'R1': p1, 'R2': p2,'addedDist': addedDist}}
    '''
    centroids1 = np.array([p.centroid for p in props1])
    centroids2 = np.array([p.centroid for p in props2])

    tree = KDTree(centroids2)
    distances = {}

    balls = tree.query_ball_point(centroids1, distanceThreshold)

    for i_p1, b in enumerate(balls):
        for i_p2 in b:
            p1 = props1[i_p1]
            p2 = props2[i_p2]
            dist = np.linalg.norm(np.array(p1.centroid) - np.array(p2.centroid))
            addedDist = 0
            while dist in distances:
                dist += 0.0001
                addedDist += 0.0001
            distances[dist] = {'R1': p1, 'R2': p2,'addedDist': addedDist}

    sortedDistances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[0] + item[1]['addedDist'])}
    p1Used = set()
    p2Used = set()
    resultDistances = {}
    for k, v in sortedDistances.items():
        if v['R1'].label in p1Used or v['R2'].label in p2Used:
            continue
        resultDistances[k] = v
        p1Used.add(v['R1'].label)
        p2Used.add(v['R2'].label)

    return resultDistances

def create_patches(distances, image_shape, overlapThreshold):
    '''
        - Creates a patch around each matched centroids by adding 3 pixels around the max coordinates of the patches (limited by image bounds).
        - Merges the patches that show more overlap than the overlap threshold to avoid duplicate evaluations on the same area.

        Parameters
        ------------------------
        distances: Dictionary of shape {distance:{'R1': p1, 'R2': p2,'addedDist': addedDist}}
        
        image_shape: shape of the two images must be the same. Either can be passed in the form Tuple(z,y,x)

        overlapThreshold: volume overlap theshold above which two patches are merged to their max coordinates (float)

        Returns
        ------------------------
        coordinates of the merged_patches (z_min, y_min, x_min, z_max, y_max, x_max)
    '''
    patches = []
    #get bounding coordinates of each patch while staying within image shape
    for distance, info in distances.items():
        bbox1 = info['R1'].bbox
        bbox2 = info['R2'].bbox
        z_min = 0 if min(bbox1[0],bbox2[0])<4 else (min(bbox1[0],bbox2[0])-3) 
        y_min = 0 if min(bbox1[1],bbox2[1])<4 else (min(bbox1[1],bbox2[1])-3)
        x_min = 0 if min(bbox1[2],bbox2[2])<4 else (min(bbox1[2],bbox2[2])-3)
        z_max = image_shape[0] if max(bbox1[3],bbox2[3])>(image_shape[0]-4) else (max(bbox1[3],bbox2[3])+3)
        y_max = image_shape[1] if max(bbox1[4],bbox2[4])>(image_shape[1]-4) else (max(bbox1[4],bbox2[4])+3)
        x_max = image_shape[2] if max(bbox1[5],bbox2[5])>(image_shape[2]-4) else (max(bbox1[5],bbox2[5])+3)
        patches.append((z_min, y_min, x_min, z_max, y_max, x_max))
    
    merged_patches = []
    #when all patches added or merged, stop
    while patches:
        z_min, y_min, x_min, z_max, y_max, x_max = patches.pop(0)
        merged = False
        for i, (mz_min, my_min, mx_min, mz_max, my_max, mx_max) in enumerate(merged_patches):
            #if there is overlap between two patches
            if not (z_max < mz_min or z_min > mz_max or y_max < my_min or y_min > my_max or x_max < mx_min or x_min > mx_max):
                overlap_patch = (max(z_min, mz_min), max(y_min, my_min), max(x_min, mx_min), min(z_max, mz_max), min(y_max, my_max), min(x_max, mx_max))
                max_patch = (min(z_min, mz_min), min(y_min, my_min), min(x_min, mx_min), max(z_max, mz_max), max(y_max, my_max), max(x_max, mx_max))
                volume_overlap = (overlap_patch[3] - overlap_patch[0]) * (overlap_patch[4] - overlap_patch[1]) * (overlap_patch[5] - overlap_patch[2])
                volume_max_patch = (max_patch[3] - max_patch[0]) * (max_patch[4] - max_patch[1]) * (max_patch[5] - max_patch[2])
                #if the overlap is more than the given threshold -> merge the two patches into one
                if volume_overlap/volume_max_patch > overlapThreshold:
                    merged_patches[i] = max_patch

                    #if you merge within the final patches array, reevaluate all following patches if they (also) have a big overlap with the new merged patch
                    #for that we remove them from the final array and append them back to the original patch array
                    if i<(len(merged_patches)-1):
                        patches += merged_patches[i+1:]
                        merged_patches = merged_patches[:i+1]
                    merged = True
                    break
        if not merged:
            merged_patches.append((z_min, y_min, x_min, z_max, y_max, x_max))
    return merged_patches

def create_random_patches(distances, image_1_shape):
    '''
    - Generates random patches around centroids from two images.
    - Shuffles centroids from the second image and pairs them with centroids from the first image.
    - Creates patches of fixed size (31x31x31) around each centroid pair if the centroid is within image bounds.

    Parameters
    ------------------------
    distances: Dictionary of shape {distance: {'R1': p1, 'R2': p2, 'addedDist': addedDist}}
        Dictionary containing region properties for the two images. The dictionary values should have keys 'R1' and 'R2' representing the properties of regions in image 1 and image 2 respectively.

    image_1_shape: Tuple of shape (z, y, x)
        Shape of the image to ensure generated patches are within the image bounds.

    Returns
    ------------------------
    List of tuples
        Each tuple contains two sets of coordinates representing the patches around centroids from the two images:
        (coord1, coord2)
        where coord1 and coord2 are tuples of the form (z_min, y_min, x_min, z_max, y_max, x_max) defining the patch boundaries.
    '''
    patches = []
    r1_values = [v['R1'] for v in distances.values()]
    r2_values = [v['R2'] for v in distances.values()]
    
    # Shuffle R2 values
    random.shuffle(r2_values)
    
    # Create pairs of R1 and shuffled R2
    paired_list = list(zip(r1_values, r2_values))
    for r1,r2 in paired_list:
        c1_float, c2_float = r1.centroid, r2.centroid
        c1 = (np.round(c1_float[0]).astype(int),np.round(c1_float[1]).astype(int),np.round(c1_float[2]).astype(int))
        c2 = (np.round(c2_float[0]).astype(int),np.round(c2_float[1]).astype(int),np.round(c2_float[2]).astype(int))
        if c1[0]-15<0 or c1[1]-15<0 or c1[2]-15<0 or c1[0]+15>image_1_shape[0] or c1[1]+15>image_1_shape[1] or c1[2]+15>image_1_shape[2]: continue
        if c2[0]-15<0 or c2[1]-15<0 or c2[2]-15<0 or c2[0]+15>image_1_shape[0] or c2[1]+15>image_1_shape[1] or c2[2]+15>image_1_shape[2]: continue
        coord1 = (c1[0]-15, c1[1]-15, c1[2]-15, c1[0]+15, c1[1]+15, c1[2]+15)
        coord2 = (c2[0]-15, c2[1]-15, c2[2]-15, c2[0]+15, c2[1]+15, c2[2]+15)
        patches.append((coord1,coord2))
    return patches

def get_ROIs_cellpose(img1, img2, maxCentroidDistance=10, make_mask=False, img1_binaryThreshold=0.05, img2_binaryThreshold=0.008, overlapThreshold=0.3):
    '''
    - Extracts regions of interest (ROIs) from two images based on their centroids and a specified maximum centroid distance.
    - Computes centroid distances between two images and creates patches around matched centroids.
    - Generates a dictionary of patch coordinates and their corresponding volumes.

    Parameters
    ------------------------
    img1: 3D numpy array
        The first image from which to extract cell properties.

    img2: 3D numpy array
        The second image from which to extract cell properties.
    
    make_mask: bool
        If true will create the mask based on the provided binary thresholds. Else it will interpret the given images as masks already

    img1_binaryThreshold: int
        Binary threshold for creating a binary image from the first image to identify bright areas.

    img2_binaryThreshold: int
        Binary threshold for creating a binary image from the second image to identify bright areas.

    maxCentroidDistance: float
        Maximum allowed distance between centroids from the two images to consider them as matching.

    overlapThreshold: float
        Volume overlap threshold above which two patches are merged to avoid duplicate evaluations.

    Returns
    ------------------------
    patch_dict: Dictionary
        A dictionary where the keys are tuples representing the coordinates of the patches (z_min, y_min, x_min, z_max, y_max, x_max)
        and the values are the volumes of these patches.

    len_img1_regions: int
        The number of regions identified in the first image.

    len_img2_regions: int
        The number of regions identified in the second image.

    len_cp_centroid_dist: int
        The number of centroid matches found between the two images based on the maximum centroid distance.
    '''
    if make_mask:
        img1_regions = get_props(img1, img1_binaryThreshold)
        img2_regions = get_props(img2, img2_binaryThreshold)
    else:
        img1_regions = measure.regionprops(img1)
        img2_regions = measure.regionprops(img2)
    cp_centroid_dist = get_centroid_distances(img1_regions, img2_regions, maxCentroidDistance)
    cp_patches = create_patches(cp_centroid_dist, img1.shape, overlapThreshold)
    patch_dict = {r: ((r[3]-r[0]) * (r[4]-r[1]) * (r[5]-r[2])) for r in cp_patches}
    return patch_dict, len(img1_regions), len(img2_regions), len(cp_centroid_dist)


def get_ROIs_random_matching(img1, img2, maxCentroidDistance):
    '''
    - Extracts regions of interest (ROIs) from two images by randomly matching centroids.
    - Computes centroid distances between two images and creates random patches around matched centroids.
    - Generates a list of patch coordinates for random matching of centroids between the images.

    Parameters
    ------------------------
    img1: 3D numpy array
        The first image from which to extract cell properties.

    img2: 3D numpy array
        The second image from which to extract cell properties.

    maxCentroidDistance: float
        Maximum allowed distance between centroids from the two images to consider them as matching.

    Returns
    ------------------------
    cp_patches: List of tuples
        A list of tuples where each tuple contains two sets of coordinates (coord1, coord2) representing the patches around matched centroids.

    len_img1_regions: int
        The number of regions identified in the first image.

    len_img2_regions: int
        The number of regions identified in the second image.

    len_cp_centroid_dist: int
        The number of centroid matches found between the two images based on the maximum centroid distance.
    '''
    img1_regions = measure.regionprops(img1)
    img2_regions = measure.regionprops(img2)
    cp_centroid_dist = get_centroid_distances(img1_regions, img2_regions, maxCentroidDistance)
    cp_patches = create_random_patches(cp_centroid_dist, img1.shape)
    return cp_patches, len(img1_regions), len(img2_regions), len(cp_centroid_dist)


def create_matching_mask(image1 , image2, maxCentroidDistance, saveImages=False):
    '''
    - Creates a matching mask for two images by labeling and highlighting the regions corresponding to matching centroids.
    - Generates binary masks where pixels corresponding to matched centroids are labeled with unique identifiers.
    - Optionally saves the resulting masks as TIFF images.

    Parameters
    ------------------------
    image1: 3D numpy array
        The first image on which to create a matching mask.

    image2: 3D numpy array
        The second image on which to create a matching mask.

    maxCentroidDistance: float
        Maximum allowed distance between centroids from the two images to consider them as matching.

    saveImages: bool, optional
        Whether to save the resulting masks as TIFF images. Defaults to False.

    Returns
    ------------------------
    Tuple of 3D numpy arrays
        - The first array contains the matching mask for the first image.
        - The second array contains the matching mask for the second image.
    '''
    img1_regions = measure.regionprops(image1)
    img2_regions = measure.regionprops(image2)
    centroid_dist = get_centroid_distances(img1_regions, img2_regions, maxCentroidDistance)

    img1 = np.zeros(image1.shape, dtype=np.uint16)
    img2 = np.zeros(image2.shape, dtype=np.uint16)

    label = 1
    totalPixels1 = 0
    totalPixels2 = 0
    for k, v in centroid_dist.items():
        r1 = v['R1']
        r2 = v['R2']
        for c in r1.coords:
            z,y,x = c
            img1[z,y,x] = label
            totalPixels1 += 1
        for c in r2.coords:
            z,y,x = c
            img2[z,y,x] = label
            totalPixels2 += 1
        label += 1

    if saveImages:
        tiff.imwrite('/scratch/img1_matching_mask.tif', img1)
        tiff.imwrite('/scratch/img2_matching_mask.tif', img2)

    return (img1,img2)
