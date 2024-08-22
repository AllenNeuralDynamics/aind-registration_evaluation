import numpy as np
import tifffile as tiff
from scipy.ndimage import affine_transform
from aind_registration_evaluation.util.extract_roi import get_ROIs_cellpose, create_matching_mask, get_centroid_distances
from skimage import measure

class CellMasks():

    def __init__(self,
                 image1_mask,
                 image2_mask,
                 transformation_matrix=None,
                 maxCentroidDistance=10):
        '''
        Initializes the CellMasks class with masks for two images, optionally applying a transformation to the second image mask.

        Parameters
        ------------------------
        image1_mask: str or 3D numpy array
            Path to the TIFF file or a 3D numpy array representing the mask for the first image.

        image2_mask: str or 3D numpy array
            Path to the TIFF file or a 3D numpy array representing the mask for the second image.

        transformation_matrix: 4x4 numpy array, optional
            An optional 4x4 transformation matrix to apply to the second image mask. If provided, the matrix will be inverted and used for affine transformation.

        maxCentroidDistance: float, optional, default=10
            Maximum allowed distance between centroids from the two images to consider them as matching.

        Attributes
        ------------------------
        image1_mask: 3D numpy array
            The mask for the first image, read from file or provided as an array.

        image2_mask: 3D numpy array
            The mask for the second image, read from file or provided as an array, transformed if a matrix is provided.

        maxCentroidDistance: float
            The maximum allowed distance between centroids for matching.

        matching_patches: list of tuples or None
            List of patches matching between the two images. Initialized to None.

        num_img1_cells: int or None
            Number of cells in the first image. Initialized to None.

        num_img2_cells: int or None
            Number of cells in the second image. Initialized to None.

        num_matching_cells: int or None
            Number of matching cells between the two images. Initialized to None.

        image1_matching_mask: 3D numpy array or None
            Mask of the first image with matched cells highlighted. Initialized to None.

        image2_matching_mask: 3D numpy array or None
            Mask of the second image with matched cells highlighted. Initialized to None.
        '''

        if isinstance(image1_mask, str): self.image1_mask = tiff.imread(image1_mask)
        else: self.image1_mask = image1_mask

        if isinstance(image2_mask, str): self.image2_mask = tiff.imread(image2_mask)
        else: self.image2_mask = image2_mask

        min_shape = np.minimum(self.image1_mask.shape, self.image2_mask.shape)
    
        # Clip both arrays to the smaller size
        self.image1_mask = self.image1_mask[:min_shape[0], :min_shape[1], :min_shape[2]]
        self.image2_mask = self.image2_mask[:min_shape[0], :min_shape[1], :min_shape[2]]

        if transformation_matrix is not None:
            self.transformation_matrix = transformation_matrix
            inverse_matrix = np.linalg.inv(transformation_matrix)
            self.image2_mask = affine_transform(self.image2_mask, inverse_matrix)
        else: self.transformation_matrix = None

        self.maxCentroidDistance = maxCentroidDistance

        self.matching_patches = None
        self.num_img1_cells = None
        self.num_img2_cells = None
        self.num_matching_cells = None
        self.image1_matching_mask = None
        self.image2_matching_mask = None


    def get_matching(self, maxCentroidDistance):
        '''
        Computes or retrieves the matching cell patches between the two images, based on the centroid distance. It can optionally use a different distance threshold.
        - If the matching patches and counts are not already computed and stored in the instance, the method calculates them using `get_ROIs_cellpose` with the provided or default `maxCentroidDistance`.
        - If the `maxCentroidDistance` parameter is different from the instance's `maxCentroidDistance`, the method recalculates the matching patches with the new distance threshold.

        Parameters
        ------------------------
        maxCentroidDistance: float, optional
            The maximum distance allowed between centroids to consider them as matching. If provided, it overrides the instance's `maxCentroidDistance`.

        Returns
        ------------------------
        resultDict: dict
            A dictionary containing the following keys:
            - 'NumImg1Cells': int
                The number of cells identified in the first image.
            - 'NumImg2Cells': int
                The number of cells identified in the second image.
            - 'maxCentroidDistance': float
                The centroid distance threshold used for matching.
            - 'NumMatchingCells': int
                The number of matching cells between the two images.
            - 'NumOfPatches': int
                The number of patches created based on the cell matches.
            - 'Patches': list of tuples
                List of matching patches between the two images. Each patch is represented by a tuple containing coordinates.
        '''
        if (self.matching_patches is None or
            self.num_img1_cells is None or
            self.num_img2_cells is None or
            self.num_matching_cells is None):

            if maxCentroidDistance is not None and self.maxCentroidDistance != maxCentroidDistance:
                matching_patches, num_img1_cells, num_img2_cells, num_matching_cells = (
                    get_ROIs_cellpose(self.image1_mask, self.image2_mask, maxCentroidDistance)
                )              

            else:
                matching_patches, num_img1_cells, num_img2_cells, num_matching_cells = (
                    get_ROIs_cellpose(self.image1_mask, self.image2_mask, self.maxCentroidDistance)
                )

                self.matching_patches, self.num_img1_cells, self.num_img2_cells, self.num_matching_cells = matching_patches, num_img1_cells, num_img2_cells, num_matching_cells
            
        else:
            if maxCentroidDistance is not None and self.maxCentroidDistance != maxCentroidDistance:
                matching_patches, num_img1_cells, num_img2_cells, num_matching_cells = (
                    get_ROIs_cellpose(self.image1_mask, self.image2_mask, maxCentroidDistance)
                )
            else:
                matching_patches, num_img1_cells, num_img2_cells, num_matching_cells = self.matching_patches, self.num_img1_cells, self.num_img2_cells, self.num_matching_cells

        resultDict = {
            'NumImg1Cells': num_img1_cells,
            'NumImg2Cells': num_img2_cells,
            'maxCentroidDistance': maxCentroidDistance,
            'NumMatchingCells': num_matching_cells,
            'NumOfPatches': len(matching_patches),
            'Patches': matching_patches,
            'MatchingValue': (num_matching_cells**2/(num_img1_cells*num_img2_cells))
        }

        return resultDict



    def get_matching_mask_images(self, maxCentroidDistance):
        '''
        Retrieves or computes the matching mask images for the two input images, based on the centroid distance. 
        - If the matching masks are not already computed and stored in the instance, the method computes them using `create_matching_mask` with either the provided or default `maxCentroidDistance`.
        - If the `maxCentroidDistance` parameter is different from the instance's `maxCentroidDistance`, the method recalculates the matching masks using the new distance threshold.
        
        Parameters
        ------------------------
        maxCentroidDistance: float, optional
            The maximum distance allowed between centroids to consider them as matching. If provided and different from the instance's `maxCentroidDistance`, the method will recompute the masks with this new distance threshold.

        Returns
        ------------------------
        image1_matching_mask: numpy.ndarray
            The binary mask for the first image indicating the matching cells.

        image2_matching_mask: numpy.ndarray
            The binary mask for the second image indicating the matching cells.
        '''
        if (self.image1_matching_mask is None or self.image2_matching_mask is None):
            if maxCentroidDistance is not None and maxCentroidDistance != self.maxCentroidDistance:
                image1_matching_mask, image2_matching_mask = (
                    create_matching_mask(self.image1_mask, self.image2_mask, maxCentroidDistance)
                )
            else:
                image1_matching_mask, image2_matching_mask = (
                    create_matching_mask(self.image1_mask, self.image2_mask, self.maxCentroidDistance)
                )
                self.image1_matching_mask, self.image2_matching_mask = image1_matching_mask, image2_matching_mask
        else:
            if maxCentroidDistance is not None and maxCentroidDistance != self.maxCentroidDistance:
                image1_matching_mask, image2_matching_mask = (
                    create_matching_mask(self.image1_mask, self.image2_mask, maxCentroidDistance)
                )
            else:
                image1_matching_mask, image2_matching_mask = self.image1_matching_mask, self.image2_matching_mask
        
        return image1_matching_mask, image2_matching_mask

    def matching_cells_by_distance_plot(self):
        """
        Plots the number of matching cells as a function of the centroid distance threshold.

        This method calculates the number of cell matches between the masks of two images
        for different centroid distance thresholds and returns the results for plotting.

        Returns
        -------
        centroid_distances : list of int
            List of centroid distance thresholds used for matching.
        
        num_of_matches : list of int
            List of the number of matches corresponding to each centroid distance threshold.
        
        min_cells : int
            The minimum number of cells between the two images.
        """
        # Extract region properties
        img1_regions = measure.regionprops(self.image1_mask)
        img2_regions = measure.regionprops(self.image2_mask)

        # Initialize lists for storing results
        centroid_distances = list(range(0, 50, 2))
        num_of_matches = []

        # Calculate number of matches for each centroid distance
        for i in centroid_distances:   
            cp_centroid_dist = get_centroid_distances(img1_regions, img2_regions, i)
            num_of_matches.append(len(cp_centroid_dist))

        return centroid_distances, num_of_matches, len(img1_regions), len(img2_regions)





        




                

                







