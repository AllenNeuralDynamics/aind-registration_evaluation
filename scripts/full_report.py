import multiprocessing
from dask.distributed import Client, LocalCluster
from aind_registration_evaluation.metric.image_analysis import ImageAnalysis

def main(cell_match_count=True, full_image=True, roi_patches=True, full_image_matching_mask=True, roi_patches_matching_mask=True):
    '''
    Main function to set up a Dask cluster and perform image analysis by initializing the ImageAnalysis class
    and running the analysis with the given parameters.
    To run it, move it in src folder and run python -u full_report.py from the src folder

    Parameters
    ------------------------
    cell_match_count: bool, optional, default=True
        Whether to count the number of cell matches between the images.

    full_image: bool, optional, default=True
        Whether to process the full images.

    roi_patches: bool, optional, default=True
        Whether to process ROI patches.

    full_image_matching_mask: bool, optional, default=True
        Whether to evaluate a matching mask for the full images.

    roi_patches_matching_mask: bool, optional, default=True
        Whether to evaluate a matching mask for the ROI patches.
    '''
    n_workers = multiprocessing.cpu_count()
    threads_per_worker = 1
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
        memory_limit="auto",
    )

    client = Client(cluster)
    path1 = '/root/capsule/data/718431_coregistration/confocal_s0_cropped_dsLev2.tif'
    path2 = '/root/capsule/data/718431_coregistration/cortical_s0_cropped_dsLev2.tif'
    m_path1 = '/root/capsule/data/718431_coregistration/confocal_s0_cropped_dsLev2_cp_masks.tif'
    m_path2 = '/root/capsule/data/718431_coregistration/cortical_s0_cropped_dsLev2_cp_masks.tif'

    matrix = [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]

    IA = ImageAnalysis(
        image1 = path1, 
        image2 = path2,
        image1_mask = m_path1,
        image2_mask = m_path2,
        transformation_matrix=matrix,
        maxCentroidDistance=10,
        cell_match_count=True, 
        full_image=True, 
        roi_patches=True, 
        full_image_matching_mask=True,
        roi_patches_matching_mask=True,
        num_matches_plot=False, 
        show_image=False,
        nmi=True, 
        mi=False
    )

    IA.run_to_excel(maxCentroidDistance=10)

    client.close()

if __name__ == "__main__":
    main()