"""
Evaluate stitching misalignment
of large scale data.
"""
import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
import yaml
from argschema import ArgSchemaParser
from scipy.ndimage import gaussian_filter

from aind_registration_evaluation import sample, util
from aind_registration_evaluation._shared.types import ArrayLike
from aind_registration_evaluation.io import extract_data, get_data
from aind_registration_evaluation.metric import (
    ImageMetricsFactory, compute_feature_space_distances,
    get_pairs_from_distances)
from aind_registration_evaluation.params import EvalRegSchema
from aind_registration_evaluation.sample import *

# IO types
PathLike = Union[str, Path]

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler("test.log", "a"),
    ],
)
logging.disable("DEBUG")

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def generate_overlap_slices(
    shapes: List[Tuple], orientation: str, overlap_ratio: float
):
    image_1_shape = shapes[0]
    image_2_shape = shapes[1]
    # Getting overlapping area for images
    overlap_area_1 = (np.array(image_1_shape) * overlap_ratio).astype(int)
    overlap_area_2 = (np.array(image_2_shape) * overlap_ratio).astype(int)

    if orientation == "x":
        # Left
        offset_img_1 = image_1_shape[1] - overlap_area_1[1]

        slices_1 = (
            slice(0, image_1_shape[0]),
            slice(offset_img_1, image_1_shape[1]),
        )
        # Right
        slices_2 = (slice(0, image_2_shape[0]), slice(0, overlap_area_2[1]))

    else:
        # Top
        offset_img_1 = image_1_shape[0] - overlap_area_1[0]
        slices_1 = (
            slice(offset_img_1, image_1_shape[0]),
            slice(0, image_1_shape[1]),
        )
        # Bottom
        slices_2 = (slice(0, overlap_area_2[0]), slice(0, image_2_shape[1]))

    return slices_1, slices_2, offset_img_1


def generate_feature_decriptors(
    img: ArrayLike,
    n_keypoints: int,
    pad_width: int,
    mode: Optional[str] = "energy",
    gss_sigma: Optional[int] = 8,
):
    """
    Generates feature decriptors for 2D/3D
    images.

    Parameters
    ----------
    img: ArrayLike
        Array with image data

    n_keypoints: int
        Number of keypoints to sample.

    pad_width: int
        Image padding for necessary for
        non-linear filtering

    mode: Optional[str]
        Image filtering approach. We have two
        options, one is energy with generates
        the image energy using a Laplacian of
        Gaussian-based approach while the second
        uses maximum filtering to get maximas and
        minimas.

        Default: "energy"

    gss_sigma: Optional[int]
        Sigma in the gaussian filtering before
        computing image derivatives.
        Default: 8

    Raises
    ------------
    NotImplementedError:
        Whenever the image is not 2D or 3D

    Returns
    ------------
    dict:
        Dictionary with the feature decriptors,
        keypoint locations and filtered image
        (which represents the response image).
    """

    mode = mode.casefold()
    mode_options = ["energy", "maxima"]

    if img.ndim not in [2, 3]:
        raise NotImplementedError("We require a 2D or 3D image")

    if mode not in mode_options:
        raise NotImplementedError(f"We only have 2 options: {mode_options}")

    if mode == "energy":
        img_keypoints, img_response = kd_fft_energy_keypoints(
            image=img,
            pad_width=pad_width,
            n_keypoints=n_keypoints,
        )
    else:
        img_keypoints, img_response = kd_fft_keypoints(
            image=img,
            pad_width=pad_width,
            n_keypoints=n_keypoints,
        )

    # Getting image derivatives
    derivatives = derivate_image_axis(
        # Smoothing before
        image=gaussian_filter(img, sigma=gss_sigma),
        axis=list(range(img.ndim)),  # [0, 1, ..., Ndims]
    )

    # Getting image magnitude and orientations
    (
        gradient_magnitude,
        gradient_orientation,
        gradient_orientation_polar,
    ) = kd_gradient_magnitudes_and_orientations(
        derivated_images=derivatives  # [img_2d_dy, img_2d_dx]
    )

    if img.ndim == 2:
        gradient_orientations = [gradient_orientation]

    else:
        gradient_orientations = [
            gradient_orientation_polar,
            gradient_orientation,
        ]

    # Getting keypoint feature decriptors
    img_keypoints_features = [
        kd_compute_keypoints_hog(
            image_gradient_magnitude=gradient_magnitude,
            image_gradient_orientation=gradient_orientations,
            keypoint=keypoint,
            n_dims=img.ndim,
            window_size=16,
            bins=[8],
        )
        for keypoint in img_keypoints
    ]

    keypoints = []
    features = []
    for key_feat in img_keypoints_features:
        keypoints.append(key_feat["keypoint"])
        features.append(key_feat["feature_vector"])

    return {
        "keypoints": np.array(keypoints),
        "features": np.array(features),
        "response_img": img_response,
    }


def remove_outliers(data, outlier_threshold=2):
    mean = np.mean(data)
    std_dev = np.std(data)

    # Define the lower and upper bounds for identifying outliers
    lower_bound = mean - outlier_threshold * std_dev
    upper_bound = mean + outlier_threshold * std_dev

    # Filter out the outliers
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]

    return filtered_data


def calculate_central_value(data, central_type="mean", outlier_threshold=2):
    filtered_data = remove_outliers(data, outlier_threshold)

    if central_type == "mean":
        central_value = np.mean(filtered_data)
    elif central_type == "median":
        central_value = np.median(filtered_data)
    else:
        raise ValueError("Invalid central_type. Choose 'mean' or 'median'.")

    return central_value


class EvalStitching(ArgSchemaParser):
    """
    Class to Evaluate Stitching.
    """

    default_schema = EvalRegSchema

    def run(self):
        """
        Args:
            Evaluate block
        """

        print(self.args)

        image_1_data = None
        image_2_data = None

        # read data/pointers and linear transform
        image_1, image_2, transform = get_data(
            path_image_1=self.args["image_1"],
            path_image_2=self.args["image_2"],
            data_type=self.args["data_type"],
            transform_matrix=self.args["transform_matrix"],
        )

        if self.args["data_type"] == "large":
            # Load dask array
            image_1_data = extract_data(image_1.as_dask_array())
            image_2_data = extract_data(image_2.as_dask_array())

        elif self.args["data_type"] == "small":
            image_1_data = extract_data(image_1.as_numpy_array())
            image_2_data = extract_data(image_2.as_numpy_array())

        elif "dummy" in self.args["data_type"]:
            image_1_data = image_1
            image_2_data = image_2

        util.validate_image_transform(
            image_1=image_1_data,
            image_2=image_2_data,
            transform_matrix=transform,
        )

        image_1_shape = image_1_data.shape
        image_2_shape = image_2_data.shape
        print(image_1_shape)

        # calculate extent of overlap using transforms
        # in common coordinate system (assume for image 1)
        bounds_1, bounds_2 = util.calculate_bounds(
            image_1_shape, image_2_shape, transform
        )
        print("Bounds: ", bounds_1, bounds_2)

        # Sample points in overlapping bounds
        points = sample.sample_points_in_overlap(
            bounds_1=bounds_1,
            bounds_2=bounds_2,
            numpoints=self.args["sampling_info"]["numpoints"],
            sample_type=self.args["sampling_info"]["sampling_type"],
        )

        pruned_points = sample.prune_points_to_fit_window(
            image_1_shape, points, self.args["window_size"]
        )

        discarded_points_window = points.shape[0] - pruned_points.shape[0]
        LOGGER.info(
            f"Number of discarded points when prunning points to window: {discarded_points_window}",
        )

        # calculate metrics per images
        metrics_results = {}
        metrics = []

        for metric in self.args["metrics"]:
            m = metric.casefold()
            metrics_results[m] = {"selected_points": [], "point_metric": []}
            metrics.append(
                ImageMetricsFactory().create(
                    image_1_data,
                    image_2_data,
                    m,
                    self.args["window_size"],
                )
            )

        for pruned_point in pruned_points:
            for metric in metrics:
                metric_name = metric.metric_type
                point_metric = metric.calculate_metric(
                    point=pruned_point, transform=transform
                )

                if point_metric is not None:
                    metrics_results[metric_name]["point_metric"].append(
                        point_metric
                    )
                    metrics_results[metric_name]["selected_points"].append(
                        pruned_point
                    )

        # compute statistics
        for metric in metrics:
            metric_name = metric.metric_type

            computed_points = len(
                metrics_results[metric_name]["selected_points"]
            )

            dscrd_pts = (
                points.shape[0] - discarded_points_window - computed_points
            )
            message = f"""Computed metric: {metric_name}
            \nMean: {np.mean(metrics_results[metric_name]["point_metric"])}
            \nStd: {np.std(metrics_results[metric_name]["point_metric"])}
            \nNumber of calculated points: {computed_points}
            \nDiscarded points by metric: {dscrd_pts}"""
            LOGGER.info(message)

            if self.args["visualize"]:
                util.visualize_images(
                    image_1_data,
                    image_2_data,
                    [bounds_1, bounds_2],
                    pruned_points,
                    metrics_results[metric_name]["selected_points"],
                    transform,
                    metric_name,
                )

    def run_misalignment_2d(
        self,
        n_keypoints: int,
        pad_width: int,
        overlap_ratio: Optional[float] = 0.10,
        orientation: Optional[str] = "x",
    ) -> List[np.ndarray]:
        """
        Runs misalignment metric for stitching
        evaluation using the overlap ratio that
        goes from the left image to the right image.
        Note: This example is for a 2D image

        Parameters
        -----------
        overlap_ratio: Optional[float]
            Overlap between images.
            Default: 0.1 -> 10%

        orientation: Optional[str]
            Overlap orientation
            ["x", "y", "z"]

        Raises
        -----------
        NotImplementedError:
            If the image extension is not
            ['.zarr', '.tiff', '.tif'].
            Please, check the image factory
            class inside the io package

        Returns
        -----------
        List[float]
            List of distances between the identified
            keypoints in the images
        """

        image_1_data = None
        image_2_data = None

        # read data/pointers and linear transform
        image_1, image_2, transform = get_data(
            path_image_1=self.args["image_1"],
            path_image_2=self.args["image_2"],
            data_type=self.args["data_type"],
            transform_matrix=self.args["transform_matrix"],
        )

        if self.args["data_type"] == "large":
            # Load dask array
            image_1_data = extract_data(image_1.as_dask_array())
            image_2_data = extract_data(image_2.as_dask_array())

        elif self.args["data_type"] == "small":
            image_1_data = extract_data(image_1.as_numpy_array())
            image_2_data = extract_data(image_2.as_numpy_array())

        elif "dummy" in self.args["data_type"]:
            image_1_data = image_1
            image_2_data = image_2

        util.validate_image_transform(
            image_1=image_1_data,
            image_2=image_2_data,
            transform_matrix=transform,
        )

        image_1_shape = image_1_data.shape
        image_2_shape = image_2_data.shape

        # calculate extent of overlap using transforms
        # in common coordinate system (assume for image 1)
        bounds_1, bounds_2 = util.calculate_bounds(
            image_1_shape, image_2_shape, transform
        )

        # Compute keypoints between images

        slices_1, slices_2, offset_img_1 = generate_overlap_slices(
            shapes=[image_1_shape, image_2_shape],
            orientation=orientation,
            overlap_ratio=overlap_ratio,
        )

        img_1_dict = generate_feature_decriptors(
            image_1_data[slices_1],
            n_keypoints=n_keypoints,
            pad_width=pad_width,
        )
        img_2_dict = generate_feature_decriptors(
            image_2_data[slices_2],
            n_keypoints=n_keypoints,
            pad_width=pad_width,
        )

        left_image_keypoints = img_1_dict["keypoints"]
        right_image_keypoints = img_2_dict["keypoints"]

        feature_vector_img_1 = (
            img_1_dict["features"],
            left_image_keypoints,
        )
        feature_vector_img_2 = (
            img_2_dict["features"],
            right_image_keypoints,
        )

        distances = compute_feature_space_distances(
            feature_vector_img_1, feature_vector_img_2, feature_weight=0.2
        )

        point_matches_pruned = get_pairs_from_distances(
            distances=distances, delete_points=True, metric_threshold=0.1
        )

        if not len(point_matches_pruned):
            print("No keypoints found!")
            return np.nan, np.nan

        # Tomorrow map points to the same
        # coordinate system
        # Working only with translation at the moment
        offset_ty = transform[0, -1]
        offset_tx = transform[1, -1]

        # Moving image keypoints back to intersection area
        if orientation == "y":
            left_image_keypoints[:, 0] += offset_img_1
        else:
            # x
            left_image_keypoints[:, 1] += offset_img_1

        right_image_keypoints[:, 0] += offset_ty
        right_image_keypoints[:, 1] += offset_tx

        # distance between points
        point_distances = np.array([])
        picked_left_points = []
        picked_right_points = []

        for left_idx, right_idx in point_matches_pruned.items():
            picked_left_points.append(left_image_keypoints[left_idx])
            picked_right_points.append(right_image_keypoints[right_idx])

            loc_dif = np.sqrt(
                np.sum(
                    np.power(
                        left_image_keypoints[left_idx]
                        - right_image_keypoints[right_idx],
                        2,
                    ),
                    axis=-1,
                )
            )
            point_distances = np.append(point_distances, loc_dif)

        picked_left_points = np.array(picked_left_points)
        picked_right_points = np.array(picked_right_points)

        median = calculate_central_value(
            point_distances, central_type="median", outlier_threshold=1
        )
        mean = calculate_central_value(
            point_distances, central_type="mean", outlier_threshold=1
        )

        threshold = 5
        point_distances_median_idx = np.where(
            (point_distances >= median - threshold)
            & (point_distances <= median + threshold)
        )
        point_distances_mean_idx = np.where(
            (point_distances >= mean - threshold)
            & (point_distances <= mean + threshold)
        )

        print(f"\n[!] Median euclidean distance in pixels/voxels: {median}")
        print(f"[!] Mean euclidean distance in pixels/voxels: {mean}")

        if self.args["visualize"]:
            util.visualize_misalignment_images(
                image_1_data,
                image_2_data,
                [bounds_1, bounds_2],
                picked_left_points[point_distances_median_idx],
                picked_right_points[point_distances_median_idx],
                transform,
                f"Misalignment metric - Error {median}",
            )

            util.visualize_misalignment_images(
                image_1_data,
                image_2_data,
                [bounds_1, bounds_2],
                picked_left_points[point_distances_mean_idx],
                picked_right_points[point_distances_mean_idx],
                transform,
                f"Misalignment metric - Error {mean}",
            )

        return mean, median


def get_default_config(filename: PathLike = None):
    """
    Gets the default configuration for the package.

    Parameters
    ------------------------
    filename: str
        command name to check the installation. Default: 'terastitcher'

    Returns
    ------------------------
    bool:
        True if the command was correctly executed, False otherwise.

    """

    if filename is None:
        filename = Path(os.path.dirname(__file__)).joinpath(
            "default_config.yaml"
        )

    config = None
    try:
        with open(filename, "r") as stream:
            config = yaml.safe_load(stream)
    except Exception as error:
        raise error

    return config


def main():
    """
    Main function to test the evaluation performance
    """
    # Get same configuration from yaml file to apply it over a dataset
    default_config = get_default_config()

    BASE_PATH = "/Users/camilo.laiton/Documents/images/"

    default_config["image_1"] = (
        BASE_PATH + "Ex_488_Em_525_468770_468770_830620_012820.zarr"
    )
    default_config["image_2"] = (
        BASE_PATH + "Ex_488_Em_525_501170_501170_830620_012820.zarr"
    )

    # default_config["image_1"] = BASE_PATH + "block_10.tif"
    # default_config["image_2"] = BASE_PATH + "block_10.tif"

    import time

    mod = EvalStitching(default_config)

    time_start = time.time()
    mod.run()
    time_end = time.time()
    print(f"Time: {time_end-time_start}")


if __name__ == "__main__":
    main()
