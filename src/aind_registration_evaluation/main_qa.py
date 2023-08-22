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


def generate_key_features_per_img2d(
    img_2d, n_keypoints, pad_width, mode="energy"
):
    if mode == "energy":
        img_2d_keypoints_energy, img_response = kd_fft_energy_keypoints(
            image=img_2d,
            pad_width=pad_width,
            n_keypoints=n_keypoints,
        )
    else:
        # maximum
        img_2d_keypoints_energy, img_response = kd_fft_keypoints(
            image=img_2d,
            pad_width=pad_width,
            n_keypoints=n_keypoints,
        )

    dy_val, dx_val = derivate_image_axis(
        gaussian_filter(img_2d, sigma=8), [0, 1]
    )

    # img_2d_dy = np.zeros(img_2d.shape, dtype=img_2d.dtype)
    # img_2d_dx = np.zeros(img_2d.shape, dtype=img_2d.dtype)

    # dy_val = np.sqrt(dy_val)
    # dx_val = np.sqrt(dx_val)

    # f, ax = plt.subplots(1,2)

    # ax[0].imshow(dy_val, cmap="gray")#vmin=0, vmax=0.2)
    # ax[1].imshow(dx_val, cmap="gray") #vmin=0, vmax=0.2)
    # plt.show()

    # img_2d_dy[:-1, :] = np.float32(dy_val)
    # img_2d_dx[:, :-1] = np.float32(dx_val)

    (
        gradient_magnitude,
        gradient_orientation,
        gradient_orientation_polar,
    ) = kd_gradient_magnitudes_and_orientations(
        derivated_images=[dy_val, dx_val]  # [img_2d_dy, img_2d_dx]
    )

    img_keypoints_features = [
        kd_compute_keypoints_hog(
            image_gradient_magnitude=gradient_magnitude,
            image_gradient_orientation=[gradient_orientation],
            keypoint=keypoint,
            n_dims=2,
            window_size=16,
            bins=[8],
        )
        for keypoint in img_2d_keypoints_energy
    ]

    keypoints = []
    features = []
    for key_feat in img_keypoints_features:
        # print(f"Keypoint {key_feat['keypoint']} feat shape: {key_feat['feature_vector'].shape}")
        keypoints.append(key_feat["keypoint"])
        features.append(key_feat["feature_vector"])

    return {
        "keypoints": np.array(keypoints),
        "features": np.array(features),
        "response_img": img_response,
    }


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

        # calculate extent of overlap using transforms
        # in common coordinate system (assume for image 1)
        bounds_1, bounds_2 = util.calculate_bounds(
            image_1_shape, image_2_shape, transform
        )

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

    def run_misalignment(self):
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

        # calculate extent of overlap using transforms
        # in common coordinate system (assume for image 1)
        bounds_1, bounds_2 = util.calculate_bounds(
            image_1_shape, image_2_shape, transform
        )

        # Compute keypoints between images
        n_keypoints = 200
        pad_width = 40
        img_1_dict = generate_key_features_per_img2d(
            image_1_data, n_keypoints=n_keypoints, pad_width=pad_width
        )
        img_2_dict = generate_key_features_per_img2d(
            image_2_data, n_keypoints=n_keypoints, pad_width=pad_width
        )

        feature_vector_img_1 = (
            img_1_dict["features"],
            img_1_dict["keypoints"],
        )
        feature_vector_img_2 = (
            img_2_dict["features"],
            img_2_dict["keypoints"],
        )

        distances = compute_feature_space_distances(
            feature_vector_img_1, feature_vector_img_2, feature_weight=0.2
        )

        point_matches_pruned = get_pairs_from_distances(
            distances=distances, delete_points=True, metric_threshold=0.1
        )

        # Tomorrow map points to the same
        # coordinate system
        # Working only with translation at the moment
        offset_ty = transform[0, -1]
        offset_tx = transform[1, -1]

        left_image_keypoints = img_1_dict["keypoints"]
        right_image_keypoints = img_2_dict["keypoints"]

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

        median = np.median(point_distances)
        mean = np.mean(point_distances)
        print(
            f"\n[!] Median euclidean distance in pixels/voxels: {median}"
        )
        print(
            f"[!] Mean euclidean distance in pixels/voxels: {mean}"
        )

        if self.args["visualize"]:
            util.visualize_misalignment_images(
                image_1_data,
                image_2_data,
                [bounds_1, bounds_2],
                picked_left_points,
                picked_right_points,
                transform,
                f"Misalignment metric ch 445 - ch 561 - Error {median}",
            )


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
