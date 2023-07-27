""" Evaluate stitching of large scale data.
"""
import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
import yaml
from argschema import ArgSchemaParser

from . import io_utils, utils
from .metrics import ImageMetricsFactory
from .params import EvalRegSchema

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
        image_1, image_2, transform = io_utils.get_data(
            path_image_1=self.args["image_1"],
            path_image_2=self.args["image_2"],
            data_type=self.args["data_type"],
            transform_matrix=self.args["transform_matrix"],
        )

        if self.args["data_type"] == "large":
            # Load dask array
            image_1_data = utils.extract_data(image_1.as_dask_array())
            image_2_data = utils.extract_data(image_2.as_dask_array())

        elif self.args["data_type"] == "small":
            image_1_data = utils.extract_data(image_1.as_numpy_array())
            image_2_data = utils.extract_data(image_2.as_numpy_array())

        elif "dummy" in self.args["data_type"]:
            image_1_data = image_1
            image_2_data = image_2

        utils.validate_image_transform(
            image_1=image_1_data,
            image_2=image_2_data,
            transform_matrix=transform,
        )

        image_1_shape = image_1_data.shape
        image_2_shape = image_2_data.shape

        # calculate extent of overlap using transforms
        # in common coordinate system (assume for image 1)
        bounds_1, bounds_2 = utils.calculate_bounds(
            image_1_shape, image_2_shape, transform
        )

        # #Sample points in overlapping bounds
        points = utils.sample_points_in_overlap(
            bounds_1=bounds_1,
            bounds_2=bounds_2,
            numpoints=self.args["sampling_info"]["numpoints"],
            sample_type=self.args["sampling_info"]["sampling_type"],
            image_shape=image_1_shape,
        )

        print("N Points: ", len(points))
        pruned_points = utils.prune_points_to_fit_window(
            image_1_shape, points, self.args["window_size"]
        )

        discarded_points_window = points.shape[0] - pruned_points.shape[0]
        LOGGER.info(
            f"""Number of discarded points when prunning
            points to window: {discarded_points_window}""",
        )

        # calculate metrics per images
        metric_per_point = []

        metric_calculator = ImageMetricsFactory().create(
            image_1_data,
            image_2_data,
            self.args["metric"],
            self.args["window_size"],
        )

        selected_pruned_points = []

        for pruned_point in pruned_points:
            met = metric_calculator.calculate_metrics(
                point=pruned_point, transform=transform
            )

            if met:
                selected_pruned_points.append(pruned_point)
                metric_per_point.append(met)

        # compute statistics
        metric = self.args["metric"]
        computed_points = len(metric_per_point)

        dscrd_pts = points.shape[0] - discarded_points_window - computed_points
        message = f"""Computed metric: {metric}
        \nMean: {np.mean(metric_per_point)}
        \nStd: {np.std(metric_per_point)}
        \nNumber of calculated points: {computed_points}
        \nDiscarded points by metric: {dscrd_pts}"""
        LOGGER.info(message)

        # utils.visualize_images(
        #     image_1_data,
        #     image_2_data,
        #     [bounds_1, bounds_2],
        #     pruned_points,
        #     selected_pruned_points,
        #     transform
        # )


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
