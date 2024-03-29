"""
Evaluation example with 2D and 3D images
"""

from aind_registration_evaluation.main_qa import (EvalStitching,
                                                  get_default_config)


def main():
    """
    Main function to test the evaluation performance
    """
    # Get same configuration from yaml file to apply it over a dataset

    default_config = get_default_config()

    # print(default_config)

    BASE_PATH = "/Users/camilo.laiton/Documents/images/"

    # default_config["image_1"] = (
    #     BASE_PATH + "Ex_488_Em_525_468770_468770_830620_012820.zarr"
    # )
    # default_config["image_2"] = (
    #     BASE_PATH + "Ex_488_Em_525_501170_501170_830620_012820.zarr"
    # )

    # default_config["image_1"] = (
    #     BASE_PATH + "Ex_445_Em_469_440050_440050_479500_012120.png"
    # )
    # default_config["image_2"] = (
    #     BASE_PATH + "Ex_561_Em_593_440050_440050_479500_012120.png"
    # )

    # default_config["transform_matrix"] = [
    #     [1, 0, 0],  # Y -17
    #     [0, 1, 0],  # X 1800
    #     [0, 0, 1],
    # ]

    # BASE_PATH = "/Users/camilo.laiton/Documents/Stitching datasets/SmartSPIM_AK030_sample/"

    default_config["image_1"] = BASE_PATH + "block_10.tif"
    default_config["image_2"] = BASE_PATH + "block_10.tif"

    # default_config["image_1"] = os.path.abspath(
    #     BASE_PATH + "test_black_3d_image.tiff"
    # )
    # default_config["image_2"] = os.path.abspath(
    #     BASE_PATH + "test_black_3d_image.tiff"
    # )

    default_config["transform_matrix"] = [
        [1, 0, 0, 0],  # Z
        [0, 1, 0, 0],  # Y
        [0, 0, 1, 0],  # X
        [0, 0, 0, 1],
    ]

    default_config["visualize"] = True

    print(default_config)

    import time

    mod = EvalStitching(default_config)

    time_start = time.time()
    mod.run()
    time_end = time.time()
    print(f"Time: {time_end-time_start}")


if __name__ == "__main__":
    main()
