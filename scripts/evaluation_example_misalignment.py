"""
Evaluation example with 2D and 3D images
"""

from aind_registration_evaluation.main_qa import (EvalStitching,
                                                  get_default_config)


def run_3d_example(base_path: str):
    """
    Runs a 3D image example
    """
    default_config = get_default_config()

    default_config["image_1"] = base_path + "block_10.tif"
    default_config["image_2"] = base_path + "block_10.tif"

    misalignment_parameters = {
        "overlap_ratio": 1,
        "n_keypoints": 200,
        "pad_width": 20,
        "filter_size": 5,
        "gss_sigma": 9,
        "orientation": "x",
        "max_relative_threshold": 0.2,
    }

    default_config["transform_matrix"] = [
        [1, 0, 0, 0],  # Z
        [0, 1, 0, 0],  # Y
        [0, 0, 1, 0],  # X
        [0, 0, 0, 1],
    ]

    return default_config, misalignment_parameters


def run_2d_tile_example(base_path: str):
    """
    Runs a 2D tile image example
    """
    default_config = get_default_config()

    default_config["image_1"] = (
        base_path + "Ex_488_Em_525_468770_468770_830620_012820.zarr"
    )
    default_config["image_2"] = (
        base_path + "Ex_488_Em_525_501170_501170_830620_012820.zarr"
    )

    misalignment_parameters = {
        "overlap_ratio": 0.1,
        "n_keypoints": 200,
        "pad_width": 30,
        "filter_size": 5,
        "gss_sigma": 9,
        "orientation": "x",
        "max_relative_threshold": 0.2,
    }

    default_config["transform_matrix"] = [
        [1, 0, 0],  # Y -17
        [0, 1, 1800],  # X 1800
        [0, 0, 1],
    ]

    return default_config, misalignment_parameters


def run_2d_multichannel_tile_example_png(base_path: str):
    """
    Runs a 2D multichannel tile image example.
    Here the microscope image output had an
    offset due to the configuration.
    """
    default_config = get_default_config()

    default_config["image_1"] = (
        base_path + "Ex_445_Em_469_440050_440050_479500_012120.png"
    )
    default_config["image_2"] = (
        base_path + "Ex_561_Em_593_440050_440050_479500_012120.png"
    )

    misalignment_parameters = {
        "overlap_ratio": 1.0,
        "n_keypoints": 200,
        "pad_width": 30,
        "filter_size": 5,
        "gss_sigma": 9,
        "orientation": "x",
        "max_relative_threshold": 0.2,
    }

    default_config["transform_matrix"] = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]

    return default_config, misalignment_parameters


def run_multichannel_example(base_path: str):
    """
    Runs a 2D full resolution multichannel
    image example
    """
    default_config = get_default_config()

    default_config["image_1"] = base_path + "Ex_445_Em_469_sample.tif"
    default_config["image_2"] = base_path + "Ex_561_Em_593_sample.tif"

    misalignment_parameters = {
        "overlap_ratio": 1.0,
        "n_keypoints": 200,
        "pad_width": 30,
        "filter_size": 5,
        "gss_sigma": 9,
        "orientation": "x",
        "max_relative_threshold": 0.2,
    }

    default_config["transform_matrix"] = [
        [1, 0, 0],
        [0, 1, 20],
        [0, 0, 1],
    ]

    return default_config, misalignment_parameters


def run_individual():
    """
    Main function to test the evaluation performance
    """

    BASE_PATH = "/Users/camilo.laiton/Documents/images/"

    # default_config, misalignment_parameters = run_3d_example(BASE_PATH)
    # default_config, misalignment_parameters = run_2d_tile_example(BASE_PATH)
    # default_config, misalignment_parameters = run_multichannel_example(BASE_PATH)
    (
        default_config,
        misalignment_parameters,
    ) = run_2d_multichannel_tile_example_png(BASE_PATH)

    default_config["visualize"] = True
    misalignment_method = "energy"  # Options = ["energy", "maxima"]

    import time

    mod = EvalStitching(default_config)

    time_start = time.time()
    mod.run_misalignment(
        **misalignment_parameters,
        mode=misalignment_method,
    )
    time_end = time.time()
    print(f"Time: {time_end-time_start}")


def run_all():
    """
    Main function to test the evaluation performance
    in different images
    """
    BASE_PATH = "/Users/camilo.laiton/Documents/images/"

    all_confs = [
        run_3d_example(BASE_PATH),
        run_2d_tile_example(BASE_PATH),
        run_multichannel_example(BASE_PATH),
        run_2d_multichannel_tile_example_png(BASE_PATH),
    ]

    for default_config, misalignment_parameters in all_confs:
        for misalignment_method in ["energy", "maxima"]:
            print(f"Run for misalignment method: {misalignment_method}")
            default_config["visualize"] = False
            import time

            mod = EvalStitching(default_config)

            time_start = time.time()
            mod.run_misalignment(
                **misalignment_parameters,
                mode=misalignment_method,
            )
            time_end = time.time()
            print(f"Time: {time_end-time_start}")


def main():
    """
    Main function
    """
    # run_individual()
    run_all()


if __name__ == "__main__":
    main()
