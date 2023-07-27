import os

from xml_parser import TeraStitcherXMLParser

from eval_reg.eval_stitching import EvalStitching, get_default_config


def main():
    """
    Main function to test the evaluation performance
    """
    # Get same configuration from yaml file to apply it over a dataset

    # flake8: noqa: E501
    # merge_xml_ak030 = "/Users/camilo.laiton/Documents/Stitching datasets/SmartSPIM_AK030_sample/xml_merging_Ex_488_Em_525.xml"

    # parsed_xml = TeraStitcherXMLParser().parse_xml(xml_path=merge_xml_ak030)
    # parsed_xml = TeraStitcherXMLParser().parse_terastitcher_xml(
    #     teras_dict=parsed_xml
    # )

    # print(parsed_xml)

    default_config = get_default_config()

    # print(default_config)

    BASE_PATH = "/Users/camilo.laiton/Documents/images/"

    default_config["image_1"] = (
        BASE_PATH + "Ex_488_Em_525_468770_468770_830620_012820.zarr"
    )
    default_config["image_2"] = (
        BASE_PATH + "Ex_488_Em_525_501170_501170_830620_012820.zarr"
    )

    default_config["transform_matrix"] = [
        [1, 0, -100],  # Y
        [0, 1, 1900],  # X
        [0, 0, 1],
    ]

    # BASE_PATH = "/Users/camilo.laiton/Documents/Stitching datasets/SmartSPIM_AK030_sample/"

    # default_config["image_1"] = os.path.abspath(BASE_PATH + "block_10.tif")
    # default_config["image_2"] = os.path.abspath(BASE_PATH + "block_10.tif")
    # default_config["transform_matrix"] = [
    #     [1, 0, 0, 100],  # Z
    #     [0, 1, 0, 100],  # Y
    #     [0, 0, 1, 0],  # X
    #     [0, 0, 0, 1],
    # ]

    print(default_config)

    import time

    mod = EvalStitching(default_config)

    time_start = time.time()
    mod.run()
    time_end = time.time()
    print(f"Time: {time_end-time_start}")


if __name__ == "__main__":
    main()
