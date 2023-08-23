"""
XML parser for stitching transformation matrices
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Union

import xmltodict

# IO types
PathLike = Union[str, Path]


def save_dict_as_json(
    filename: str,
    dictionary: dict,
) -> None:
    """
    Saves a dictionary as a json file.

    Parameters
    ------------------------
    filename: str
        Name of the json file.
    dictionary: dict
        Dictionary that will be saved as json.
    verbose: Optional[bool]
        True if you want to print the path where the file was saved.

    """

    if dictionary is None:
        dictionary = {}

    with open(filename, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)


class TeraStitcherXMLParser:
    """
    Class to parse from XML to JSON format
    """

    def __init__(self) -> None:
        """
        Class constructor
        """

        # terastitcher reference order
        self.terastitcher_reference_order = {
            1: ["Y", "V"],
            2: ["X", "H"],
            3: ["D", "Z"],
        }

        # Terastitcher string reference relation
        self.terastitcher_str_reference = {
            "H": ["X", 1],
            "V": ["Y", 2],
            "D": ["Z", 3],
        }

    @staticmethod
    def parse_xml(xml_path: PathLike, encoding: str = "utf-8") -> dict:
        """
        Static method to parse XML to dictionary

        Parameters
        --------------
        xml_path: PathLike
            Path where the XML is stored

        encoding: str
            XML encoding system. Default: utf-8

        Returns
        --------------
        Dict
            Dictionary with the parsed XML
        """
        with open(xml_path, "r", encoding=encoding) as xml_reader:
            xml_file = xml_reader.read()

        xml_dict = xmltodict.parse(xml_file)

        return xml_dict

    def __map_terastitcher_volume_info(self, teras_dict: dict) -> Tuple[Dict]:
        """
        Maps the terastitcher image volume information

        Parameters
        --------------
        teras_dict: dict
            Dictionary with the information extracted
            from the XML

        Returns
        --------------
        dict
            Dictionary with the volume information
        """
        # Mapping reference system
        refs = {}
        for ref, val in teras_dict["TeraStitcher"]["ref_sys"].items():
            val = int(val)
            negative = ""
            if val < 0:
                negative = "-"

            refs[
                ref.replace("@", "")
            ] = f"{negative}{self.terastitcher_reference_order[abs(val)][0]}"

        # Mapping voxel dims
        voxels = {}
        voxels["unit"] = "microns"
        for ref, val in teras_dict["TeraStitcher"]["voxel_dims"].items():
            ref = ref.replace("@", "")
            voxels[self.terastitcher_str_reference[ref][0]] = float(val)

        # Mapping volume origin for computations
        origin = {}
        origin["unit"] = "milimeters"
        for ref, val in teras_dict["TeraStitcher"]["origin"].items():
            ref = ref.replace("@", "")
            origin[self.terastitcher_str_reference[ref][0]] = float(val)

        return refs, voxels, origin

    def __map_terastitcher_stacks_displacements(
        self, teras_dict: dict
    ) -> dict:
        """
        Map the terastitcher stacks displacements

        Parameters
        --------------
        teras_dict: dict
            Dictionary with the information extracted
            from the XML

        Returns
        --------------
        dict
            Dictionary with the displacements per
            stacks of tiles
        """

        def map_displ(displacement: dict) -> dict:
            """
            Helper function to map displacements in each
            direction

            Parameters
            --------------
            displacement: dict
                Dictionary with the displacements in each
                direction. (e.g. NORTH, SOUTH, WEST, EAST)

            Returns
            --------------
            dict
                Dictionary with the parsed displacements
                per stack of tiles
            """
            data = {}

            if displacement is not None:
                for key, val in displacement["Displacement"].items():
                    if key in ["V", "H", "D"]:
                        axis = self.terastitcher_str_reference[key][0]
                        data[axis] = val["@displ"]

            return data

        stack_displacements = teras_dict["TeraStitcher"]["STACKS"]["Stack"]
        stacks = {}

        for idx in range(len(stack_displacements)):
            # print(stack_displacements[idx])
            if stack_displacements[idx]["@STITCHABLE"] == "yes":
                dir_name = stack_displacements[idx]["@DIR_NAME"]
                stacks[dir_name] = {
                    "north": map_displ(
                        stack_displacements[idx]["NORTH_displacements"]
                    ),
                    "east": map_displ(
                        stack_displacements[idx]["EAST_displacements"]
                    ),
                    "south": map_displ(
                        stack_displacements[idx]["SOUTH_displacements"]
                    ),
                    "west": map_displ(
                        stack_displacements[idx]["WEST_displacements"]
                    ),
                }

        return stacks

    def parse_terastitcher_xml(
        self, xml_path: PathLike, encoding: str = "utf-8"
    ) -> dict:
        """
        Parses the terastitcher XML file

        Parameters
        --------------
        teras_dict: dict
            Dictionary with the information extracted
            from the XML

        Returns
        --------------
        Dictionary with the parsed terastitcher xml
        """
        teras_dict = TeraStitcherXMLParser.parse_xml(xml_path, encoding)

        stitch_dict = {}

        stitch_dict["dataset_path"] = teras_dict["TeraStitcher"]["stacks_dir"][
            "@value"
        ]
        refs, voxels, origin = self.__map_terastitcher_volume_info(teras_dict)

        # Getting important info to identify dataset
        stitch_dict["reference_axis"] = refs
        stitch_dict["voxels_size_"] = voxels
        stitch_dict["origin"] = origin

        stacks_displacement = self.__map_terastitcher_stacks_displacements(
            teras_dict
        )
        stitch_dict["stacks_displacements"] = stacks_displacement

        return stitch_dict
