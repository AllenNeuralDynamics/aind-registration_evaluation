import os
import sys

import marshmallow as mm
from argschema import ArgSchema
from argschema.fields import InputFile, Int, Nested, Str
from argschema.schemas import DefaultSchema
from marshmallow import fields


class InputImage(fields.Str):
    """
    InputImage is :class:`marshmallow.fields.Str` subclass which is a path to a
    a file location that exists and that the user can access
    (presently checked with os.access)
    """

    def _validate(self, value: str):
        """
        Method to validate if the input image actually exists
        """
        # If it is a folder (e.g., .zarr) or a file (e.g., .png)
        check = os.path.isdir(value) or os.path.isfile(value)

        if not check:
            raise mm.ValidationError("%s is not a directory")

        if sys.platform == "win32":
            try:
                list(os.scandir(value))
            except PermissionError:
                raise mm.ValidationError(
                    "%s is not a readable directory" % value
                )
        else:
            if not os.access(value, os.R_OK):
                raise mm.ValidationError(
                    "%s is not a readable directory" % value
                )


class SamplingArgsSchema(DefaultSchema):
    """
    Nested schema for sampling args.
    """

    sampling_type = Str(
        required=True,
        metadata={"description": "Type of sampling"},
        dump_default="random",
    )

    numpoints = Int(
        required=False,
        metadata={"description": "Number of points to sample"},
        dump_default=200,
    )


class EvalRegSchema(ArgSchema):
    """
    Schema format for Evaluate Stitching.
    """

    image_1 = InputImage(
        required=True,
        metadata={"description": "Path to the file where the data is located"},
    )

    image_2 = InputImage(
        required=True,
        metadata={"description": "Path to the file where the data is located"},
    )

    transform = InputFile(
        required=False,  # True,
        metadata={
            "description": "Json with transformation relating images 1 and 2"
        },
    )

    data_type = Str(
        required=True,
        metadata={
            "description": """
            Type of data: dummy (dummy_2D, dummy_3D),
            small (Read into memory),
            large (not loaded in memory)
            """
        },
        dump_default="small",
    )

    metric = Str(
        required=True,
        metadata={"description": "SSD / NCC"},
        dump_default="large",
    )

    window_size = Int(
        required=False,
        metadata={
            "description": "Size of window across which to calculate metric"
        },
        dump_default=2,
    )

    image_channel = Int(
        required=False,
        metadata={
            "description": """Integer that indicates the
            channel that will be processed"""
        },
        dump_default=0,
    )

    sampling_info = Nested(
        SamplingArgsSchema,
        required=False,
        description="schema for sampling points",
    )
