from argschema import ArgSchema
from argschema.fields import Int, Str, Nested, InputFile, InputDir
from argschema.schemas import DefaultSchema
from marshmallow import validate

class SamplingArgsSchema(DefaultSchema):
    """
    Nested schema for sampling args.
    """

    sampling_type = Str(
        required=True, 
        metadata={
            'description':"Type of sampling"
        },
        dump_default="random"
    )
    
    numpoints = Int(
        required=False, 
        metadata={
            'description':'Number of points to sample'
        },
        dump_default=200
    )
    
class EvalRegSchema(ArgSchema):
    """
    Schema format for Evaluate Stitching.
    """
    image_1 = InputDir(
        required=True, 
        metadata={
            'description':'Path to the file where the data is located'
        }
    )
    
    image_2 = InputDir(
        required=True, 
        metadata={
            'description':'Path to the file where the data is located'
        }
    )
    
    transform = InputFile(
        required=False,#True, 
        metadata={
            'description':'Json with transformation relating images 1 and 2'
        }
    )
    
    data_type = Str(
        required=True, 
        metadata={
            'description':"Type of data: dummy (dummy_2D, dummy_3D), small (Read into memory), large (not loaded in memory)"
        },
        dump_default="small"
    )
    
    metric = Str(
        required=True, 
        metadata={
            'description':"SSD / NCC"
        },
        dump_default="large"
    )
    
    window_size = Int(
        required=False, 
        metadata={
            'description':'Size of window across which to calculate metric'
        },
        dump_default=2
    )
    
    image_channel = Int(
        required=False, 
        metadata={
            'description':'Integer that indicates the channel that will be processed'
        },
        dump_default=0
    )
    
    sampling_info = Nested(
        SamplingArgsSchema,
        required=False,
        description='schema for sampling points'
    )