""" Evaluate stitching of large scale data.
"""

from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import Str , Int, Nested
import random
import numpy as np
import metrics
import iotools
import utils

example_input = {
    "image1": "test1/",
    "image2": "test2/",
    "transform": "transform.json",
    "datatype": "dummy",
    "metric": "SSD",
    "windowsize": 2,
    "samplinginfo": { "type": "random", "numpoints": 10}
}

class SamplingArgsSchema(ArgSchema):
    """
    Nested schema for sampling args.
    """

    type = Str(metadata={"required":False, "description":"Type of "})
    numpoints = Int(metadata={"required":False, "description":"Number of points to sample"})

class EvalRegSchema(ArgSchema):
    """
    Schema format for Evaluate Stitching.
    """
    image1 = Str(metadata={"required":True, "description":"Image 1 location"})
    image2 = Str(metadata={"required":True, "description":"Image 2 location"})
    transform = Str(metadata={"required":True, "description":"json with transformation relating Images 1 and 2"})
    datatype = Str(metadata={"required":True, "description":"Type of data: Dummy, Small (Read into memory), Large (not loaded in memory)"})
    metric = Str(metadata={"required":True, "description":"SSD / NCC"})
    windowsize = Int(metadata={"required":True, "description":"Size of window across which to calculate metric"})
    samplinginfo = Nested(SamplingArgsSchema,required=False,default={},description='schema for sampling points')

class EvalStitching (ArgSchemaParser):
    """
    Class to Evaluate Stitching.
    """
    default_schema = EvalRegSchema

    def run(self):
        """
        Args:
            Evaluate block
        """

        #read data/pointers and linear transform 
        I1, I2, transform = iotools.get_data(self.args)

        #calculate extent of overlap using transforms in common coordinate system (assume for image 1)
        bounds1,bounds2 = utils.calculate_bounds(I1.shape, I2.shape,transform)

        #Sample points in overlapping bounds
        pts = utils.sample_points_in_overlap(bounds1, bounds2,self.args['samplinginfo'])
        pruned_pts = utils.prune_points_to_fit_window(pts, self.args['windowsize'], I1.shape)
        print(pruned_pts)
        #calculate metrics
        M = []
        for pt in pruned_pts:
            met = metrics.calculate_metrics(pt, I1, I2, transform,self.args)
            if met is not None:
                M.append(met)
            
        #compute statistics
        print("Mean : ", np.mean(M), " ,std: ", np.std(M))

        
if __name__ == '__main__':
    mod = EvalStitching(example_input)
    mod.run()