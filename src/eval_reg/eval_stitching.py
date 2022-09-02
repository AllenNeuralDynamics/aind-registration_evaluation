""" Evaluate stitching of large scale data.
"""

from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import Str , Int, Nested
import random
import numpy as np
import metrics
import iotools

example_input = {
    "image1": "test1/",
    "image2": "test2/",
    "transform": "transform.json",
    "datatype": "dummy",
    "metric": "SSD",
    "windowsize": 0,
    "samplinginfo": { "type": "random", "numpoints": 1}
}
class SamplingArgsSchema(ArgSchema):
    type = Str(metadata={"required":False, "description":"Type of "})
    numpoints = Int(metadata={"required":False, "description":"Number of points to sample"})
class EvalRegSchema(ArgSchema):
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

    def sample_points_in_overlap(self,bounds1, bounds2, args ):
        """samples points in the overlap regions 
           and returns a list of points
           sampling types : random, grid, feature extracted
        """
        
        #check if there is intersection
        #######NEED TO DO!!!!

        #if there is, then : 
        if args['type'] == "random":
            o_min = [np.max([bounds1[0][0], bounds2[0][0]]),
                        np.max([bounds1[0][1], bounds2[0][1]]),
                            np.max([bounds1[0][2], bounds2[0][2]])]

            o_max = [np.min([bounds1[1][0], bounds2[1][0]]),
                        np.min([bounds1[1][1], bounds2[1][1]]),
                            np.min([bounds1[1][2], bounds2[1][2]])]

            print(range(o_min[0], o_max[0]))
            x = list(np.random.choice( range(o_min[0], o_max[0]), args['numpoints'] ))
            y = list(np.random.choice(range(o_min[1], o_max[1]), args['numpoints']))
            z = list(np.random.choice(range(o_min[2], o_max[2]), args['numpoints']))

        return list(np.array([x,y,z]).transpose())

    

    def calculate_bounds(self,image_shape, transform):
        pt_min = np.matrix([0,0,0,1]).transpose()
        pt_max = np.matrix(list(image_shape)+[1]).transpose()
        ret = [np.squeeze(transform*pt_min).tolist()[0][:3],
               np.squeeze(transform*pt_max).tolist()[0][:3]]  
             
        return ret 

    def run(self):
        """
       
        Args:
            Evaluate block
        
        """

        #read data/pointers and linear transform 
        I1, I2, transform = iotools.get_data(self.args)

        #calculate extent of overlap using transforms in common coordinate system (assume for image 1)
        bounds1 = [[0,0,0],list(I1.shape)]
        bounds2 = self.calculate_bounds(I2.shape,transform)

        #Sample points in overlapping bounds
        pts = self.sample_points_in_overlap(bounds1, bounds2,self.args['samplinginfo'])
        
        #calculate metrics
        M = []
        for pt in pts:
            M.append(metrics.calculate_metrics(pt, I1, I2, transform,self.args))
            
        #compute statistics
        print("Mean : ", np.mean(M), " ,std: ", np.std(M))

        
if __name__ == '__main__':
    mod = EvalStitching(example_input)
    mod.run()