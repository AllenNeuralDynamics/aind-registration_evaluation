""" Evaluate stitching of large scale data.
"""

from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import Str
import random
import numpy as np
import metrics
import scipy

example_input = {
    "image1": "test1/",
    #"image2": "test2/",
    #"transformjson": "transform.json"
}
class EvalRegSchema(ArgSchema):
    image1 = Str(metadata={"required":True, "description":"Image 1 location"})
    image2 = Str(metadata={"required":True, "description":"Image 2 location"})
    transformjson = Str(metadata={"required":True, "description":"json with transformation relating Images 1 and 2"})
    


class EvalStitching (ArgSchemaParser):
    """
    Class to Evaluate Stitching.
    """

    default_schema = EvalRegSchema

    def sample_points_in_overlap(self,bounds1, bounds2, args={"numpoints":20, "type": "random"} ):
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

    def calculate_metrics(self,pt1, I1, I2, transform, windowsize = 0):
        """Given a pair of points, the images and the transform, 
           Calculate metrics for window size. If windowsize = 0, calculate just for one point"""

        
        pt2 = (np.linalg.inv(transform)*np.matrix(list(pt1)+[1]).transpose())[:3]
        print(pt2)
        X = np.linspace(0, I2.shape[0], I2.shape[0])  
        Y = np.linspace(0, I2.shape[1], I2.shape[1])  
        Z = np.linspace(0, I2.shape[2], I2.shape[2])   
        
        if windowsize == 0:
            Patch1 = I1[pt1[0],pt1[1],pt1[2]]
            Patch2check = I2[int(pt2[0]),int(pt2[1]),int(pt2[2])]
            Patch2 = scipy.interpolate.interpn((X,Y,Z), I2, [pt2])

            
            
        #return metrics.mean_squared_error(Patch1,Patch2)
        print(Patch1, Patch2, Patch2check)
        

    def calculate_bounds(self,I, transform):
        pt_min = np.matrix([0,0,0,1]).transpose()
        pt_max = np.matrix(list(I.shape)+[1]).transpose()
        ret = [np.squeeze(transform*pt_min).tolist()[0][:3],
               np.squeeze(transform*pt_max).tolist()[0][:3]]  
             
        return ret 

    def run(self):
        """
       
        Args:
            Evaluate block
        
        """
        #get sizes of images and transform
        I1 = np.random.random((10,10,10))
        I2 = I1
        transform = np.matrix([[1,0,0,3],[0,1,0,3],[0,0,1,3],[0,0,0,1]])

        #calculate extent of overlap using transforms in common coordinate system (assume for image 1)
        bounds1 = [[0,0,0],list(I1.shape)]
        bounds2 = self.calculate_bounds(I2,transform)

        #Sample points in overlapping bounds
        pts = self.sample_points_in_overlap(bounds1, bounds2)
        print(pts)
        #calculate metrics
        for pt in pts:
            self.calculate_metrics(pt, I1, I2, transform)
            break

        #compute statistics

        return 1

if __name__ == '__main__':
    mod = EvalStitching(example_input)
    mod.run()