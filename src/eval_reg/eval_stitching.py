""" Evaluate stitching of large scale data.
"""
from argschema import ArgSchemaParser
from params import EvalRegSchema
from typing import Union
import random
import numpy as np
import metrics
import io_utils
import utils
import yaml
from pathlib import Path
import os

# IO types 
PathLike = Union[str, Path]
    
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
        
        #read data/pointers and linear transform 
        image_1, image_2, transform = io_utils.get_data(
            path_image_1=self.args['image_1'], 
            path_image_2=self.args['image_2'],
            data_type=self.args['data_type']
        )
        
        print("Got data: ", image_1.shape, image_2.shape, transform)

        #calculate extent of overlap using transforms in common coordinate system (assume for image 1)
        bounds_1, bounds_2 = utils.calculate_bounds(image_1, image_2, transform)
        print("BOUNDS1: ", bounds_1, "BOUNDS2: ", bounds_2)

        # #Sample points in overlapping bounds
        points = utils.sample_points_in_overlap(
            bounds_1=bounds_1, 
            bounds_2=bounds_2, 
            numpoints=self.args['sampling_info']['numpoints'],
            sample_type=self.args['sampling_info']['sampling_type']
        )
        
        pruned_points = utils.prune_points_to_fit_window(
            image_1.shape, 
            points,
            self.args['window_size']
        )

        print("Number of discarded points: ", points.shape[0] - pruned_points.shape[0])
        
        # #calculate metrics
        # M = []
        # for pt in pruned_pts:
        #     met = metrics.calculate_metrics(pt, data,self.args)
        #     if met is not None:
        #         M.append(met)
            
        
        # #compute statistics
        # print("Mean : ", np.mean(M), " ,std: ", np.std(M), "number of points: ", len(M))

def get_default_config(filename:PathLike=None):
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
    
    if filename == None:
        filename = Path(os.path.dirname(__file__)).joinpath('default_config.yaml')
    
    config = None
    try:
        with open(filename, "r") as stream:
            config = yaml.safe_load(stream)
    except Exception as error:
        raise error
    
    return config

def main():
    # Get same configuration from yaml file to apply it over a dataset
    default_config = get_default_config()
    
    default_config['image_1'] = 'C:/Users/camilo.laiton/Documents/images/Ex_488_Em_525_468770_468770_830620_012820.zarr'
    default_config['image_2'] = 'C:/Users/camilo.laiton/Documents/images/Ex_488_Em_525_468770_468770_830620_012820.zarr'
    
    mod = EvalStitching(default_config)
    mod.run()

if __name__ == '__main__':
    main()