""" Evaluate stitching of large scale data.
"""
from argschema import ArgSchemaParser
from params import EvalRegSchema
from typing import Union
import random
import numpy as np
from metrics import ImageMetricsFactory
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
        
        image_1_data = None
        image_2_data = None
        
        #read data/pointers and linear transform 
        image_1, image_2, transform = io_utils.get_data(
            path_image_1=self.args['image_1'], 
            path_image_2=self.args['image_2'],
            data_type=self.args['data_type']
        )
        
        if self.args['data_type'] == 'large':
            # Load dask array
            image_1_data = utils.extract_data(image_1.as_dask_array())
            image_2_data = utils.extract_data(image_2.as_dask_array())
        
        elif self.args['data_type'] == 'small':
            image_1_data = utils.extract_data(image_1.as_numpy_array())
            image_2_data = utils.extract_data(image_2.as_numpy_array())
        
        elif 'dummy' in self.args['data_type']:
            image_1_data = image_1
            image_2_data = image_2
             
        # print(type(image_1_data), image_1_data.shape)
        
        # exit()
        image_1_shape = image_1_data.shape
        image_2_shape = image_2_data.shape
        
        print("Got data: ", image_1_shape, image_2_shape, transform)

        #calculate extent of overlap using transforms in common coordinate system (assume for image 1)
        bounds_1, bounds_2 = utils.calculate_bounds(image_1_shape, image_2_shape, transform)
        print("BOUNDS1: ", bounds_1, "BOUNDS2: ", bounds_2)

        # #Sample points in overlapping bounds
        points = utils.sample_points_in_overlap(
            bounds_1=bounds_1, 
            bounds_2=bounds_2, 
            numpoints=self.args['sampling_info']['numpoints'],
            sample_type=self.args['sampling_info']['sampling_type']
        )
        
        pruned_points = utils.prune_points_to_fit_window(
            image_1_shape, 
            points,
            self.args['window_size']
        )

        print("Number of discarded points: ", points.shape[0] - pruned_points.shape[0])
        
        #calculate metrics
        # metric_per_point_old = []
        metric_per_point = []
        
        for pruned_point in pruned_points:
            
            met = metrics.new_calculate_metrics(
                point=points[0],
                image_1=image_1_data,
                image_2=image_2_data,
                transform=transform,
                window_size=self.args['window_size'],
                metric=self.args['metric']
            )
            # met_old = metrics.calculate_metrics(pruned_point, [image_1, image_2, transform],self.args)
            
            metric_per_point.append(met)
            
            # if met_old is not None:
            #     metric_per_point_old.append(met_old)
            
        print(None in metric_per_point)
        # #compute statistics
        # print("Mean : ", np.mean(metric_per_point_old), " ,std: ", np.std(metric_per_point_old), "number of points: ", len(metric_per_point_old))
        print("Mean : ", np.mean(metric_per_point), " ,std: ", np.std(metric_per_point), "number of points: ", len(metric_per_point))
        
        
    def run_2(self):
        """
        Args:
            Evaluate block
        """
        
        image_1_data = None
        image_2_data = None
        
        #read data/pointers and linear transform 
        image_1, image_2, transform = io_utils.get_data(
            path_image_1=self.args['image_1'], 
            path_image_2=self.args['image_2'],
            data_type=self.args['data_type']
        )
        
        if self.args['data_type'] == 'large':
            # Load dask array
            image_1_data = utils.extract_data(image_1.as_dask_array())
            image_2_data = utils.extract_data(image_2.as_dask_array())
            
            chunk_sizes = {idx:'auto' for idx in range(len(image_1_data.shape))}
            
            image_1_data = image_1_data.rechunk(chunk_sizes)
            image_2_data = image_2_data.rechunk(chunk_sizes)
        
        elif self.args['data_type'] == 'small':
            image_1_data = utils.extract_data(image_1.as_numpy_array())
            image_2_data = utils.extract_data(image_2.as_numpy_array())
        
        elif 'dummy' in self.args['data_type']:
            image_1_data = image_1
            image_2_data = image_2
             
        # print(type(image_1_data), image_1_data.shape)
        
        # exit()
        image_1_shape = image_1_data.shape
        image_2_shape = image_2_data.shape
        
        # print("Got data: ", image_1_shape, image_2_shape, transform)

        #calculate extent of overlap using transforms in common coordinate system (assume for image 1)
        bounds_1, bounds_2 = utils.calculate_bounds(image_1_shape, image_2_shape, transform)
        # print("BOUNDS1: ", bounds_1, "BOUNDS2: ", bounds_2)

        # #Sample points in overlapping bounds
        points = utils.sample_points_in_overlap(
            bounds_1=bounds_1, 
            bounds_2=bounds_2, 
            numpoints=self.args['sampling_info']['numpoints'],
            sample_type=self.args['sampling_info']['sampling_type']
        )
        
        pruned_points = utils.prune_points_to_fit_window(
            image_1_shape, 
            points,
            self.args['window_size']
        )

        print("Number of discarded points: ", points.shape[0] - pruned_points.shape[0])
        
        # calculate metrics per images
        metric_per_point = []
        
        metric_calculator_dask = ImageMetricsFactory().create(
            image_1_data, 
            image_2_data, 
            self.args['metric']
        )
            
        for pruned_point in pruned_points:
            
            met = metric_calculator_dask.calculate_metrics(
                point=pruned_point,
                transform=transform,
                window_size=self.args['window_size']
            )
            
            if met:
                metric_per_point.append(met)
            
        # compute statistics
        print("Mean : ", np.mean(metric_per_point), " ,std: ", np.std(metric_per_point), "number of points: ", len(metric_per_point))

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
    
    import time
    
    mod = EvalStitching(default_config)
    
    dask_start = time.time()
    mod.run_2()
    dask_end = time.time()
    print(f"Time: {dask_end-dask_start}")
    

if __name__ == '__main__':
    main()