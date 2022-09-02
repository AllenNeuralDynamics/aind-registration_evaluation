""" Modules to calculate Metrics between features
"""

from skimage import metrics
import numpy as np
import scipy


def get_patches(I1, I2, pt1,pt2,X,Y,Z,datatype):
    """
    Function to return patches to be compared.
    """
    if datatype != "large":
        Patch1 = I1[pt1[0],pt1[1],pt1[2]]
        Patch2 = scipy.interpolate.interpn((X,Y,Z), I2, [pt2])[0]
    return Patch1, Patch2


def calculate_metrics(pt1, I1, I2, transform, args):
    """Given a pair of points, the images and the transform, 
        Calculate metrics for window size. If windowsize = 0, calculate just for one point"""

        
        X = np.linspace(0, I2.shape[0], I2.shape[0])  
        Y = np.linspace(0, I2.shape[1], I2.shape[1])  
        Z = np.linspace(0, I2.shape[2], I2.shape[2])   
        
        if args['windowsize'] == 0:
            pt2 = (np.linalg.inv(transform)*np.matrix(list(pt1)+[1]).transpose())[:3]
            pt2 = np.squeeze(np.asarray(pt2))
            Patch1,Patch2 = get_patches(I1,I2, pt1,pt2,X,Y,Z, args['datatype'])
            
        else:
            print("need to work on this")
            #pt1_X = np.linspace(pt1[0]-args['windowsize'], pt1[0]+args['windowsize'], pt1[0]-args['windowsize'])  
            #pt1_Y = np.linspace(pt1[1]-args['windowsize'], pt1[1]-args['windowsize'], pt1[1]-args['windowsize'])  
            #pt1_Z = np.linspace(pt1[2]-args['windowsize'], pt1[2]-args['windowsize'], pt1[2]-args['windowsize'])
            #print(pt1_X.shape)
            #pt1_win = np.vstack ([pt1_X, pt1_Y, pt1_Z])
            #print(pt1_win)
        met = compute_metric_for_patch(Patch1,Patch2,args['metric'])

        return met

def compute_metric_for_patch(f1,f2,metrictype):
    """
    Function that decides which metric to calculate
    """
    if metrictype == "SSD":
        met = mean_squared(f1,f2)
    return met    

def mean_squared(f1,f2):
    """Calculate mean squared error between two image patches."""
    return metrics.mean_squared_error(f1,f2)




