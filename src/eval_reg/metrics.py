""" Modules to calculate Metrics between features
"""

from skimage import metrics
import numpy as np
import scipy
from dask import delayed
import utils
import dask.array as da



def get_patch_from_dask_array(I, mins, maxs):
    '''test function  --- to be updated'''
    X = np.linspace(int(mins[0]), int(maxs[0]),int (maxs[0]-mins[0]) + 1)  
    Y = np.linspace(int(mins[1]), int(maxs[1]), int(maxs[1]-mins[1]) + 1)
    Z = np.linspace(int(mins[2]), int(maxs[2]), int(maxs[2]-mins[2]) + 1)
    points = (X, Y, Z)
    res = np.meshgrid(*points, indexing='ij')
    #print("This is res: ", res[0].shape, res[1].shape, res[2].shape)
    Patch2 = I.vindex[res[0],res[1],res[2]]
    #print("This is patch 2: ", Patch2.shape)
    return Patch2, X-mins[0],Y-mins[1],Z-mins[2]

def get_patches(data, pt1,pt2,X,Y,Z,args):
    """
    Function to return patches to be compared.
    """
    transform = data[2]
    if args['datatype'] != "large":
        I1 = data[0]
        I2 = data[1]
        
        pt1 = pt1.astype(int)
        try:
            X = np.linspace(0, I2.shape[0], I2.shape[0])  
            Y = np.linspace(0, I2.shape[1], I2.shape[1]) 
            Z = np.linspace(0, I2.shape[2], I2.shape[2]) 
            Patch1 = I1[pt1[0], pt1[1], pt1[2]]
            Patch2 = scipy.interpolate.interpn((X,Y,Z), I2, pt2.transpose())
            return Patch1, Patch2
        except:
            return None,None 
    else:
            I1 = np.squeeze(data[0][:,1, :,:,:])
            I2 = np.squeeze(data[1][:,args['channel'], :,:,:])
            transform = data[2]
            print("Pseudocode")
            
            """ Patch1 = I1.vindex[pt1[0], pt1[1], pt1[2]]
            mins = np.floor(np.min(pt2, axis = 1))
            maxs = np.ceil(np.max(pt2, axis = 1))
            
            I2_patch,pX,pY,pZ = get_patch_from_dask_array(I2, mins, maxs)
            pt2_adjusted = pt2-mins
            X = np.linspace(0, I2_patch.shape[0]-1, I2_patch.shape[0])  
            Y = np.linspace(0, I2_patch.shape[1]-1, I2_patch.shape[1])  
            Z = np.linspace(0, I2_patch.shape[2]-1, I2_patch.shape[2])
            
            Patch2 = delayed(scipy.interpolate.interpn)((X,Y,Z), I2_patch, pt2_adjusted.transpose())
            Patch2.compute()
            
            Patch2 = scipy.interpolate.interpn((X,Y,Z), I2_patch, (pt2_adjusted.transpose())
            return Patch1, Patch2 """
        



def calculate_metrics(pt1, data, args):
    """Given a pair of points, the images and the transform, 
        Calculate metrics for window size. If windowsize = 0, calculate just for one point"""
        
    if args['datatype'] == 'large':
        I1 = np.squeeze(data[0][:,args['channel'], :,:,:])
        I2 = np.squeeze(data[1][:,args['channel'], :,:,:])
    else:
        I1 = data[0]
        I2 = data[1]
    transform = data[2]

    X = np.linspace(0, I2.shape[0], I2.shape[0])  
    Y = np.linspace(0, I2.shape[1], I2.shape[1])  
    Z = np.linspace(0, I2.shape[2], I2.shape[2])   
    
    if args['windowsize'] == 0:
        pt2 = (np.linalg.inv(transform)*np.matrix(list(pt1)+[1]).transpose())[:3]
        pt2 = np.squeeze(np.asarray(pt2))
        Patch1,Patch2 = get_patches(I1,I2, np.expand(pt1,axis=0),np.expand_dims(pt2,axis=0),X,Y,Z, args)
        
    else:
        #print("need to work on this")
        pt1_X = np.expand_dims(np.linspace(pt1[0]-args['windowsize'], pt1[0]+args['windowsize'], 2*args['windowsize']+1)  , axis=0)
        pt1_Y = np.expand_dims(np.linspace(pt1[1]-args['windowsize'], pt1[1]+args['windowsize'], 2*args['windowsize']+1)  , axis=0)
        pt1_Z = np.expand_dims(np.linspace(pt1[2]-args['windowsize'], pt1[2]+args['windowsize'], 2*args['windowsize']+1)  , axis=0)
        points = (pt1_X, pt1_Y, pt1_Z)
        X,Y,Z = np.meshgrid(*points, indexing='ij')
        
        pt1_win = np.vstack ([X.flatten(), Y.flatten(), Z.flatten()])
        homogenous_pts = np.matrix(np.vstack([pt1_win, np.ones(X.flatten().shape)]))
        pt2_win = (np.linalg.inv(transform)*homogenous_pts)[:3,:]
        Patch1,Patch2 = get_patches(data, pt1_win,pt2_win,np.squeeze(pt1_X), np.squeeze(pt1_Y), np.squeeze(pt1_Z), args)
        
    if Patch1 is not None:
        met = compute_metric_for_patch(Patch1,Patch2,args['metric'])
        return met
    else:
        return None

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




