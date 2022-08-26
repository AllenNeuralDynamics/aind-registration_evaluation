""" Modules to calculate Metrics between features
"""

from skimage import metrics

def mean_squared(f1,f2):
    """Calculate mean squared error between two image patches."""
    return metrics.mean_squared_error(f1,f2)




