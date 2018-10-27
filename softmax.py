import numpy as np

"""
Unstable version
"""

def calc(x):
    """
    Caclculate the sofmax of an numpy array

    Parameters:
    x: Input value of shape (n_x, m)
    Return:
    softmaxt values of shape (n_x, m)
    """
    z = np.exp(x)
    ret = z / np.sum(z, axis=0, keepdims=True)
    return ret
