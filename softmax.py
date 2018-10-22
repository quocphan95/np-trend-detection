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
    return z / np.sum(z, axis=1, Keepdims=True)

def derivative(p):
    """
    Calculate the derivative of the softmaxt function w.r.t x

    Parameters:
    p: Output of the softmax function (n_x, m)
    Return:
    Dsoftmax/Dx shape = (n_x, m)
    """
    return p * (1 - p)
