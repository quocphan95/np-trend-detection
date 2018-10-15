import numpy as np
import pandas as pd


def generate_array(array_len, trend = None):
    """
    Generate an array with a given trend

    Parameters:
    array_len: lengh of the generated array
    trend: trend of the array ("inc", "dec", None)
    Return:
    An array with the given trend
    """
    assert trend is None or trend == "inc" or trend == "dec", "Unrecognized trend"
    array = []
    begin = 0
    end = array_len
    step = 1
    elevation = 5
    scale = 20
    label = 1

    if trend is None:
        elevation = 0
        label = 0
    elif trend == "dec":
        step = -1
        begin, end = (end, begin)
        label = 2
        
    for i in range(begin, end, step):
        value = np.random.rand() * scale + i * elevation
        array = array + [value]
    array = [label] + array
    return array

if __name__ == "__main__":
    # Generate increasing array
    inc_array = []
    for i in range(0, 3000):
        inc_array = inc_array + [generate_array(20, "inc")]

    # Generate decreasing array
    dec_array = []
    for i in range(0, 3000):
        dec_array = dec_array + [generate_array(20, "dec")]

    # Generate untrend array
    unt_array = []
    for i in range(0, 3000):
        unt_array = unt_array + [generate_array(20)]

    array = inc_array + dec_array + unt_array
    array = np.asarray(array)
    frame = pd.DataFrame(data=array, index=None)
    frame = frame.sample(frac=1).reset_index(drop=True)
    frame.to_csv("train.csv")

    
    
