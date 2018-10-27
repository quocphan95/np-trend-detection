import pandas as pd
import numpy as np
from rnn import *
from optimizer import *

if __name__ == "__main__":
    "0: None, 1:inc, 2:desc"
    frame = pd.read_csv("training/train.csv")
    training_set = frame.values[:, 1:]
    train = training_set[0:8000, :]
    test = training_set[8000:, :]
    X = train[:, 1:].reshape((1, 8000, 20))
    Y_index = train[:, 0]
    Y = np.zeros((3, 8000)).astype(np.int32)

    for i in range(8000):
        Y[int(Y_index[i]), i] = 1

    optimizer = GradientDescentOptimizer(0.01, 1, 10, 3, 20, 8000)
    optimizer.fit(X,Y, 8000, 10, True)

    
    
    
