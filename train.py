import pandas as pd
import numpy as np
from rnn import *
from optimizer import *
from preprocess import *

if __name__ == "__main__":
    "0: None, 1:inc, 2:desc"
    train_index = 8000
    frame = pd.read_csv("training/train.csv")
    training_set = frame.values[:, 1:]
    train = training_set[0:train_index, :]
    test = training_set[train_index:, :]
    X = train[:, 1:].reshape((1, train_index, 20))
    Y_index = train[:, 0]
    Y = np.zeros((3, train_index)).astype(np.int32)
    X = preprocess(X)

    for i in range(train_index):
        Y[int(Y_index[i]), i] = 1

    optimizer = GradientDescentOptimizer(0.01, 1, 10, 3, 20, train_index)
    optimizer.fit(X,Y, train_index, 100, True)

    
    
    
