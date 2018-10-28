import pandas as pd
import numpy as np
import pickle as pk
from sklearn import metrics
from rnn import *
from optimizer import *
from preprocess import *

def print_summary(y, y_predict):
    target_names = ["No Trend", "Increase", "Decrease"]
    print("==========")
    print(metrics.classification_report(y.astype(np.int32), y_predict, target_names=target_names))

if __name__ == "__main__":
    "0: None, 1:inc, 2:desc"
    m = 9000
    train_index = 8000
    frame = pd.read_csv("training/train.csv")
    training_set = frame.values[:, 1:]
    train = training_set[0:train_index, :]
    test = training_set[train_index:, :]
    
    X = train[:, 1:].reshape((1, train_index, 20))
    Y_index = train[:, 0]
    Y = np.zeros((3, train_index)).astype(np.int32)
    X = preprocess(X)
    X_test = test[:, 1:].reshape((1, m-train_index, 20))
    Y_index_test = test[:, 0]
    X_test = preprocess(X_test)

    for i in range(train_index):
        Y[int(Y_index[i]), i] = 1

    optimizer = GradientDescentOptimizer(0.01, 1, 10, 3, 20, train_index)
    parameters = optimizer.fit(X,Y, train_index, 1000, True)
    with open("parameters/parameters.pkl", "wb") as file:
        pk.dump(parameters, file)
    Y_predicts = predict(X_test, parameters)
    print_summary(Y_index_test, Y_predicts)    
    
    
