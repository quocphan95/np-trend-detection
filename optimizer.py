import numpy as np
from rnn import *

class GradientDescentOptimizer:
    """
    Implement the GD algorithim
    """
    def __init__(self, lr, nx, na, ny, Tx, m):
        """
        Init the optimizer

        Parameters:
        lr: learning rate alpha
        nx: dimmension of input nx
        na: dimmension of hidden states na
        ny: dimmension of output y (is 3 in this problem)
        Tx: the length of the x sequence Tx
        m: size of training set
        Return:
        the optimizer object
        """
        self.lr = lr
        self.nx = nx
        self.na = na
        self.ny = ny
        self.Tx = Tx
        self.m = m
        self.parameters = {
                "Wax" : np.random.randn(self.na, self.nx),
                "Waa" : np.random.randn(self.na, self.na),
                "ba"  : np.zeros((na, 1)),
                "Wya" : np.random.randn(self.ny, self.na),
                "by"  : np.zeros((ny, 1))
            }

    def fit(self, X, Y, batch_size, epoch, print_costs=False):
        """
        Train the model with given data

        Parameters:
        X: the input of shape(nx, m)
        Y: the labels of training examples
        batch_size: batch size (<m)
        epoch: number of epoch to optimize
        print_costs: print the cost after each time the parameters is updated
        
        Return:
        The dictionary contains the trained parameters
        """
        # X (nx, m, Tx)
        self.a0 = np.zeros((self.na, batch_size))
        for e in range(epoch):
            for batch_number in range(self.m // batch_size):
                begin_index = batch_number * batch_size
                batch_x = X[:, begin_index: begin_index + batch_size, :]
                batch_y = Y[:, begin_index: begin_index + batch_size]
                # forward propagation
                a, caches = rnn_forward(self.a0, batch_x, self.parameters)
                a_last = a[:, :, -1]
                yhat, cache_y = rnn_y_forward(a_last, self.parameters)
                J = 1 / batch_size * np.sum(-batch_y * np.log(yhat))
                #print(-batch_y * np.log(yhat))
                # print result here
                if print_costs:
                    print("Epoch {0:>4}, iter {1:>8}, J = {2:>.4}".format(e, batch_number, J))
                # backward propagation
                d_ay = 1 / batch_size * (yhat - batch_y)
                gradients_y = rnn_y_backward(d_ay, cache_y)
                gradients = rnn_backward(gradients_y["da"], caches)
                # update parameters
                self.parameters["Wax"] = self.parameters["Wax"] - self.lr * gradients["dWax"]
                self.parameters["Waa"] = self.parameters["Waa"] - self.lr * gradients["dWaa"]
                self.parameters["ba"]  = self.parameters["ba"]  - self.lr * gradients["dba"]
                self.parameters["Wya"] = self.parameters["Wya"] - self.lr * gradients_y["dWya"]
                self.parameters["by"]  = self.parameters["by"]  - self.lr * gradients_y["dby"]
        return self.parameters
                
                
            
        
