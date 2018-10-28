import numpy as np

"""
Activation functions

Each class contains 2 class function:
calc: calculate the output from the input
derivative: calculate the derivative of the output w.r.t the input
"""
class Relu:
    @classmethod
    def calc(x):
        return (x>0).astype(np.int32) * x
    
    @classmethod
    def derivative(x):
        return (x>0).astype(np.int32)
    
class Tanh:
    @classmethod
    def calc(x):
        return np.tanh(x)

    @classmethod
    def derivative(x):
        return (1 - Tanh.calc(x)**2)

if __name__ == "__main__":
    x = np.random.randn(3,3)
    y = relu(x)
    print(y)
    x = relu_derivative(y)
    print(x)
