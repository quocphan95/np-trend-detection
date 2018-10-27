import numpy as np

def relu(x):
    return (x>0).astype(np.int32) * x

def relu_derivative(y):
    return (y>0).astype(np.int32)

if __name__ == "__main__":
    x = np.random.randn(3,3)
    y = relu(x)
    print(y)
    x = relu_derivative(y)
    print(x)
