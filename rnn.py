import numpy as np
import softmax

def rnn_cell_forward(a_prev, xt, parameters):
    """
    Calculate the forward propagation at time t

    Parameters:
    a_prev: hidden unit a at time t-1 of shape (n_a, m)
    xt: input at time t of shape (n_x, m)
    parameters: dictionary contains 3 parameters: Wax, Waa, ba
    Return:
    a_next: hidden unit at time t
    cache: information at time t (a_prev, a_next, xt, parameters) used for backprobagation phase
    """
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    ba = parameters["ba"]

    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    cache = (a_prev, a_next, xt, parameters)
    return a_next, cache

def rnn_forward(a0, X, parameters):
    """
    Calculate the forward propagation of RNN net

    Parameters:
    a0: initial state for the first cell of the net (shape n_a, m)
    X: Training input X, shape nx, m, Tx
    parameters: dictionary contains 3 parameters: Wax, Waa, ba
    Return:
    a: numpy array of shape (n_a, m, Tx) contains the hidden activations
    caches: A list contain all caches return by rnn_cell_forward
    """
    # X.shape = (nx, m, Tx)
    n_a, m = a0.shape
    (Tx, m) = X.shape
    a = np.zeros((n_a, m, Tx))
    caches = []
    a_prev = a0
    for t in range(Tx):
        a_next, cache = rnn_cell_fordward(a_prev, X[:, :, t])
        a[:, :, t] = a_next
        caches = caches + [cache]
    return a, caches

def rnn_cell_backward(da_next, cache):
    """
    Calculate 1 stage of backward propagation

    Parameters:
    da_next: Upstream derivative DJ/Dat, shape = (n_a, m)
    cache: contain necessary values returned by rnn_cell_forward
    Return:
    A dictionary contain resultant gradients:
        dWax: Dj/DWax(t)
        dWaa: Dj/DWaa(t)
        dba: Dj/Dba(t)
        da_prev: DJ/Da(t-1)
    """
    # da_next is DJ/Da_next
    a_prev, a_next, xt, parameters = cache
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    ba = parameters["ba"]
    # use chain rule to calculate DJ/DWax(t), DJ/DWaa(t), Dj/Dba(t)
    dtanh = da_next * (1 - a_next ** 2) 
    dWax = np.dot(dtanh, xt.T)
    dWaa = np.dot(dtanh, a_prev.T)
    dba = np.sum(dtanh, axis=1, keepdims = True)
    da_prev = np.dot(Waa.T, dtanh)
    return {"dWax": dWax, "dWaa" : dWaa, "dba": dba, "da_prev" : da_prev}

def rnn_backward(da, caches):
    """
    BPTT (back propagation through time)

    Caclculate the gradients of J w.r.t Wax, Waa, ba
    Parameters:
    da: Upstream derivative DJ/Dan, shape = (n_a, m)
    caches: contains all caches of the forward phases
    Return:
    dWax: DJ/Dax
    dWaa: DJ/Daa
    dba: DJ/Dba
    """
    # da.shape is n_a, m
    Tx = len(caches)
    cache = caches[0]
    a_prev, a_next, xt, parameters = cache
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    ba = parameters["ba"]
    n_a, nx = Wax.shape
    dWax = np.zeros(Wax.shape)
    dWaa = np.zeros(Waa.shape)
    dba = np.zeros((n_a, 1))
    da_prevt = da
    for t in reversed(range(Tx)):
        gradients = rnn_cell_backward(da_prevt, caches[t])
        dWax += gradients["dWax"]
        dWaa += gradients["dWaa"]
        dba += gradients["dba"]
        da_prevt = gradients["da_prev"]
    return {"dWax" : dWax, "dWaa" : dWaa, "dba" : dba}
        
def rnn_y_forward(a, parameters):
    """
    Caclculate y from an

    Parameters:
    a: an, the output of the last cell in rnn net (n_a, m)
    parameters: contain Way and by
    Return:
    y: prediction result (n_y, m)
    cache: values used in calculating the derivatives in backward pass
    """
    Way = parameters["Way"]
    by = paramters["by"]
    tanh = np.tanh(np.dot(Way, a) + by)
    y = softmax.calc(tanh)
    cache = (a, tanh, y, parameters)
    return y, cache

def rnn_y_backward(dy, cache):
    """
    Calculate the derivatives of J w.r.t Way, by, an

    Parameters:
    dy: DJ/Dy
    cache: cache returned by rnn_y_forward
    Return: dWay: DJ/DWay
            dby: DJ/Dby
            da: DJ/Da
    """
    # dy is DJ/Dy
    # dy shape is 3, m
    a, tanh, y, parameters = cache
    Way = parameters["Way"]
    by = paramters["by"]
    # DJ/Da
    dtanh = dy * softmax.derivative(y)
    # temporary derivative used to calculate dWay, dby and da
    # dtanh shape is 3, m
    dtanh = dtanh * (1 - tanh**2)
    dWay = np.dot(dtand, a.T)
    dby = np.sum(dtanh, axis=1, keepsdim=True)
    da = np.dot(Way.T, dtanh)
    return {"dWay" : dWay, "dby" : dby, "da" : da}
    
    
    
    
            
    
