def preprocess(X):
    """
    Preprocess the input X

    Covert the input to number in (-1, 1) to increase the training performance
    Parameters:
    X: Input X of shape(nx, m, Tx)
    Return:
    X in range (-1, 1) of shape (nx, m, Tx)
    """
    old_max = 115
    old_min = 0
    new_max = 1
    new_min = -1
    X = (X - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return X
