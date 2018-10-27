def preprocess(X):
    old_max = 115
    old_min = 0
    new_max = 1
    new_min = -1
    X = (X - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return X
