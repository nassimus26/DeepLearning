import numpy as np

def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    # Divide x by its norm.
    x = x/x_norm
    ### END CODE HERE ###
    print(np.shape(x))
    print(np.shape(x_norm))
    return x
x = np.array([
[0, 3, 4],
[1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))


