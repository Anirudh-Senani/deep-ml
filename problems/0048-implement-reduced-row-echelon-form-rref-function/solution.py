import numpy as np


def rref(matrix):
    # Your code here
    n = len(matrix)
    matrix = np.asarray(matrix, dtype=np.float32)

    for i in range(n):
        if matrix[i, i] == 0:
            j = i
            while j < n-1 and matrix[j, i] == 0:
                j += 1
            if matrix[j, i] != 0:
                matrix[i], matrix[j] = matrix[j].copy(), matrix[i].copy()

        if matrix[i, i] != 0:
            matrix[i] /= matrix[i, i]

        for j in range(n):
            if i != j:
                scale = 1.0
                if matrix[i, i] != 0:
                    scale = matrix[j, i]/matrix[i, i]
                matrix[j] -= scale * matrix[i]

    i = 0
    last = n-1
    while i < last:
        if np.all(matrix[i, :] == 0):
            matrix[i], matrix[last] = matrix[last].copy(), matrix[i].copy()
            last -= 1
        i += 1

    return matrix.tolist()
