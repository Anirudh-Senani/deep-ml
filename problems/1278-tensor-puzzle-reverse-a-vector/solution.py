import numpy as np

def flip(a: np.ndarray) -> np.ndarray:
    """Reverse 1-D array a without slicing a[::-1]."""
    # Your code here
    for i in range(len(a)//2):
        a[i], a[len(a)-i-1] = a[len(a)-i-1], a[i]

    return a
