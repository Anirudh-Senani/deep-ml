import numpy as np


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with an intercept.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        np.ndarray: (p + 1,) coefficients, intercept first.
    """
    # Your code here
    feats = np.column_stack([np.ones(X.shape[0]), X])
    coeffs, _, _, _ = np.linalg.lstsq(feats, y)
    return coeffs


def omitted_variable_bias(X: np.ndarray, y: np.ndarray, omit_idx: int) -> tuple:
    """Return (full_kept, short, bias) for a two-column X."""
    # Your code here
    full_kept = ols(X, y)

    if omit_idx == 0:
        short = ols(X[:,1], y)[-1]
        bias = full_kept[1]
        full_kept = full_kept[2]
        delta = ols(X[:,1], X[:,0])[-1]

    else:
        short = ols(X[:,0], y)[-1]
        bias = full_kept[2]
        full_kept = full_kept[1]
        delta = ols(X[:,0], X[:,1])[-1]

    return full_kept, short, bias*delta
