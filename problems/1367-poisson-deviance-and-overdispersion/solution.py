import numpy as np


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance, using the convention 0 * log(0) = 0."""
    # Your code here
    return 2 * float(np.sum(np.where(y==0, mu, y*np.log(y/mu) - (y - mu))))


def dispersion_ratio(y: np.ndarray, mu: np.ndarray, n_params: int) -> float:
    """Pearson chi-square divided by (n - n_params)."""
    # Your code here
    n = y.shape[0]
    if n_params == n:
        return 0.0

    return 1.0/(n - n_params) * float(np.sum(((y - mu)**2)/mu))
