import numpy as np

def tile_coding_step(base_alpha: float, n_tilings: int, weights: list, active_tiles: list, target: float) -> dict:
    """
    Perform a single value function update using tile coding with adjusted step size.
    
    Args:
        base_alpha: The base learning rate before adjustment
        n_tilings: Number of tilings used in the tile coding scheme
        weights: List of current weight values for all tiles
        active_tiles: List of indices of tiles active for the current state
        target: The target value (e.g., a return or TD target)
    
    Returns:
        Dictionary with 'adjusted_alpha', 'prediction_before',
        'prediction_after', and 'updated_weights'
    """
    weights = np.asarray(weights)
    adjusted_alpha = base_alpha/n_tilings
    prediction_before = weights[active_tiles].sum()
    error = target - prediction_before
    update = error * adjusted_alpha

    weights[active_tiles] += update
    prediction_after = weights[active_tiles].sum()

    return {
        'adjusted_alpha':adjusted_alpha,
        'prediction_before':float(prediction_before),
        'prediction_after':float(prediction_after),
        'updated_weights':np.round(weights, 4).tolist()
    }
