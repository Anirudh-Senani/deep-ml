import torch
from typing import Tuple

def find_best_split(X: torch.Tensor, y: torch.Tensor) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    # ✏️ TODO: implement
    best_feat_ind = X.shape[1]
    best_thres = 1e9
    best_gini = 1e9
    n = X.shape[0]
    for ind in range(X.shape[1]):
        col = X[:, ind]
        for thres in col:
            left = y[col <= thres]
            right = y[col > thres]

            left_gini = (left==0).float().mean().item()**2 + (left==1).float().mean().item()**2
            left_gini = 1 - left_gini

            right_gini = (right==0).float().mean().item()**2 + (right==1).float().mean().item()**2
            right_gini = 1 - right_gini

            gini = left.shape[0]/n * left_gini + right.shape[0] * right_gini

            if gini < best_gini:
                best_feat_ind = ind
                best_thres = thres
                best_gini = gini

    return best_feat_ind, best_thres.item()
