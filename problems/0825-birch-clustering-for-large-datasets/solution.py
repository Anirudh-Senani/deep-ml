import numpy as np

def birch_cluster(X, threshold):
    """
    Single-level BIRCH clustering.
    X: array-like of shape (n_samples, n_features)
    threshold: float, max allowed subcluster radius
    Returns: list of centroids (each a list of floats), sorted lexicographically.
    """
    X = np.asarray(X)
    if X.shape[0] == 0:
        return []

    cluster_features = [(1, X[0], X[0]**2)]

    for i in range(1, X.shape[0]):
        centroids = np.stack([cf[1]/cf[0] for cf in cluster_features])
        dists = np.sqrt(((centroids - X[i][None, :])**2).sum(axis=-1))
        centroid = int(np.argmin(dists))

        n = cluster_features[centroid][0] + 1
        ls = cluster_features[centroid][1] + X[i]
        ss = cluster_features[centroid][2] + X[i]**2
        radius = np.sqrt((ss/n - (ls/n)**2).sum())

        if radius <= threshold:
            cluster_features[centroid] = (n, ls, ss)
        else:
            cluster_features.append((1, X[i], X[i]**2))

    centroids = [tuple((cf[1]/cf[0]).tolist()) for cf in cluster_features]

    return sorted(centroids)
