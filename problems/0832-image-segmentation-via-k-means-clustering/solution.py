import numpy as np

def segment_image(image, k: int, max_iters: int = 100):
    """
    Segment an RGB image using K-Means clustering on pixel colors.

    Args:
        image: nested list / array of shape (H, W, 3) with pixel values.
        k: number of clusters.
        max_iters: maximum number of K-Means iterations.

    Returns:
        Segmented image as a nested list of shape (H, W, 3).
    """
    image = np.asarray(image)
    H, W, C = image.shape
    flat_image = image.reshape(-1, C)
    centroids = flat_image[:k, :]

    for _ in range(max_iters):
        assigned = np.argmin(((flat_image[:, None, :] - centroids[None, :, :])**2).sum(axis=-1), axis=-1)
        new_centroids = np.zeros_like(centroids, dtype=np.float32)
        for ki in range(k):
            if np.any(assigned==ki):
                new_centroids[ki] = flat_image[assigned==ki].mean(axis=0)
            else:
                new_centroids[ki] = centroids[ki]

        if np.all(new_centroids == centroids):
            break

        centroids = new_centroids

    assigned = np.argmin(((flat_image[:, None, :] - centroids[None, :, :])**2).sum(axis=-1), axis=-1)

    return centroids[assigned].reshape(W, H, C).tolist()
