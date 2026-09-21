def rejection_sampling_best_of_k(candidates, scores):
    """
    Select the highest-scoring candidate per prompt.

    Args:
        candidates: list of N lists, each containing K candidate outputs.
        scores: list of N lists, each containing K reward scores.

    Returns:
        List of N selected candidates.
    """
    selected = []

    for candidate, score in zip(candidates, scores):
        ind = max(range(len(score)), key=lambda x: score[x])
        selected.append(candidate[ind])

    return selected
