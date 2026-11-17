def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """

    top_k = recommended[:k]

    relevant_set = set(relevant)

    hits = sum(1 for item in top_k if item in relevant_set)

    precision = hits / k if k > 0 else 0.0
    recall = hits / len(relevant_set) if relevant_set else 0.0

    return [float(precision), float(recall)]