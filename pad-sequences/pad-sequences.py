import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L)
    """

    # Handle empty input
    if len(seqs) == 0:
        return np.empty((0, 0), dtype=int)

    # Find maximum sequence length
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    # Create output matrix filled with pad_value
    result = np.full((len(seqs), max_len), pad_value, dtype=int)

    # Copy and truncate sequences
    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)
        result[i, :length] = seq[:length]

    return result