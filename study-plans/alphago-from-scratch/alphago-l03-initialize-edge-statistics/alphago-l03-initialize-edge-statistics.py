import numpy as np

def initialize_mcts_edges(policy_logits: np.ndarray, legal_mask: np.ndarray) -> dict:
    """
    Returns: N as an int64 array; W, Q, and P as float64 arrays in a dictionary.
    """

    # Convert legal mask to boolean
    legal_mask = np.asarray(legal_mask, dtype=bool)

    # Mask illegal actions
    masked_logits = np.where(legal_mask, policy_logits, -np.inf)

    # Stable softmax over legal actions
    if np.any(legal_mask):
        max_logit = np.max(masked_logits[legal_mask])
        exp_logits = np.zeros(policy_logits.shape, dtype=np.float64)

        exp_logits[legal_mask] = np.exp(
            masked_logits[legal_mask] - max_logit
        )

        P = exp_logits / np.sum(exp_logits)
    else:
        P = np.zeros(policy_logits.shape, dtype=np.float64)

    # Initialize MCTS statistics
    N = np.zeros(policy_logits.shape, dtype=np.int64)
    W = np.zeros(policy_logits.shape, dtype=np.float64)
    Q = np.zeros(policy_logits.shape, dtype=np.float64)

    return {
        "N": N,
        "W": W,
        "Q": Q,
        "P": P
    }