def gae(rewards: list, values: list, gamma: float, lam: float) -> list:
    """
    Returns the generalized advantage estimate at every timestep.
    """
    advantages = [0.0] * len(rewards)
    gae_value = 0.0

    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae_value = delta + gamma * lam * gae_value
        advantages[t] = gae_value

    return advantages