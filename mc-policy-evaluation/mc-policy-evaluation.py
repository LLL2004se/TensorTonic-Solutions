import numpy as np

def mc_policy_evaluation(episodes: list, gamma: float, n_states: int) -> np.ndarray:
    returns = [[] for _ in range(n_states)]

    for episode in episodes:
        G = 0
        episode_returns = []

        for state, reward in reversed(episode):
            G = reward + gamma * G
            episode_returns.append((state, G))

        episode_returns.reverse()

        visited = set()

        for state, G in episode_returns:
            if state not in visited:
                returns[state].append(G)
                visited.add(state)

    values = np.zeros(n_states, dtype=float)

    for state in range(n_states):
        if returns[state]:
            values[state] = np.mean(returns[state])

    return values