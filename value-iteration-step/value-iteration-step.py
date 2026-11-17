def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    new_values = []

    for s in range(len(values)):
        max_value = float('-inf')

        for a in range(len(transitions[s])):
            total = 0.0

            for next_state in range(len(values)):
                probability = transitions[s][a][next_state]
                total += probability * values[next_state]

            q_value = rewards[s][a] + gamma * total

            max_value = max(max_value, q_value)

        new_values.append(float(max_value))

    return new_values