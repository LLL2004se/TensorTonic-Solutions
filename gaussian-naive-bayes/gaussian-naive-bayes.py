
import math

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns a predicted class label for every test sample.
    """

    classes = sorted(set(y_train))
    n_features = len(X_train[0])

    # Calculate mean and variance for each class
    statistics = {}

    for cls in classes:
        samples = [
            X_train[i] for i in range(len(X_train))
            if y_train[i] == cls
        ]

        class_stats = []

        for j in range(n_features):
            values = [sample[j] for sample in samples]

            mean = sum(values) / len(values)

            variance = sum(
                (x - mean) ** 2 for x in values
            ) / len(values)

            # Avoid division by zero
            variance = max(variance, 1e-9)

            class_stats.append((mean, variance))

        statistics[cls] = (len(samples), class_stats)

    predictions = []

    for sample in X_test:
        best_class = None
        best_score = float("-inf")

        for cls in classes:
            count, class_stats = statistics[cls]

            # Log prior probability
            score = math.log(count / len(y_train))

            for j in range(n_features):
                mean, variance = class_stats[j]

                x = sample[j]

                # Gaussian log likelihood
                score += -0.5 * math.log(2 * math.pi * variance)
                score -= ((x - mean) ** 2) / (2 * variance)

            if score > best_score:
                best_score = score
                best_class = cls

        predictions.append(best_class)

    return predictions
