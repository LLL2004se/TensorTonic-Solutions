
import torch

def train_tiny_micrograd_mlp(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
    learning_rate: float,
    steps: int
) -> tuple:
    """
    Returns final predictions, final loss, trained weights,
    trained biases, and pre-update loss history.
    """

    W = [w.clone() for w in weights]
    b = [v.clone() for v in biases]

    loss_history = []

    for _ in range(steps):

        # Handle empty batch
        if inputs.shape[0] == 0:
            loss_history.append(inputs.new_zeros(()))
            continue

        # Forward pass
        activations = [inputs]
        
        for i in range(len(W)):
            z = activations[-1] @ W[i].T + b[i]
            activations.append(torch.tanh(z))

        predictions = activations[-1].squeeze(-1)

        # Summed squared error
        error = predictions - targets
        loss = torch.sum(error ** 2)

        # Save pre-update loss
        loss_history.append(loss.clone())

        # Backward pass: manually calculate gradients
        grad = (2 * error).unsqueeze(-1)

        grad_W = [None] * len(W)
        grad_b = [None] * len(b)

        for i in reversed(range(len(W))):

            # Derivative through tanh
            grad_z = grad * (1 - activations[i + 1] ** 2)

            grad_W[i] = grad_z.T @ activations[i]
            grad_b[i] = grad_z.sum(dim=0)

            if i > 0:
                grad = grad_z @ W[i]

        # Simultaneous gradient descent updates
        W = [
            w - learning_rate * gw
            for w, gw in zip(W, grad_W)
        ]

        b = [
            v - learning_rate * gb
            for v, gb in zip(b, grad_b)
        ]

    # Final forward pass
    with torch.no_grad():
        h = inputs

        for i in range(len(W)):
            h = torch.tanh(h @ W[i].T + b[i])

        final_predictions = h.squeeze(-1)
        final_loss = torch.sum((final_predictions - targets) ** 2)

    return (
        final_predictions,
        final_loss,
        W,
        b,
        loss_history
    )
