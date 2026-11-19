import torch

def dense_layer_forward(inputs: torch.Tensor, weight_matrix: torch.Tensor, biases: torch.Tensor, nonlinear: bool) -> torch.Tensor:
    output = torch.mv(weight_matrix, inputs) + biases

    if nonlinear:
        output = torch.tanh(output)

    return output