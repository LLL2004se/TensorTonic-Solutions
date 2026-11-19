import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_bwd_kernel(
    x_ptr, gamma_ptr, dy_ptr,
    dx_ptr, dgamma_ptr, dbeta_ptr,
    stride_x_row, stride_dy_row, stride_dx_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_SIZE)

    mask = col < N

    x = tl.load(x_ptr + row * stride_x_row + col, mask, 0)
    gamma = tl.load(gamma_ptr + col, mask, 0)
    dy = tl.load(dy_ptr + row * stride_dy_row + col, mask, 0)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / N
    centered = x - mean

    var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / N
    rstd = tl.rsqrt(var + eps)

    # Normalize input
    x_hat = centered * rstd

    # Gradient calculations
    dx_hat = dy * gamma

    mean_dx_hat = tl.sum(tl.where(mask, dx_hat, 0.0), axis=0) / N
    mean_dx_hat_xhat = tl.sum(
        tl.where(mask, dx_hat * x_hat, 0.0), axis=0
    ) / N

    dx = rstd * (
        dx_hat - mean_dx_hat - x_hat * mean_dx_hat_xhat
    )

    # Store input gradient
    tl.store(
        dx_ptr + row * stride_dx_row + col,
        dx,
        mask
    )

    # Parameter gradients across rows
    tl.atomic_add(
        dgamma_ptr + col,
        dy * x_hat,
        mask=mask,
        sem="relaxed"
    )

    tl.atomic_add(
        dbeta_ptr + col,
        dy,
        mask=mask,
        sem="relaxed"
    )


def solve(
    x: torch.Tensor, gamma: torch.Tensor, dy: torch.Tensor,
    dx_out: torch.Tensor, dgamma_out: torch.Tensor, dbeta_out: torch.Tensor,
    eps: float,
) -> None:
    """Launch the LayerNorm backward kernel: one program per row, atomic reductions for dgamma and dbeta."""

    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)

    dgamma_out.zero_()
    dbeta_out.zero_()

    grid = (M,)

    layernorm_bwd_kernel[grid](
        x, gamma, dy,
        dx_out, dgamma_out, dbeta_out,
        x.stride(0), dy.stride(0), dx_out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )