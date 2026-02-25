import os
import hashlib
import warnings

import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.utils.cpp_extension import load


module_path = os.path.dirname(__file__)

# Build a deterministic, per-path extension name to avoid collisions if multiple copies exist.
_ext_suffix = hashlib.md5(module_path.encode("utf-8")).hexdigest()[:8]
_ext_name = f"fused_bias_act_{_ext_suffix}"

_fused = None
if torch.cuda.is_available() and os.name != "nt":
    try:
        _fused = load(
            name=_ext_name,
            sources=[
                os.path.join(module_path, "fused_bias_act.cpp"),
                os.path.join(module_path, "fused_bias_act_kernel.cu"),
            ],
            verbose=False,
        )
    except Exception as e:
        warnings.warn(
            "Could not build/load fused_bias_act CUDA extension; "
            "falling back to pure PyTorch ops.\n"
            f"Reason: {e}"
        )
        _fused = None


def _bias_add(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if bias is None:
        return x

    # bias is [C]
    if x.ndim == 2:
        return x + bias.view(1, -1)
    elif x.ndim == 3:
        return x + bias.view(1, -1, 1)
    else:
        # Assume channel dim is 1 (N, C, ...)
        view = [1, -1] + [1] * (x.ndim - 2)
        return x + bias.view(*view)


class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = float(negative_slope)
        ctx.scale = float(scale)

        empty = grad_output.new_empty(0)
        grad_input = _fused.fused_bias_act(
            grad_output, empty, out, 3, 1, ctx.negative_slope, ctx.scale
        )

        dim = [0]
        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()
        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        (out,) = ctx.saved_tensors
        gradgrad_out = _fused.fused_bias_act(
            gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
        )
        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        out = _fused.fused_bias_act(
            input, bias, empty, 3, 0, float(negative_slope), float(scale)
        )
        ctx.save_for_backward(out)
        ctx.negative_slope = float(negative_slope)
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.negative_slope, ctx.scale
        )
        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = float(negative_slope)
        self.scale = float(scale)

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    negative_slope = float(negative_slope)
    scale = float(scale)

    # Use CUDA extension only if it successfully built and we're on CUDA tensors.
    if _fused is not None and input.is_cuda:
        return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)

    # Pure PyTorch fallback (works on CPU or CUDA, no compilation required).
    x = _bias_add(input, bias)
    return F.leaky_relu(x, negative_slope=negative_slope) * scale
