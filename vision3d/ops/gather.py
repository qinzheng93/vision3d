from torch import Tensor
from torch.autograd import Function

from vision3d.utils.misc import load_ext


ext_module = load_ext("vision3d.ext", ["gather_forward", "gather_backward"])


class GatherFunction(Function):
    @staticmethod
    def forward(ctx, inputs: Tensor, indices: Tensor, transposed: bool = False) -> Tensor:
        """Gather by indices.

        If not transposed:

        outputs[i][c][j] = inputs[i][c][indices[i][j]]

        If transposed:

        outputs[i][j][c] = inputs[i][indices[i][j]][c]

        Args:
            inputs (Tensor): The input tensor in the shape of (B, C, N).
            indices (Tensor): The indices to gather in the shape of (B, M).
            transposed (bool=False): If True, the inputs is in the shape of (B, N, C).

        Returns:
            A Tensor of the gathered inputs in the shape of (B, C, M) or (B, M, C) if transposed.
        """
        if transposed:
            inputs = inputs.transpose(1, 2)  # (B, N, C) -> (B, C, N)
        inputs = inputs.contiguous()
        indices = indices.contiguous()

        ctx.save_for_backward(indices)
        ctx.num_inputs = inputs.shape[-1]
        ctx.transposed = transposed

        outputs = inputs.new_zeros(size=(inputs.shape[0], inputs.shape[1], indices.shape[1]))  # (B, C, M)

        ext_module.gather_forward(inputs, indices, outputs)

        if transposed:
            outputs = outputs.transpose(1, 2).contiguous()  # (B, C, M) -> (B, M, C)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs: Tensor):
        indices = ctx.saved_tensors[0]
        num_inputs = ctx.num_inputs
        transposed = ctx.transposed

        if transposed:
            grad_outputs = grad_outputs.transpose(1, 2)  # (B, M, C) -> (B, C, M)
        grad_outputs = grad_outputs.contiguous()

        # (B, C, N)
        grad_inputs = grad_outputs.new_zeros(size=(grad_outputs.shape[0], grad_outputs.shape[1], num_inputs))

        ext_module.gather_backward(grad_outputs, indices, grad_inputs, num_inputs)

        if transposed:
            grad_inputs = grad_inputs.transpose(1, 2).contiguous()  # (B, C, N) -> (B, N, C)

        return grad_inputs, None, None


gather = GatherFunction.apply
