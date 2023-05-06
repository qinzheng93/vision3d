#ifndef VISION3D_CUDA_THREE_INTERPOLATE_H_
#define VISION3D_CUDA_THREE_INTERPOLATE_H_

#include "torch_helper.h"

namespace vision3d {

void three_interpolate_forward(
    const at::Tensor& sources, const at::Tensor& indices, const at::Tensor& weights, at::Tensor& targets);

void three_interpolate_forward_cuda_launcher(
    const at::Tensor& sources, const at::Tensor& indices, const at::Tensor& weights, at::Tensor& targets);

void three_interpolate_backward(
    const at::Tensor& target_grads,
    const at::Tensor& indices,
    const at::Tensor& weights,
    at::Tensor& source_grads,
    const int num_sources);

void three_interpolate_backward_cuda_launcher(
    const at::Tensor& target_grads,
    const at::Tensor& indices,
    const at::Tensor& weights,
    at::Tensor& source_grads,
    const int num_sources);

}  // namespace vision3d

#endif
