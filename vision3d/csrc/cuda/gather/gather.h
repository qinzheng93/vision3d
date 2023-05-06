#ifndef VISION3D_CUDA_GATHER_H_
#define VISION3D_CUDA_GATHER_H_

#include "torch_helper.h"

namespace vision3d {

void gather_forward(const at::Tensor& sources, const at::Tensor& indices, at::Tensor& targets);

void gather_forward_cuda_launcher(const at::Tensor& sources, const at::Tensor& indices, at::Tensor& targets);

void gather_backward(
    const at::Tensor& target_grads, const at::Tensor& indices, at::Tensor& source_grads, const int num_sources);

void gather_backward_cuda_launcher(
    const at::Tensor& target_grads, const at::Tensor& indices, at::Tensor& source_grads, const int num_sources);

}  // namespace vision3d

#endif
