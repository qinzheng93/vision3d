#ifndef VISION3D_CUDA_GROUP_GATHER_H_
#define VISION3D_CUDA_GROUP_GATHER_H_

#include "torch_helper.h"

namespace vision3d {

void group_gather_forward(const at::Tensor& sources, const at::Tensor& indices, at::Tensor& targets);

void group_gather_forward_cuda_launcher(const at::Tensor& sources, const at::Tensor& indices, at::Tensor& targets);

void group_gather_backward(
    const at::Tensor& target_grads, const at::Tensor& indices, at::Tensor& source_grads, const int num_sources);

void group_gather_backward_cuda_launcher(
    const at::Tensor& target_grads, const at::Tensor& indices, at::Tensor& source_grads, const int num_sources);

}  // namespace vision3d

#endif
