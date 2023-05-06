#include "group_gather.h"

namespace vision3d {

void group_gather_forward(const at::Tensor& sources, const at::Tensor& indices, at::Tensor& targets) {
  group_gather_forward_cuda_launcher(sources, indices, targets);
}

void group_gather_backward(
    const at::Tensor& target_grads, const at::Tensor& indices, at::Tensor& source_grads, const int num_sources) {
  group_gather_backward_cuda_launcher(target_grads, indices, source_grads, num_sources);
}

}  // namespace vision3d
