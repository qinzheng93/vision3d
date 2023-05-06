#include "three_interpolate.h"

namespace vision3d {

void three_interpolate_forward(
    const at::Tensor& sources, const at::Tensor& indices, const at::Tensor& weights, at::Tensor& targets) {
  three_interpolate_forward_cuda_launcher(sources, indices, weights, targets);
}

void three_interpolate_backward(
    const at::Tensor& target_grads,
    const at::Tensor& indices,
    const at::Tensor& weights,
    at::Tensor& source_grads,
    const int num_sources) {
  three_interpolate_backward_cuda_launcher(target_grads, indices, weights, source_grads, num_sources);
}

}  // namespace vision3d
