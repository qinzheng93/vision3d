#include "three_nn.h"

namespace vision3d {

void three_nn(
    const at::Tensor& q_points, const at::Tensor& s_points, at::Tensor& tnn_distances, at::Tensor& tnn_indices) {
  three_nn_cuda_launcher(q_points, s_points, tnn_distances, tnn_indices);
}

}  // namespace vision3d
