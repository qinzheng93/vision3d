#ifndef VISION3D_CUDA_THREE_NN_H_
#define VISION3D_CUDA_THREE_NN_H_

#include <tuple>

#include "torch_helper.h"

namespace vision3d {

void three_nn(
    const at::Tensor& q_points, const at::Tensor& s_points, at::Tensor& tnn_distances, at::Tensor& tnn_indices);

void three_nn_cuda_launcher(
    const at::Tensor& q_points, const at::Tensor& s_points, at::Tensor& tnn_distances, at::Tensor& tnn_indices);

}  // namespace vision3d

#endif
