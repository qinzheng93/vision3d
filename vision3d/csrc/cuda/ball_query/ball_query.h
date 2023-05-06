#ifndef VISION3D_CUDA_BALL_QUERY_H_
#define VISION3D_CUDA_BALL_QUERY_H_

#include "torch_helper.h"

namespace vision3d {

void ball_query(
    const at::Tensor& q_points,
    const at::Tensor& s_points,
    at::Tensor& indices,
    const float max_radius,
    const int num_samples);

void ball_query_cuda_launcher(
    const at::Tensor& q_points,
    const at::Tensor& s_points,
    at::Tensor& indices,
    const float max_radius,
    const int num_samples);

}  // namespace vision3d

#endif
