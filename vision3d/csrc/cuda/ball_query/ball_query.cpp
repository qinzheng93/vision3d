#include "ball_query.h"

namespace vision3d {

void ball_query(
    const at::Tensor& q_points,
    const at::Tensor& s_points,
    at::Tensor& indices,
    const float max_radius,
    const int num_samples) {
  ball_query_cuda_launcher(q_points, s_points, indices, max_radius, num_samples);
}

}  // namespace vision3d
