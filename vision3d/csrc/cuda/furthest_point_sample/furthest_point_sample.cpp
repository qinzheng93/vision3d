#include "furthest_point_sample.h"

namespace vision3d {

void furthest_point_sample(
    const at::Tensor& points, at::Tensor& distances, at::Tensor& indices, const int num_samples) {
  furthest_point_sample_cuda_launcher(points, distances, indices, num_samples);
}

}  // namespace vision3d
