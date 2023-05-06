#include "knn_points.h"

namespace vision3d {

void knn_points(
    const at::Tensor& q_points,
    const at::Tensor& s_points,
    at::Tensor& knn_distances,
    at::Tensor& knn_indices,
    int num_neighbors) {
  knn_points_cuda_launcher(q_points, s_points, knn_distances, knn_indices, num_neighbors);
}

}  // namespace vision3d
