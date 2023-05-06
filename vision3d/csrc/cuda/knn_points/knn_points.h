#ifndef VISION3D_CUDA_KNN_POINTS_H_
#define VISION3D_CUDA_KNN_POINTS_H_

#include <tuple>

#include "torch_helper.h"

namespace vision3d {

void knn_points(
    const at::Tensor& q_points,
    const at::Tensor& s_points,
    at::Tensor& knn_distances,
    at::Tensor& knn_indices,
    int num_neighbors);

void knn_points_cuda_launcher(
    const at::Tensor& q_points,
    const at::Tensor& s_points,
    at::Tensor& knn_distances,
    at::Tensor& knn_indices,
    int num_neighbors);

}  // namespace vision3d

#endif
