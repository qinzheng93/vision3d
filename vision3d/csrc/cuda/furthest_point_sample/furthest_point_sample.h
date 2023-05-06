#ifndef VISION3D_CUDA_FURTHEST_POINT_SAMPLE_H_
#define VISION3D_CUDA_FURTHEST_POINT_SAMPLE_H_

#include "torch_helper.h"

namespace vision3d {

void furthest_point_sample(const at::Tensor& points, at::Tensor& distances, at::Tensor& indices, const int num_samples);

void furthest_point_sample_cuda_launcher(
    const at::Tensor& points, at::Tensor& distances, at::Tensor& indices, const int num_samples);

}  // namespace vision3d

#endif
