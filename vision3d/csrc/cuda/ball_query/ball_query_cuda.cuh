#ifndef VISION3D_CUDA_BALL_QUERY_CUDA_CUH_
#define VISION3D_CUDA_BALL_QUERY_CUDA_CUH_

#include "cuda_helper.h"
#include "torch_helper.h"

namespace vision3d {

// input: q_points(b, m, 3) s_points(b, n, 3)
// output: idx(b, m, num_samples)
template <typename T>
__global__ void ball_query_kernel(
    int batch_size,
    int num_q_points,
    int num_s_points,
    int num_samples,
    float max_radius,
    const T* __restrict__ q_points,
    const T* __restrict__ s_points,
    long* indices) {
  int batch_index = blockIdx.y;
  if (batch_index >= batch_size) return;

  q_points += batch_index * num_q_points * 3;
  s_points += batch_index * num_s_points * 3;
  indices += batch_index * num_q_points * num_samples;

  T max_radius2 = static_cast<T>(max_radius) * static_cast<T>(max_radius);
  CUDA_1D_KERNEL_LOOP(q_index, num_q_points) {
    const T* cur_q_points = q_points + q_index * 3;
    long* cur_indices = indices + q_index * num_samples;

    T qx = cur_q_points[0];
    T qy = cur_q_points[1];
    T qz = cur_q_points[2];

    int cnt = 0;
    for (int s_index = 0; s_index < num_s_points; ++s_index) {
      T sx = s_points[s_index * 3 + 0];
      T sy = s_points[s_index * 3 + 1];
      T sz = s_points[s_index * 3 + 2];
      T sq_dist = (qx - sx) * (qx - sx) + (qy - sy) * (qy - sy) + (qz - sz) * (qz - sz);
      if (sq_dist < max_radius2) {
        if (cnt == 0) {
          for (int i = 0; i < num_samples; ++i) {
            cur_indices[i] = s_index;
          }
        }
        cur_indices[cnt] = s_index;
        ++cnt;
        if (cnt >= num_samples) break;
      }
    }
  }
}

}  // namespace vision3d

#endif
