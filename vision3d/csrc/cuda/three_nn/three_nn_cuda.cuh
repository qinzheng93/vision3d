#ifndef VISION3D_CUDA_THREE_NN_CUDA_CUH_
#define VISION3D_CUDA_THREE_NN_CUDA_CUH_

#include "cuda_helper.h"
#include "torch_helper.h"

namespace vision3d {

// input: q_points(b, n, 3) s_points(b, m, 3)
// output: dist2(b, n, 3), idx(b, n, 3)
template <typename T>
__global__ void three_nn_kernel(
    int batch_size,
    int num_q_points,
    int num_s_points,
    const T* __restrict__ q_points,
    const T* __restrict__ s_points,
    T* tnn_distances,
    long* tnn_indices) {
  int batch_index = blockIdx.y;
  if (batch_index >= batch_size) return;

  q_points += batch_index * num_q_points * 3;
  s_points += batch_index * num_s_points * 3;
  tnn_distances += batch_index * num_q_points * 3;
  tnn_indices += batch_index * num_q_points * 3;

  CUDA_1D_KERNEL_LOOP(q_index, num_q_points) {
    const T* cur_q_points = q_points + q_index * 3;
    T* cur_tnn_distances = tnn_distances + q_index * 3;
    long* cur_tnn_indices = tnn_indices + q_index * 3;

    T qx = cur_q_points[0];
    T qy = cur_q_points[1];
    T qz = cur_q_points[2];

    double best1 = 1e40;
    double best2 = 1e40;
    double best3 = 1e40;
    int besti1 = -1;
    int besti2 = -1;
    int besti3 = -1;

    for (int s_index = 0; s_index < num_s_points; ++s_index) {
      T sx = s_points[s_index * 3 + 0];
      T sy = s_points[s_index * 3 + 1];
      T sz = s_points[s_index * 3 + 2];
      T sq_dist = (qx - sx) * (qx - sx) + (qy - sy) * (qy - sy) + (qz - sz) * (qz - sz);
      if (sq_dist < best1) {
        best3 = best2;
        besti3 = besti2;
        best2 = best1;
        besti2 = besti1;
        best1 = sq_dist;
        besti1 = s_index;
      } else if (sq_dist < best2) {
        best3 = best2;
        besti3 = besti2;
        best2 = sq_dist;
        besti2 = s_index;
      } else if (sq_dist < best3) {
        best3 = sq_dist;
        besti3 = s_index;
      }
    }

    cur_tnn_distances[0] = static_cast<T>(sqrt(best1));
    cur_tnn_distances[1] = static_cast<T>(sqrt(best2));
    cur_tnn_distances[2] = static_cast<T>(sqrt(best3));

    cur_tnn_indices[0] = besti1;
    cur_tnn_indices[1] = besti2;
    cur_tnn_indices[2] = besti3;
  }
}

}  // namespace vision3d

#endif
