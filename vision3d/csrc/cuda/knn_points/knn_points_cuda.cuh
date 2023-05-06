#ifndef VISION3D_CUDA_KNN_POINTS_CUDA_CUH_
#define VISION3D_CUDA_KNN_POINTS_CUDA_CUH_

#include "cuda_helper.h"
#include "torch_helper.h"

namespace vision3d {

template <typename T>
inline __device__ void swap(T *x, T *y) {
  T tmp = *x;
  *x = *y;
  *y = tmp;
}

template <typename T>
__device__ void reheap(T *dist, int *idx, int k) {
  int root = 0;
  int child = root * 2 + 1;
  while (child < k) {
    if (child + 1 < k && dist[child + 1] > dist[child]) child++;
    if (dist[root] > dist[child]) return;
    swap<T>(&dist[root], &dist[child]);
    swap<int>(&idx[root], &idx[child]);
    root = child;
    child = root * 2 + 1;
  }
}

template <typename T>
__device__ void heap_sort(T *dist, int *idx, int k) {
  for (int i = k - 1; i > 0; --i) {
    swap<T>(&dist[0], &dist[i]);
    swap<int>(&idx[0], &idx[i]);
    reheap(dist, idx, i);
  }
}

// input: xyz (b, n, 3) new_xyz (b, m, 3)
// output: idx (b, m, nsample) dist2 (b, m, nsample)
template <typename T>
__global__ void knn_points_kernel(
    int batch_size,
    int num_q_points,
    int num_s_points,
    int num_neighbors,
    const T *q_points,
    const T *s_points,
    T *knn_distances,
    long *knn_indices) {
  int batch_index = blockIdx.y;
  if (batch_index >= batch_size) return;

  q_points += batch_index * num_q_points * 3;
  s_points += batch_index * num_s_points * 3;
  knn_indices += batch_index * num_q_points * num_neighbors;
  knn_distances += batch_index * num_q_points * num_neighbors;

  CUDA_1D_KERNEL_LOOP(q_index, num_q_points) {
    const T *cur_q_points = q_points + q_index * 3;
    T *cur_knn_distances = knn_distances + q_index * num_neighbors;
    long *cur_knn_indices = knn_indices + q_index * num_neighbors;

    T qx = cur_q_points[0];
    T qy = cur_q_points[1];
    T qz = cur_q_points[2];

    T best_dist[100];
    int best_idx[100];
    for (int n_index = 0; n_index < num_neighbors; ++n_index) {
      best_dist[n_index] = 1e10;
      best_idx[n_index] = 0;
    }

    for (int s_index = 0; s_index < num_s_points; ++s_index) {
      T sx = s_points[s_index * 3 + 0];
      T sy = s_points[s_index * 3 + 1];
      T sz = s_points[s_index * 3 + 2];
      T sq_dist = (qx - sx) * (qx - sx) + (qy - sy) * (qy - sy) + (qz - sz) * (qz - sz);
      if (sq_dist < best_dist[0]) {
        best_dist[0] = sq_dist;
        best_idx[0] = s_index;
        reheap(best_dist, best_idx, num_neighbors);
      }
    }

    heap_sort(best_dist, best_idx, num_neighbors);
    for (int n_index = 0; n_index < num_neighbors; ++n_index) {
      cur_knn_indices[n_index] = best_idx[n_index];
      cur_knn_distances[n_index] = sqrt(best_dist[n_index]);
    }
  }
}

}  // namespace vision3d

#endif
