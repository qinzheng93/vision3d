#ifndef VISION3D_CUDA_FURTHEST_POINT_SAMPLE_CUDA_CUH_
#define VISION3D_CUDA_FURTHEST_POINT_SAMPLE_CUDA_CUH_

#include "cuda_helper.h"
#include "torch_helper.h"

namespace vision3d {

inline int opt_n_threads(int work_size) {
  int pow_2 = static_cast<int>(std::log(static_cast<double>(work_size)) / std::log(2.0));
  return std::max(std::min(1 << pow_2, 512), 1);
}

__device__ void __update(float* __restrict__ dists, int* __restrict__ dists_i, int idx1, int idx2) {
  float v1 = dists[idx1];
  float v2 = dists[idx2];
  int i1 = dists_i[idx1];
  int i2 = dists_i[idx2];
  if (v2 > v1) {
    v1 = v2;
    i1 = i2;
  }
  dists[idx1] = v1;
  dists_i[idx1] = i1;
}

// Input points: (b, n, 3), distances: (b, n)
// Ouput idx (b, m)
template <unsigned int block_size>
__global__ void furthest_point_sample_kernel(
    int batch_size,
    int num_points,
    int num_samples,
    const float* __restrict__ points,
    float* distances,
    long* indices) {
  if (num_samples <= 0) return;

  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  int batch_index = blockIdx.x;
  points += batch_index * num_points * 3;
  distances += batch_index * num_points;
  indices += batch_index * num_samples;

  int tid = threadIdx.x;
  const int stride = block_size;

  int old = 0;
  if (threadIdx.x == 0) indices[0] = old;

  __syncthreads();
  for (int s_index = 1; s_index < num_samples; s_index++) {
    int besti = 0;
    float best = -1;
    float x1 = points[old * 3 + 0];
    float y1 = points[old * 3 + 1];
    float z1 = points[old * 3 + 2];
    for (int p_index = tid; p_index < num_points; p_index += stride) {
      float x2 = points[p_index * 3 + 0];
      float y2 = points[p_index * 3 + 1];
      float z2 = points[p_index * 3 + 2];
      float sq_dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
      sq_dist = min(sq_dist, distances[p_index]);
      distances[p_index] = sq_dist;
      if (sq_dist > best) {
        best = sq_dist;
        besti = p_index;
      }
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

#pragma unroll
    for (int block_size_thresh = 512; block_size_thresh >= 2; block_size_thresh >>= 1) {
      const int tid_thresh = block_size_thresh >> 1;
      if (block_size >= block_size_thresh && tid < tid_thresh) {
        __update(dists, dists_i, tid, tid + tid_thresh);
      }
      __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0) indices[s_index] = old;
  }
}

}  // namespace vision3d

#endif
