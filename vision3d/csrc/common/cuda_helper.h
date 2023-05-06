#ifndef VISION3D_COMMON_CUDA_HELPER_H_
#define VISION3D_COMMON_CUDA_HELPER_H_

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/cuda/Atomic.cuh>
#include <cassert>
#include <cmath>

namespace vision3d {

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
  int optimal_block_num = (N + num_threads - 1) / num_threads;
  int max_block_num = 4096;
  return std::min(optimal_block_num, max_block_num);
}

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

}  // namespace vision3d

#endif
