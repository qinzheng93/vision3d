#ifndef VISION3D_CUDA_GATHER_CUDA_CUH_
#define VISION3D_CUDA_GATHER_CUDA_CUH_

#include "cuda_helper.h"
#include "torch_helper.h"

namespace vision3d {

// input: points(b, c, n) idx(b, m)
// output: out(b, c, m)
template <typename T>
__global__ void gather_forward_kernel(
    int batch_size,
    int num_channels,
    int num_sources,
    int num_samples,
    const T* __restrict__ sources,
    const long* __restrict__ indices,
    T* targets) {
  int batch_index = blockIdx.z;
  int channel_index = blockIdx.y;

  if (batch_index >= batch_size || channel_index >= num_channels) return;

  sources += batch_index * num_channels * num_sources + channel_index * num_sources;
  indices += batch_index * num_samples;
  targets += batch_index * num_channels * num_samples + channel_index * num_samples;

  CUDA_1D_KERNEL_LOOP(t_index, num_samples) {
    int s_index = indices[t_index];
    targets[t_index] = sources[s_index];
  }
}

// input: grad_out(b, c, m) idx(b, m)
// output: grad_points(b, c, n)
template <typename T>
__global__ void gather_backward_kernel(
    int batch_size,
    int num_channels,
    int num_sources,
    int num_samples,
    const T* __restrict__ target_grads,
    const long* __restrict__ indices,
    T* source_grads) {
  int batch_index = blockIdx.z;
  int channel_index = blockIdx.y;

  if (batch_index >= batch_size || channel_index >= num_channels) return;

  source_grads += batch_index * num_channels * num_sources + channel_index * num_sources;
  indices += batch_index * num_samples;
  target_grads += batch_index * num_channels * num_samples + channel_index * num_samples;

  CUDA_1D_KERNEL_LOOP(t_index, num_samples) {
    int s_index = indices[t_index];
    gpuAtomicAdd(source_grads + s_index, target_grads[t_index]);
  }
}

}  // namespace vision3d

#endif
