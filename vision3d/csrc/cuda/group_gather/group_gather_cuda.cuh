#ifndef VISION3D_CUDA_GROUP_GATHER_CUDA_CUH_
#define VISION3D_CUDA_GROUP_GATHER_CUDA_CUH_

#include "cuda_helper.h"
#include "torch_helper.h"

namespace vision3d {

// input: sources(b, c, n) idx(b, num_sources, num_samples)
// output: out(b, c, num_sources, num_samples)
template <typename T>
__global__ void group_gather_forward_kernel(
    int batch_size,
    int num_channels,
    int num_sources,
    int num_targets,
    int num_samples,
    const T* __restrict__ sources,
    const long* __restrict__ indices,
    T* targets) {
  int batch_index = blockIdx.z;
  int channel_index = blockIdx.y;
  if (batch_index >= batch_size || channel_index >= num_channels) return;

  sources += batch_index * num_channels * num_sources + channel_index * num_sources;
  targets += batch_index * num_channels * num_targets * num_samples + channel_index * num_targets * num_samples;
  indices += batch_index * num_targets * num_samples;

  CUDA_1D_KERNEL_LOOP(target_index, num_targets * num_samples) {
    int source_index = indices[target_index];
    targets[target_index] = sources[source_index];
  }
}

// input: target_grads(b, c, num_sources, num_samples), idx(b, num_sources,
// num_samples) output: source_grads(b, c, n)
template <typename T>
__global__ void group_gather_backward_kernel(
    int batch_size,
    int num_channels,
    int num_sources,
    int num_targets,
    int num_samples,
    const T* __restrict__ target_grads,
    const long* __restrict__ indices,
    T* source_grads) {
  int batch_index = blockIdx.z;
  int channel_index = blockIdx.y;
  if (batch_index >= batch_size || channel_index >= num_channels) return;

  source_grads += batch_index * num_channels * num_sources + channel_index * num_sources;
  target_grads += batch_index * num_channels * num_targets * num_samples + channel_index * num_targets * num_samples;
  indices += batch_index * num_targets * num_samples;

  CUDA_1D_KERNEL_LOOP(target_index, num_targets * num_samples) {
    int source_index = indices[target_index];
    gpuAtomicAdd(source_grads + source_index, target_grads[target_index]);
  }
}

}  // namespace vision3d

#endif
