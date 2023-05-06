#ifndef VISION3D_CUDA_THREE_INTERPOLATE_CUDA_CUH_
#define VISION3D_CUDA_THREE_INTERPOLATE_CUDA_CUH_

#include "cuda_helper.h"
#include "torch_helper.h"

namespace vision3d {

// input: points(b, c, m), indices(b, n, 3), weight(b, n, 3)
// output: out(b, c, n)
template <typename T>
__global__ void three_interpolate_forward_kernel(
    int batch_size,
    int num_channels,
    int num_sources,
    int num_targets,
    const T* __restrict__ sources,
    const long* __restrict__ indices,
    const T* __restrict__ weights,
    T* targets) {
  int batch_index = blockIdx.z;
  int channel_index = blockIdx.y;

  if (batch_index >= batch_size || channel_index >= num_channels) return;

  sources += batch_index * num_channels * num_sources + channel_index * num_sources;
  indices += batch_index * num_targets * 3;
  weights += batch_index * num_targets * 3;
  targets += batch_index * num_channels * num_targets + channel_index * num_targets;

  CUDA_1D_KERNEL_LOOP(t_index, num_targets) {
    const long* cur_indices = indices + t_index * 3;
    const T* cur_weights = weights + t_index * 3;

    T w1 = cur_weights[0];
    T w2 = cur_weights[1];
    T w3 = cur_weights[2];
    int s_index1 = cur_indices[0];
    int s_index2 = cur_indices[1];
    int s_index3 = cur_indices[2];

    targets[t_index] = sources[s_index1] * w1 + sources[s_index2] * w2 + sources[s_index3] * w3;
  }
}

// input: grad_out(b, c, n), indices(b, n, 3), weight(b, n, 3)
// output: grad_points(b, c, m)
template <typename T>
__global__ void three_interpolate_backward_kernel(
    int batch_size,
    int num_channels,
    int num_sources,
    int num_targets,
    const T* __restrict__ target_grads,
    const long* __restrict__ indices,
    const T* __restrict__ weights,
    T* source_grads) {
  int batch_index = blockIdx.z;
  int channel_index = blockIdx.y;

  if (batch_index >= batch_size || channel_index >= num_channels) return;

  source_grads += batch_index * num_channels * num_sources + channel_index * num_sources;
  indices += batch_index * num_targets * 3;
  weights += batch_index * num_targets * 3;
  target_grads += batch_index * num_channels * num_targets + channel_index * num_targets;

  CUDA_1D_KERNEL_LOOP(t_index, num_targets) {
    const long* cur_indices = indices + t_index * 3;
    const T* cur_weights = weights + t_index * 3;

    T w1 = cur_weights[0];
    T w2 = cur_weights[1];
    T w3 = cur_weights[2];
    int s_index1 = cur_indices[0];
    int s_index2 = cur_indices[1];
    int s_index3 = cur_indices[2];

    gpuAtomicAdd(source_grads + s_index1, target_grads[t_index] * w1);
    gpuAtomicAdd(source_grads + s_index2, target_grads[t_index] * w2);
    gpuAtomicAdd(source_grads + s_index3, target_grads[t_index] * w3);
  }
}

}  // namespace vision3d

#endif
