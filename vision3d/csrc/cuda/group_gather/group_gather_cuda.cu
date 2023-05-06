#include "group_gather_cuda.cuh"

namespace vision3d {

void group_gather_forward_cuda_launcher(const at::Tensor& sources, const at::Tensor& indices, at::Tensor& targets) {
  CHECK_CUDA_AND_CONTIGUOUS(sources);
  CHECK_CUDA_AND_CONTIGUOUS(indices);
  CHECK_CUDA_AND_CONTIGUOUS(targets);
  CHECK_SCALAR_TYPE_LONG(indices);

  at::cuda::CUDAGuard device_guard(sources.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int batch_size = sources.size(0);
  int num_channels = sources.size(1);
  int num_sources = sources.size(2);
  int num_targets = indices.size(1);
  int num_samples = indices.size(2);

  dim3 grid_dim(GET_BLOCKS(num_targets * num_samples, THREADS_PER_BLOCK), num_channels, batch_size);
  dim3 block_dim(THREADS_PER_BLOCK);

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool, sources.scalar_type(), "group_gather_forward_kernel", [&] {
    group_gather_forward_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
        batch_size,
        num_channels,
        num_sources,
        num_targets,
        num_samples,
        sources.data_ptr<scalar_t>(),
        indices.data_ptr<long>(),
        targets.data_ptr<scalar_t>());
  });

  AT_CUDA_CHECK(cudaGetLastError());
}

void group_gather_backward_cuda_launcher(
    const at::Tensor& target_grads, const at::Tensor& indices, at::Tensor& source_grads, const int num_sources) {
  CHECK_CUDA_AND_CONTIGUOUS(target_grads);
  CHECK_CUDA_AND_CONTIGUOUS(indices);
  CHECK_CUDA_AND_CONTIGUOUS(source_grads);
  CHECK_SCALAR_TYPE_LONG(indices);

  at::cuda::CUDAGuard device_guard(target_grads.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int batch_size = target_grads.size(0);
  int num_channels = target_grads.size(1);
  int num_targets = target_grads.size(2);
  int num_samples = target_grads.size(3);

  dim3 grid_dim(GET_BLOCKS(num_targets * num_samples, THREADS_PER_BLOCK), num_channels, batch_size);
  dim3 block_dim(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES(target_grads.scalar_type(), "group_gather_backward_kernel", [&] {
    group_gather_backward_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
        batch_size,
        num_channels,
        num_sources,
        num_targets,
        num_samples,
        target_grads.data_ptr<scalar_t>(),
        indices.data_ptr<long>(),
        source_grads.data_ptr<scalar_t>());
  });

  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vision3d
