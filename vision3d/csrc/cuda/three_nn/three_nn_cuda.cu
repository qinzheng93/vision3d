#include "three_nn_cuda.cuh"

namespace vision3d {

void three_nn_cuda_launcher(
    const at::Tensor& q_points, const at::Tensor& s_points, at::Tensor& tnn_distances, at::Tensor& tnn_indices) {
  CHECK_CUDA_AND_CONTIGUOUS(q_points);
  CHECK_CUDA_AND_CONTIGUOUS(s_points);
  CHECK_CUDA_AND_CONTIGUOUS(tnn_distances);
  CHECK_CUDA_AND_CONTIGUOUS(tnn_indices);
  CHECK_SCALAR_TYPE_LONG(tnn_indices);

  at::cuda::CUDAGuard device_guard(q_points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int batch_size = q_points.size(0);
  int num_q_points = q_points.size(1);
  int num_s_points = s_points.size(1);

  dim3 grid_dim(GET_BLOCKS(num_q_points, THREADS_PER_BLOCK), batch_size);
  dim3 block_dim(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES(q_points.scalar_type(), "three_nn_kernel", [&] {
    three_nn_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
        batch_size,
        num_q_points,
        num_s_points,
        q_points.data_ptr<scalar_t>(),
        s_points.data_ptr<scalar_t>(),
        tnn_distances.data_ptr<scalar_t>(),
        tnn_indices.data_ptr<long>());
  });

  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vision3d
