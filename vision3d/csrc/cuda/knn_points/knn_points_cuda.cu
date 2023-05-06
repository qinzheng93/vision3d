#include "knn_points_cuda.cuh"

namespace vision3d {

void knn_points_cuda_launcher(
    const at::Tensor& q_points,
    const at::Tensor& s_points,
    at::Tensor& knn_distances,
    at::Tensor& knn_indices,
    int num_neighbors) {
  CHECK_CUDA_AND_CONTIGUOUS(q_points);
  CHECK_CUDA_AND_CONTIGUOUS(s_points);
  CHECK_CUDA_AND_CONTIGUOUS(knn_distances);
  CHECK_CUDA_AND_CONTIGUOUS(knn_indices);
  CHECK_SCALAR_TYPE_LONG(knn_indices);

  at::cuda::CUDAGuard device_guard(q_points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int batch_size = q_points.size(0);
  int num_q_points = q_points.size(1);
  int num_s_points = s_points.size(1);

  dim3 grid_dim(GET_BLOCKS(num_q_points, THREADS_PER_BLOCK), batch_size);
  dim3 block_dim(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES(q_points.scalar_type(), "knn_points_kernel", [&] {
    knn_points_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
        batch_size,
        num_q_points,
        num_s_points,
        num_neighbors,
        q_points.data_ptr<scalar_t>(),
        s_points.data_ptr<scalar_t>(),
        knn_distances.data_ptr<scalar_t>(),
        knn_indices.data_ptr<long>());
  });

  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vision3d
