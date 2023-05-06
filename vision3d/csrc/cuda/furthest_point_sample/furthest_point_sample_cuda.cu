#include "furthest_point_sample_cuda.cuh"

namespace vision3d {

void furthest_point_sample_cuda_launcher(
    const at::Tensor& points, at::Tensor& distances, at::Tensor& indices, const int num_samples) {
  CHECK_CUDA_AND_CONTIGUOUS(points);
  CHECK_CUDA_AND_CONTIGUOUS(distances);
  CHECK_CUDA_AND_CONTIGUOUS(indices);
  CHECK_SCALAR_TYPE_FLOAT(points);
  CHECK_SCALAR_TYPE_FLOAT(distances);
  CHECK_SCALAR_TYPE_LONG(indices);

  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int batch_size = points.size(0);
  int num_points = points.size(1);

  const float* points_ptr = points.data_ptr<float>();
  float* distances_ptr = distances.data_ptr<float>();
  long* indices_ptr = indices.data_ptr<long>();

  size_t block_dim = opt_n_threads(num_points);

  switch (block_dim) {
    case 512:
      furthest_point_sample_kernel<512><<<batch_size, block_dim, 0, stream>>>(
          batch_size, num_points, num_samples, points_ptr, distances_ptr, indices_ptr);
      break;
    case 256:
      furthest_point_sample_kernel<256><<<batch_size, block_dim, 0, stream>>>(
          batch_size, num_points, num_samples, points_ptr, distances_ptr, indices_ptr);
      break;
    case 128:
      furthest_point_sample_kernel<128><<<batch_size, block_dim, 0, stream>>>(
          batch_size, num_points, num_samples, points_ptr, distances_ptr, indices_ptr);
      break;
    case 64:
      furthest_point_sample_kernel<64><<<batch_size, block_dim, 0, stream>>>(
          batch_size, num_points, num_samples, points_ptr, distances_ptr, indices_ptr);
      break;
    case 32:
      furthest_point_sample_kernel<32><<<batch_size, block_dim, 0, stream>>>(
          batch_size, num_points, num_samples, points_ptr, distances_ptr, indices_ptr);
      break;
    case 16:
      furthest_point_sample_kernel<16><<<batch_size, block_dim, 0, stream>>>(
          batch_size, num_points, num_samples, points_ptr, distances_ptr, indices_ptr);
      break;
    case 8:
      furthest_point_sample_kernel<8><<<batch_size, block_dim, 0, stream>>>(
          batch_size, num_points, num_samples, points_ptr, distances_ptr, indices_ptr);
      break;
    case 4:
      furthest_point_sample_kernel<4><<<batch_size, block_dim, 0, stream>>>(
          batch_size, num_points, num_samples, points_ptr, distances_ptr, indices_ptr);
      break;
    case 2:
      furthest_point_sample_kernel<2><<<batch_size, block_dim, 0, stream>>>(
          batch_size, num_points, num_samples, points_ptr, distances_ptr, indices_ptr);
      break;
    case 1:
      furthest_point_sample_kernel<1><<<batch_size, block_dim, 0, stream>>>(
          batch_size, num_points, num_samples, points_ptr, distances_ptr, indices_ptr);
      break;
    default:
      furthest_point_sample_kernel<512><<<batch_size, block_dim, 0, stream>>>(
          batch_size, num_points, num_samples, points_ptr, distances_ptr, indices_ptr);
  }

  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vision3d
