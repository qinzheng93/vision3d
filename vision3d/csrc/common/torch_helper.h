#ifndef VISION3D_COMMON_TORCH_HELPER_H_
#define VISION3D_COMMON_TORCH_HELPER_H_

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace vision3d {

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CPU(x) TORCH_CHECK(!x.device().is_cuda(), #x " must be a CPU tensor")

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_CUDA_AND_CONTIGUOUS(x) \
  CHECK_CUDA(x);                     \
  CHECK_CONTIGUOUS(x)

#define CHECK_SCALAR_TYPE_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")

#define CHECK_SCALAR_TYPE_LONG(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Long, #x " must be an long tensor")

#define CHECK_SCALAR_TYPE_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be a float tensor")

}  // namespace vision3d

#endif
