#include "voxel.h"

namespace vision3d {

Voxel min_voxel(const std::vector<Voxel>& voxels) {
  Voxel result(INT_MAX, INT_MAX, INT_MAX);
  for (const auto& voxel : voxels) {
    if (voxel.x < result.x) {
      result.x = voxel.x;
    }
    if (voxel.y < result.y) {
      result.y = voxel.y;
    }
    if (voxel.z < result.z) {
      result.z = voxel.z;
    }
  }
  return result;
}

Voxel max_voxel(const std::vector<Voxel>& voxels) {
  Voxel result(INT_MIN, INT_MIN, INT_MIN);
  for (const auto& voxel : voxels) {
    if (voxel.x > result.x) {
      result.x = voxel.x;
    }
    if (voxel.y > result.y) {
      result.y = voxel.y;
    }
    if (voxel.z > result.z) {
      result.z = voxel.z;
    }
  }
  return result;
}

Voxel voxelize(const Point& point, float voxel_size) {
  int x = static_cast<int>(std::floor(point.x / voxel_size));
  int y = static_cast<int>(std::floor(point.y / voxel_size));
  int z = static_cast<int>(std::floor(point.z / voxel_size));
  return Voxel(x, y, z);
}

}  // namespace vision3d
