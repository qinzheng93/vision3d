#ifndef VISION3D_CPU_VOXEL_H_
#define VISION3D_CPU_VOXEL_H_

#include <climits>
#include <cmath>
#include <vector>

#include "../cloud/cloud.h"

namespace vision3d {

class Voxel {
 public:
  int x, y, z;

  Voxel() : x(0), y(0), z(0) {}
  Voxel(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}

  Voxel& operator-=(const Voxel& other) {
    this->x -= other.x;
    this->y -= other.y;
    this->z -= other.z;
    return *this;
  }

  Voxel& operator+=(const Voxel& other) {
    this->x += other.x;
    this->y += other.y;
    this->z += other.z;
    return *this;
  }
};

inline Voxel operator+(const Voxel& A, const Voxel& B) { return Voxel(A.x + B.x, A.y + B.y, A.z + B.z); }

inline Voxel operator-(const Voxel& A, const Voxel& B) { return Voxel(A.x - B.x, A.y - B.y, A.z - B.z); }

Voxel min_voxel(const std::vector<Voxel>& voxels);

Voxel max_voxel(const std::vector<Voxel>& voxels);

Voxel voxelize(const Point& point, float voxel_size);

}  // namespace vision3d

#endif
