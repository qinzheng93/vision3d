// Modified from https://github.com/HuguesTHOMAS/KPConv-PyTorch
#ifndef VISION3D_CPU_CLOUD_H_
#define VISION3D_CPU_CLOUD_H_

#include <time.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace vision3d {

class Point {
 public:
  float x, y, z;

  Point() {
    x = 0;
    y = 0;
    z = 0;
  }

  Point(float x0, float y0, float z0) {
    x = x0;
    y = y0;
    z = z0;
  }

  float operator[](int i) const {
    if (i == 0) {
      return x;
    } else if (i == 1) {
      return y;
    } else {
      return z;
    }
  }

  float dot(const Point& P) const { return x * P.x + y * P.y + z * P.z; }

  float sq_norm() { return x * x + y * y + z * z; }

  Point cross(const Point& P) const { return Point(y * P.z - z * P.y, z * P.x - x * P.z, x * P.y - y * P.x); }

  Point& operator+=(const Point& P) {
    x += P.x;
    y += P.y;
    z += P.z;
    return *this;
  }

  Point& operator-=(const Point& P) {
    x -= P.x;
    y -= P.y;
    z -= P.z;
    return *this;
  }

  Point& operator*=(const float& a) {
    x *= a;
    y *= a;
    z *= a;
    return *this;
  }
};

inline Point operator+(const Point& A, const Point& B) { return Point(A.x + B.x, A.y + B.y, A.z + B.z); }

inline Point operator-(const Point& A, const Point& B) { return Point(A.x - B.x, A.y - B.y, A.z - B.z); }

inline Point operator*(const Point& P, const float a) { return Point(P.x * a, P.y * a, P.z * a); }

inline Point operator*(const float a, const Point& P) { return Point(P.x * a, P.y * a, P.z * a); }

inline Point operator/(const Point& P, const float a) { return Point(P.x / a, P.y / a, P.z / a); }

inline std::ostream& operator<<(std::ostream& os, const Point P) {
  return os << "[" << P.x << ", " << P.y << ", " << P.z << "]";
}

inline bool operator==(const Point& A, const Point& B) { return A.x == B.x && A.y == B.y && A.z == B.z; }

inline Point floor(const Point& P) { return Point(std::floor(P.x), std::floor(P.y), std::floor(P.z)); }

Point max_point(const std::vector<Point>& points);

Point min_point(const std::vector<Point>& points);

struct PointCloud {
  std::vector<Point> pts;

  inline size_t kdtree_get_point_count() const { return pts.size(); }

  // Returns the dim'th component of the idx'th point in the class:
  // Since this is inlined and the "dim" argument is typically an immediate
  // value, the
  //  "if/else's" are actually solved at compile time.
  inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (dim == 0) {
      return pts[idx].x;
    } else if (dim == 1) {
      return pts[idx].y;
    } else {
      return pts[idx].z;
    }
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop.
  //   Return true if the BBOX was already computed by the class and returned in
  //   "bb" so it can be avoided to redo it again. Look at bb.size() to find out
  //   the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /* bb */) const {
    return false;
  }
};

}  // namespace vision3d

#endif
