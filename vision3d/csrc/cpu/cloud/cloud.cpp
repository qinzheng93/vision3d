// Modified from https://github.com/HuguesTHOMAS/KPConv-PyTorch
#include "cloud.h"

namespace vision3d {

Point max_point(const std::vector<Point>& points) {
  Point maxP(points[0]);

  for (const auto& p : points) {
    if (p.x > maxP.x) {
      maxP.x = p.x;
    }
    if (p.y > maxP.y) {
      maxP.y = p.y;
    }
    if (p.z > maxP.z) {
      maxP.z = p.z;
    }
  }

  return maxP;
}

Point min_point(const std::vector<Point>& points) {
  Point minP(points[0]);

  for (const auto& p : points) {
    if (p.x < minP.x) {
      minP.x = p.x;
    }
    if (p.y < minP.y) {
      minP.y = p.y;
    }
    if (p.z < minP.z) {
      minP.z = p.z;
    }
  }

  return minP;
}

}  // namespace vision3d
