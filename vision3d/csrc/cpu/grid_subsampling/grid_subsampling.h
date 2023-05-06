#ifndef VISION3D_CPU_GRID_SUBSAMPLING_H_
#define VISION3D_CPU_GRID_SUBSAMPLING_H_

#include <tuple>
#include <vector>

#include "../cloud/cloud.h"
#include "numpy_helper.h"
#include "torch_helper.h"

namespace vision3d {

class SampledData {
 public:
  int count;
  Point point;

  SampledData() {
    count = 0;
    point = Point();
  }

  void update(const Point& p) {
    count += 1;
    point += p;
  }
};

std::tuple<py::array_t<float>, py::array_t<long>> grid_subsampling(
    const py::array_t<float>& points, const py::array_t<long>& lengths, float voxel_size);

}  // namespace vision3d

#endif