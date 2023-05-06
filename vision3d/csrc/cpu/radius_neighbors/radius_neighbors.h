#ifndef VISION3D_CPU_RADIUS_NEIGHBORS_H_
#define VISION3D_CPU_RADIUS_NEIGHBORS_H_

#include <nanoflann.hpp>
#include <tuple>
#include <vector>

#include "../cloud/cloud.h"
#include "numpy_helper.h"
#include "torch_helper.h"

namespace vision3d {

typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3> kd_tree_t;

py::array_t<long> radius_neighbors(
    const py::array_t<float>& q_points,
    const py::array_t<float>& s_points,
    const py::array_t<long>& q_lengths,
    const py::array_t<long>& s_lengths,
    float radius);

}  // namespace vision3d

#endif
