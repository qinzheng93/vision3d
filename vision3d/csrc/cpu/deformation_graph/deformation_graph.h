#ifndef VISION3D_CPU_DEFORMATION_GRAPH_H_
#define VISION3D_CPU_DEFORMATION_GRAPH_H_

#include <climits>
#include <cmath>
#include <map>
#include <queue>
#include <set>
#include <tuple>
#include <vector>

#include "../cloud/cloud.h"
#include "../voxel/voxel.h"
#include "numpy_helper.h"
#include "torch_helper.h"

namespace vision3d {

const int HASH_BASE = 19997;

std::tuple<
    py::array_t<long>,
    py::array_t<float>,
    py::array_t<float>,
    py::array_t<long>,
    py::array_t<float>,
    py::array_t<float>>
build_deformation_graph_from_point_cloud(
    const py::array_t<float>& vertices,
    const py::array_t<long>& node_indices,
    int num_neighbors,
    int num_anchors,
    float max_distance,
    float node_coverage);

}  // namespace vision3d

#endif
