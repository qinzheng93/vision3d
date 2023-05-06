#ifndef VISION3D_CPU_NODE_SAMPLING_H_
#define VISION3D_CPU_NODE_SAMPLING_H_

#include <tuple>
#include <vector>

#include "../cloud/cloud.h"
#include "numpy_helper.h"
#include "torch_helper.h"

namespace vision3d {

py::array_t<long> sample_nodes_with_fps(
    const py::array_t<float>& points, const float min_distance, const int num_samples);

}  // namespace vision3d

#endif
