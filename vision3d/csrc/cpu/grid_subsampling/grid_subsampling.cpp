#include "grid_subsampling.h"

namespace vision3d {

std::tuple<py::array_t<float>, py::array_t<long>> grid_subsampling(
    const py::array_t<float>& points, const py::array_t<long>& lengths, float voxel_size) {
  std::size_t batch_size = lengths.shape(0);
  std::size_t total_points = points.shape(0);
  std::size_t num_channels = points.shape(1);

  // prepare sample points, use slightly more memory, but it is fine.
  std::vector<Point> s_points_vec;
  std::vector<long> s_lengths_vec;
  s_points_vec.reserve(total_points);
  s_lengths_vec.reserve(batch_size);

  // grid subsampling
  std::size_t start_index = 0;
  for (std::size_t batch_index = 0; batch_index < batch_size; batch_index++) {
    // read array
    std::size_t cur_length = *lengths.data(batch_index);
    std::vector<Point> cur_points_vec(cur_length);
    for (std::size_t i = 0; i < cur_length; ++i) {
      cur_points_vec[i].x = *points.data(start_index + i, 0);
      cur_points_vec[i].y = *points.data(start_index + i, 1);
      cur_points_vec[i].z = *points.data(start_index + i, 2);
    }

    // grid subsample
    Point min_corner = min_point(cur_points_vec);
    Point max_corner = max_point(cur_points_vec);
    Point origin = floor(min_corner / voxel_size) * voxel_size;

    auto dim_x = static_cast<std::size_t>(std::floor((max_corner.x - origin.x) / voxel_size) + 1.);
    auto dim_y = static_cast<std::size_t>(std::floor((max_corner.y - origin.y) / voxel_size) + 1.);

    std::unordered_map<std::size_t, SampledData> data;
    for (const auto& point : cur_points_vec) {
      auto vx = static_cast<std::size_t>(std::floor((point.x - origin.x) / voxel_size));
      auto vy = static_cast<std::size_t>(std::floor((point.y - origin.y) / voxel_size));
      auto vz = static_cast<std::size_t>(std::floor((point.z - origin.z) / voxel_size));
      auto data_index = vx + dim_x * vy + dim_x * dim_y * vz;

      if (data.find(data_index) == data.end()) {
        data.emplace(data_index, SampledData());
      }

      data[data_index].update(point);
    }

    for (const auto& v : data) {
      s_points_vec.push_back(v.second.point * (1.0 / v.second.count));
    }
    s_lengths_vec.push_back(data.size());

    // increment start index
    start_index += cur_length;
  }

  // write array
  std::size_t total_s_points = s_points_vec.size();
  py::array_t<float> s_points = py::array_t<float>({total_s_points, num_channels});
  for (std::size_t i = 0; i < total_s_points; ++i) {
    *s_points.mutable_data(i, 0) = s_points_vec[i].x;
    *s_points.mutable_data(i, 1) = s_points_vec[i].y;
    *s_points.mutable_data(i, 2) = s_points_vec[i].z;
  }

  py::array_t<long> s_lengths = py::array_t<long>({batch_size});
  for (std::size_t i = 0; i < batch_size; ++i) {
    *s_lengths.mutable_data(i) = s_lengths_vec[i];
  }

  return {s_points, s_lengths};
}

}  // namespace vision3d
