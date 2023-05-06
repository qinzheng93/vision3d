#include "node_sampling.h"

namespace vision3d {

int pop(std::vector<int>& data, const int num_data, const int index) {
  std::swap(data[index], data[num_data - 1]);
  return num_data - 1;
}

py::array_t<long> sample_nodes_with_fps(
    const py::array_t<float>& points, const float min_distance, const int num_samples) {
  int num_points = points.shape(0);

  // read arary
  std::vector<Point> points_vec(num_points);
  for (int i = 0; i < num_points; ++i) {
    points_vec[i].x = *points.data(i, 0);
    points_vec[i].y = *points.data(i, 1);
    points_vec[i].z = *points.data(i, 2);
  }

  // furthest point sampling
  std::vector<int> selected_indices;

  std::vector<int> unselected_indices(num_points);
  for (int i = 0; i < num_points; ++i) {
    unselected_indices[i] = i;
  }
  int num_unselected_indices = num_points;

  std::vector<float> distances(num_points);
  for (int i = 0; i < num_points; ++i) {
    distances[i] = 1e40;
  }
  distances[0] = 0.;

  int best_index = 0;
  float best_distance = 0.;
  while (1) {
    int cur_point_index = unselected_indices[best_index];
    selected_indices.push_back(cur_point_index);
    num_unselected_indices = pop(unselected_indices, num_unselected_indices, best_index);

    if (num_samples > 0 && selected_indices.size() >= num_samples) break;

    best_index = -1;
    best_distance = 0.;
    int index = 0;
    while (index < num_unselected_indices) {
      int point_index = unselected_indices[index];
      float distance = std::sqrt((points_vec[cur_point_index] - points_vec[point_index]).sq_norm());
      distance = std::min(distances[point_index], distance);
      distances[point_index] = distance;

      if (distance > best_distance) {
        best_index = index;
        best_distance = distance;
      }

      if (distance < min_distance) {
        num_unselected_indices = pop(unselected_indices, num_unselected_indices, index);
      } else {
        ++index;
      }
    }

    if (best_distance < min_distance) break;
  }

  // write array
  int num_nodes = selected_indices.size();
  py::array_t<long> node_indices = py::array_t<long>({num_nodes});
  for (int i = 0; i < num_nodes; ++i) {
    *node_indices.mutable_data(i) = selected_indices[i];
  }

  return node_indices;
}

}  // namespace vision3d
