#include "deformation_graph.h"

namespace vision3d {

inline int ravel_hash_func(const Voxel &voxel, const Voxel &dimension) {
  return ((voxel.x * dimension.y + voxel.y) * dimension.z + voxel.z) % HASH_BASE;
}

inline float compute_skinning_weight(float distance, float node_coverage) {
  return std::exp(-(distance * distance) / (2. * node_coverage * node_coverage));
}

void build_edges_from_point_cloud_impl(
    const std::vector<Point> &points, std::vector<std::vector<std::pair<int, float>>> &edges, float max_distance) {
  int num_points = points.size();

  // voxelize
  std::vector<Voxel> voxels(num_points);
  for (int i = 0; i < num_points; ++i) {
    voxels[i] = voxelize(points[i], max_distance);
  }

  Voxel origin = min_voxel(voxels);
  for (int i = 0; i < num_points; ++i) {
    voxels[i] -= origin;
  }
  Voxel dimension = max_voxel(voxels) + Voxel(1, 1, 1);

  // compute hash list
  std::vector<int> voxel_id_to_hash_list_id(HASH_BASE, -1);
  std::vector<std::vector<int>> hash_list;
  for (int i = 0; i < num_points; ++i) {
    int voxel_id = ravel_hash_func(voxels[i], dimension);
    if (voxel_id_to_hash_list_id[voxel_id] == -1) {
      hash_list.emplace_back(std::vector<int>());
      voxel_id_to_hash_list_id[voxel_id] = hash_list.size() - 1;
    }
    int hash_list_id = voxel_id_to_hash_list_id[voxel_id];
    hash_list[hash_list_id].push_back(i);
  }

  // compute edges
  for (int i = 0; i < num_points; ++i) {
    for (int vx = -1; vx <= 1; ++vx) {
      int new_x = voxels[i].x + vx;
      if (new_x < 0 || new_x >= dimension.x) continue;
      for (int vy = -1; vy <= 1; ++vy) {
        int new_y = voxels[i].y + vy;
        if (new_y < 0 || new_y >= dimension.y) continue;
        for (int vz = -1; vz <= 1; ++vz) {
          int new_z = voxels[i].z + vz;
          if (new_z < 0 || new_z >= dimension.z) continue;
          Voxel new_voxel(new_x, new_y, new_z);
          int voxel_id = ravel_hash_func(new_voxel, dimension);
          if (voxel_id_to_hash_list_id[voxel_id] == -1) continue;
          int hash_list_id = voxel_id_to_hash_list_id[voxel_id];
          const auto &cur_hash_list = hash_list[hash_list_id];
          int hash_list_size = cur_hash_list.size();
          for (int j = 0; j < hash_list_size; ++j) {
            int k = cur_hash_list[j];
            if (i >= k) continue;
            float distance = std::sqrt((points[i] - points[k]).sq_norm());
            if (distance < max_distance) {
              edges[i].emplace_back(std::make_pair(k, distance));
              edges[k].emplace_back(std::make_pair(i, distance));
            }
          }
        }
      }
    }
  }
}

void build_deformation_graph_from_point_cloud_impl(
    const std::vector<Point> &points,
    const std::vector<int> &node_indices,
    std::vector<int> &neighbor_indices,
    std::vector<float> &neighbor_distances,
    std::vector<float> &neighbor_weights,
    std::vector<int> &anchor_indices,
    std::vector<float> &anchor_distances,
    std::vector<float> &anchor_weights,
    int num_neighbors,
    int num_anchors,
    float max_distance,
    float node_coverage) {
  int num_points = points.size();
  int num_nodes = node_indices.size();

  std::vector<int> point_to_node_indices(num_points, -1);
  for (int i = 0; i < num_nodes; ++i) {
    point_to_node_indices[node_indices[i]] = i;
  }

  std::vector<std::vector<std::pair<int, float>>> edges(num_points);
  build_edges_from_point_cloud_impl(points, edges, max_distance);

  // for each node, run dijkstra
  std::vector<std::vector<std::pair<float, int>>> point_anchors(num_points);
  std::vector<int> timestamp(num_points, -1);
  int neighbor_start_index = 0;
  for (int i = 0; i < num_nodes; ++i) {
    int cur_num_neighbors = 0;

    std::priority_queue<std::pair<float, int>> heap;
    heap.push(std::make_pair(0., node_indices[i]));

    while (!heap.empty()) {
      auto item = heap.top();
      heap.pop();

      int x = item.second;
      float distance = -item.first;

      if (timestamp[x] == i) continue;
      timestamp[x] = i;

      if (point_to_node_indices[x] != -1) {
        // if node, the first one is always itself
        if (cur_num_neighbors < num_neighbors) {
          int node_index = point_to_node_indices[x];
          int cur_neighbor_index = neighbor_start_index + cur_num_neighbors;
          neighbor_indices[cur_neighbor_index] = node_index;
          neighbor_distances[cur_neighbor_index] = distance;
          neighbor_weights[cur_neighbor_index] = compute_skinning_weight(distance, node_coverage);
          ++cur_num_neighbors;
        }
      }

      point_anchors[x].emplace_back(std::make_pair(distance, i));

      const auto &cur_edges = edges[x];
      for (const auto &edge : cur_edges) {
        int y = edge.first;
        if (timestamp[y] == i) continue;
        float next_distance = distance + edge.second;
        if (next_distance > 2. * node_coverage) continue;
        heap.push(std::make_pair(-next_distance, y));
      }
    }

    neighbor_start_index += num_neighbors;
  }

  // for each point, compute anchors
  int anchor_start_index = 0;
  for (int i = 0; i < num_points; ++i) {
    auto &cur_point_anchors = point_anchors[i];
    if (cur_point_anchors.size() == 0) continue;

    std::sort(cur_point_anchors.begin(), cur_point_anchors.end());

    int cur_num_anchors = 0;
    for (const auto &anchor : cur_point_anchors) {
      int cur_anchor_index = anchor_start_index + cur_num_anchors;
      anchor_indices[cur_anchor_index] = anchor.second;
      anchor_distances[cur_anchor_index] = anchor.first;
      anchor_weights[cur_anchor_index] = compute_skinning_weight(anchor.first, node_coverage);
      ++cur_num_anchors;
      if (cur_num_anchors == num_anchors) break;
    }

    // normalize
    float weight_sum = 0;
    for (int j = 0; j < cur_num_anchors; ++j) {
      weight_sum += anchor_weights[anchor_start_index + j];
    }
    if (weight_sum > 0) {
      for (int j = 0; j < cur_num_anchors; ++j) {
        anchor_weights[anchor_start_index + j] /= weight_sum;
      }
    } else {
      for (int j = 0; j < cur_num_anchors; ++j) {
        anchor_weights[anchor_start_index + j] = 1. / static_cast<float>(cur_num_anchors);
      }
    }

    anchor_start_index += num_anchors;
  }
}

std::tuple<
    py::array_t<long>,
    py::array_t<float>,
    py::array_t<float>,
    py::array_t<long>,
    py::array_t<float>,
    py::array_t<float>>
build_deformation_graph_from_point_cloud(
    const py::array_t<float> &points,
    const py::array_t<long> &node_indices,
    int num_neighbors,
    int num_anchors,
    float max_distance,
    float node_coverage) {
  int num_points = points.shape(0);
  int num_nodes = node_indices.shape(0);

  // read array
  std::vector<Point> points_vec(num_points);
  for (int i = 0; i < num_points; ++i) {
    points_vec[i].x = *points.data(i, 0);
    points_vec[i].y = *points.data(i, 1);
    points_vec[i].z = *points.data(i, 2);
  }

  std::vector<int> node_indices_vec(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    node_indices_vec[i] = *node_indices.data(i);
  }

  // build deformation graph
  int total_neighbor_size = num_nodes * num_neighbors;
  std::vector<int> neighbor_indices_vec(total_neighbor_size, -1);
  std::vector<float> neighbor_distances_vec(total_neighbor_size, 0.);
  std::vector<float> neighbor_weights_vec(total_neighbor_size, 0.);

  int total_anchor_size = num_points * num_anchors;
  std::vector<int> anchor_indices_vec(total_anchor_size, -1);
  std::vector<float> anchor_distances_vec(total_anchor_size, 0.);
  std::vector<float> anchor_weights_vec(total_anchor_size, 0.);

  build_deformation_graph_from_point_cloud_impl(
      points_vec,
      node_indices_vec,
      neighbor_indices_vec,
      neighbor_distances_vec,
      neighbor_weights_vec,
      anchor_indices_vec,
      anchor_distances_vec,
      anchor_weights_vec,
      num_neighbors,
      num_anchors,
      max_distance,
      node_coverage);

  // write array
  py::array_t<long> neighbor_indices = py::array_t<long>({num_nodes, num_neighbors});
  py::array_t<float> neighbor_distances = py::array_t<float>({num_nodes, num_neighbors});
  py::array_t<float> neighbor_weights = py::array_t<float>({num_nodes, num_neighbors});

  for (int i = 0; i < num_nodes; ++i) {
    for (int j = 0; j < num_neighbors; ++j) {
      int index = i * num_neighbors + j;
      *neighbor_indices.mutable_data(i, j) = neighbor_indices_vec[index];
      *neighbor_distances.mutable_data(i, j) = neighbor_distances_vec[index];
      *neighbor_weights.mutable_data(i, j) = neighbor_weights_vec[index];
    }
  }

  py::array_t<long> anchor_indices = py::array_t<long>({num_points, num_anchors});
  py::array_t<float> anchor_distances = py::array_t<float>({num_points, num_anchors});
  py::array_t<float> anchor_weights = py::array_t<float>({num_points, num_anchors});

  for (int i = 0; i < num_points; ++i) {
    for (int j = 0; j < num_anchors; ++j) {
      int index = i * num_anchors + j;
      *anchor_indices.mutable_data(i, j) = anchor_indices_vec[index];
      *anchor_distances.mutable_data(i, j) = anchor_distances_vec[index];
      *anchor_weights.mutable_data(i, j) = anchor_weights_vec[index];
    }
  }

  return {neighbor_indices, neighbor_distances, neighbor_weights, anchor_indices, anchor_distances, anchor_weights};
}

}  // namespace vision3d
