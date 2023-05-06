#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "cpu/deformation_graph/deformation_graph.h"
#include "cpu/graph_proc/graph_proc.h"
#include "cpu/grid_subsampling/grid_subsampling.h"
#include "cpu/node_sampling/node_sampling.h"
#include "cpu/radius_neighbors/radius_neighbors.h"
#include "cuda/ball_query/ball_query.h"
#include "cuda/furthest_point_sample/furthest_point_sample.h"
#include "cuda/gather/gather.h"
#include "cuda/group_gather/group_gather.h"
#include "cuda/knn_points/knn_points.h"
#include "cuda/three_interpolate/three_interpolate.h"
#include "cuda/three_nn/three_nn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // NumPy Extensions
  m.def("depth_to_mesh", &graph_proc::depthToMesh, "Convert Depth Image to Mesh");
  m.def("erode_mesh", &graph_proc::erode_mesh, "Erode Mesh");
  m.def("sample_nodes", &graph_proc::sample_nodes, "Sample Graph Nodes");
  m.def("compute_edges_geodesic", &graph_proc::compute_edges_geodesic, "Compute Graph Edges via Geodesic Distance");
  m.def("compute_edges_euclidean", &graph_proc::compute_edges_euclidean, "Compute Graph Edges via Euclidean Distance");
  m.def(
      "compute_pixel_anchors_geodesic",
      &graph_proc::compute_pixel_anchors_geodesic,
      "Compute Pixel Anchors via Geodesic Distance");
  m.def("node_and_edge_clean_up", &graph_proc::node_and_edge_clean_up, "Remove Nodes and Edges with Too Few Neighbors");
  m.def("update_pixel_anchors", &graph_proc::update_pixel_anchors, "Update Pixel Anchor IDs");
  m.def("compute_clusters", &graph_proc::compute_clusters, "Compute Graph Clusters");

  m.def(
      "radius_neighbors",
      &vision3d::radius_neighbors,
      "Radius Neighbors",
      py::arg("q_points"),
      py::arg("s_points"),
      py::arg("q_lengths"),
      py::arg("s_lenghts"),
      py::arg("radius"));

  m.def(
      "grid_subsampling",
      &vision3d::grid_subsampling,
      "Grid Subsampling",
      py::arg("points"),
      py::arg("lengths"),
      py::arg("voxel_size"));

  m.def(
      "build_deformation_graph_from_point_cloud",
      &vision3d::build_deformation_graph_from_point_cloud,
      "Build Deformation Graph from Point Cloud",
      py::arg("vertices"),
      py::arg("node_indices"),
      py::arg("num_neighbors"),
      py::arg("num_anchors"),
      py::arg("max_distance"),
      py::arg("node_coverage"));

  m.def(
      "sample_nodes_with_fps",
      &vision3d::sample_nodes_with_fps,
      "Sample Nodes with Furthest Point Sampling.",
      py::arg("points"),
      py::arg("min_distance"),
      py::arg("num_samples"));

  // CUDA extensions
  m.def(
      "ball_query",
      &vision3d::ball_query,
      "Ball Query",
      py::arg("q_points"),
      py::arg("s_points"),
      py::arg("indices"),
      py::arg("max_radius"),
      py::arg("num_samples"));

  m.def(
      "group_gather_forward",
      &vision3d::group_gather_forward,
      "Group Gather Points By Index (Forward)",
      py::arg("sources"),
      py::arg("indices"),
      py::arg("targets"));
  m.def(
      "group_gather_backward",
      &vision3d::group_gather_backward,
      "Group Gather Points By Index (Backward)",
      py::arg("target_grads"),
      py::arg("indices"),
      py::arg("source_grads"),
      py::arg("num_sources"));

  m.def(
      "furthest_point_sample",
      &vision3d::furthest_point_sample,
      "Furthest Point Sampling",
      py::arg("points"),
      py::arg("distances"),
      py::arg("indices"),
      py::arg("num_samples"));

  m.def(
      "gather_forward",
      &vision3d::gather_forward,
      "Gather Points By Index (Forward)",
      py::arg("sources"),
      py::arg("indices"),
      py::arg("targets"));
  m.def(
      "gather_backward",
      &vision3d::gather_backward,
      "Gather Points By Index (Backward)",
      py::arg("target_grads"),
      py::arg("indices"),
      py::arg("source_grads"),
      py::arg("num_sources"));

  m.def(
      "three_nn",
      &vision3d::three_nn,
      "Three Nearest Neighbors",
      py::arg("q_points"),
      py::arg("s_points"),
      py::arg("tnn_distances"),
      py::arg("tnn_indices"));

  m.def(
      "three_interpolate_forward",
      &vision3d::three_interpolate_forward,
      "Three Nearest Neighbors Interpolate (Forward)",
      py::arg("sources"),
      py::arg("indices"),
      py::arg("weights"),
      py::arg("targets"));
  m.def(
      "three_interpolate_backward",
      &vision3d::three_interpolate_backward,
      "Three Nearest Neighbors Interpolate (Backward)",
      py::arg("target_grads"),
      py::arg("indices"),
      py::arg("weights"),
      py::arg("source_grads"),
      py::arg("num_sources"));

  m.def(
      "knn_points",
      &vision3d::knn_points,
      "k Nearest Neighbors",
      py::arg("q_points"),
      py::arg("s_points"),
      py::arg("knn_distances"),
      py::arg("knn_indices"),
      py::arg("num_neighbors"));
}
