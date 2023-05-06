#include "graph_proc.h"

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <numeric>  //std::iota
#include <random>
#include <set>
#include <vector>

using std::vector;

namespace graph_proc {

std::tuple<py::array_t<float>, py::array_t<int>, py::array_t<int>> depthToMesh(
    const py::array_t<float>& pointImage, float maxTriangleEdgeDistance) {
  int height = pointImage.shape(0);
  int width = pointImage.shape(1);

  // Compute valid pixel vertices and faces.
  // We also need to compute the pixel -> vertex index mapping for
  // computation of faces.
  // We connect neighboring pixels on the square into two triangles.
  // We only select valid triangles, i.e. with all valid vertices and
  // not too far apart.
  // Important: The triangle orientation is set such that the normals
  // point towards the camera.
  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3i> faces;
  std::vector<Eigen::Vector2i> pixels;

  int vertexIdx = 0;
  std::vector<int> mapPixelToVertexIdx(width * height, -1);

  for (int y = 0; y < height - 1; y++) {
    for (int x = 0; x < width - 1; x++) {
      // clang-format off
      Eigen::Vector3f obs00(
          *pointImage.data(y, x, 0),
          *pointImage.data(y, x, 1),
          *pointImage.data(y, x, 2));
      Eigen::Vector3f obs01(
          *pointImage.data(y + 1, x, 0),
          *pointImage.data(y + 1, x, 1),
          *pointImage.data(y + 1, x, 2));
      Eigen::Vector3f obs10(
          *pointImage.data(y, x + 1, 0),
          *pointImage.data(y, x + 1, 1),
          *pointImage.data(y, x + 1, 2));
      Eigen::Vector3f obs11(
          *pointImage.data(y + 1, x + 1, 0),
          *pointImage.data(y + 1, x + 1, 1),
          *pointImage.data(y + 1, x + 1, 2));
      // clang-format on

      int idx00 = y * width + x;
      int idx01 = (y + 1) * width + x;
      int idx10 = y * width + (x + 1);
      int idx11 = (y + 1) * width + (x + 1);

      bool valid00 = obs00.z() > 0;
      bool valid01 = obs01.z() > 0;
      bool valid10 = obs10.z() > 0;
      bool valid11 = obs11.z() > 0;

      if (valid00 && valid01 && valid10) {
        float d0 = (obs00 - obs01).norm();
        float d1 = (obs00 - obs10).norm();
        float d2 = (obs01 - obs10).norm();

        if (d0 <= maxTriangleEdgeDistance && d1 <= maxTriangleEdgeDistance && d2 <= maxTriangleEdgeDistance) {
          int vIdx0 = mapPixelToVertexIdx[idx00];
          int vIdx1 = mapPixelToVertexIdx[idx01];
          int vIdx2 = mapPixelToVertexIdx[idx10];

          if (vIdx0 == -1) {
            vIdx0 = vertexIdx;
            mapPixelToVertexIdx[idx00] = vertexIdx;
            vertices.push_back(obs00);
            pixels.push_back(Eigen::Vector2i(x, y));
            vertexIdx++;
          }
          if (vIdx1 == -1) {
            vIdx1 = vertexIdx;
            mapPixelToVertexIdx[idx01] = vertexIdx;
            vertices.push_back(obs01);
            pixels.push_back(Eigen::Vector2i(x, y + 1));
            vertexIdx++;
          }
          if (vIdx2 == -1) {
            vIdx2 = vertexIdx;
            mapPixelToVertexIdx[idx10] = vertexIdx;
            vertices.push_back(obs10);
            pixels.push_back(Eigen::Vector2i(x + 1, y));
            vertexIdx++;
          }

          faces.push_back(Eigen::Vector3i(vIdx0, vIdx1, vIdx2));
        }
      }

      if (valid01 && valid10 && valid11) {
        float d0 = (obs10 - obs01).norm();
        float d1 = (obs10 - obs11).norm();
        float d2 = (obs01 - obs11).norm();

        if (d0 <= maxTriangleEdgeDistance && d1 <= maxTriangleEdgeDistance && d2 <= maxTriangleEdgeDistance) {
          int vIdx0 = mapPixelToVertexIdx[idx11];
          int vIdx1 = mapPixelToVertexIdx[idx10];
          int vIdx2 = mapPixelToVertexIdx[idx01];

          if (vIdx0 == -1) {
            vIdx0 = vertexIdx;
            mapPixelToVertexIdx[idx11] = vertexIdx;
            vertices.push_back(obs11);
            pixels.push_back(Eigen::Vector2i(x + 1, y + 1));
            vertexIdx++;
          }
          if (vIdx1 == -1) {
            vIdx1 = vertexIdx;
            mapPixelToVertexIdx[idx10] = vertexIdx;
            vertices.push_back(obs10);
            pixels.push_back(Eigen::Vector2i(x + 1, y));
            vertexIdx++;
          }
          if (vIdx2 == -1) {
            vIdx2 = vertexIdx;
            mapPixelToVertexIdx[idx01] = vertexIdx;
            vertices.push_back(obs01);
            pixels.push_back(Eigen::Vector2i(x, y + 1));
            vertexIdx++;
          }

          faces.push_back(Eigen::Vector3i(vIdx0, vIdx1, vIdx2));
        }
      }
    }
  }

  // Convert to numpy array.
  int nVertices = vertices.size();
  int nFaces = faces.size();

  py::array_t<float> vertexPositions = py::array_t<float>({nVertices, 3});
  py::array_t<int> vertexPixels = py::array_t<int>({nVertices, 2});
  py::array_t<int> faceIndices = py::array_t<int>({nFaces, 3});

  if (nVertices > 0 && nFaces > 0) {
    for (int i = 0; i < nVertices; i++) {
      *vertexPositions.mutable_data(i, 0) = vertices[i].x();
      *vertexPositions.mutable_data(i, 1) = vertices[i].y();
      *vertexPositions.mutable_data(i, 2) = vertices[i].z();

      *vertexPixels.mutable_data(i, 0) = pixels[i].x();
      *vertexPixels.mutable_data(i, 1) = pixels[i].y();
    }

    for (int i = 0; i < nFaces; i++) {
      *faceIndices.mutable_data(i, 0) = faces[i].x();
      *faceIndices.mutable_data(i, 1) = faces[i].y();
      *faceIndices.mutable_data(i, 2) = faces[i].z();
    }
  }

  return std::make_tuple(vertexPositions, faceIndices, vertexPixels);
}

void erode_mesh(
    const py::array_t<float>& vertexPositions,
    const py::array_t<int>& faceIndices,
    py::array_t<bool>& nonErodedVertices,
    int nIterations,
    int minNeighbors) {
  int nVertices = vertexPositions.shape(0);
  int nFaces = faceIndices.shape(0);

  // Init output
  std::vector<bool> nonErodedVerticesVec(nVertices, false);

  // Init list of eroded face indices with original list
  std::vector<Eigen::Vector3i> erodedFaceIndicesVec;
  erodedFaceIndicesVec.reserve(nFaces);
  for (int FaceIdx = 0; FaceIdx < nFaces; ++FaceIdx) {
    // clang-format off
    Eigen::Vector3i face(
        *faceIndices.data(FaceIdx, 0),
        *faceIndices.data(FaceIdx, 1),
        *faceIndices.data(FaceIdx, 2));
    // clang-format on
    erodedFaceIndicesVec.push_back(face);
  }

  // Erode mesh for a total of nIterations
  for (int i = 0; i < nIterations; i++) {
    nFaces = erodedFaceIndicesVec.size();

    // We compute the number of neighboring vertices for each vertex.
    vector<int> numNeighbors(nVertices, 0);
    for (int i = 0; i < nFaces; i++) {
      const auto& face = erodedFaceIndicesVec[i];
      numNeighbors[face[0]] += 1;
      numNeighbors[face[1]] += 1;
      numNeighbors[face[2]] += 1;
    }

    std::vector<Eigen::Vector3i> tmp;
    tmp.reserve(nFaces);

    for (int i = 0; i < nFaces; i++) {
      const auto& face = erodedFaceIndicesVec[i];
      if (numNeighbors[face[0]] >= minNeighbors && numNeighbors[face[1]] >= minNeighbors &&
          numNeighbors[face[2]] >= minNeighbors) {
        tmp.push_back(face);
      }
    }

    // We kill the faces with border vertices.
    erodedFaceIndicesVec.clear();
    erodedFaceIndicesVec = std::move(tmp);
  }

  // Mark non isolated vertices as not eroded.
  nFaces = erodedFaceIndicesVec.size();

  for (int i = 0; i < nFaces; i++) {
    const auto& face = erodedFaceIndicesVec[i];
    nonErodedVerticesVec[face[0]] = true;
    nonErodedVerticesVec[face[1]] = true;
    nonErodedVerticesVec[face[2]] = true;
  }

  // Store into python array
  for (int i = 0; i < nVertices; i++) {
    *nonErodedVertices.mutable_data(i) = nonErodedVerticesVec[i];
  }
}

std::tuple<py::array_t<float>, py::array_t<int>> sample_nodes(
    const py::array_t<float>& vertexPositions,
    const py::array_t<bool>& nonErodedVertices,
    float nodeCoverage,
    const bool useOnlyNonErodedIndices = true,
    const bool randomShuffle = true) {
  // assert(vertexPositions.ndim() == 2);

  float nodeCoverage2 = nodeCoverage * nodeCoverage;
  int nVertices = vertexPositions.shape(0);

  std::vector<int> nodeIDs;

  // create list of shuffled indices
  std::vector<int> shuffledVertices(nVertices);
  std::iota(std::begin(shuffledVertices), std::end(shuffledVertices), 0);

  if (randomShuffle) {
    std::default_random_engine re{std::random_device{}()};
    std::shuffle(std::begin(shuffledVertices), std::end(shuffledVertices), re);
  }

  std::vector<Eigen::Vector3f> nodePositionsVec;
  for (int vertexIdx : shuffledVertices) {
    // clang-format off
    Eigen::Vector3f point(
        *vertexPositions.data(vertexIdx, 0),
        *vertexPositions.data(vertexIdx, 1),
        *vertexPositions.data(vertexIdx, 2));
    // clang-format on

    if (useOnlyNonErodedIndices && !(*nonErodedVertices.data(vertexIdx))) {
      continue;
    }

    bool bIsNode = true;
    for (int nodeIdx = 0; nodeIdx < nodePositionsVec.size(); ++nodeIdx) {
      if ((point - nodePositionsVec[nodeIdx]).squaredNorm() <= nodeCoverage2) {
        bIsNode = false;
        break;
      }
    }

    if (bIsNode) {
      nodePositionsVec.push_back(point);
      nodeIDs.push_back(vertexIdx);
    }
  }

  // convert to numpy array
  int nNodes = nodePositionsVec.size();
  py::array_t<float> nodePositions = py::array_t<float>({nNodes, 3});
  py::array_t<int> nodeIndices = py::array_t<int>({nNodes});

  for (int i = 0; i < nNodes; i++) {
    *nodePositions.mutable_data(i, 0) = nodePositionsVec[i].x();
    *nodePositions.mutable_data(i, 1) = nodePositionsVec[i].y();
    *nodePositions.mutable_data(i, 2) = nodePositionsVec[i].z();
    *nodeIndices.mutable_data(i) = nodeIDs[i];
  }

  return std::make_tuple(nodePositions, nodeIndices);
}

/**
 * Custom comparison operator for geodesic priority queue.
 */
struct CustomCompare {
  bool operator()(const std::pair<int, float>& left, const std::pair<int, float>& right) {
    return left.second > right.second;
  }
};

inline float compute_anchor_weight(
    const Eigen::Vector3f& pointPosition, const Eigen::Vector3f& nodePosition, float nodeCoverage) {
  return std::exp(-(nodePosition - pointPosition).squaredNorm() / (2.f * nodeCoverage * nodeCoverage));
}

inline float compute_anchor_weight(float dist, float nodeCoverage) {
  return std::exp(-(dist * dist) / (2.f * nodeCoverage * nodeCoverage));
}

void compute_edges_geodesic(
    const py::array_t<float>& vertexPositions,
    const py::array_t<bool>& validVertices,
    const py::array_t<int>& faceIndices,
    const py::array_t<int>& nodeIndices,
    py::array_t<int>& graphEdges,
    py::array_t<float>& graphEdgesWeights,
    py::array_t<float>& graphEdgesDistances,
    py::array_t<float>& nodeToVertexDistances,
    const int nMaxNeighbors,
    const float nodeCoverage,
    const bool allow_only_valid_vertices,
    const bool enforce_total_num_neighbors) {
  int nVertices = vertexPositions.shape(0);
  int nFaces = faceIndices.shape(0);
  int nNodes = nodeIndices.shape(0);

  float maxInfluence = 2.f * nodeCoverage;

  // Preprocess vertex neighbors.
  vector<std::set<int>> vertexNeighbors(nVertices);
  for (int faceIdx = 0; faceIdx < nFaces; faceIdx++) {
    for (int j = 0; j < 3; j++) {
      int v_idx = *faceIndices.data(faceIdx, j);

      for (int k = 0; k < 3; k++) {
        int n_idx = *faceIndices.data(faceIdx, k);

        if (v_idx == n_idx) continue;
        vertexNeighbors[v_idx].insert(n_idx);
      }
    }
  }

  // Compute inverse vertex -> node relationship.
  vector<int> mapVertexToNode(nVertices, -1);

  for (int nodeId = 0; nodeId < nNodes; nodeId++) {
    int vertexIdx = *nodeIndices.data(nodeId);
    if (vertexIdx >= 0) {
      mapVertexToNode[vertexIdx] = nodeId;
    }
  }

  // #pragma omp parallel for
  for (int nodeId = 0; nodeId < nNodes; nodeId++) {
    // vertex queue
    std::priority_queue<std::pair<int, float>, vector<std::pair<int, float>>, CustomCompare> nextVerticesWithIds;

    std::set<int> visitedVertices;

    // Add node vertex as the first vertex to be visited.
    int nodeVertexIdx = *nodeIndices.data(nodeId);
    if (nodeVertexIdx < 0) continue;
    nextVerticesWithIds.push(std::make_pair(nodeVertexIdx, 0.f));

    // Traverse all neighbors in the monotonically increasing order.
    vector<int> neighborNodeIds;
    vector<float> neighborNodeWeights;
    vector<float> neighborNodeDistances;
    while (!nextVerticesWithIds.empty()) {
      auto nextVertex = nextVerticesWithIds.top();
      nextVerticesWithIds.pop();

      int nextVertexIdx = nextVertex.first;
      float nextVertexDist = nextVertex.second;

      // We skip the vertex, if it was already visited before.
      if (visitedVertices.find(nextVertexIdx) != visitedVertices.end()) continue;

      // We check if the vertex is a node.
      int nextNodeId = mapVertexToNode[nextVertexIdx];
      if (nextNodeId >= 0 && nextNodeId != nodeId) {
        neighborNodeIds.push_back(nextNodeId);
        neighborNodeWeights.push_back(compute_anchor_weight(nextVertexDist, nodeCoverage));
        neighborNodeDistances.push_back(nextVertexDist);
        if (neighborNodeIds.size() >= nMaxNeighbors) break;
      }

      // Note down the node-vertex distance.
      *nodeToVertexDistances.mutable_data(nodeId, nextVertexIdx) = nextVertexDist;

      // We visit the vertex, and check all his neighbors.
      // We add only valid vertices under a certain distance
      visitedVertices.insert(nextVertexIdx);
      Eigen::Vector3f nextVertexPos(
          *vertexPositions.data(nextVertexIdx, 0),
          *vertexPositions.data(nextVertexIdx, 1),
          *vertexPositions.data(nextVertexIdx, 2));

      const auto& nextNeighbors = vertexNeighbors[nextVertexIdx];
      for (int neighborIdx : nextNeighbors) {
        bool is_valid_vertex = *validVertices.data(neighborIdx);
        if (allow_only_valid_vertices && !is_valid_vertex) {
          continue;
        }

        Eigen::Vector3f neighborVertexPos(
            *vertexPositions.data(neighborIdx, 0),
            *vertexPositions.data(neighborIdx, 1),
            *vertexPositions.data(neighborIdx, 2));
        float dist = nextVertexDist + (nextVertexPos - neighborVertexPos).norm();

        if (enforce_total_num_neighbors) {
          nextVerticesWithIds.push(std::make_pair(neighborIdx, dist));
        } else {
          // std::cout << dist << " " << maxInfluence << std::endl;
          if (dist <= maxInfluence) {
            nextVerticesWithIds.push(std::make_pair(neighborIdx, dist));
          }
        }
      }
    }

    // Store the nearest neighbors.
    int nNeighbors = neighborNodeIds.size();

    float weightSum = 0.f;
    for (int i = 0; i < nNeighbors; i++) {
      *graphEdges.mutable_data(nodeId, i) = neighborNodeIds[i];
      weightSum += neighborNodeWeights[i];
    }

    // Normalize weights
    if (weightSum > 0) {
      for (int i = 0; i < nNeighbors; i++) {
        *graphEdgesWeights.mutable_data(nodeId, i) = neighborNodeWeights[i] / weightSum;
      }
    } else if (nNeighbors > 0) {
      for (int i = 0; i < nNeighbors; i++) {
        *graphEdgesWeights.mutable_data(nodeId, i) = neighborNodeWeights[i] / nNeighbors;
      }
    }

    // Store edge distance.
    for (int i = 0; i < nNeighbors; i++) {
      *graphEdgesDistances.mutable_data(nodeId, i) = neighborNodeDistances[i];
    }
  }
}

py::array_t<int> compute_edges_euclidean(
    const py::array_t<float>& nodePositions, int nMaxNeighbors, float maxInfluence) {
  int nNodes = nodePositions.shape(0);

  py::array_t<int> graphEdges = py::array_t<int>({nNodes, nMaxNeighbors});

  // Find nearest Euclidean neighbors for each node.
  for (int nodeId = 0; nodeId < nNodes; nodeId++) {
    Eigen::Vector3f nodePos(
        *nodePositions.data(nodeId, 0), *nodePositions.data(nodeId, 1), *nodePositions.data(nodeId, 2));

    // Keep only the k nearest Euclidean neighbors.
    std::list<std::pair<int, float>> nearestNodesWithSquaredDistances;

    for (int neighborId = 0; neighborId < nNodes; neighborId++) {
      if (neighborId == nodeId) continue;

      // clang-format off
      Eigen::Vector3f neighborPos(
          *nodePositions.data(neighborId, 0),
          *nodePositions.data(neighborId, 1),
          *nodePositions.data(neighborId, 2));
      // clang-format on

      float distance2 = (nodePos - neighborPos).squaredNorm();
      if (distance2 > maxInfluence) {
        continue;
      }
      bool bInserted = false;
      for (auto it = nearestNodesWithSquaredDistances.begin(); it != nearestNodesWithSquaredDistances.end(); ++it) {
        // We insert the element at the first position where its distance is smaller than the other element's distance,
        // which enables us to always keep a sorted list of at most k nearest neighbors.
        if (distance2 <= it->second) {
          it = nearestNodesWithSquaredDistances.insert(it, std::make_pair(neighborId, distance2));
          bInserted = true;
          break;
        }
      }

      if (!bInserted && nearestNodesWithSquaredDistances.size() < nMaxNeighbors) {
        nearestNodesWithSquaredDistances.emplace_back(std::make_pair(neighborId, distance2));
      }

      // We keep only the list of k nearest elements.
      if (bInserted && nearestNodesWithSquaredDistances.size() > nMaxNeighbors) {
        nearestNodesWithSquaredDistances.pop_back();
      }
    }

    // Store nearest neighbor ids.
    int idx = 0;
    for (auto it = nearestNodesWithSquaredDistances.begin(); it != nearestNodesWithSquaredDistances.end(); ++it) {
      int neighborId = it->first;
      *graphEdges.mutable_data(nodeId, idx) = neighborId;
      idx++;
    }

    for (idx = nearestNodesWithSquaredDistances.size(); idx < nMaxNeighbors; idx++) {
      *graphEdges.mutable_data(nodeId, idx) = -1;
    }
  }

  return graphEdges;
}

inline int traverse_neighbors(
    const std::vector<std::set<int>>& node_neighbors, std::vector<int>& cluster_ids, int cluster_id, int node_id) {
  if (cluster_ids[node_id] != -1) return 0;

  std::set<int> active_node_indices;

  // Initialize with current node.
  int cluster_size = 0;
  active_node_indices.insert(node_id);

  // Process until we have no active nodes anymore.
  while (!active_node_indices.empty()) {
    int active_node_id = *active_node_indices.begin();
    active_node_indices.erase(active_node_indices.begin());

    if (cluster_ids[active_node_id] == -1) {
      cluster_ids[active_node_id] = cluster_id;
      ++cluster_size;
    }

    // Look if we need to process any of the neighbors
    for (const auto& n_idx : node_neighbors[active_node_id]) {
      if (cluster_ids[n_idx] == -1) {  // If it doesn't have a cluster yet
        active_node_indices.insert(n_idx);
      }
    }
  }

  return cluster_size;
}

void node_and_edge_clean_up(const py::array_t<int>& graph_edges, py::array_t<bool>& valid_nodes_mask) {
  int num_nodes = graph_edges.shape(0);
  int max_num_neighbors = graph_edges.shape(1);

  std::list<int> removed_nodes;

  while (true) {
    int num_newly_removed_nodes = 0;

    for (int node_id = 0; node_id < num_nodes; ++node_id) {
      if (*valid_nodes_mask.data(node_id) == false) {
        // if node has been already removed, continue
        continue;
      }

      int num_neighbors = 0;
      for (int i = 0; i < max_num_neighbors; ++i) {
        int neighbor_id = *graph_edges.data(node_id, i);

        // if neighboring node is -1, break, since by design 'graph_edges' has
        // the shape [2, 3, 6, -1, -1, -1, -1, -1]
        if (neighbor_id == -1) {
          break;
        }

        // if neighboring node has been marked as invalid, continue
        if (std::find(removed_nodes.begin(), removed_nodes.end(), neighbor_id) != removed_nodes.end()) {
          continue;
        }

        ++num_neighbors;
      }

      if (num_neighbors <= 1) {
        // remove node
        *valid_nodes_mask.mutable_data(node_id) = false;
        removed_nodes.emplace_back(node_id);
        ++num_newly_removed_nodes;
      }
    }

    if (num_newly_removed_nodes == 0) {
      break;
    }
  }
}

std::vector<int> compute_clusters(const py::array_t<int>& graph_edges, py::array_t<int>& graph_clusters) {
  int num_nodes = graph_edges.shape(0);
  int max_num_neighbors = graph_edges.shape(1);

  // convert graph_edges to a vector of sets
  std::vector<std::set<int>> node_neighbors(num_nodes);

  for (int node_id = 0; node_id < num_nodes; ++node_id) {
    for (int neighbor_idx = 0; neighbor_idx < max_num_neighbors; ++neighbor_idx) {
      int neighbor_id = *graph_edges.data(node_id, neighbor_idx);

      if (neighbor_id == -1) {
        break;
      }

      node_neighbors[node_id].insert(neighbor_id);
      node_neighbors[neighbor_id].insert(node_id);
    }
  }

  std::vector<int> cluster_ids(num_nodes, -1);
  std::vector<int> clusters_size;

  int cluster_id = 0;
  for (int node_id = 0; node_id < num_nodes; ++node_id) {
    int cluster_size = traverse_neighbors(node_neighbors, cluster_ids, cluster_id, node_id);
    if (cluster_size > 0) {
      cluster_id++;
      clusters_size.push_back(cluster_size);
    }
  }

  for (int node_id = 0; node_id < num_nodes; ++node_id) {
    *graph_clusters.mutable_data(node_id) = cluster_ids[node_id];
  }

  return clusters_size;
}

inline void compute_nearest_geodesic_nodes(
    const py::array_t<float>& node_to_vertex_distance,
    const py::array_t<int>& valid_nodes_mask,
    const int vertex_id,
    const int num_anchors,
    std::vector<int>& nearest_geodesic_node_ids,
    std::vector<float>& dist_to_nearest_geodesic_nodes) {
  int num_nodes = node_to_vertex_distance.shape(0);

  std::map<int, float> node_map;

  for (int n = 0; n < num_nodes; ++n) {
    // discard node if it was marked as invalid (due to not having enough neighbors)
    if (*valid_nodes_mask.data(n) == false) {
      continue;
    }

    float dist = *node_to_vertex_distance.data(n, vertex_id);

    if (dist >= 0) {
      node_map.emplace(n, dist);
    }
  }

  // Sort the map by distance
  // Declaring the type of Predicate that accepts 2 pairs and return a bool
  typedef std::function<bool(std::pair<int, float>, std::pair<int, float>)> Comparator;

  // Defining a lambda function to compare two pairs. It will compare two pairs using second field
  Comparator comp_functor = [](std::pair<int, float> node1, std::pair<int, float> node2) {
    return node1.second <= node2.second;
  };

  // Declaring a set that will store the pairs using above comparision logic
  std::set<std::pair<int, float>, Comparator> node_set(node_map.begin(), node_map.end(), comp_functor);

  for (auto n : node_map) {
    nearest_geodesic_node_ids.push_back(n.first);
    dist_to_nearest_geodesic_nodes.push_back(n.second);

    if (nearest_geodesic_node_ids.size() == num_anchors) {
      break;
    }
  }
}

void compute_pixel_anchors_geodesic(
    const py::array_t<float>& node_to_vertex_distance,
    const py::array_t<int>& valid_nodes_mask,
    const py::array_t<float>& vertices,
    const py::array_t<int>& vertex_pixels,
    py::array_t<int>& pixel_anchors,
    py::array_t<float>& pixel_weights,
    const int width,
    const int height,
    const int num_anchors,
    const float node_coverage) {
  int num_vertices = vertices.shape(0);

  for (int vertex_id = 0; vertex_id < num_vertices; vertex_id++) {
    // Get corresponding pixel location
    int u = *vertex_pixels.data(vertex_id, 0);
    int v = *vertex_pixels.data(vertex_id, 1);

    // Initialize some variables
    std::vector<int> nearest_geodesic_node_ids;
    std::vector<float> dist_to_nearest_geodesic_nodes;
    std::vector<float> skinning_weights;

    nearest_geodesic_node_ids.reserve(num_anchors);
    dist_to_nearest_geodesic_nodes.reserve(num_anchors);
    skinning_weights.reserve(num_anchors);

    // Find closest geodesic nodes
    compute_nearest_geodesic_nodes(
        node_to_vertex_distance,
        valid_nodes_mask,
        vertex_id,
        num_anchors,
        nearest_geodesic_node_ids,
        dist_to_nearest_geodesic_nodes);

    int num_anchors = nearest_geodesic_node_ids.size();

    // Compute skinning weights.
    float weight_sum{0.f};
    for (int i = 0; i < num_anchors; ++i) {
      float geodesic_dist_to_node = dist_to_nearest_geodesic_nodes[i];

      float weight = compute_anchor_weight(geodesic_dist_to_node, node_coverage);
      weight_sum += weight;

      skinning_weights.push_back(weight);
    }

    // Normalize the skinning weights.
    if (weight_sum > 0) {
      for (int i = 0; i < num_anchors; i++) skinning_weights[i] /= weight_sum;
    } else if (num_anchors > 0) {
      for (int i = 0; i < num_anchors; i++) skinning_weights[i] = 1.f / num_anchors;
    }

    // Store the results.
    for (int i = 0; i < num_anchors; i++) {
      *pixel_anchors.mutable_data(v, u, i) = nearest_geodesic_node_ids[i];
      *pixel_weights.mutable_data(v, u, i) = skinning_weights[i];
    }
  }
}

void compute_pixel_anchors_euclidean(
    const py::array_t<float>& graphNodes,
    const py::array_t<float>& pointImage,
    const int num_anchors,
    const float nodeCoverage,
    py::array_t<int>& pixelAnchors,
    py::array_t<float>& pixelWeights) {
  int nNodes = graphNodes.shape(0);
  int height = pointImage.shape(0);
  int width = pointImage.shape(1);

  // Compute anchors for every pixel.
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      // Query 3d pixel position.
      // clang-format off
      Eigen::Vector3f pixelPos(
          *pointImage.data(y, x, 0),
          *pointImage.data(y, x, 1),
          *pointImage.data(y, x, 2));
      // clang-format on
      if (pixelPos.z() <= 0) continue;

      // Keep only the k nearest Euclidean neighbors.
      std::list<std::pair<int, float>> nearestNodesWithSquaredDistances;

      for (int nodeId = 0; nodeId < nNodes; nodeId++) {
        // clang-format off
        Eigen::Vector3f nodePos(
            *graphNodes.data(nodeId, 0),
            *graphNodes.data(nodeId, 1),
            *graphNodes.data(nodeId, 2));
        // clang-format on

        float distance2 = (pixelPos - nodePos).squaredNorm();
        bool bInserted = false;
        for (auto it = nearestNodesWithSquaredDistances.begin(); it != nearestNodesWithSquaredDistances.end(); ++it) {
          // We insert the element at the first position where its distance is
          // smaller than the other element's distance, which enables us to
          // always keep a sorted list of at most k nearest neighbors.
          if (distance2 <= it->second) {
            it = nearestNodesWithSquaredDistances.insert(it, std::make_pair(nodeId, distance2));
            bInserted = true;
            break;
          }
        }

        if (!bInserted && nearestNodesWithSquaredDistances.size() < num_anchors) {
          nearestNodesWithSquaredDistances.emplace_back(std::make_pair(nodeId, distance2));
        }

        // We keep only the list of k nearest elements.
        if (bInserted && nearestNodesWithSquaredDistances.size() > num_anchors) {
          nearestNodesWithSquaredDistances.pop_back();
        }
      }

      // Compute skinning weights.
      std::vector<int> nearestEuclideanNodeIds;
      nearestEuclideanNodeIds.reserve(nearestNodesWithSquaredDistances.size());

      std::vector<float> skinningWeights;
      skinningWeights.reserve(nearestNodesWithSquaredDistances.size());

      float weightSum{0.f};
      for (auto it = nearestNodesWithSquaredDistances.begin(); it != nearestNodesWithSquaredDistances.end(); ++it) {
        int nodeId = it->first;

        // clang-format off
        Eigen::Vector3f nodePos(
            *graphNodes.data(nodeId, 0),
            *graphNodes.data(nodeId, 1),
            *graphNodes.data(nodeId, 2));
        // clang-format on

        float weight = compute_anchor_weight(pixelPos, nodePos, nodeCoverage);
        weightSum += weight;

        nearestEuclideanNodeIds.push_back(nodeId);
        skinningWeights.push_back(weight);
      }

      // Normalize the skinning weights.
      int nAnchors = nearestEuclideanNodeIds.size();

      if (weightSum > 0) {
        for (int i = 0; i < nAnchors; i++) skinningWeights[i] /= weightSum;
      } else if (nAnchors > 0) {
        for (int i = 0; i < nAnchors; i++) skinningWeights[i] = 1.f / nAnchors;
      }

      // Store the results.
      for (int i = 0; i < nAnchors; i++) {
        *pixelAnchors.mutable_data(y, x, i) = nearestEuclideanNodeIds[i];
        *pixelWeights.mutable_data(y, x, i) = skinningWeights[i];
      }
    }
  }
}

void update_pixel_anchors(const std::map<int, int>& node_id_mapping, py::array_t<int>& pixel_anchors) {
  int height = pixel_anchors.shape(0);
  int width = pixel_anchors.shape(1);
  int num_anchors = pixel_anchors.shape(2);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int a = 0; a < num_anchors; a++) {
        int current_anchor_id = *pixel_anchors.data(y, x, a);

        if (current_anchor_id != -1) {
          int mapped_anchor_id = node_id_mapping.at(current_anchor_id);

          // update anchor only if it would actually change something
          if (mapped_anchor_id != current_anchor_id) {
            *pixel_anchors.mutable_data(y, x, a) = mapped_anchor_id;
          }
        }
      }
    }
  }
}

}  // namespace graph_proc