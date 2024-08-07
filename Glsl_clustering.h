//-------------------------------------------------------------------------------
//
// Copyright (c) 2024, Dan Israel Malta <malta.dan@gmail.com>
// All rights reserved.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy 
// of this software and associated documentation files (the "Software"), to deal 
// in the Software without restriction, including without limitation the rights 
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
// copies of the Software, and to permit persons to whom the Software is 
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all 
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
// SOFTWARE.
// 
//-------------------------------------------------------------------------------
#pragma once
#include "Glsl.h"
#include "Glsl_space_partitioning.h"
#include <iterator>
#include <vector>

//
// collection of clustering algorithms
//
namespace Clustering {

    /**
    * \brief perform density-based spatial clustering of applications with noise (DBSCAN)
    *        on collection of points using Euclidean distance as metric.
    *        notice that this function uses ISpacePartitioning object from "Glsl_space_partitioning.h" for neighbour query.
    * @param {forward_iterator,                             in}  iterator to point cloud collection first point
    * @param {forward_iterator,                             in}  iterator to point cloud collection last point
    * @param {ISpacePartitioning,                           in}  space partitioning object for neighbour query
    * @param {value_type,                                   in}  minimal distance between two points to be considered in same cluster
    * @param {size_t,                                       in}  minimal amount of points for a cluster to be formed
    * @param {{vector<vector<integral>>, vector<integral>}, out} {vector of vectors of cluster id's. id at index 'i' marks cluster id of point at *(first + i),
    *                                                             vector of indices of points which are noise}
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>, class T = typename VEC::value_type,
             SpacePartitioning::ISpacePartitioning<VEC, InputIt> SP>
        requires(GLSL::is_fixed_vector_v<VEC>)
    constexpr auto get_density_based_clusters(const InputIt first, const InputIt last, SP& spacePartition,
                                              const T minimal_distance, const std::size_t minimal_points) {
        using query_t = decltype(spacePartition.range_query(SpacePartitioning::RangeSearchType::Radius, *first, minimal_distance));
        using out_t = struct { std::vector<std::vector<std::size_t>> clusters; std::vector<std::size_t> noise; };

        // partition space
        spacePartition.construct(first, last);

        // estimate cluster indices
        const std::size_t len{ static_cast<std::size_t>(std::distance(first, last)) };
        std::vector<bool> visited(len, false);
        std::vector<bool> noise(len, false);
        std::vector<std::vector<std::size_t>> clusterIndices;
        for (std::size_t i{}; i < len; ++i) {
            if (visited[i]) {
                continue;
            }

            visited[i] = true;
            const VEC point{ *(first + i) };
            const query_t neighbors{ spacePartition.range_query(SpacePartitioning::RangeSearchType::Radius, point, minimal_distance) };
            if (neighbors.size() < minimal_points) {
                noise[i] = true;
            }
            else {
                clusterIndices.emplace_back(std::vector<std::size_t>{});
                
                // expand cluster
                std::deque<std::size_t> neighborDeque;
                for (const auto& n : neighbors) {
                    neighborDeque.push_back(n.second);
                }
                
                auto& lastCluster = clusterIndices.back();
                while (!neighborDeque.empty()) {
                    const std::size_t curIdx{ neighborDeque.front() };
                    neighborDeque.pop_front();

                    if (noise[curIdx]) {
                        lastCluster.emplace_back(curIdx);
                        continue;
                    }

                    if (!visited[curIdx]) {
                        visited[curIdx] = true;
                        lastCluster.emplace_back(curIdx);

                        const VEC curPoint{ *(first + curIdx) };
                        const query_t curNeighbors{ spacePartition.range_query(SpacePartitioning::RangeSearchType::Radius, curPoint, minimal_distance) };
                        if (curNeighbors.size() < minimal_points) {
                            continue;
                        }

                        for (const auto& n : curNeighbors) {
                            neighborDeque.push_back(n.second);
                        }
                    }
                }
            }
        }

        // accumelate noise
        std::vector<std::size_t> notClusters;
        for (std::size_t i{}; i < len; ++i) {
            if (noise[i]) {
                notClusters.emplace_back(i);
            }
        }

        // output
        return out_t{ clusterIndices, notClusters };
    }

    /**
    * \brief perform clustering operation using k-means and Euclidean distance as metric.
    *        notice that this function uses ISpacePartitioning object from "Glsl_space_partitioning.h" for neighbour query.
    * @param {forward_iterator,          in}  iterator to point cloud collection first point
    * @param {forward_iterator,          in}  iterator to point cloud collection last point
    * @param {size_t,                    in}  number of clusters
    * @param {size_t,                    in}  maximal number of iterations
    * @param {size_t,                    in}  convergence tolerance for operation stoppage (minimal movement of each cluster center in two consecutive iterations)
    * @param {vector<vector<integral>>}, out} vector of vectors of cluster id's. id at index 'i' marks cluster id of point at *(first + i)
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>, class T = typename VEC::value_type>
        requires(GLSL::is_fixed_vector_v<VEC>)
    constexpr std::vector<std::vector<std::size_t>> k_means(const InputIt first, const InputIt last, const std::size_t k,
                                                            const std::size_t max_iterations, const T tol) {
        // place centers in random manner
        std::vector<VEC> centers(k);
        const auto aabb = AxisLignedBoundingBox::point_cloud_aabb(first, last);
        for (std::size_t i{}; i < k; ++i) {
            Utilities::static_for<0, 1, VEC::length()>([&centers, &aabb, i](std::size_t j) {
                centers[i][j] = aabb.min[j] + std::fmod(static_cast<T>(rand()), aabb.max[j] - aabb.min[j] + static_cast<T>(1));
            });
        }
  
        // k-means
        std::size_t i{};
        bool converged{ false };
        const std::size_t len{ static_cast<std::size_t>(std::distance(first, last)) };
        std::vector<T> dist(k); // temporary vectot which holds distance from given point to all clusters
        std::vector<VEC> sum(k); // temporary vector to sum distances in every cluster
        std::vector<std::size_t> count(k); // temporary vector to hold amount of points in each cluster
        std::vector<std::size_t> init(len); // point 'i' in cloud point belongs to cluster center in 'dist' at position initi[i]
        while (!converged && i < max_iterations) {
            // for each point - calculate its distance to the centers
            std::size_t x{};
            for (auto it{ first }; it != last; ++it) {
                const VEC p{ *it };
                for (std::size_t j{}; j < k; ++j) {
                    dist[j] = GLSL::dot(p - centers[j]);
                }
  
                // assign cluster with closest center
                init[x] = std::distance(dist.begin(), std::min_element(dist.begin(), dist.end()));
  
                // update loop iterator
                ++x;
            }
  
            // update centers
            bool check_tolerance{ true };
            std::ranges::fill(sum, VEC());
            std::ranges::fill(count, 0);
            for (std::size_t j{}; j < len; ++j) {
                const std::size_t cluster_index{ init[j] };
                sum[cluster_index] += *(first + j);
                ++count[cluster_index];
            }
            for (std::size_t j{}; j < k; ++j) {
                assert(count[j] > 0);
                const VEC newCenter{ sum[j] / static_cast<T>(count[j]) };
                check_tolerance &= GLSL::max(GLSL::abs(centers[j] - newCenter)) <= tol;
                centers[j] = newCenter;
            }
            
            // update loop iterator
            ++i;
            converged = check_tolerance;
        }
  
        // outputs
        std::vector<std::vector<std::size_t>> out(k);
        for (std::size_t i{}; i < len; ++i) {
            out[init[i]].push_back(i);
        }
        return out;
    }
}
