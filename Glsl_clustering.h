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
#include "Glsl_axis_aligned_bounding_box.h"
#include "Glsl_extra.h"
#include "Algorithms.h"
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
        using indices_t = std::vector<std::size_t>;
        using out_t = struct { std::vector<indices_t> clusters; indices_t noise; };

        // partition space
        spacePartition.construct(first, last);

        // estimate cluster indices
        const std::size_t len{ static_cast<std::size_t>(std::distance(first, last)) };
        std::vector<bool> visited(len, false);
        std::vector<bool> noise(len, false);
        std::vector<indices_t> clusterIndices;
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
                clusterIndices.emplace_back(indices_t{});
                
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

        // accumulate noise
        indices_t notClusters;
        for (std::size_t i{}; i < len; ++i) {
            if (noise[i]) {
                notClusters.emplace_back(i);
            }
        }

        // remove possible empty clusters
        std::vector<std::size_t> to_remove;
        to_remove.reserve(clusterIndices.size());
        for (std::size_t i{}; i < clusterIndices.size(); ++i) {
            if (clusterIndices[i].empty()) {
                to_remove.emplace_back(i);
            }
        }
        Algoithms::remove(clusterIndices, to_remove);

        // output
        return out_t{ clusterIndices, notClusters };
    }

    /**
    * \brief perform clustering operation using k-means algorithm (Euclidean distance is used as clustering metric).
    * @param {forward_iterator,          in}  iterator to point cloud collection first point
    * @param {forward_iterator,          in}  iterator to point cloud collection last point
    * @param {size_t,                    in}  number of clusters
    * @param {size_t,                    in}  maximal number of iterations
    * @param {size_t,                    in}  convergence tolerance for operation stoppage (minimal movement of each cluster center in two consecutive iterations)
    * @param {vector<IFixedVector>,      in}  initial cluster centers, if empty - will be calculated by function (default is empty)
    * @param {vector<vector<integral>>}, out} vector of vectors of cluster id's. id at index 'i' marks cluster id of point at *(first + i)
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>, class T = typename VEC::value_type>
        requires(GLSL::is_fixed_vector_v<VEC>)
    constexpr std::vector<std::vector<std::size_t>> k_means(const InputIt first, const InputIt last, const std::size_t k,
                                                            const std::size_t max_iterations, const T tol,
                                                            std::vector<VEC> centers = std::vector<VEC>{}) {
        using point_cloud_aabb_t = decltype(AxisLignedBoundingBox::point_cloud_aabb(first, last));

        // place centers in random manner
        if (centers.empty()) {
            centers.reserve(k);
            const point_cloud_aabb_t aabb{ AxisLignedBoundingBox::point_cloud_aabb(first, last) };
            for (std::size_t i{}; i < k; ++i) {
                VEC center;
                Utilities::static_for<0, 1, VEC::length()>([&center, &aabb, i](std::size_t j) {
                    center[j] = aabb.min[j] + std::fmod(static_cast<T>(rand()), aabb.max[j] - aabb.min[j] + static_cast<T>(1));
                });
                centers.emplace_back(center);
            }
        }
  
        // k-means
        std::size_t i{};
        bool converged{ false };
        const std::size_t len{ static_cast<std::size_t>(std::distance(first, last)) };
        std::vector<T> dist(k); // temporary vector which holds distance from given point to all clusters
        std::vector<VEC> sum(k); // temporary vector to sum distances in every cluster
        std::vector<std::size_t> count(k); // temporary vector to hold amount of points in each cluster
        std::vector<std::size_t> assigned_clusters(len); // point 'i' in cloud point belongs to cluster center in 'dist' at position initi[i]
        while (!converged && i < max_iterations) {
            // for each point - calculate its distance to the centers
            std::size_t x{};
            for (auto it{ first }; it != last; ++it) {
                const VEC p{ *it };
                for (std::size_t j{}; j < k; ++j) {
                    dist[j] = GLSL::dot(p - centers[j]);
                }
  
                // assign cluster with closest center
                const auto minElementIter = Algoithms::min_element(dist.begin(), dist.end(), [](const T& a, const T& b) { return a < b; });
                assigned_clusters[x] = std::distance(dist.begin(), minElementIter);
  
                // update loop iterator
                ++x;
            }
  
            // update centers
            bool check_tolerance{ true };
            Algoithms::fill(sum.begin(), sum.end(), VEC());
            Algoithms::fill(count.begin(), count.end(), 0);
            for (std::size_t j{}; j < len; ++j) {
                const std::size_t cluster_index{ assigned_clusters[j] };
                sum[cluster_index] += *(first + j);
                ++count[cluster_index];
            }
            for (std::size_t j{}; j < k; ++j) {
                if (count[j] > 0) [[likely]] {
                    const VEC newCenter{ sum[j] / static_cast<T>(count[j]) };
                    check_tolerance &= GLSL::max(GLSL::abs(centers[j] - newCenter)) <= tol;
                    centers[j] = newCenter;
                }
            }
            
            // update loop iterator
            ++i;
            converged = check_tolerance;
        }
  
        // outputs
        std::vector<std::vector<std::size_t>> out(k);
        for (i = 0; i < len; ++i) {
            out[assigned_clusters[i]].push_back(i);
        }
        return out;
    }

    /**
    * \brief perform clustering operation using mean-shift algorithm.
    * 
    *        remarks:
    *        > Euclidean distance is used as clustering metric and the kernel is of Gaussian form.
    *        > disadvantage of this non-parametric clustering method is that it is relatively slow compared to
    *          other clustering algorithms in this namespace since it is more complex than k-means
    *          and does not use accelerated spatial query structures as DBSCAN does.
    * 
    * @param {forward_iterator,          in}  iterator to point cloud collection first point
    * @param {forward_iterator,          in}  iterator to point cloud collection last point
    * @param {value_type,                in}  kernel bandwidth / window size
    * @param {value_type,                in}  tolerance for point shifting to stop (default is 1e-4)
    * @param {value_type,                in}  tolerance for point to be considered in cluster (default is 1e-1)
    * @param {{vector<vector<integral>>,      {vector of vectors of cluster id's. id at index 'i' marks cluster id of point at *(first + i),
    *          vector<IFixedVector>},    out}  vector holding clusters centers. center at index 'i' is the center of cluster 'i' in first output argument}
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>,
             class T = typename VEC::value_type>
        requires(GLSL::is_fixed_vector_v<VEC>)
    constexpr auto get_mean_shift_based_clusters(const InputIt first, const InputIt last, const T bandwidth,
                                                 const T shifting_tolerance = static_cast<T>(1e-4),
                                                 const T cluster_tolerance = static_cast<T>(1e-1)) {
        using out_t = struct { std::vector<std::vector<std::size_t>> clusters; std::vector<VEC> centers; };

        const T sqrt_tau{ std::sqrt(static_cast<T>(6.283185307179586476925286766559)) };
        assert(bandwidth > T{});
        [[assume(bandwidth > T{})]];

        // Gaussian kernel
        const auto kernel = [sqrt_tau, bandwidth](const T distance) -> T {
            const T ratio{ distance / bandwidth };
            const T exp{ std::exp(static_cast<T>(-0.5) * ratio * ratio) };
            const T coeff{ static_cast<T>(1.0) / (bandwidth * sqrt_tau) };
            return (coeff * exp);
        };

         // lambda to calculate how much to shift a point
         const auto mean_shift_point = [&first, &last, _kernel = FWD(kernel), bandwidth]
                                       (const VEC point) -> VEC {
            VEC shifted;
            T scale{};

            for (InputIt it{ first }; it != last; ++it) {
                const VEC pt{ *it };
                const T dist{ GLSL::distance(point, pt) };
                if (Numerics::areEquals(dist, T{})) {
                    continue;
                }

                const T weight{ _kernel(dist) };
                shifted += weight * pt;
                scale += weight;
            }

            shifted /= scale;
            return shifted;
         };

         // housekeeping
         const std::size_t len{ static_cast<std::size_t>(std::distance(first, last)) };
         std::vector<bool> shifting(len, true);
         std::vector<VEC> shift_points;
         shift_points.reserve(len);
         for (InputIt it{ first }; it != last; ++it) {
             shift_points.emplace_back(*it);
         }

         //
         // mean-shift
         //

         while (true) {
             T max_dist{};

             for (std::size_t i{}; i < len; ++i) {
                 if (!shifting[i]) {
                     continue;
                 }

                 const VEC p_shift_init{ shift_points[i] };
                 shift_points[i] = mean_shift_point(shift_points[i]);
                 const T dist{ GLSL::distance(p_shift_init, shift_points[i]) };
                 max_dist = Numerics::max(max_dist, dist);
                 shifting[i] = dist > shifting_tolerance;
             }

             if (max_dist < shifting_tolerance) {
                 break;
             }
         }

         //
         // clustering
         // 
         
         std::vector<std::size_t> cluster_ids;
         std::vector<VEC> cluster_centers;
         cluster_ids.reserve(len);
         std::size_t cluster_index{};
         for (std::size_t i{}; i < len; ++i) {
             const VEC point{ shift_points[i] };

             if (!cluster_ids.empty()) [[likely]] {
                 std::size_t c{};
                 for (const VEC center : cluster_centers) {
                     const T dist{ GLSL::distance(point, center) };
                     if (dist < cluster_tolerance) {
                         cluster_ids.emplace_back(c);
                     }
                     ++c;
                 }

                 if (cluster_ids.size() < i + 1) {
                     cluster_ids.emplace_back(cluster_index);
                     cluster_centers.emplace_back(shift_points[i]);
                     ++cluster_index;
                 }
             } else {
                 cluster_ids.emplace_back(cluster_index);
                 cluster_centers.emplace_back(shift_points[i]);
                 ++cluster_index;
             }
         }

         // output
         std::vector<std::vector<std::size_t>> clusters(cluster_centers.size(), std::vector<std::size_t>{});
         for (std::size_t i{}; i < len; ++i) {
             clusters[cluster_ids[i]].emplace_back(i);
         }
         return out_t{ clusters, cluster_centers };
    }
}
