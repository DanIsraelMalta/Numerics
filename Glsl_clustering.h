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
	* @param {forward_iterator,         in}  iterator to point cloud collection first point
	* @param {forward_iterator,         in}  iterator to point cloud collection last point
	* @param {ISpacePartitioning,       in}  space partitioning object for neighbour query
	* @param {value_type,               in}  minimal distance between two points to be considered in same cluster
	* @param {size_t,                   in}  minimal amount of points for a cluster to be formed
	* @param {vector<vector<integral>>, out} vector of vectors of cluster id's. id at index 'i' marks cluster id of point at *(first + i)
	**/
	template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>, class T = typename VEC::value_type,
	         SpacePartitioning::ISpacePartitioning<VEC, InputIt> SP>
		requires(GLSL::is_fixed_vector_v<VEC>)
	constexpr std::vector<std::vector<std::size_t>> get_density_based_clusters(const InputIt first, const InputIt last, SP& spacePartition,
		                                                                       const T minimal_distance, const std::size_t minimal_points) {
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
			const auto neighbors = spacePartition.range_query(SpacePartitioning::RangeSearchType::Radius, point, minimal_distance);
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
						const auto curNeighbors = spacePartition.range_query(SpacePartitioning::RangeSearchType::Radius, curPoint, minimal_distance);
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

		// output
		return clusterIndices;
	}
}
