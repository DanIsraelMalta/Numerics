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
#include "Algorithms.h"
#include <vector>
#include <map>
#include <queue>
#include <memory> // unique_ptr

//
// collection of Euclidean space partitioning data structure
//
namespace SpacePartitioning {

    /**
    * type of range search
    **/
    enum class RangeSearchType : std::int8_t {
        Manhattan = 0, // search neighbors within a cube
        Radius    = 1  // search neighbors within a sphere
    };

    /**
    * concept of a Euclidean space partitioning data structure for static scenes, i.e. - you cant add points after its construction.
    **/
    template<class SP, typename VEC, typename ITER>
    concept ISpacePartitioning = GLSL::is_fixed_vector_v<VEC> &&                                          // VEC is a GLSL vector
                                 std::is_same_v<VEC, typename SP::point_t> &&                             // GLSL vector type partitioned by data structure
                                 std::is_same_v<typename VEC::value_type, typename SP::coordinate_t> &&   // GLSL vector coordinate type
        requires(SP sp, VEC point, typename VEC::value_type t, RangeSearchType type, ITER it, std::size_t i) {           
            /**
            * \brief given iterators to collection of points - construct space partitioning structure
            * @param {forward_iterator, in} iterator to point collection first point
            * @param {forward_iterator, in} iterator to point collection last point
            **/
            { sp.construct(it, it) } -> std::same_as<void>;

            /**
            * \brief clear kd-tree
            **/
            { sp.clear() } -> std::same_as<void>;

            /**
            * \brief given point and query type, return all its neighbors within this area/volume whose extent/radius is 't'
            * @param {RangeSearchType,                 in} type of range query
            * @param {point_t,                         in} point to search around
            * @param {coordinate_t,                    in} extent of search cube around point
            * @param {vector<{coordinate_t, size_t}>, out} collection of pairs of {squared Euclidean distance from 'point' to point in this pair,
            *                                                                      index of this point in the collection from which ISpacePartitioning was built}
            *                                              if  we queries something relative to point 'P' then the following holds: GLSL::dot(P - *(it + second)) = first
            **/
            { sp.range_query(type, point, t) } -> std::same_as<std::vector<std::pair<typename VEC::value_type, std::size_t>>>;

            /**
            * \brief given point, 'i' of its nearest neighbors
            * @param {point_t,                         in} point to search around
            * @param {size_t,                          in} amount of closest points
            * @param {vector<{coordinate_t, size_t}>, out} collection of pairs of {squared Euclidean distance from 'point' to point in this pair,
            *                                                                      index of this point in the collection from which ISpacePartitioning was built}
            *                                              if we query something relative to point 'P' then the following holds: GLSL::dot(P - *(it + second)) = first
            **/
            { sp.nearest_neighbors_query(point, i) } -> std::same_as<std::vector<std::pair<typename VEC::value_type, std::size_t>>>;
    };

    /**
    * \brief balanced KD-tree for GLSL::IFixedVector objects.
    *        notice that tree doesn't hold the points themselves, rather it points to their position in their original structure.
    * @param {IFixedVector} point held in tree
    **/
    template<GLSL::IFixedVector VEC>
    struct KDTree {
        // aliases
        using point_t = VEC;
        using coordinate_t = typename VEC::value_type;
        using pair_t = std::pair<typename VEC::value_type, std::size_t>;
        using vector_queries_t = std::vector<pair_t>;

        /**
        * \brief given iterators to collection of points - construct kd-tree in recursive manner using median partitioning
        * @param {forward_iterator, in} iterator to point cloud collection first point
        * @param {forward_iterator, in} iterator to point cloud collection last point
        **/
        template<std::forward_iterator InputIt>
            requires(std::is_same_v<point_t, typename std::decay_t<decltype(*std::declval<InputIt>())>>)
        constexpr void construct(const InputIt begin, const InputIt end) {
            this->first = Utilities::addressof(*begin);
            std::vector<std::size_t> indices(static_cast<std::size_t>(std::distance(begin, end)));
            Algoithms::iota(indices.begin(), indices.end(), 0);
            this->root = this->build_tree(0, indices);
        }

        /**
        * \brief clear kd-tree
        **/
        constexpr void clear() noexcept {
            this->clear_node(MOV(this->root));
            this->root.release();
            this->first = nullptr;
        }

        /**
        * \brief given point and query type, return all its neighbors within a specific enclosing area/volume around it
        * @param {RangeSearchType,              in} query type
        * @param {point_t,                      in} point to search around
        * @param {coordinate_t,                 in} extent/radius of search area/volume around point
        * @param {vector<coordinate_t, size_t>, out} collection of query nodes holding point squared distance and index in point cloud
        **/
        constexpr vector_queries_t range_query(const RangeSearchType type, const point_t point, const coordinate_t distance) const {
            // housekeeping
            const coordinate_t distance_criteria{ type == RangeSearchType::Manhattan ? distance : distance * distance };
            auto metric_function = (type == RangeSearchType::Manhattan) ?
                          this->distance_metric<RangeSearchType::Manhattan>() :
                          this->distance_metric<RangeSearchType::Radius>();
            vector_queries_t out{};

            // search kd-tree
#ifdef _DEBUG
            std::size_t count{};
#endif
            std::vector<Node*> stack;
            stack.push_back(this->root.get());
            while (!stack.empty()) {
                const Node* node{ stack.back() };
                stack.pop_back();

                const point_t nodePoint(*(this->first + node->index));
                const point_t diff{ point - nodePoint };
                if (metric_function(diff) <= distance_criteria) {
                    out.emplace_back(std::make_pair(GLSL::dot(diff), node->index));
                }

                const std::size_t split{ node->splitAxis };
                const coordinate_t dist_per_split{ diff[split] };
                const bool inside{ dist_per_split <= coordinate_t{} };

                std::array<Node*, 2> nodes{ {node->right.get(), node->left.get()} };
                if (nodes[inside]) {
                    stack.push_back(nodes[inside]);
                }
                if (nodes[!inside] && std::abs(dist_per_split) < distance) {
                    stack.push_back(nodes[!inside]);
                }
#ifdef _DEBUG
                ++count;
#endif
            }

            return out;
        }

        /**
        * \brief given point, 'k' of its nearest neighbors. notice that this method is recursive.
        * @param {point_t,                      in}  point to search around
        * @param {size_t,                       in}  amount of closest points
        * @param {vector<coordinate_t, size_t>, out} collection of query nodes holding point squared distance and index in point cloud
        **/
        constexpr vector_queries_t nearest_neighbors_query(const point_t point, const std::size_t k) const {
            // housekeeping
            std::priority_queue<pair_t, vector_queries_t, less_pair_t> kMaxHeap;
            coordinate_t minDistance{ GLSL::dot(*(this->first + this->root->index) - point) };

            // lambda implementing nearest neighbors query logic for one node
#ifdef _DEBUG
            std::size_t depth{};
            const auto nearest_neighbors_query_recursive = [this, &kMaxHeap, point, k, &minDistance, &depth]
                                                           (const Node* node, auto&& recursive_driver) -> void {
#else
            const auto nearest_neighbors_query_recursive = [this, &kMaxHeap, point, k, &minDistance]
                                                           (const Node* node, auto&& recursive_driver) -> void {
#endif
                const point_t nodePoint(*(this->first + node->index));
                const point_t diff{ point - nodePoint };
                if (const coordinate_t distance{ GLSL::dot(diff) };
                    distance < minDistance) {
                    while (kMaxHeap.size() >= k) {
                        kMaxHeap.pop();
                    }
                    kMaxHeap.emplace(std::make_pair(distance, node->index));
                    minDistance = kMaxHeap.top().first;
                }

                const std::size_t split{ node->splitAxis };
                const coordinate_t dist_per_split{ diff[split] };
                const std::size_t inside{ dist_per_split <= coordinate_t{} };

#ifdef _DEBUG
                std::array<Node*, 2> nodes{ {node->right.get(), node->left.get()} };
                if (nodes[inside]) {
                    ++depth;
                    recursive_driver(nodes[inside], recursive_driver);
                }
                if (nodes[!inside] && std::abs(dist_per_split) <= minDistance) {
                    ++depth;
                    recursive_driver(nodes[!inside], recursive_driver);
                }
#else
                std::array<Node*, 2> nodes{ {node->right.get(), node->left.get()} };
                if (nodes[inside]) {
                    recursive_driver(nodes[inside], recursive_driver);
                }
                if (nodes[!inside] && std::abs(dist_per_split) <= minDistance) {
                    recursive_driver(nodes[!inside], recursive_driver);
                }
#endif
            };

            // find k nearest neighbors using depth search
            nearest_neighbors_query_recursive(this->root.get(), nearest_neighbors_query_recursive);

            // output
            vector_queries_t out;
            out.reserve(k);
            while (!kMaxHeap.empty()) {
                out.emplace_back(kMaxHeap.top());
                kMaxHeap.pop();
            }
            Algoithms::reverse(out.begin(), out.end());
            out.shrink_to_fit();
            return out;
        }
        
        // internals
        private:

            // KDTree node
            struct Node {
                std::size_t index{};         // node position in original cloud point, i.e. - at *(first + index)
                std::size_t splitAxis{};     // dimension along which this node was split
                std::unique_ptr<Node> left;  // pointer to left child
                std::unique_ptr<Node> right; // pointer to right child
            };
            
            // properties
            std::unique_ptr<Node> root;       // tree root
            const point_t* first{ nullptr };  // iterator for point cloud start 
        
            /**
            * \brief spatially divide the collection of points in recursive manner.
            * @param {size_t,           in}  node depth
            * @param {vector<size_t>,   in}  cloud points vector of indices
            * @param {unique_ptr<Node>, out} pointer to new tree node
            **/
            constexpr std::unique_ptr<Node> build_tree(std::size_t depth, std::vector<std::size_t> indices) {
                constexpr std::size_t N{ point_t::length() };
                if (indices.empty()) {
                    return nullptr;
                }
        
                // partition range according to its first coordinate
                const std::size_t medianIndex{ indices.size() / 2 };
                const std::size_t axis{ depth % N };
                Algoithms::nth_element(indices, medianIndex,
                    [axis, this](std::size_t a, std::size_t b) {
                        const point_t point_a(*(this->first + a));
                        const point_t point_b(*(this->first + b));
                        return (point_a[axis] <= point_b[axis]);
                    });

                // build and return node
                const std::size_t nodeIndex{ indices[medianIndex] };
                return std::make_unique<Node>(Node{
                    .index = nodeIndex,
                    .splitAxis = axis,
                    .left = this->build_tree(depth + 1, std::vector<std::size_t>(indices.begin(), indices.begin() + medianIndex)),
                    .right = this->build_tree(depth + 1, std::vector<std::size_t>(indices.begin() + medianIndex + 1, indices.end()))
                });
            }

            /**
            * \brief erase a given node and all its children
            * @param {Node*, in} node to delete
            **/
            constexpr void clear_node(std::unique_ptr<Node> node) {
                if (!node) {
                    return;
                }
        
                if (node->left) {
                    this->clear_node(MOV(node.get()->left));
                }
        
                if (node->right) {
                    this->clear_node(MOV(node.get()->right));
                }
        
                node.release();
            }

            /**
            * \brief std::less override for 'pair_t' type
            **/
            struct less_pair_t {
                constexpr bool operator()(const pair_t& a, const pair_t& b) const {
                    return a.first < b.first;
                }
            };

            /**
            * \brief given range query type - return metric used in searching
            * @param {invocable, out} distance metric used for range query
            **/
            template<RangeSearchType TYPE>
            constexpr auto distance_metric() const {
                if constexpr (TYPE == RangeSearchType::Manhattan) {
                    return [](const point_t a) -> coordinate_t { return GLSL::max(GLSL::abs(a)); };
                }
                else if constexpr (TYPE == RangeSearchType::Radius) {
                    return [](const point_t a) -> coordinate_t { return GLSL::dot(a); };
                }
            }
    };
    static_assert(ISpacePartitioning<KDTree<vec2>, vec2, std::vector<vec2>::iterator>);


    /**
    * \brief uniform grid ("Bin-Lattice spatial subdivision structure") for GLSL::IFixedVector objects of second or third dimension.
    *        notice that tree doesn't hold the points themselves, rather it points to their position in their original structure.
    * @param {IFixedVector} point held in tree (2D/3D)
    **/
    template<GLSL::IFixedVector VEC>
        requires(VEC::length() <= 3)
    struct Grid {
        // aliases
        using point_t = VEC;
        using coordinate_t = typename VEC::value_type;
        using index_array_t = std::array<std::size_t, VEC::length()>;
        using pair_t = std::pair<typename VEC::value_type, std::size_t>;
        using vector_queries_t = std::vector<pair_t>;

        /**
        * \brief given iterators to collection of points - construct grid
        * @param {forward_iterator, in} iterator to point cloud collection first point
        * @param {forward_iterator, in} iterator to point cloud collection last point
        **/
        template<std::forward_iterator InputIt>
            requires(std::is_same_v<point_t, typename std::decay_t<decltype(*std::declval<InputIt>())>>)
        constexpr void construct(const InputIt begin, const InputIt end) {
            using aabb_t = decltype(AxisLignedBoundingBox::point_cloud_aabb(begin, end));

            // build grid
            const aabb_t aabb{ AxisLignedBoundingBox::point_cloud_aabb(begin, end) };
            this->min = aabb.min;
            this->max = aabb.max;
            this->offset = GLSL::ceil(-aabb.min);
            this->gridMin = this->to_grid_position(aabb.min);
            this->gridMax = this->to_grid_position(aabb.max);
            if constexpr (k == 2) {
                this->numCells = { {static_cast<std::size_t>(std::ceil((std::abs(aabb.max[0] - aabb.min[0]) + one) / this->cellSize)),
                                    static_cast<std::size_t>(std::ceil((std::abs(aabb.max[1] - aabb.min[1]) + one) / this->cellSize))} };
            }
            else {
                this->numCells = { {static_cast<std::size_t>(std::ceil((std::abs(aabb.max[0] - aabb.min[0]) + one) / this->cellSize)),
                                    static_cast<std::size_t>(std::ceil((std::abs(aabb.max[1] - aabb.min[1]) + one) / this->cellSize)),
                                    static_cast<std::size_t>(std::ceil((std::abs(aabb.max[2] - aabb.min[2]) + one) / this->cellSize))} };
            }

            // fill grid
            this->first = Utilities::addressof(*begin);
            std::size_t i{};
            for (auto it{ begin }; it != end; ++it) {
                const point_t p{ *it };
                const std::size_t key{ this->to_index(p) };
                if (const auto node = this->bins.find(key); 
                    node == this->bins.end()) {
                    this->bins.insert({ key, std::vector<std::size_t>{} });
                }
                this->bins[key].emplace_back(i);
                ++i;
            }
        }

        /**
        * \brief clear grid
        **/
        constexpr void clear() noexcept {
            this->bins.clear();
            this->numCells = { {0} };
            this->gridMin = { {0} };
            this->gridMax = { {0} };
            this->min = VEC(coordinate_t{});
            this->max = VEC(coordinate_t{});
            this->offset = VEC(coordinate_t{});
            this->first = nullptr;
        }

        /**
        * \brief given point and query type, return all its neighbors within a specific enclosing area/volume around it
        * @param {RangeSearchType,              in} query type
        * @param {point_t,                      in} point to search around
        * @param {coordinate_t,                 in} extent/radius of search area/volume around point
        * @param {vector<coordinate_t, size_t>, out} collection of query nodes holding point squared distance and index in point cloud
        **/
        constexpr vector_queries_t range_query(const RangeSearchType type, const point_t point, const coordinate_t distance) const {
            // housekeeping
            const coordinate_t distance_criteria{ type == RangeSearchType::Manhattan ? distance : distance * distance };
            auto metric_function = (type == RangeSearchType::Manhattan) ?
                                   this->distance_metric<RangeSearchType::Manhattan>() :
                                   this->distance_metric<RangeSearchType::Radius>();
            vector_queries_t out;

            // lambda to check if point should be included in query
            const auto query_point = [this, METRIC_FUNC = FWD(metric_function), point, distance_criteria, &out]
                                     (index_array_t cellPosition) {
                const std::size_t key{ this->to_index(cellPosition) };
                const auto node = this->bins.find(key);
                if (node == this->bins.end()) {
                    return;
                }

                for (const std::size_t i : node->second) {
                    const point_t pos{ *(this->first + i) };
                    const point_t diff{ point - pos };
                    if (METRIC_FUNC(diff) <= distance_criteria) {
                        out.emplace_back(std::make_pair(GLSL::dot(diff), i));
                    }
                }
            };

            // world bounds for search
            const point_t radiusVec(distance);
            const point_t max1{ this->max + point_t(one) };
            const point_t _min{ GLSL::clamp(point - radiusVec, this->min, max1) };
            const point_t _max{ GLSL::clamp(point + radiusVec, this->min, max1) };

            // grid bound for search
            index_array_t minCell{ this->to_grid_position(_min) };
            index_array_t maxCell{ this->to_grid_position(_max) };
            Utilities::static_for<0, 1, VEC::length()>([this, &minCell, &maxCell](std::size_t i) {
                minCell[i] = Numerics::max(minCell[i], static_cast<std::size_t>(0));
                maxCell[i] = Numerics::min(maxCell[i] + 1, this->numCells[i]);
            });

            // search
            index_array_t cellpos{ {0} };
            if constexpr (this->k == 2) {
                for (cellpos[1] = minCell[1]; cellpos[1] < maxCell[1]; ++cellpos[1]) {
                    for (cellpos[0] = minCell[0]; cellpos[0] < maxCell[0]; ++cellpos[0]) {
                        query_point(cellpos);
                    }
                }
            }
            else {
                for (cellpos[2] = minCell[2]; cellpos[2] < maxCell[2]; ++cellpos[2]) {
                    for (cellpos[1] = minCell[1]; cellpos[1] < maxCell[1]; ++cellpos[1]) {
                        for (cellpos[0] = minCell[0]; cellpos[0] < maxCell[0]; ++cellpos[0]) {
                            query_point(cellpos);
                        }
                    }
                }
            }

            // output
            return out;
        }

        /**
        * \brief given point, 'k' of its nearest neighbors. notice that this method is recursive.
        * @param {point_t,                      in}  point to search around
        * @param {size_t,                       in}  amount of closest points
        * @param {vector<coordinate_t, size_t>, out} collection of query nodes holding point squared distance and index in point cloud
        **/
        constexpr vector_queries_t nearest_neighbors_query(const point_t point, const std::size_t N) const {
            // housekeeping
            const coordinate_t maxExtent{ GLSL::max(this->max - this->min) };
            coordinate_t cellSizeSearch{ this->cellSize };

            // search
#ifdef _DEBUG
            std::size_t iter{};
#endif
            vector_queries_t out;
            while ((out.size() < N) && (cellSizeSearch < maxExtent)) {
                out = range_query(RangeSearchType::Manhattan, point, cellSizeSearch);
                cellSizeSearch *= static_cast<coordinate_t>(2.0);
#ifdef _DEBUG
                ++iter;
#endif
            }

            Algoithms::sort(out.begin(), out.end(), [](const pair_t& a, const pair_t& b) {
                return a.first < b.first;
            });

            while (out.size() > N) {
                out.pop_back();
            }

            return out;
        }
        
        // internals
        private:
            // constants
            static const std::size_t k{ point_t::length() };
            inline static const coordinate_t one{ static_cast<coordinate_t>(1.0) };
            inline static const coordinate_t cellSize{ static_cast<coordinate_t>(1 << k) };

            // properties
            std::map<std::size_t, std::vector<std::size_t>> bins;
            index_array_t numCells{ {0} };           // number of cells in each dimension
            index_array_t gridMin{ {0} };            // grid cells minimal values
            index_array_t gridMax{ {0} };            // grid cells maximal values
            point_t min{};                           // grid min position
            point_t max{};                           // grid max position
            point_t offset{};                        // grid offset from coordinate system zero
            const point_t* first{ nullptr };         // iterator for point cloud start 

            /**
            * \brief given point in space, offset and k-value, return position in grid
            * @param {point_t,  in}  point
            * @param {numCells, out} position in grid
            **/
            constexpr index_array_t to_grid_position(const point_t pos) const {
                if constexpr (this->k == 2) {
                    return index_array_t{{ static_cast<std::size_t>(pos[0] + this->offset[0]) >> this->k,
                                           static_cast<std::size_t>(pos[1] + this->offset[1]) >> this->k }};
                }
                else {
                    return index_array_t{{ static_cast<std::size_t>(pos[0] + this->offset[0]) >> this->k,
                                           static_cast<std::size_t>(pos[1] + this->offset[1]) >> this->k,
                                           static_cast<std::size_t>(pos[2] + this->offset[2]) >> this->k }};
                }
            }

            /**
            * \brief given point and number of cells - return its cell index in the grid
            * @param {point_t, in}  point
            * @param {size_t,  out} index of cell holding point in grid
            **/
            constexpr std::size_t to_index(const point_t pos) const {
                return this->to_index(this->to_grid_position(pos));
            }

            /**
            * \brief given grid cell - return its index
            * @param {index_array_t, in}  cell
            * @param {size_t,        out} index of cell holding point in grid
            **/
            constexpr std::size_t to_index(const index_array_t cell) const {
                if constexpr (this->k == 2) {
                    return cell[0] + this->numCells[0] * cell[1];
                }
                else {
                    return cell[0] + this->numCells[0] * (cell[1] + this->numCells[1] * cell[2]);
                }
            }

            /**
            * \brief given range query type - return metric used in searching
            * @param {invocable, out} distance metric used for range query
            **/
            template<RangeSearchType TYPE>
            constexpr auto distance_metric() const {
                if constexpr (TYPE == RangeSearchType::Manhattan) {
                    return [](const point_t a) -> coordinate_t { return GLSL::max(GLSL::abs(a)); };
                }
                else if constexpr (TYPE == RangeSearchType::Radius) {
                    return [](const point_t a) -> coordinate_t { return GLSL::dot(a); };
                }
            }
    };
    static_assert(ISpacePartitioning<Grid<vec2>, vec2, std::vector<vec2>::iterator>);
}
