#pragma once
#include "Glsl.h"
#include <vector>
#include <queue>
#include <stack>
#include <algorithm> // nth_element
#include <numeric> // iota
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
            *                                              if  we queries something relative to point 'P' then the following holds: GLSL::dot(P - *(it + second)) = first
            **/
            { sp.nearest_neighbors_query(point, i) } -> std::same_as<std::vector<std::pair<typename VEC::value_type, std::size_t>>>;
    };

    /**
    * \brief balanced KD-tree for GLSL::IFixedVector objects.
    *        notice that tree doesn't hold the points themselvs, rather it points to their position in their original structure.
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
            this->first = std::addressof(*begin);
            std::vector<std::size_t> indices(static_cast<std::size_t>(std::distance(begin, end)));
            std::iota(indices.begin(), indices.end(), 0);
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
            std::stack<Node*> stack;
            stack.push(this->root.get());
            while (!stack.empty()) {
                const Node* node{ stack.top() };
                stack.pop();

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
                    stack.push(nodes[inside]);
                }
                if (nodes[!inside] && std::abs(dist_per_split) < distance) {
                    stack.push(nodes[!inside]);
                }
            }

            return out;
        }

        /**
        * \brief given point, 'k' of its nearest neighbors. notice that this method is recursive.
        * @param {point_t,                      in}  point to search around
        * @paran {size_t,                       in}  amount of closest points
        * @param {vector<coordinate_t, size_t>, out} collection of query nodes holding point squared distance and index in point cloud
        **/
        constexpr vector_queries_t nearest_neighbors_query(const point_t point, const std::size_t k) const {
            // housekeeping
            std::priority_queue<pair_t, vector_queries_t, less_pair_t> kMaxHeap;
            coordinate_t minDistance{ GLSL::dot(*(this->first + this->root->index) - point) };

            // lambda implementing nearest neighbors query logic for one node
            const auto nearest_neighbors_query_recursive = [this, &kMaxHeap, point, k](const Node* node, coordinate_t& mindistance, auto&& recursive_driver) -> void {
                const point_t nodePoint(*(this->first + node->index));
                const point_t diff{ point - nodePoint };
                if (const coordinate_t distance{ GLSL::dot(diff) }; distance < mindistance) {
                    while (kMaxHeap.size() >= k) {
                        kMaxHeap.pop();
                    }
                    kMaxHeap.emplace(std::make_pair(distance, node->index));
                    mindistance = kMaxHeap.top().first;
                }

                const std::size_t split{ node->splitAxis };
                const coordinate_t dist_per_split{ diff[split] };
                const std::size_t inside{ dist_per_split <= coordinate_t{} };

                std::array<Node*, 2> nodes{ {node->right.get(), node->left.get()} };
                if (nodes[inside]) {
                    recursive_driver(nodes[inside], mindistance, recursive_driver);
                }
                if (nodes[!inside] && std::abs(dist_per_split) <= mindistance) {
                    recursive_driver(nodes[!inside], mindistance, recursive_driver);
                }
            };

            // find k nearest neighbors
            nearest_neighbors_query_recursive(this->root.get(), minDistance, nearest_neighbors_query_recursive);

            // output
            vector_queries_t out(k);
            std::size_t i{ k-1 };
            for (; !kMaxHeap.empty(); kMaxHeap.pop()) {
                out[i] = kMaxHeap.top();
                --i;
            }
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
            * @param {size_t,           in} node depth
            * @param {vector<int32_t>,  in} cloud points vector of indices
            **/
            constexpr std::unique_ptr<Node> build_tree(std::size_t depth, std::vector<std::size_t> indices) {
                constexpr std::size_t N{ point_t::length() };
                if (indices.empty()) {
                    return nullptr;
                }
        
                // partition range according to its first coordinate
                const std::size_t medianIndex{ indices.size() / 2 };
                const std::size_t axis{ depth % N };
                std::ranges::nth_element(indices.begin(),
                                         indices.begin() + medianIndex,
                                         indices.end(),
                    [&](std::size_t a, std::size_t b) {
                        const point_t point_a(*(this->first + a));
                        const point_t point_b(*(this->first + b));
                        return (point_a[axis] < point_b[axis]);
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
}