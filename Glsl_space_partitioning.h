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
    concept ISpacePartitioning = GLSL::is_fixed_vector_v<VEC> &&                                                             // VEC is a GLSL vector
                               std::is_same_v<VEC, typename SP::point_t> &&                                                  // GLSL vector type partitioned by data structure
                               std::is_same_v<typename VEC::value_type, typename SP::coordinate_t> &&                        // GLSL vector coordinate type
        requires(SP sp, VEC point, typename VEC::value_type t, RangeSearchType type, ITER it, std::size_t i) {
            /**
            * \brief given iterators to collection of points - construct space partitioning structure
            * @param {forward_iterator, in} iterator to point cloud collection first point
            * @param {forward_iterator, in} iterator to point cloud collection last point
            **/
            { sp.construct(it, it) } -> std::same_as<void>;

            /**
            * \brief clear kd-tree
            **/
            { sp.clear() } -> std::same_as<void>;

            /**
            * \brief given point and query type, return all its neighbors within this area/volume whose extent/radius is 't'
            * @param {RangeSearchType,                 in}  type of range query
            * @param {point_t,                         in}  point to search around
            * @param {coordinate_t,                    in}  extent of search cube around point
            * @param {vector<{point_t, coordinate_t}>, out} collection of pairs {point in range, squared distance between current point and queried points}
            **/
            { sp.range_query(type, point, t) } -> std::same_as<std::vector<std::pair<VEC, typename VEC::value_type>>>;

            /**
            * \brief given point, 'i' of its nearest neighbors
            * @param {point_t,                         in} point to search around
            * @param {size_t,                          in} amount of closest points
            * @param {vector<{point_t, coordinate_t}>, out} collection of pairs {point, squared distance between current point and queried points}
            **/
            { sp.nearest_neighbors_query(point, i) } -> std::same_as<std::vector<std::pair<VEC, typename VEC::value_type>>>;
    };

    /**
    * \brief balanced KD-tree for GLSL::IFixedVector objects.
    *
    * @param {IFixedVector} point held in tree
    **/
    template<GLSL::IFixedVector VEC>
    struct KDTree {
        // aliases
        using point_t = VEC;
        using coordinate_t = typename VEC::value_type;
        using pair_t = std::pair<point_t, coordinate_t>;
        using vector_pairs_t = std::vector<pair_t>;

        /**
        * \brief given iterators to collection of points - construct kd-tree in recursive manner using median partitioning
        * @param {forward_iterator, in} iterator to point cloud collection first point
        * @param {forward_iterator, in} iterator to point cloud collection last point
        **/
        template<std::forward_iterator InputIt>
            requires(std::is_same_v<point_t, typename std::decay_t<decltype(*std::declval<InputIt>())>>)
        constexpr void construct(const InputIt first, const InputIt last) {
            InputIt f(first);
            InputIt l(last);
            const auto len{ std::distance(f, l) };
            if (len < 0) {
                std::swap(f, l);
            }
            std::vector<std::int32_t> indices(len);
            std::iota(indices.begin(), indices.end(), 0);
            this->root = MOV(this->build_tree(f, 0, MOV(indices)));
        }

        /**
        * \brief clear kd-tree
        **/
        constexpr void clear() noexcept {
            this->clear_node(MOV(this->root));
            this->root.release();
        }

        /**
        * \brief given point and query type, return all its neighbors within a specific enclosing area/volume around it
        * @param {RangeSearchType,                 in} query type
        * @param {point_t,                         in} point to search around
        * @param {coordinate_t,                    in} extent/radius of search area/volume around point
        * @param {vector<{point_t, coordinate_t}>, out} collection of pairs {point in range, squared distance between current point and queried points}
        **/
        constexpr vector_pairs_t range_query(const RangeSearchType type, const point_t point, const coordinate_t distance) const {
            vector_pairs_t out{};
            std::stack<Node*> stack;
            stack.push(this->root.get());

            while (!stack.empty()) {
                const Node* node = stack.top();
                stack.pop();

                const point_t diff{ point - node->point };
                if (type == RangeSearchType::Manhattan) {
                    if (GLSL::max(GLSL::abs(diff)) <= distance) {
                        out.push_back(std::make_pair(node->point, GLSL::dot(diff)));
                    }
                }
                else if (type == RangeSearchType::Radius) {
                    if (const coordinate_t d2{ GLSL::dot(diff) }; d2 <= distance * distance) {
                        out.push_back(std::make_pair(node->point, d2));
                    }
                }

                const std::size_t split{ node->splitAxis };
                const coordinate_t dist_per_split{ diff[split] };

                if (node->left && dist_per_split <= coordinate_t{}) {
                    stack.push(node->left.get());
                }
                else if (node->right && dist_per_split > coordinate_t{}) {
                    stack.push(node->right.get());
                }

                if (std::abs(dist_per_split) > distance) {
                    continue;
                }

                if (node->right && dist_per_split <= coordinate_t{}) {
                    stack.push(node->right.get());
                }
                else if (node->left && dist_per_split > coordinate_t{}) {
                    stack.push(node->left.get());
                }
            }

            return out;
        }

        /**
        * \brief given point, 'k' of its nearest neighbors
        * @param {point_t,                         in}  point to search around
        * @paran {size_t,                          in}  amount of closest points
        * @param {vector<{point_t, coordinate_t}>, out} collection of 'k' nearest pairs {point in range, squared distance between current point and queried points}
        **/
        constexpr vector_pairs_t nearest_neighbors_query(const point_t point, const std::size_t k) const {
            std::priority_queue<pair_t, vector_pairs_t, less_pair_t> kMaxHeap;
            coordinate_t minDistance{ GLSL::dot(point - this->root->point) };

            std::stack<Node*> stack;
            stack.push(this->root.get());
            while (!stack.empty()) {
                const Node* node = stack.top();
                stack.pop();
                if (kMaxHeap.size() == k) {
                    minDistance = kMaxHeap.top().second;
                }

                const point_t diff{ point - node->point };
                if (const coordinate_t distance{ GLSL::dot(point - node->point) }; distance < minDistance) {
                    while (kMaxHeap.size() >= k) {
                        kMaxHeap.pop();
                    }
                    kMaxHeap.emplace(std::make_pair(node->point, distance));
                    minDistance = distance;
                }

                const std::size_t split{ node->splitAxis };
                const coordinate_t dist_per_split{ diff[split] };

                if (node->left && dist_per_split <= coordinate_t{}) {
                    stack.push(node->left.get());
                }
                else if (node->right && dist_per_split > coordinate_t{}) {
                    stack.push(node->right.get());
                }

                if (node->right && dist_per_split <= coordinate_t{}) {
                    stack.push(node->right.get());
                }
                else if (node->left && dist_per_split > coordinate_t{}) {
                    stack.push(node->left.get());
                }
            }

            // output
            vector_pairs_t out(k);
            std::size_t i{k-1};
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
                 VEC point;
                 std::unique_ptr<Node> left;
                 std::unique_ptr<Node> right;
                 std::size_t splitAxis{};
             };
             
             // tree root
             std::unique_ptr<Node> root;
        
             /**
             * \brief spatially divide the collection of points in recursive manner.
             * @param {forward_iterator, in} iterator to point cloud collection first point
             * @param {size_t,           in} node depth
             * @param {vector<int32_t>,  in} cloud points vector of indices
             **/
             template<std::forward_iterator InputIt>
                 requires(std::is_same_v<point_t, typename std::decay_t<decltype(*std::declval<InputIt>())>>)
             constexpr std::unique_ptr<Node> build_tree(const InputIt first, std::size_t depth, std::vector<std::int32_t>&& indices) {
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
                     [&](std::int32_t a, std::int32_t b) {
                         const point_t point_a(*(first + a));
                         const point_t point_b(*(first + b));
                         return (point_a[axis] < point_b[axis]);
                     });

                 // build and return node
                 return std::make_unique<Node>(Node{
                     .point = *(first + indices[medianIndex]),
                     .left = this->build_tree(first, depth + 1, MOV(std::vector<std::int32_t>(indices.begin(), indices.begin() + medianIndex))),
                     .right = this->build_tree(first, depth + 1, MOV(std::vector<std::int32_t>(indices.begin() + medianIndex + 1, indices.end()))),
                     .splitAxis = axis
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
                 constexpr bool operator()(const pair_t& a, const pair_t& b) const { return a.second < b.second; }
             };
             
    };

    static_assert(ISpacePartitioning<KDTree<vec2>, vec2, std::vector<vec2>::iterator>);
}
