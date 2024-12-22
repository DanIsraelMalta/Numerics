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
#include "Algorithms.h"
#include "Glsl.h"
#include "Glsl_extra.h"
#include "Glsl_point_distance.h"
#include "Glsl_axis_aligned_bounding_box.h"
#include "DiamondAngle.h"
#include "Glsl_triangle.h"
#include <limits>
#include <vector>
#include <iterator>
#include <random>
#include <queue> // earcut

//
// collection of algorithms for 2D cloud points and shapes
//
namespace Algorithms2D {

    //
    // utilities
    //
    namespace Internals {

        /**
        * \brief check if a point is counter clock wise relative to two other points
        * @param {IFixedVector, in}  point a
        * @param {IFixedVector, in}  point b
        * @param {IFixedVector, in}  point c
        * @param {value_type,   out} negative value means 'c' is counter clockwise to segment a-b,
        *                            positive value means 'c' is clockwise to segment a-b,
        *                            zero means 'c' is colinear with segment a-b,
        **/
        template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
            requires(VEC::length() == 2)
        constexpr T are_points_ordered_counter_clock_wise(const VEC& a, const VEC& b, const VEC& c) noexcept {
            return Numerics::diff_of_products(b.y - a.y, c.x - b.x, b.x - a.x, c.y - b.y);
        }

        /**
        * \brief check if a point is "left" to another point
        * @param {IFixedVector, in}  point a
        * @param {IFixedVector, in}  point b
        * @param {bool,         out} true if point 'a' is left to point 'b'
        **/
        template<GLSL::IFixedVector VEC>
            requires(VEC::length() == 2)
        constexpr bool is_point_left_of(const VEC& a, const VEC& b) noexcept {
            return (a.x < b.x || (a.x == b.x && a.y < b.y));
        }

        /**
        * \brief return twice the triangle signed area
        * @param {IFixedVector, in}  point a
        * @param {IFixedVector, in}  point b
        * @param {IFixedVector, in}  point c
        * @param {floating,     out} twice triangle signed area
        **/
        template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
            requires(VEC::length() == 2)
        constexpr T triangle_twice_signed_area(const VEC& a, const VEC& b, const VEC& c) noexcept {
            const VEC v1{ a - c };
            const VEC v2{ b - c };
            return Numerics::diff_of_products(v1.x, v2.y, v1.y, v2.x);
        }

        /**
        * \brief return twice a triangle area
        * @param {IFixedVector, in}  point a
        * @param {IFixedVector, in}  point b
        * @param {IFixedVector, in}  point c
        * @param {floating,     out} twice triangle area
        **/
        template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
            requires(VEC::length() == 2)
        constexpr T triangle_twice_area(const VEC& a, const VEC& b, const VEC& c) noexcept {
            return std::abs(triangle_twice_signed_area(a, b, c));
        }

        /**
        * \brief return triangle area
        * @param {IFixedVector, in}  ray #1 origin
        * @param {IFixedVector, in}  ray #1 direction
        * @param {IFixedVector, in}  ray #2 origin
        * @param {IFixedVector, in}  ray #2 direction
        * @param {IFixedVector, out} ray #1 and ray #2 intersection point (vector filled with std::numeric_limits<T>::max() if intersection does not occure)
        **/
        template<GLSL::IFixedVector VEC>
            requires(VEC::length() == 2)
        constexpr VEC get_rays_intersection_point(const VEC& ro0, const VEC& rd0, const VEC& ro1, const VEC& rd1) noexcept {
            using T = typename VEC::value_type;
            assert(Extra::is_normalized(rd0));
            assert(Extra::is_normalized(rd1));
            if (GLSL::equal(ro0, ro1)) [[unlikely]] {
                return ro0;
            }

            const T det{ rd1.x * rd0.y - rd1.y * rd0.x };
            if (Numerics::areEquals(det, T{})) [[unlikely]] {
                return VEC(std::numeric_limits<T>::max());
            }
            [[assume(det > T{})]];

            const VEC d{ ro1 - ro0 };
            const T u{ (d.y * rd1.x - d.x * rd1.y) / det };
            if (const T v{ (d.y * rd0.x - d.x * rd0.y) / det }; u < T{} || v < T{}) [[unlikely]] {
                return VEC(std::numeric_limits<T>::max());
            }
            return ro0 + u * rd0;
        }

        /**
        * \brief test if two segments intersect
        * @param {IFixedVector, in}  segment #1 point #1
        * @param {IFixedVector, in}  segment #1 point #2
        * @param {IFixedVector, in}  segment #1 point #1
        * @param {IFixedVector, in}  segment #1 point #2
        * @param {bool,         out} true if two segments intersect, false otherwise
        **/
        template<GLSL::IFixedVector VEC>
            requires(VEC::length() == 2)
        constexpr bool do_segments_intersect(const VEC& a0, const VEC& a1, const VEC& b0, const VEC& b1) noexcept {
            using T = typename VEC::value_type;
            const VEC s1{ a1 - a0 };
            const VEC s2{ b1 - b0 };
            const T denom{ s1.x * s2.y - s2.x * s1.y};
            if (Numerics::areEquals(denom, T{})) {
                return false;
            }

            const VEC ab{ a0 - b0 };
            const T s{ Numerics::diff_of_products(s1.x, ab.y, s1.y, ab.x) / denom };
            const T t{ Numerics::diff_of_products(s2.x, ab.y, s2.y, ab.x) / denom };
            return (s >= T{} && s <= static_cast<T>(1.0) &&
                    t >= T{} && t <= static_cast<T>(1.0));
        }

        /**
        * \brief project point on segment
        * @param {IFixedVector,               in}  segment point #1
        * @param {IFixedVector,               in}  segment point #2
        * @param {IFixedVector,               in}  point
        * @param {{IFixedVector, value_type}, out} {point projected on segment, interpolant along segment from point #1 to projected point}
        **/
        template<GLSL::IFixedVector VEC>
            requires(VEC::length() == 2)
        constexpr auto project_point_on_segment(const VEC& a, const VEC& b, const VEC& p) noexcept {
            using T = typename VEC::value_type;
            using out_t = struct { VEC point; T t; };

            const VEC ap{ p - a };
            const VEC ab{ b - a };

            const T ap_dot_ab{ GLSL::dot(ap, ab) };
            const T ab_dot{ GLSL::dot(ab) };
            assert(ab_dot > T{});

            const T t{ ap_dot_ab / ab_dot };
            return out_t{ a + t * ab, t };
        }

        /**
        * \brief get the circumcircle of two/three points
        * @param {IFixedVector,               in}  point #1
        * @param {IFixedVector,               in}  point #2
        * @param {IFixedVector,               in}  point #3 (optional)
        * @param {{IFixedVector, value_type}, out} {circumcircle center, circumcircle squared radius}
        *                                          center will be at center and squared radius will be negative in case of invalid calculation.
        **/
        template<GLSL::IFixedVector VEC>
            requires(VEC::length() == 2)
        constexpr auto get_circumcircle(const VEC& a, const VEC& b) noexcept {
            using T = typename VEC::value_type;
            using out_t = struct { VEC center; T radius_squared; };
            return out_t{ (a + b) / static_cast<T>(2), GLSL::dot(a - b) / static_cast<T>(4) };
        }
        template<GLSL::IFixedVector VEC>
            requires(VEC::length() == 2)
        constexpr auto get_circumcircle(const VEC& a, const VEC& b, const VEC& c) noexcept {
            using T = typename VEC::value_type;
            using out_t = decltype(Algorithms2D::Internals::get_circumcircle(a, b));

            const VEC ba{ b - a };
            const VEC ca{ c - a };
            const T B{ GLSL::dot(ba) };
            const T C{ GLSL::dot(ca) };
            const T D{ GLSL::cross(ba, ca) };
            if (Numerics::areEquals(D, T{})) {
                return out_t{ VEC(), static_cast<T>(-1) };
            }
            const VEC center{ a + VEC(Numerics::diff_of_products(ca.y, B, ba.y, C),
                                      Numerics::diff_of_products(ba.x, C, ca.x, B)) / (static_cast<T>(2) * D) };
            return out_t{ center, GLSL::dot(center - a) };
        }
        template<GLSL::IFixedVector VEC>
            requires(VEC::length() == 2)
        constexpr auto get_circumcircle(const VEC& a, const VEC& b, const VEC& c, const VEC& d) noexcept {
            using T = typename VEC::value_type;
            using out_t = decltype(Algorithms2D::Internals::get_circumcircle(a, b));

            // does 'abc' circle include 'd'?
            out_t circle{ Internals::get_circumcircle(a, b, c) };
            bool inside{ GLSL::dot(d - circle.center) <= circle.radius_squared };
            if (inside) {
                return circle;
            }

            // does 'abd' circle include 'c'?
            circle = Internals::get_circumcircle(a, b, d);
            inside = GLSL::dot(c - circle.center) <= circle.radius_squared;
            if (inside) {
                return circle;
            }

            // does 'acd' circle include 'b'?
            circle = Internals::get_circumcircle(a, c, d);
            inside = GLSL::dot(b - circle.center) <= circle.radius_squared;
            if (inside) {
                return circle;
            }

            // 'bcd' circle must include 'a'...
            return Internals::get_circumcircle(b, c, d);
        }

        /**
        * \brief given collection of points, return it sorted in clock wise manner
        * @param {forward_iterator, in}  iterator to first point in collection
        * @param {forward_iterator, in}  iterator to last point in collection
        * @param {IFixedVector,     out} centroid
        **/
        template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
            requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
        constexpr VEC get_centroid(const InputIt first, const InputIt last) {
            using T = typename VEC::value_type;
            assert(std::distance(first, last) > 0);

            VEC centroid;
            for (auto it{ first }; it != last; ++it) {
                centroid += *it;
            }

            return (centroid / static_cast<T>(std::distance(first, last)));
        }

        /**
        * \brief given a polygon, return its area
        * @param {forward_iterator, in}  iterator to polygon first point
        * @param {forward_iterator, in}  iterator to polygon last point
        * @param {value_type,       out} polygon area
        **/
        template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>, class T = VEC::value_type>
            requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
        constexpr T get_polygon_area(const InputIt first, const InputIt last) {
            // vertex has no area
            if (std::distance(first, last) == 0) {
                return T{};
            }

            // housekeeping
            T area{};
            const auto accumulate_area = [&area](const VEC v0, const VEC v1) {
                area += (v0.x * v1.y - v1.x * v0.y);
            };

            // accumulate polygon triangles area
            for (auto it{ first }, nt{first + 1}; nt != last; ++it, ++nt) {
                accumulate_area(*it, *nt);
            }
            accumulate_area(*(last - 1), *first);

            // output
            return std::abs(area / static_cast<T>(2.0));
        }

        /**
        * \brief given a collection of points which define segments of closed polygon and a point, return the indices of points which define the segment which is closest to the point
        * @param {vector<IFixedVector>, in}  cloud of points, each two consecutive points define a segment
        * @param {IFixedVector,         in}  point
        * @param {{value_type, size_t}, out} {squared distance, index of point #1 in collection which define closest segment}
        **/
        template<GLSL::IFixedVector VEC>
            requires(VEC::length() == 2)
        constexpr auto get_index_of_closest_segment(const std::vector<VEC>& cloud, const VEC point) noexcept {
            using T = typename VEC::value_type;
            using out_t = struct { T distance_squared; std::size_t index; };

            // housekeeping
            std::size_t index{};
            T distSquared{ std::numeric_limits<T>::max() };
            const auto update_point = [&point, &index, &distSquared](const VEC& a, const VEC& b, const std::size_t i) {
                const T dist2{ PointDistance::squared_udf_to_segment(point, a, b) };
                const bool closest{ dist2 < distSquared };
                distSquared = closest ? dist2 : distSquared;
                index = closest ? i : index;
            };

            // iterate over all segments
            const std::size_t N{ cloud.size() - 1};
            for (std::size_t i{}; i < N - 1; ++i) {
                update_point(cloud[i], cloud[i + 1], i);
            }

            // check "closed segment"
            update_point(cloud[0], cloud[N], N);

            // output
            return out_t{ distSquared, index };
        }

        /**
        * \brief given a closed polygon (as a collection of points) return iterators to two monotone chains
        *        going from minimum of each coordinate to the maximum of that coordinate
        * @param {forward_iterator,             in}  iterator to first point in polygon
        * @param {forward_iterator,             in}  iterator to last point in polygon
        * @param {{array<forward_iterator, 2>,       {iterator from minimal x coordinate to maximal x coordinate,
        *          array<forward_iterator, 2>,        iterator from minimal y coordinate to maximal y coordinate}
        *          array<value_type, 2>,              minimal and maximal x values,
        *          array<value_type, 2>,},      out}  minimal and maximal y values}
        **/
        template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
            requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
        constexpr auto get_monotone_chains(const InputIt first, const InputIt last) {
            using T = typename VEC::value_type;
            using out_t = struct {
                std::array<InputIt, 2> x_monotone_chain;
                std::array<InputIt, 2> y_monotone_chain;
                std::array<T, 2> x_min_max;
                std::array<T, 2> y_min_max;
            };

            // project points on line and find "leftmost" (min) and "rightmost" (max) points on each coordinate
            T xMin{ std::numeric_limits<T>::max() };
            T yMin{ std::numeric_limits<T>::max() };
            T xMax{ -std::numeric_limits<T>::max() };
            T yMax{ -std::numeric_limits<T>::max() };
            InputIt xMinIterator;
            InputIt xMaxIterator;
            InputIt yMinIterator;
            InputIt yMaxIterator;
            for (InputIt it{ first }; it != last; ++it) {
                const VEC p{ *it };

                if (p.x < xMin) {
                    xMin = p.x;
                    xMinIterator = it;
                }
                if (p.x > xMax) {
                    xMax = p.x;
                    xMaxIterator = it;
                }

                if (p.y < yMin) {
                    yMin = p.y;
                    yMinIterator = it;
                }
                if (p.y > yMax) {
                    yMax = p.y;
                    yMaxIterator = it;
                }
            }

            if (std::distance(xMinIterator, xMaxIterator) < 0) {
                Utilities::swap(xMinIterator, xMaxIterator);
            }

            if (std::distance(yMinIterator, yMaxIterator) < 0) {
                Utilities::swap(yMinIterator, yMaxIterator);
            }

            // output
            return out_t{ std::array<InputIt, 2>{{xMinIterator, xMaxIterator}}, std::array<InputIt, 2>{{yMinIterator, yMaxIterator}},
                          std::array<T, 2>{{xMin, xMax}}, std::array<T, 2>{{yMin, yMax}} };
        }
    };

    /**
    * \brief calculate the convex hull of collection of 2D points (using Graham scan algorithm).
    *       
    * @param {forward_iterator,     in}  iterator to point cloud collection first point
    * @param {forward_iterator,     in}  iterator to point cloud collection last point
    * @param {vector<IFixedVector>, out} collection of points which define point cloud convex hull (points are sorted counter clock wise)
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC> && (VEC::length() == 2))
    constexpr std::vector<VEC> get_convex_hull(const InputIt first, const InputIt last) {
        using T = typename VEC::value_type;
        std::vector<VEC> points(first, last);

        // place left most point at start of point cloud
        const InputIt minElementIterator{ Algoithms::min_element(points.begin(), points.end(),
                                                [](const VEC& a, const VEC& b) noexcept -> bool {
            return (a.x < b.x || (a.x == b.x && a.y < b.y));
        }) };
        Utilities::swap(points[0], *minElementIterator);

        // lexicographically sort all points using the smallest point as pivot
        const VEC v0( points[0] );
        Algoithms::sort(points.begin() + 1, points.end(), [v0](const VEC& b, const VEC& c) noexcept -> bool {
            return Numerics::diff_of_products(b.y - v0.y, c.x - b.x, b.x - v0.x, c.y - b.y) < T{};
        });

        // build hull
        auto it = points.begin();
        std::vector<VEC> hull{ {*it++, *it++, *it++} };
        while (it != points.end()) {
            while (Internals::are_points_ordered_counter_clock_wise(*(hull.rbegin() + 1), *(hull.rbegin()), *it) >= T{}) {
                hull.pop_back();
            }
            hull.push_back(*it++);
        }

        return hull;
    }

    /**
    * \brief given convex hull, return its minimal area bounding rectangle
    * @param {vector<IFixedVector>,                                     in}  convex hull
    * @param {{IFixedVector, IFixedVector, IFixedVector, IFixedVector}, out} vertices of minimal area bounding rectangle (ordererd counter clock wise)
    **/
    template<GLSL::IFixedVector VEC>
        requires(VEC::length() == 2)
    constexpr auto get_convex_hull_minimum_area_bounding_rectangle(const std::vector<VEC>& hull) {
        using T = typename VEC::value_type;
        using out_t = struct { VEC p0; VEC p1; VEC p2; VEC p3; };

        // iterate over all convex hull edges and find minimal area bounding rectangle
        T area{ std::numeric_limits<T>::max() };
        VEC p0;
        VEC p1;
        VEC maxNormal;
        for (std::size_t i{}; i < hull.size() - 2; ++i) {
            // segment
            VEC v0{ hull[i] };
            VEC v1{ hull[i + 1] };

            // segment distance to center
            const VEC center{ (v0 + v1) / static_cast<T>(2) };
            T v0_dist{ GLSL::dot(v0 - center) };
            T v1_dist{ v0_dist };
            T vNormal_dist{};

            // segment tangential and orthogonal directions
            const VEC dir{ GLSL::normalize(v1 - v0) };
            const VEC normal(-dir.y, dir.x);

            // point on orthogonal segment
            const VEC v2_0{ v0 + normal };
            const VEC v2_1{ v1 + normal };
            VEC vNormal(center);
            VEC ref;
            
            // project points on line connecting convex hull edge and find minimal/maximal points.
            // project points on orthogonal line to edge (should be going inside the convex hull) and find maximal point.
            for (const VEC point: hull) {
                // project points on segment tangential and orthogonal directions (orthogonal towards inside hull)
                const auto projOnDir{ Internals::project_point_on_segment(v0, v1, point) };
                const auto projOnNormal_0{ Internals::project_point_on_segment(v0, v2_0, point) };
                const auto projOnNormal_1{ Internals::project_point_on_segment(v1, v2_1, point) };
                
                // find furthest points along v0v1 segments which can constitute an extent to tight bounding box
                if (const T projOnDir_dist{ GLSL::dot(projOnDir.point - center) }; projOnDir.t < T{} && projOnDir_dist > v0_dist) {
                    v0_dist = projOnDir_dist;
                    v0 = projOnDir.point;
                } else if (projOnDir.t > T{} && projOnDir_dist > v1_dist) {
                    v1_dist = projOnDir_dist;
                    v1 = projOnDir.point;
                }
                
                // find furthest point from v0v1 segment along orthogonal direction to v0v1
                if (const T dist{ GLSL::dot(projOnNormal_0.point - v2_0) }; dist > vNormal_dist) {
                    vNormal_dist = dist;
                    vNormal = projOnNormal_0.point;
                    ref = v0;
                }
                if (const T dist{ GLSL::dot(projOnNormal_1.point - v2_1) }; dist > vNormal_dist) {
                    vNormal_dist = dist;
                    vNormal = projOnNormal_1.point;
                    ref = v1;
                }
            }

            // rectangle area
            const T rectangle_area{ GLSL::dot(v1 - v0) * GLSL::dot(vNormal - ref) };
            if (rectangle_area < area) {
                area = rectangle_area;
                p0 = v0;
                p1 = v1;
                maxNormal = vNormal;
            }
        }

        // calculate bounding rectangle vertices
        const VEC dir{ GLSL::normalize(p1 - p0) };
        const VEC normal{ -dir.y, dir.x };
        const VEC p2{ Internals::get_rays_intersection_point(p1, normal, maxNormal,  dir) };
        const VEC p3{ Internals::get_rays_intersection_point(p0, normal, maxNormal, -dir) };

        return out_t{ p0, p1, p2, p3 };
    }
    
    /**
    * \brief given convex hull of collection of points, return its diameter
    * @param {vector<IFixedVector>,           in}  points convex hull
    * @param {{value_type, array<size_t, 2>}, out} {squared diameter, <index of anti podal oint #1, index of anti podal point #2>}
    **/
    template<GLSL::IFixedVector VEC>
        requires(VEC::length() == 2)
    constexpr auto get_convex_diameter(const std::vector<VEC>& hull) {
        using T = typename VEC::value_type;
        using out_t = struct { T diamater_squared; std::array<std::size_t, 2> indices; };

        // housekeeping
        const std::size_t N{ hull.size() };
        out_t out{
            .diamater_squared = T{},
            .indices = std::array<std::size_t, 2>{{0, 0}}
        };
        const auto checkPoints = [N, &out, &hull](const std::size_t i, const std::size_t j) {
            const VEC a{ hull[i % N] };
            const VEC b{ hull[j % N] };
            const T furthest{ GLSL::dot(a - b) };
            const bool update{ furthest > out.diamater_squared };
            out.diamater_squared = update ? furthest : out.diamater_squared;
            out.indices = update ? std::array<std::size_t, 2>{{i, j}} : out.indices;
        };

        std::size_t k{ 1 };
        const VEC hull_0{ hull[0] };
        const VEC hull_n1{ hull[N - 1] };
        while (Internals::triangle_twice_area(hull_n1, hull_0, hull[(k + 1) % N]) >
               Internals::triangle_twice_area(hull_n1, hull_0, hull[k])) {
            ++k;
            checkPoints(N - 1, k);
        }

        for (std::size_t i{}, j{ k }; i <= k && j < N; ++i) {
            const VEC hull_i{ hull[i] };
            const VEC hull_i_1{ hull[(i + 1) % N] };

            while (j < N &&
                   Internals::triangle_twice_area(hull_i, hull_i_1, hull[(j + 1) % N]) >
                   Internals::triangle_twice_area(hull_i, hull_i_1, hull[j])) {
                checkPoints(i, (j + 1) % N);
                ++j;
            }
        }

        return out;
    }

    /**
    * \brief given convex hull of collection of points, return the minimal bounding circle (circumcircle).
    *        based on https://www.cise.ufl.edu/~sitharam/COURSES/CG/kreveldnbhd.pdf.
    * @param {vector<IFixedVector>,       in}  points convex hull
    * @param {{IFixedVector, value_type}, out} {minimal bounding circle center, minimal bounding circle squared radius}
    **/
    template<GLSL::IFixedVector VEC>
        requires(VEC::length() == 2)
    constexpr auto get_minimal_bounding_circle(const std::vector<VEC>& hull) {
        using T = typename VEC::value_type;
        using out_t = decltype(Algorithms2D::Internals::get_circumcircle(hull[0], hull[0]));

        // is hulll composed of two/three/four points?
        const std::size_t N{ hull.size() };
        if (N == 2) {
            return Internals::get_circumcircle(hull[0], hull[1]);
        } else if (N == 3) {
            return Internals::get_circumcircle(hull[0], hull[1], hull[2]);
        } else if (N == 4) {
            return Internals::get_circumcircle(hull[0], hull[1], hull[2], hull[3]);
        }

        // lambda to check if a given point is inside given circle
        const auto is_point_in_circle = [](const out_t& circle, const VEC& point) -> bool {
            return GLSL::dot(point - circle.center) <= circle.radius_squared;
        };

        // find minimal bounding circle of set of points using two points
        const auto make_bounding_circle_two_points = [&hull, &is_point_in_circle](const std::size_t end, const VEC& p, const VEC& q) -> out_t {
            out_t circ{ Internals::get_circumcircle(p, q) };
            out_t left{ VEC(), static_cast<T>(-1) };
            out_t right{ VEC(), static_cast<T>(-1) };

            const VEC pq{ q - p };
            for (std::size_t i{}; i < end; ++i) {
                const VEC r{ hull[i] };
                if (is_point_in_circle(circ, r)) {
                    continue;
                }

                const out_t c{ Internals::get_circumcircle(p, q, r) };
                if (c.radius_squared <= T{}) {
                    continue;
                }

                const T cross{ GLSL::cross(pq, r - p) };
                const T pq_cross{ GLSL::cross(pq, c.center - p) };
                if (cross > T{} && (left.radius_squared < T{} || pq_cross > GLSL::cross(pq, left.center - p))) {
                    left.center = c.center;
                    left.radius_squared = c.radius_squared;
                }
                else if (cross < T{} && (right.radius_squared < T{} || pq_cross < GLSL::cross(pq, right.center - p))) {
                    right.center = c.center;
                    right.radius_squared = c.radius_squared;
                }
            }

            if (left.radius_squared <= T{} && right.radius_squared <= T{}) {
                return circ;
            }
            else if (left.radius_squared < T{}) {
                return right;
            }
            else if (right.radius_squared < T{}) {
                return left;
            }
            else {
                return (left.radius_squared <= right.radius_squared) ? left : right;
            }
        };

        // find minimal bounding circle of set of points using one point
        const auto make_bounding_circle_one_points = [&hull, &is_point_in_circle, &make_bounding_circle_two_points, N](const std::size_t end, const VEC& p) -> out_t {
            out_t circle{ p, static_cast<T>(-1) };

            for (std::size_t i{}; i < end; ++i) {
                const VEC q{ hull[i] };

                if (!is_point_in_circle(circle, q)) {
                    if (circle.radius_squared <= T{}) {
                        circle = Internals::get_circumcircle(p, q);
                    } else {
                        circle = make_bounding_circle_two_points(Numerics::min(i + 1, N), p, q);
                    }
                }
            }

            return circle;
        };

        // iterate over remaining points and find minimal enclosing circle
        out_t circle{ Internals::get_circumcircle(hull[0], hull[1], hull[2], hull[3]) };
        for (std::size_t i{ 4 }; i < N; ++i) {
            const VEC p{ hull[i] };
            if (!is_point_in_circle(circle, p)) {
                circle = make_bounding_circle_one_points(Numerics::min(i + 1, N), p);
            }
        }

        return circle;
    }

    /**
    * \brief given closed polygon and a point - check if point is inside polygon
    *        (see: https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html)
    *        should be faster than PointDistance::sdf_to_polygon function in "Glsl_point_distance.h"
    * @param {forward_iterator, in}  iterator to polygon first point
    * @param {forward_iterator, in}  iterator to polygon last point
    * @param {IFixedVector,     in}  point
    * @param {bool,             out} true if point is inside polygon, false otherwise
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC> && VEC::length() == 2)
    constexpr bool is_point_inside_polygon(const InputIt first, const InputIt last, const VEC& point) {
        using T = typename VEC::value_type;
        using iter_size_t = std::iterator_traits<InputIt>::difference_type;

        bool inside{ false };
        for (iter_size_t len{ std::distance(first, last) }, i{}, j{ len - 2 }; i < len - 1; j = i++) {
            const VEC pi{ *(first + i) };
            const VEC pj{ *(first + j) };
            const bool intersects{ pi.y > point.y != pj.y > point.y &&
                                   point.x < ((pj.x - pi.x) * (point.y - pi.y)) / (pj.y - pi.y) + pi.x };
            inside = intersects ? !inside : inside;
        }

        return inside;
    }

    /**
    * \brief given collection of points, return true if they are ordered in clock wise manner
    * @param {forward_iterator, in}  iterator to cloud points collection first point
    * @param {forward_iterator, in}  iterator to cloud points collection last point
    * @param {IFixedVector,     in}  points geometric center
    * @param {bool,             out} true if point are ordered in clock wise manner, false otherwise
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr bool are_points_ordererd_clock_wise(const InputIt first, const InputIt last, const VEC& centroid) {
        using T = typename VEC::value_type;
        using iter_size_t = std::iterator_traits<InputIt>::difference_type;

        bool clockwise{ false };
        for (InputIt f{ first }, s{ first + 1 }; s != last; ++f, ++s) {
            const VEC a{ *f };
            const VEC b{ *s };
            const T angle_a{ DiamondAngle::atan2(a.y - centroid.y, a.x - centroid.x) };
            const T angle_b{ DiamondAngle::atan2(b.y - centroid.y, b.x - centroid.x) };

            clockwise = angle_a > angle_b ? !clockwise : clockwise;
        }

        return clockwise;
    }

    /**
    * \brief given collection of points, sort them in clock wise manner
    * @param {forward_iterator,     in}  iterator to point collection first point
    * @param {forward_iterator,     in}  iterator to point collection last point
    * @param {IFixedVector,         in}  points geometric center
    * @param {vector<IFixedVector>, out} points sorted in clock wise manner
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC> && VEC::length() == 2)
    constexpr void sort_points_clock_wise(InputIt first, InputIt last, const VEC centroid) {
        using T = typename VEC::value_type;

        Algoithms::sort(first, last, [centroid](const VEC& a, const VEC& b) noexcept -> bool {
            const T angla_a{ DiamondAngle::atan2(a.y - centroid.y, a.x - centroid.x) };
            const T angla_b{ DiamondAngle::atan2(b.y - centroid.y, b.x - centroid.x) };
            return angla_a > angla_b;
        });
    }

    /**
    * \brief calculate the concave hull of collection of 2D points (using Graham scan algorithm).
    *
    * @param {forward_iterator,     in}  iterator to point cloud collection first point
    * @param {forward_iterator,     in}  iterator to point cloud collection last point
    * @param {value_type,           in}  concave threshold. the minimal ratio between segment length with added point and original segment length.
    *                                    if the value is smaller than the threshold - the point is part of the concave.
    *                                    the larger the threshold - the more points will be part of the concave.
    *                                    default is 0, i.e - convex hull
    * @param {vector<IFixedVector>, out} collection of points which define point cloud concave hull
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>, class T = typename VEC::value_type>
        requires(GLSL::is_fixed_vector_v<VEC> && (VEC::length() == 2))
    constexpr std::vector<VEC> get_concave_hull(const InputIt first, const InputIt last, T concave_threshold = T{}) {
        // get convex hull
        std::vector<VEC> hull{ Algorithms2D::get_convex_hull(first, last) };
        if (concave_threshold <= T{}) {
            return hull;
        }

        // remove convex hull points from cloud
        std::vector<VEC> cloud(first, last);
        for (const VEC h : hull) {
            for (std::size_t i{}; i < cloud.size(); ++i) {
                if (Extra::are_vectors_identical(h, cloud[i])) {
                    Utilities::swap(cloud[i], cloud.back());
                    cloud.pop_back();
                }
            }
        }

        // "dig" into convex hull to create concave hull
        std::size_t i{};
        while(i < cloud.size()) {
            // find hull segment which cloud point is closest to
            const VEC p{ cloud[i] };
            const auto closest = Algorithms2D::Internals::get_index_of_closest_segment(hull, p);

            // should point be part of concave hull?
            const std::size_t segment_index_start{ closest.index };
            const std::size_t segment_index_end{ (segment_index_start + 1) % hull.size() };
            const T segment_length{ GLSL::distance(hull[segment_index_start], hull[segment_index_end]) };
            const T new_segment_length{ GLSL::distance(p, hull[segment_index_end]) +
                                        GLSL::distance(hull[segment_index_start], p) };
            [[assume(segment_length > T{})]];
            [[assume(new_segment_length > T{})]];
            const T segment_length_ratio{ new_segment_length / segment_length };
            if (Numerics::areEquals(segment_length, new_segment_length, Numerics::equality_precision<T>()) || segment_length_ratio <= concave_threshold) {
                hull.insert(hull.begin() + segment_index_start + 1, p);
                cloud.erase(cloud.begin() + i);
                i = 0;
            }
            else {
                ++i;
            }
        }

        // output
        return hull;
    }

    /**
    * \brief given a collection of points as a signal, i.e. - x coordinate is monotonic increasing, return points defining its "envelope"
    * @param {forward_iterator,     in}  iterator to first point in collection/signal
    * @param {forward_iterator,     in}  iterator to last point in collection/signal
    * @param {vector<IFixedVector>, out} {collection of points defining the top of the signal envelope, collection of points defining the bottom of the signal envelope}
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC> && VEC::length() == 2)
    constexpr auto get_points_envelope(const InputIt first, const InputIt last) {
        using T = typename VEC::value_type;
        using out_t = struct { std::vector<VEC> top; std::vector<VEC> bottom; };

        // given vector of values - return vector holding differences between adjacent elements
        const auto diff = [](const std::vector<T>& x) -> std::vector<T> {
            const std::size_t len{ x.size() - 1 };
            std::vector<T> d(len);
            for (std::size_t i{}; i < len - 1; ++i) {
                d[i] = x[i+1] - x[i];
            }
            return d;
        };

        // given vector of values - return vector holding sign of each element
        const auto sign = [](const std::vector<T>& x) -> std::vector<T> {
            const std::size_t len{ x.size() - 1 };
            std::vector<T> d(len);
            for (std::size_t i{}; i < len - 1; ++i) {
                if (x[i] > T{}) {
                    d[i] = static_cast<T>(1);
                }
                else if (x[i] < T{}) {
                    d[i] = static_cast<T>(-1);
                }
                else {
                    d[i] = T{};
                }
            }
            return d;
        };

        // separate cloud point into XY coordinates
        const std::size_t len{ static_cast<std::size_t>(std::distance(first, last)) };
        std::vector<T> x(len);
        std::vector<T> y(len);
        for (std::size_t i{}; i < len - 1; ++i) {
            const VEC p{ *(first + i) };
            x[i] = p.x;
            y[i] = p.y;
        }

        // calculate second derivative
        std::vector<T> secondDerivative{ diff(sign(diff(y))) };

        // find peaks indices
        std::vector<std::size_t> extrMaxIndex;
        std::vector<std::size_t> extrMinIndex;
        for (std::size_t i{}; i < secondDerivative.size(); ++i) {
            if (Numerics::areEquals(secondDerivative[i], static_cast<T>(2))) {
                extrMinIndex.emplace_back(i + 1);
            }
            else if (Numerics::areEquals(secondDerivative[i], static_cast<T>(-2))) {
                extrMaxIndex.emplace_back(i + 1);
            }
        }

        // find peaks
        std::vector<T> extrMaxValue;
        for (const std::size_t i : extrMaxIndex) {
            extrMaxValue.emplace_back(y[i]);
        }
        std::vector<T> extrMinValue;
        for (const std::size_t i : extrMinIndex) {
            extrMinValue.emplace_back(y[i]);
        }

        // output
        std::vector<VEC> top(extrMaxValue.size());
        std::vector<VEC> bottom(extrMinValue.size());
        for (std::size_t i{}; i < extrMaxValue.size(); ++i) {
            top[i] = VEC(x[extrMaxIndex[i]], extrMaxValue[i]);
        }
        for (std::size_t i{}; i < extrMinValue.size(); ++i) {
            bottom[i] = VEC(x[extrMinIndex[i]], extrMinValue[i]);
        }
        return out_t{ top, bottom };
    }

    /**
    * \brief given collection of points, estimate main principle axis
    * @param {forward_iterator, in}  iterator to first point in collection
    * @param {forward_iterator, in}  iterator to last point in collection
    * @param {IFixedVector,     out} normalized axis estimating cloud point principle direction
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr VEC get_principle_axis(const InputIt first, const InputIt last) {
        using T = typename VEC::value_type;

        // housekeeping
        const VEC centroid{ Internals::get_centroid(first, last) };
        const T N{ static_cast<T>(std::distance(first, last) - 1) };

        // calculate covariance matrix elements
        T cov_xx{};
        T cov_yy{};
        T cov_xy{};
        for (auto it{ first }; it != last; ++it) {
            const VEC d{ *it - centroid };
            cov_xx += d.x * d.x;
            cov_yy += d.y * d.y;
            cov_xy += d.x * d.y;
        }
        cov_xx /= N;
        cov_yy /= N;
        cov_xy /= N;
        
        // find covariance matrix largest eigenvalue
        // (see Decomposition::eigenvalues from 'Glsl_solvers.h' for reference)
        const T diff{ cov_xx - cov_yy };
        const T center{ cov_xx + cov_yy };
        const T deltaSquared{ diff * diff + static_cast<T>(4) * cov_xy * cov_xy };
        [[assume(deltaSquared >= T{})]];
        const T delta{ std::sqrt(deltaSquared) };
        const T eigenvalue{ (center + delta) / static_cast<T>(2) };

        // principle component
        return GLSL::normalize(VEC(cov_xy, eigenvalue - cov_xx));
    }

    /**
    * \brief given a closed polygon (as a collection of points) and a line (given by two points), check if polygon is monotone relative to the line
    * @param {forward_iterator, in}  iterator to first point in polygon
    * @param {forward_iterator, in}  iterator to last point in polygon
    * @param {IFixedVector,     in}  line first point
    * @param {IFixedVector,     in}  line last point
    * @param {bool,             out} true if polygon is monotone relative to line
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr bool is_polygon_monotone_relative_to_line(const InputIt first, const InputIt last, const VEC& p0, const VEC& p1) {
        using T = typename VEC::value_type;
        using chains_t = decltype(Algorithms2D::Internals::get_monotone_chains(first, last));

        // get monotone chains
        const chains_t chains{ Internals::get_monotone_chains(first, last) };

        // extend line
        const T extent{ Numerics::max(std::abs(chains.x_min_max[0] - chains.x_min_max[1]),
                                      std::abs(chains.y_min_max[0] - chains.y_min_max[1])) };
        const VEC dir{ p1 - p0 };
        const VEC _p0{ p0 - extent * dir };
        const VEC _p1{ p1 + extent * dir };

        // monotone checking lambda
        auto check_montone = [_p0, _p1](InputIt start, InputIt end, const std::size_t index) -> bool {
            InputIt it{ start };
            VEC valuePrev{ Internals::project_point_on_segment(_p0, _p1, *it).point };
            ++it;
            for (; it != end; ++it) {
                const VEC value{ Internals::project_point_on_segment(_p0, _p1, *it).point };
                if (value[index] < valuePrev[index]) {
                    return false;
                }
                valuePrev = value;
            }
            return true;
        };

        bool isMonotne{ check_montone(chains.x_monotone_chain[0], chains.x_monotone_chain[1], 0)};
        if (isMonotne) {
            isMonotne = check_montone(chains.y_monotone_chain[0], chains.y_monotone_chain[1], 1);
        }
        return isMonotne;
    }

    /**
    * \brief given a closed polygon (as a collection of points), check if its edges a orthogonal with respect to XY axes
    * @param {forward_iterator, in}  iterator to first point in polygon
    * @param {forward_iterator, in}  iterator to last point in polygon
    * @param {value_type,       in}  minimal slope between two consecutive points for their segment would be declared not orthogonal (default is Numerics::equality_precision)
    * @param {bool,             out} true if polygon is orthogonal
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>, class T = typename VEC::value_type>
        requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr bool is_polygon_orthogonal(const InputIt first, const InputIt last, const T tol = Numerics::equality_precision<T>() ) {
        bool isOrthogonal{ true };

        InputIt it{ first };
        VEC pointPrev{ *it };
        ++it;
        for (; it != last; ++it) {
            const VEC point{ *it };
            isOrthogonal &= (std::abs(point.x - pointPrev.x) < tol) || (std::abs(point.y - pointPrev.y) < tol);
            pointPrev = point;
        }

        return isOrthogonal;
    }
    
    /**
    * \brief given a closed non intersecting polygon (as a collection of clockwise or counter clockwise ordered points), check if its convex
    * @param {forward_iterator, in}  iterator to first point in polygon
    * @param {forward_iterator, in}  iterator to last point in polygon
    * @param {int32_t,          out} true if polygon is convex, false otherwise
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr bool is_polygon_convex(const InputIt first, const InputIt last) {
        using T = typename VEC::value_type;

        // housekeeping
        const std::size_t len{ static_cast<std::size_t>(std::distance(first, last)) };
        if (len < 4) {
            return true;
        }

        const bool sign{ Internals::triangle_twice_signed_area(*(first + 2), *first , *(first + 1)) > T{} };
        for (std::size_t i{ 1 }; i < len - 1; ++i) {
            const T area{ Internals::triangle_twice_signed_area(*(first + (i + 2) % len), *(first + i) , *(first + (i + 1) % len)) };
            if (sign != (area > T{})) {
                return false;
            }
        }

        return true;
    }

    /**
    * \brief given a closed non intersecting polygon and two of its vertices, check if line connecting these vertices is inside or outside the polygon.
    *        notice that segments connecting neighboring vertices will be declared as "inside polygon".
    * @param {forward_iterator, in}  iterator to first point in polygon
    * @param {forward_iterator, in}  iterator to last point in polygon
    * @param {value_type,       in}  polygon area
    * @param {forward_iterator, in}  iterator to first vertex
    * @param {forward_iterator, in}  iterator to second vertex
    * @param {bool,             out} true if segment connecting vertices is inside polygon, false otherwise
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>, class T = typename VEC::value_type>
        requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr bool is_line_connecting_polygon_vertices_inside_polygon(const InputIt first, const InputIt last, const T area, const InputIt i0, const InputIt i1) {
        // if segment connects neighboring vertices - it is "inside polygon"
        if ((Extra::are_vectors_identical(*first, *i0) && Extra::are_vectors_identical(*(last - 1), *i1)) ||
            Extra::are_vectors_identical(*(i0 + 1), *i1)) {
            return true;
        }

        // lambda to check if any of four vectors are identical
        const auto is_any_identical = [](const VEC& a, const VEC& b, const VEC& c, const VEC& d) -> bool {
            return (Extra::are_vectors_identical(a, c) || Extra::are_vectors_identical(b, d) ||
                    Extra::are_vectors_identical(a, d) || Extra::are_vectors_identical(b, c));
        };

        // lambda to check if line connecting two vertices (i0-i1) intersects polygon edges
        const auto does_segment_intersect_polygon = [first, last, i0, i1, &is_any_identical]() -> bool {
            const VEC p0{ *i0 };
            const VEC p1{ *i1 };

            InputIt it{ first };
            InputIt nt{ first + 1 };
            for (; nt != last; ++it, ++nt) {
                if (is_any_identical(*it, *nt, p0, p1)) {
                    continue;
                }
                if (Algorithms2D::Internals::do_segments_intersect(p0, p1, *it, *nt)) {
                    return true;
                }
            }

            if (is_any_identical(*first, *(last - 1), p0, p1)) {
                return false;
            }
            return Algorithms2D::Internals::do_segments_intersect(p0, p1, *first, *(last - 1));
        };

        // lambda to check if line connecting two polygon vertices (i0, i1) is inside or outside the polygon
        const auto is_line_inside_polygon = [first, last, i0, i1, area]() -> bool {
            const T area_from_polygon_start_to_segment_start{ Algorithms2D::Internals::get_polygon_area(first, i0) };
            const T area_from_segment_end_to_polygon_end{ Algorithms2D::Internals::get_polygon_area(i1, last) };
            const VEC p0{ *i0 };
            const VEC p1{ *i1 };
            const T area_segment{ (p0.x * p1.y - p1.x * p0.y) / static_cast<T>(2.0) };
            return (area_from_polygon_start_to_segment_start + area_segment + area_from_segment_end_to_polygon_end <= area);
        };

        // test segment
        return does_segment_intersect_polygon() ? false : is_line_inside_polygon();
    }

    /**
    * \brief given a closed polygon (as a collection of points) return iterators to its reflex vertices (vertex whose angle is bigger than PI)
    * @param {forward_iterator,         in}  iterator to first point in polygon
    * @param {forward_iterator,         in}  iterator to last point in polygon
    * @param {vector<forward_iterator>, out} vector of iterators to reflex vertices
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr std::vector<InputIt> get_reflex_vertices(const InputIt first, const InputIt last) {
        using T = typename VEC::value_type;
        // housekeeping
        const VEC centroid{ Algorithms2D::Internals::get_centroid(first, last) };
        const bool is_clockwise{ Algorithms2D::are_points_ordererd_clock_wise(first, last, centroid) };

        // lambda to check if vertex 'b' in a list of three consecutive vertices (a->b->c) is reflex vertex
        const auto is_reflex_vertex = [&is_clockwise](const InputIt a, const InputIt b, const InputIt c) -> bool {
            const vec2 A{ *b - *a };
            const vec2 B{ *c - *b };
            const float det{ A.x * B.y - A.y * B.x };
            return (is_clockwise && det < T{}) || (!is_clockwise && det > T{});
        };

        // iterate over all vertices
        std::vector<InputIt> reflex;
        for (InputIt f{ first }, s{ first + 1 }, t{ first + 2 }; t != last; ++f, ++s, ++t) {
            if (is_reflex_vertex(f, s, t)) {
                reflex.emplace_back(s);
            }
        }
        if (is_reflex_vertex(last - 1, first, first + 1)) {
            reflex.emplace_back(first);
        }

        // output
        return reflex;
    }

    /**
    * \brief given a closed polygon (as a collection of points) return iterators to its cusp vertices (vertices whose adjacent vertices x or y coordinates are either both above or below it)
    * @param {forward_iterator,         in}  iterator to first point in polygon
    * @param {forward_iterator,         in}  iterator to last point in polygon
    * @param {integral,                 in}  0 - look for cusps in X coordinate, 1 - look for cusps in Y coordinate (0/x by default)
    * @param {vector<forward_iterator>, out} vector of iterators to reflex vertices
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr std::vector<InputIt> get_cusp_vertices(const InputIt first, const InputIt last, const std::int8_t coordinate = 0) {
        using T = typename VEC::value_type;
        assert(coordinate == 0 || coordinate == 1);

        // lambda to check if vertex 'b' in a list of three consecutive vertices (a->b->c) is cusp vertex
        const auto is_cusp_vertex = [coordinate](const InputIt a, const InputIt b, const InputIt c) -> bool {
            return ((*a)[coordinate] > (*b)[coordinate]) && ((*c)[coordinate] > (*b)[coordinate]);
        };

        // iterate over all vertices
        std::vector<InputIt> cusps;
        for (InputIt f{ first }, s{ first + 1 }, t{ first + 2 }; t != last; ++f, ++s, ++t) {
            if (is_cusp_vertex(f, s, t)) {
                cusps.emplace_back(s);
            }
        }
        if (is_cusp_vertex(last - 1, first, first + 1)) {
            cusps.emplace_back(first);
        }

        // output
        return cusps;
    }

    /**
    * \brief given a closed simple polygon (as a collection of points) triangulate it using "ear cut" method.
    * @param {forward_iterator,         in}  iterator to first point in polygon
    * @param {forward_iterator,         in}  iterator to last point in polygon
    * @param {vector<forward_iterator>, out} vector of iterators to vertices of polygon triangles, 
                                             every three consecutive iterators define a triangle.
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr std::vector<InputIt> triangulate_polygon_earcut(const InputIt first, const InputIt last) {
        using T = typename VEC::value_type;

        // housekeeping
        srand(1234567);
        const std::size_t len{ static_cast<std::size_t>(std::distance(first, last)) };
        assert(len >= 3);
        std::vector<VEC> poly(first, last);
        std::vector<InputIt> tris;
        std::priority_queue<std::pair<T, std::size_t>> queue_of_ears; // priority queue of ears to be cut

        // initialize doubly linked list
        std::vector<std::size_t> prev(len);
        std::vector<std::size_t> next(len);
        Algoithms::iota(prev.begin(), prev.end(), -1);
        Algoithms::iota(next.begin(), next.end(),  1);
        prev.front() = len - 1;
        next.back() = 0;    

        // helpers to keep track of ears and concave vertices.
        // notice that:
        // > corners that were not ears at the beginning may become so later on...
        // > corners that were concave at the beginning may become convex/ears so later on....
        std::vector<bool> is_ear(len, false);
        std::vector<bool> is_concave(len, false);

        // lambda that performs local concavity test around vertex (given by its id)
        const auto concave_test = [&poly, &prev, &next](const std::size_t cur) -> bool {
            const std::size_t prev_vertex_index{ prev[cur] };
            const std::size_t next_vertex_index{ next[cur] };

            const VEC t0{ poly[prev_vertex_index] };
            const VEC t1{ poly[cur] };
            const VEC t2{ poly[next_vertex_index] };

            return (Internals::triangle_twice_signed_area(t0, t1, t2) <= T{});
        };

        // lambda that performs local ear test around vertex (given by its id)
        const auto ear_test = [&poly, &prev, &next, &is_concave](const std::size_t cur) -> bool {
            if (is_concave[cur]) {
                return false;
            }

            const std::size_t prev_vertex_index{ prev[cur] };
            const std::size_t next_vertex_index{ next[cur] };

            const VEC t0{ poly[prev_vertex_index] };
            const VEC t1{ poly[cur] };
            const VEC t2{ poly[next_vertex_index] };

            // does the ear contain any other front vertex?
            const std::size_t beg{ next[cur] };
            const std::size_t end{ prev[cur] };
            std::size_t test{ next[beg] };
            while (test != end) {
                // check only concave vertices
                if (is_concave[test] && Triangle::is_point_within_triangle(poly[test], t0, t1, t2)) {
                    return false;
                }
                test = next[test];
            }
            return true;
        };

        // lambda that inserts an ear into the the queue
        const auto push_ear = [&poly, &queue_of_ears , &is_ear, &prev, &next](const std::size_t cur) {
            is_ear[cur] = true;

            const std::size_t prev_vertex_index{ prev[cur] };
            const std::size_t next_vertex_index{ next[cur] };
            const VEC c{ poly[cur] };
            const VEC u{ GLSL::normalize(poly[prev_vertex_index] - c) };
            const VEC v{ GLSL::normalize(poly[next_vertex_index] - c) };
            const T ang{ GLSL::dot(u, v) };
            queue_of_ears.push(std::make_pair(-ang, cur));
        };

        // find concave vertices and valid ears
        for (std::size_t i{}; i < len; ++i) {
            is_concave[i] = concave_test(i);
        }
        for (std::size_t i{}; i < len; ++i) {
            if (ear_test(i)) {
                push_ear(i);
            }
        }

        // iteratively triangulate the polygon while updating queue of ears
        // (a simple polygon with n vertices can be meshed with n-2 triangles)
        for (std::size_t i{}; i < len - 2; ++i) {
            // is polygon degenerate?
            if (queue_of_ears.empty()) {
                break;
            }

            // get ear
            const std::size_t curr{ queue_of_ears.top().second };
            queue_of_ears.pop();

            // the ear has already been processed
            if (!is_ear[curr]) {
                --i;
                continue;
            }

            is_ear[curr] = false;

            // make new tri
            tris.push_back(first + prev[curr]);
            tris.push_back(first + curr);
            tris.push_back(first + next[curr]);

            // exclude curr from the front, connecting prev and next
            next[prev[curr]] = next[curr];
            prev[next[curr]] = prev[curr];

            // update concavity info
            is_concave[prev[curr]] = concave_test(prev[curr]);
            is_concave[next[curr]] = concave_test(next[curr]);

            // update prev ear info
            if (ear_test(prev[curr])) {
                push_ear(prev[curr]);
            }
            else if (is_ear[prev[curr]]) {
                is_ear[prev[curr]] = false;
            }

            // update next ear
            if (ear_test(next[curr])) {
                push_ear(next[curr]);
            }
            else if (is_ear[next[curr]]) {
                is_ear[next[curr]] = false;
            }
        }

        // output
        return tris;
    }
}
