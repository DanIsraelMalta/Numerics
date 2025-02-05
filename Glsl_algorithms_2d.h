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
#include "Glsl_triangle.h"
#include "Hash.h"
#include "DiamondAngle.h"
#include "Numerical_algorithms.h"
#include <limits>
#include <vector>
#include <iterator>
#include <random>
#include <queue>  // earcut triangulation
#include <set>    // delaunay triangulation
#include <map>    // |
#include <chrono> //  \ closest pair

//
// collection of algorithms for 2D cloud points and shapes
//
namespace Algorithms2D {

    /**
    * winding order
    **/
    enum class Winding : std::uint8_t {
        None             = 0,
        ClockWise        = 1,
        CounterClockWise = 2,
    };

    //
    // utilities
    //
    namespace Internals {

        /**
        * \brief check if point 'a' is left to point 'b' in relation to point 'x', i.e. - 
        *        "is point 'a' less than point 'b' relative to 'c'".
        * @param {IFixedVector, in}  point a
        * @param {IFixedVector, in}  point b
        * @param {IFixedVector, in}  point c
        * @param {bool,         out} true if 'a' is left to 'b' relative to 'c', false otherwise
        **/
        template<GLSL::IFixedVector VEC>
            requires(VEC::length() == 2)
        constexpr bool is_point_less(const VEC& a, const VEC& b, const VEC& c) noexcept {
            using T = typename VEC::value_type;

            if (a.x - c.x >= T{} &&
                b.x - c.x < T{}) {
                return true;
            }
            if (a.x - c.x < T{} &&
                b.x - c.x >= T{}) {
                return false;
            }
            if (Numerics::areEquals(a.x - c.x, T{}) &&
                Numerics::areEquals(b.x - c.x, T{})) {
                if (a.y - c.y >= T{} ||
                    b.y - c.y >= T{}) {
                    return a.y > b.y;
                }
                return b.y > a.y;
            }

            // compute the cross product of vectors (c -> a) x (c -> b)
            const VEC ac{ a - c };
            const VEC bc{ b - c };
            const T det{ Numerics::diff_of_products(ac.x, bc.y, bc.x, ac.y) };
            if (det < T{}) {
                return true;
            }
            if (det > T{}) {
                return false;
            }

            // points 'a' and 'b' are on the same line from the 'c'
            // check which point is closer to the 'c'
            const T d1{ GLSL::dot(ac) };
            const T d2{ GLSL::dot(bc) };
            return d1 > d2;
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
            const T denom{ Numerics::diff_of_products(s1.x, s2.y, s2.x, s1.y) };
            if (Numerics::areEquals(denom, T{})) {
                return false;
            }

            [[assume(denom != T{})]];
            const VEC ab{ a0 - b0 };
            const T s{ Numerics::diff_of_products(s1.x, ab.y, s1.y, ab.x) / denom };
            const T t{ Numerics::diff_of_products(s2.x, ab.y, s2.y, ab.x) / denom };
            return (s >= T{} && s <= static_cast<T>(1.0) &&
                    t >= T{} && t <= static_cast<T>(1.0));
        }

        /**
        * \brief return intersection point of two segments
        * @param {IFixedVector, in}  segment #1 point #1
        * @param {IFixedVector, in}  segment #1 point #2
        * @param {IFixedVector, in}  segment #1 point #1
        * @param {IFixedVector, in}  segment #1 point #2
        * @param {IFixedVector, out} intersection point
        **/
        template<GLSL::IFixedVector VEC>
            requires(VEC::length() == 2)
        constexpr VEC get_segments_intersection_point(const VEC& p0, const VEC& p1, const VEC& q0, const VEC& q1) noexcept {
            using T = typename VEC::value_type;

            VEC p;
            const T a1{ p1.y - p0.y };
            const T b1{ p0.x - p1.x };
            const T c1{ a1 * p0.x + b1 * p0.y };
            const T a2{ q1.y - q0.y };
            const T b2{ q0.x - q1.x };
            const T c2{ a2 * q0.x + b2 * q0.y };
            const T denom{ Numerics::diff_of_products(a1, b2, a2, b1) };
            if (Numerics::areEquals(denom, T{})) {
                return p;
            }

            [[assume(denom != T{})]];
            p.x = Numerics::diff_of_products(b2, c1, b1, c2) / denom;
            p.y = Numerics::diff_of_products(a1, c2, a2, c1) / denom;
            return p;
        }

        /**
        * \brief given a closed polygon, test if it is simple
        * @param {forward_iterator, in}  iterator to polygon first vertex
        * @param {forward_iterator, in}  iterator to polygon last vertex
        * @param {bool,             out} true if polygon si simple, false otherwise
        **/
        template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
            requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
        constexpr bool is_simple(const InputIt first, const InputIt last) {
            // lambda to check if segment intersect any other segments in ordered polygon
            const auto check_intersection = [&first, &last](const InputIt i0, const InputIt i1, const InputIt edge) -> bool {
                const VEC a0{ *i0 };
                const VEC a1{ *i1 };

                for (InputIt j0{ i1 + 1 }, j1{ i1 + 2 }; j1 != edge; ++j0, ++j1) {
                    if (Algorithms2D::Internals::do_segments_intersect(a0, a1, *j0, *j1)) {
                        return true;
                    }
                }

                return false;
            };

            // check segments intersection
            for (InputIt i0{ first }, i1{ first + 1 }; i1 != last - 2; ++i0, ++i1) {
                if (check_intersection(i0, i1, last)) {
                    return false;
                }
            }

            // check closing segment
            return !check_intersection(last - 1, first, last - 1);
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
                                      Numerics::diff_of_products(ba.x, C, ca.x, B)) / (static_cast<T>(2.0) * D) };
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
            const T dist{ static_cast<T>(std::distance(first, last)) };
            assert(dist > T{});

            VEC centroid;
            for (auto it{ first }; it != last; ++it) {
                centroid += *it;
            }

            return (centroid / dist);
        }

        /**
        * \brief given a polygon (as a collection of points), return twice its area
        * @param {forward_iterator, in}  iterator to first point in polygon
        * @param {forward_iterator, in}  iterator to last point in polygon
        * @param {value_type,       out} twice the polygon area
        **/
        template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>,
                 class T = typename VEC::value_type>
            requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
        constexpr T get_area(const InputIt first, const InputIt last) {
            const T dist{ static_cast<T>(std::distance(first, last)) };
            assert(dist > T{});

            T area{};
            for (auto it{ first }, nt{first + 1}; nt != last; ++it, ++nt) {
                area += GLSL::cross(*it, *nt);
            }
            area += GLSL::cross(*(last - 1), *first);

            return std::abs(area);
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
            const auto update_point = [&point, &index, &distSquared](const VEC& a, const VEC& b, const std::size_t i) -> void {
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

        assert(Algorithms2D::Internals::is_simple(first, last));

        // lambda to check if 'point' intersect with segment given by its edges 'p0' and 'p1'
        const auto intersects = [&point](const VEC& p0, const VEC& p1) -> bool {
            return ((p0.y > point.y != p1.y > point.y) &&
                    (point.x < ((p1.x - p0.x) * (point.y - p0.y)) / (p1.y - p0.y) + p0.x));
        };       

        bool inside{ false };
        for (InputIt it{ first }, nt{ it + 1 }; nt != last; ++it, ++nt) {
            inside = intersects(*it, *nt) ? !inside : inside;
        }
        inside = intersects(*(last - 1), *first) ? !inside : inside;

        return inside;
    }

    /**
    * \brief given a closed non intersecting polygon (as a collection of clockwise or counter clockwise ordered points), check if its convex
    * @param {forward_iterator, in}  iterator to first point in polygon
    * @param {forward_iterator, in}  iterator to last point in polygon
    * @param {bool,             out} true if polygon is convex, false otherwise
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr bool is_polygon_convex(const InputIt first, const InputIt last) {
        using T = typename VEC::value_type;

        // check that polygon is simple
        assert(Algorithms2D::Internals::is_simple(first, last));

        // lambda to calculate triangle signed area
        const auto calculate_area = [](const VEC a, const VEC b, const VEC c) -> T {
            const VEC v1{ b - a };
            const VEC v2{ c - a };
            return Numerics::diff_of_products(v1.x, v2.y, v1.y, v2.x);
        };

        // check polygon last two triangles
        const T area0{ Numerics::sign(calculate_area(*(last - 1), *first, *(first + 1))) };
        T area1{ Numerics::sign(calculate_area(*(last - 2), *(last - 1), *first)) };
        if (area0 * area1 < T{}) {
            return false;
        }

        // check rest ot triangles
        for (InputIt i0{ first }, i1{ first + 1 }, i2{ first + 2 }; i2 != last; ++i0, ++i1, ++i2) {
            area1 = Numerics::sign(calculate_area(*i0, *i1, *i2));
            if (area0 * area1 < T{}) {
                return false;
            }
        }

        return true;
    }

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
        using vec_it = typename std::vector<VEC>::iterator;

        // lambda to check if points are ordered clockwise
        const auto are_points_ordered_counter_clock_wise = [](const VEC & a, const VEC & b, const VEC & c) -> T {
            return Numerics::diff_of_products(b.y - a.y, c.x - b.x, b.x - a.x, c.y - b.y);
        };

        // housekeeping
        std::vector<VEC> points(first, last);

        // place left most point at start of point cloud
        const vec_it minElementIterator{ Algoithms::min_element(points.begin(), points.end(),
                                                [](const VEC& a, const VEC& b) noexcept -> bool {
            return (a.x < b.x || (a.x == b.x && a.y < b.y));
        }) };
        Utilities::swap(points[0], *minElementIterator);

        // lexicographically sort all points using the smallest point as pivot
        const VEC v0( points[0] );
        std::sort(points.begin() + 1, points.end(), [v0](const VEC& b, const VEC& c) noexcept -> bool {
            return Numerics::diff_of_products(b.y - v0.y, c.x - b.x, b.x - v0.x, c.y - b.y) < T{};
        });

        // build hull
        vec_it it = points.begin();
        std::vector<VEC> hull{ {*it++, *it++, *it++} };
        while (it != points.end()) {
            while (are_points_ordered_counter_clock_wise(*(hull.rbegin() + 1), *(hull.rbegin()), *it) >= T{}) {
                hull.pop_back();
            }
            hull.push_back(*it++);
        }

        // output
        assert(Algorithms2D::is_polygon_convex(hull.begin(), hull.end()));
        return hull;
    }

    /**
    * \brief given collection of points, estimate main principle axis
    * @param {forward_iterator, in}  iterator to first point in collection
    * @param {forward_iterator, in}  iterator to last point in collection
    * @param {IFixedVector,     in}  point cloud centroid
    * @param {IFixedVector,     out} normalized axis estimating cloud point principle direction
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr VEC get_principle_axis(const InputIt first, const InputIt last, const VEC& centroid) {
        using T = typename VEC::value_type;

        // housekeeping
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
        const T deltaSquared{ diff * diff + static_cast<T>(4.0) * cov_xy * cov_xy };
        [[assume(deltaSquared >= T{})]];
        const T delta{ std::sqrt(deltaSquared) };
        const T eigenvalue{ (center + delta) / static_cast<T>(2.0) };

        // principle component
        return GLSL::normalize(VEC(cov_xy, eigenvalue - cov_xx));
    }

    /**
    * \brief given a convex polygon, return its minimal area bounding rectangle ("oriented bounding box")
    * @param {vector<IFixedVector>,                                     in}  convex polygon
    * @param {{IFixedVector, IFixedVector, IFixedVector, IFixedVector}, out} vertices of minimal area bounding rectangle (ordered counter clock wise)
    **/
    template<GLSL::IFixedVector VEC>
        requires(VEC::length() == 2)
    constexpr auto get_bounding_rectangle(const std::vector<VEC>& hull) {
        using T = typename VEC::value_type;
        using out_t = struct { VEC p0; VEC p1; VEC p2; VEC p3; };

        // housekeeping
        T area{ std::numeric_limits<T>::max() };
        VEC p0;
        VEC p1;
        VEC maxNormal;

        // lambda to project point on plane (given by point and normal)
        const auto project_point_plane = [](const VEC& point, const VEC& plane_point, const VEC& plane_normal) -> VEC {
            const VEC d{ point - plane_point };
            const T dist{ GLSL::dot(d, plane_normal) };
            return point - dist * plane_normal;
        };

        // lambda which accepts polygon edge (i-j) and find minimal area bounding rectangle using this edge
        const auto minimal_area_bounding_rectangle_per_edge = [&area, &p0, &p1, &maxNormal, &hull, &project_point_plane]
                                                              (const std::size_t i, const std::size_t j) {
            // segment
            VEC v0{ hull[i] };
            VEC v1{ hull[j] };

            // segment distance to center
            const VEC center{ (v0 + v1) / static_cast<T>(2.0) };
            T v0_dist{ GLSL::dot(v0 - center) };
            T v1_dist{ v0_dist };
            T vNormal_dist{};

            // segment tangential and orthogonal directions
            const VEC dir{ GLSL::normalize(v1 - v0) };
            const VEC normal(-dir.y, dir.x);

            // point on orthogonal segment
            VEC vNormal(center);
            VEC ref;

            // 1. project polygon points on line connecting edge and find minimal/maximal points.
            // 2. project polygon points on orthogonal line to edge (should be going inside the polygon) and find maximal point.
            for (std::size_t k{}; k < hull.size(); ++k) {
                if (k == i || k == j) {
                    continue;
                }

                const VEC point{ hull[k] };

                // project points on segment tangential and orthogonal directions (orthogonal towards inside hull)
                const VEC projOnDir{ project_point_plane(point, center, normal) };
                const VEC projOnNormal_0{ project_point_plane(point, v0, dir) };
                const VEC projOnNormal_1{ project_point_plane(point, v1, -dir) };

                // find furthest points along v0v1 segments which can constitute an extent to tight bounding box
                const bool point_on_v0_side{GLSL::dot(point - v1) > GLSL::dot(point - v0) };
                if (const T projOnDir_dist{ GLSL::dot(projOnDir - center) };
                    point_on_v0_side && projOnDir_dist > v0_dist) {
                    v0_dist = projOnDir_dist;
                    v0 = projOnDir;
                }
                else if (!point_on_v0_side && projOnDir_dist > v1_dist) {
                    v1_dist = projOnDir_dist;
                    v1 = projOnDir;
                }

                // find furthest point from v0v1 segment along orthogonal direction to v0v1
                if (const T dist{ GLSL::dot(projOnNormal_0 - v0) };
                    dist > vNormal_dist) {
                    vNormal_dist = dist;
                    vNormal = projOnNormal_0;
                    ref = v0;
                }
                if (const T dist{ GLSL::dot(projOnNormal_1 - v1) };
                    dist > vNormal_dist) {
                    vNormal_dist = dist;
                    vNormal = projOnNormal_1;
                    ref = v1;
                }
            }
            assert(!Extra::are_vectors_identical(v1, v0));

            // rectangle area
            if (const T rectangle_area{ GLSL::dot(v1 - v0) * GLSL::dot(vNormal - ref) };
                rectangle_area < area) {
                area = rectangle_area;
                p0 = v0;
                p1 = v1;
                maxNormal = vNormal;
            }
        };

        // iterate over all polygon edges, rotating caliper style, and find edge on which minimal area bounding rectangle will be built
        for (std::size_t i{}, j{i + 1}; j < hull.size(); ++i, ++j) {
            minimal_area_bounding_rectangle_per_edge(i, j);
        }
        minimal_area_bounding_rectangle_per_edge(hull.size() - 1, 0);

        // calculate bounding rectangle vertices
        const VEC dir{ GLSL::normalize(p1 - p0) };
        const VEC normal(-dir.y, dir.x);
        const VEC p2{ Internals::get_rays_intersection_point(p1, normal, maxNormal,  dir) };
        const VEC p3{ Internals::get_rays_intersection_point(p0, normal, maxNormal, -dir) };

        return out_t{ p0, p1, p2, p3 };
    }
    
    /**
    * \brief given convex polygon, return its diameter
    * @param {vector<IFixedVector>,           in}  convex polygon
    * @param {{value_type, array<size_t, 2>}, out} {squared diameter, <index of anti podal point #1, index of anti podal point #2>}
    **/
    template<GLSL::IFixedVector VEC>
        requires(VEC::length() == 2)
    constexpr auto get_convex_diameter(const std::vector<VEC>& hull) {
        using T = typename VEC::value_type;
        using out_t = struct { T diameter_squared; std::array<std::size_t, 2> indices; };

        // housekeeping
        const std::size_t N{ hull.size() };
        out_t out{
            .diameter_squared = T{},
            .indices = std::array<std::size_t, 2>{{0, 0}}
        };
        const auto checkPoints = [N, &out, &hull](const std::size_t i, const std::size_t j) {
            const VEC a{ hull[i % N] };
            const VEC b{ hull[j % N] };
            const T furthest{ GLSL::dot(a - b) };
            const bool update{ furthest > out.diameter_squared };
            out.diameter_squared = update ? furthest : out.diameter_squared;
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
    * \brief given convex polygon, return the minimal bounding circle (circumcircle).
    *        based on https://www.cise.ufl.edu/~sitharam/COURSES/CG/kreveldnbhd.pdf.
    * @param {vector<IFixedVector>,       in}  convex polygon
    * @param {{IFixedVector, value_type}, out} {minimal bounding circle center, minimal bounding circle squared radius}
    **/
    template<GLSL::IFixedVector VEC>
        requires(VEC::length() == 2)
    constexpr auto get_minimal_bounding_circle(const std::vector<VEC>& hull) {
        using T = typename VEC::value_type;
        using out_t = decltype(Algorithms2D::Internals::get_circumcircle(hull[0], hull[0]));

        // is hull composed of two/three/four points?
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
        out_t circle{ Internals::get_circumcircle(hull[0], hull[1]) };
        for (std::size_t i{ 2 }; i < N; ++i) {
            const VEC p{ hull[i] };
            if (!is_point_in_circle(circle, p)) {
                circle = make_bounding_circle_one_points(Numerics::min(i + 1, N), p);
            }
        }

        return circle;
    }

    /**
    * \brief given two convex polygons, return true if they intersect and false otherwise.
    *        this function uses Gilbert-Johnson-Keerthi algorithm (with Minkowski sum) for fast intersection calculation.
    * 
    * @param {MAX_ITER} maximal iteration for calculation to stop (default is 50)
    * @param {vector<IFixedVector>, in}  convex polygon #1
    * @param {IFixedVector        , in}  convex polygon #1 centroid
    * @param {vector<IFixedVector>, in}  convex polygon #2
    * @param {IFixedVector        , in}  convex polygon #2 centroid
    * @param {bool,                 out} true if convex hulls intersect each other, false otherwise
    **/
    template<std::size_t MAX_ITER = 50, GLSL::IFixedVector VEC>
        requires(VEC::length() == 2)
    constexpr bool do_convex_polygons_intersect(const std::vector<VEC>& hull0, const VEC& centroid0,
                                                const std::vector<VEC>& hull1, const VEC& centroid1) {
        using T = typename VEC::value_type;

        // lambda to calculate triple product
        const auto triple_product = [](const VEC& a, const VEC& b, const VEC& c) -> VEC {
            const T ac{ GLSL::dot(a, c) };
            const T bc{ GLSL::dot(b, c) };
            return VEC(Numerics::diff_of_products(b.x, ac, a.x, bc),
                       Numerics::diff_of_products(b.y, ac, a.y, bc));
        };

        // lambda to return the index of the furthest point along a given direction from a point
        const auto index_of_furthest_point = [](const std::vector<VEC>& vertices, const VEC d) -> std::size_t {
            T maxDistSquared{ GLSL::dot(d, vertices[0]) };
            std::size_t index{};

            for (size_t i{ 1 }; i < vertices.size(); ++i) {
                if (const T dist2{ GLSL::dot(d, vertices[i]) };
                    dist2 > maxDistSquared) {
                    maxDistSquared = dist2;
                    index = i;
                }
            }

            return index;
        };
        
        // lambda to calculate local Minkowski sum
        const auto minkowski_support = [&hull0, &hull1, &index_of_furthest_point](const VEC d) -> VEC {
            const std::size_t i{ index_of_furthest_point(hull0,  d) };
            const std::size_t j{ index_of_furthest_point(hull1, -d) };
            return (hull0[i] - hull1[j]);
        };

        // direction from the center of 'hull0' to 'hull1', if its zero - set it to be X axis (arbitrary)
        VEC d{ !Extra::are_vectors_identical(centroid0, centroid1) ? centroid0 - centroid1 : VEC(static_cast<T>(1.0)) };

        // first support is identical to simplex initial point
        VEC a{ minkowski_support(d) };
        std::array<VEC, 3> simplex = { {a, a, a} };
        if (GLSL::dot(a, d) <= T{}) {
            return false;
        }

        // next search direction is towards the origin
        d = -d;

        // Gilbert-Johnson-Keerthi
        std::size_t iter{};
        std::size_t index{};
        while (iter < MAX_ITER) {
            ++iter;

            a = minkowski_support(d);
            simplex[++index] = a;

            if (GLSL::dot(a, d) <= T{}) {
                return false;
            }

            // a -> origin
            const VEC ao{ -a };

            // simplex is a segment
            if (index < 2) {
                const VEC b{ simplex[0] };
                const VEC ab{ b - a };
                d = triple_product(ab, ao, ab);

                // direction will be normal to AB towards origin
                if (Numerics::areEquals(GLSL::dot(d), T{})) {
                    d.x =  ab.y;
                    d.y = -ab.x;
                }

                continue;
            } // simplex is a triangle
            else {
                const VEC b{ simplex[1] };
                const VEC c{ simplex[0] };
                const VEC ab{ b - a };
                const VEC ac{ c - a };

                // set new direction as normal to AC towards origin
                if (const VEC ac_perpendicular{ triple_product(ab, ac, ac) };
                    GLSL::dot(ac_perpendicular, ao) >= T{}) {
                    d = ac_perpendicular;
                } // swap simplex vertices and choose new direction as normal to AB towards origin?
                else if (const VEC ab_perpendicular{ triple_product(ac, ab, ab) };
                         GLSL::dot(ab_perpendicular, ao) >= T{}) {
                    simplex[0] = simplex[1];
                    d = ab_perpendicular;
                } // collision!
                else {
                    return true;
                }

                // swap simplex middle vertex
                simplex[1] = simplex[2];
                --index;
            }
        }

        // output
        return false;
    }

    /**
    * \brief given a polygon, determine its winding
    * @param {forward_iterator, in}  iterator to polygon first point
    * @param {forward_iterator, in}  iterator to polygon last point
    * @param {Winding,          out} polygon winding
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr Algorithms2D::Winding get_polygon_winding(const InputIt first, const InputIt last) {
        using T = typename VEC::value_type;
        using iter_size_t = std::iterator_traits<InputIt>::difference_type;

        VEC v1{ *first };
        VEC v2{ *(last - 1) };
        T sum{ Numerics::diff_of_products(v1.x, v2.y, v1.y, v2.x) };
        for (InputIt it{ first }, nt{first + 1}; nt != last; ++it, ++nt) {
            v1 = *it;
            v2 = *nt;
            sum += Numerics::diff_of_products(v1.x, v2.y, v1.y, v2.x);
        }

        if (sum > T{}) {
            return Algorithms2D::Winding::CounterClockWise;
        }
        else if (sum < T{}) {
            return Algorithms2D::Winding::ClockWise;
        }
        else {
            return Algorithms2D::Winding::None;
        }
    }

    /**
    * \brief given collection of points, sort them (in place) in specified winding order around their centroid
    * @param {Winding}               required winding
    * @param {forward_iterator,  in} iterator to point collection first point
    * @param {forward_iterator,  in} iterator to point collection last point
    * @param {IFixedVector,      in} points geometric center
    **/
    template<Algorithms2D::Winding WINDING, std::forward_iterator InputIt,
             class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC> && VEC::length() == 2)
    constexpr void sort_points(InputIt first, InputIt last, const VEC centroid) {
        if constexpr (WINDING == Algorithms2D::Winding::ClockWise) {
            Algoithms::sort(first, last, [centroid](const VEC& a, const VEC& b) noexcept -> bool {
                return !Algorithms2D::Internals::is_point_less(a, b, centroid);
            });
        }
        else if constexpr (WINDING == Algorithms2D::Winding::CounterClockWise) {
            Algoithms::sort(first, last, [centroid](const VEC& a, const VEC& b) noexcept -> bool {
                return Algorithms2D::Internals::is_point_less(a, b, centroid);
            });
        }
    }

    /**
    * \brief calculate the concave hull of collection of 2D points (using Graham scan algorithm).
    *
    * @param {forward_iterator,     in}  iterator to point cloud collection first point
    * @param {forward_iterator,     in}  iterator to point cloud collection last point
    * @param {value_type,           in}  concave threshold. the minimal ratio between segment length with added point and original segment length.
    *                                    if the value is smaller than the threshold - the point is part of the concave.
    *                                    the larger the threshold - the more points will be part of the concave.
    *                                    default is 0, i.e - convex hull.
    *                                    value of 1.3 means that if the new segment is smaller than 130% current segment - it will be added to concave hull.
    * @param {vector<IFixedVector>, out} collection of points which define point cloud concave hull
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>, class T = typename VEC::value_type>
        requires(GLSL::is_fixed_vector_v<VEC> && (VEC::length() == 2))
    constexpr std::vector<VEC> get_concave_hull(const InputIt first, const InputIt last, const T concave_threshold = T{}) {
        using closest_t = decltype(Algorithms2D::Internals::get_index_of_closest_segment(std::vector<VEC>{}, VEC()));

        // get convex hull
        std::vector<VEC> hull{ Algorithms2D::get_convex_hull(first, last) };
        if (concave_threshold <= T{}) {
            return hull;
        }

        // Euclidean distance approximation
        // (https://en.wikibooks.org/wiki/Algorithms/Distance_approximations)
        const auto distance = [](const VEC& a, const VEC& b) -> T {
            constexpr T c1{ static_cast<T>(1007.0) / static_cast<T>(1024) };
            constexpr T c2{ static_cast<T>(441.0) / static_cast<T>(1024) };
            constexpr T c3{ static_cast<T>(40.0) / static_cast<T>(1024) };
            constexpr T c4{ static_cast<T>(16.0) };

            const T x{ Numerics::max(std::abs(a.x), std::abs(b.x)) };
            const T y{ Numerics::max(std::abs(a.y), std::abs(b.y)) };
            const T _max{ Numerics::max(x, y) };
            const T _min{ Numerics::min(x, y) };

            return (_max < c4 * _min) ? ((c1 - c3) * _max + c2 * _min) : (c1 * _max + c2 * _min);
        };

        // remove convex hull points from cloud
        std::vector<VEC> cloud(first, last);
        std::vector<std::size_t> to_remove;
        to_remove.reserve(hull.size());
        for (const VEC h : hull) {
            for (std::size_t i{}; i < cloud.size(); ++i) {
                if (Extra::are_vectors_identical(h, cloud[i])) {
                    to_remove.emplace_back(i);
                    break;
                }
            }
        }
        Algoithms::remove(cloud, to_remove);
        
        // "dig" into convex hull to create concave hull
#ifdef _DEBUG
        std::size_t restarts{};
#endif
        for (std::size_t i{}; i < cloud.size(); ++i) {
            // find hull segment which cloud point is closest to
            const VEC p{ cloud[i] };
            const closest_t closest{ Algorithms2D::Internals::get_index_of_closest_segment(hull, p) };

            // should point be part of concave hull?
            const std::size_t segment_index_start{ closest.index };
            const std::size_t segment_index_end{ (segment_index_start + 1) % hull.size() };
            const T segment_length{ distance(hull[segment_index_start], hull[segment_index_end]) };
            const T new_segment_length{ distance(p, hull[segment_index_end]) +
                                        distance(hull[segment_index_start], p) };
            [[assume(segment_length > T{})]];
            [[assume(new_segment_length > T{})]];
            if (new_segment_length <= segment_length * concave_threshold ||
                Numerics::areEquals(segment_length, new_segment_length, Numerics::equality_precision<T>())) {
                // update hull
                assert(segment_index_start + 1 <= hull.size());
                hull.insert(hull.begin() + segment_index_start + 1, p);

                // remove point from cloud
                Utilities::swap(cloud[i], cloud.back());
                cloud.pop_back();

                // restart search
                i = 0;
#ifdef _DEBUG
                ++restarts;
#endif
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
    * \brief given a closed polygon (as a collection of points), check if its edges are orthogonal with respect to XY axes
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

        // check that polygon is simple
        assert(Algorithms2D::Internals::is_simple(first, last));

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
    * @param {Algorithms2D::Winding,    in}  polygon winding
    * @param {vector<forward_iterator>, out} vector of iterators to reflex vertices
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr std::vector<InputIt> get_reflex_vertices(const InputIt first, const InputIt last, const Algorithms2D::Winding winding) {
        using T = typename VEC::value_type;

        // check that polygon is simple
        assert(Algorithms2D::Internals::is_simple(first, last));

        // housekeeping
        assert(winding != Algorithms2D::Winding::None);
        const bool is_clockwise{ winding == Algorithms2D::Winding::ClockWise };

        // lambda to check if vertex 'b' in a list of three consecutive vertices (a->b->c) is reflex vertex
        const auto is_reflex_vertex = [&is_clockwise](const InputIt a, const InputIt b, const InputIt c) -> bool {
            const vec2 A{ *b - *a };
            const vec2 B{ *c - *a };
            const T det{ GLSL::cross(B, A) };
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
    * @param {vector<forward_iterator>, out} vector of iterators to cusp vertices
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr std::vector<InputIt> get_cusp_vertices(const InputIt first, const InputIt last, const std::int8_t coordinate = 0) {
        using T = typename VEC::value_type;
        assert(coordinate == 0 || coordinate == 1);

        // check that polygon is simple
        assert(Algorithms2D::Internals::is_simple(first, last));

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

        // check that polygon is simple
        assert(Algorithms2D::Internals::is_simple(first, last));

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
        assert(tris.size() % 3 == 0);
        return tris;
    }

    /**
    * \brief given a collection of points, triangulate it using "Delaunay" method.
    *        notice that this function uses Bowyer-Watson algorithm, which means its O(n*long(n)) and might fail
    *        in cases of polygons with degenerate segments.
    * 
    * @param {forward_iterator,     in}  iterator to first point in collection
    * @param {forward_iterator,     in}  iterator to last point in collection
    * @param {point_cloud_aabb,     in}  point cloud axis aligned bounding box (type signature is of AxisLignedBoundingBox::point_cloud_aabb)
    * @param {vector<IFixedVector>, out} vector of vertices of triangles, every three consecutive iterators define a triangle.
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>,
             class aabb_t = decltype(AxisLignedBoundingBox::point_cloud_aabb<InputIt>(std::declval<InputIt>(), std::declval<InputIt>()))>
        requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr std::vector<VEC> triangulate_points_delaunay(const InputIt first, const InputIt last, const aabb_t aabb ) {
        using T = typename VEC::value_type;
        using edge_t = struct { VEC first; VEC second; };
        using circle_t = decltype(Algorithms2D::Internals::get_circumcircle(*first, *first, *first));
        constexpr T super_triangle_margin{ static_cast<T>(20.0) };

        // housekeeping
        std::vector<VEC> tris;

        // super triangle (covers all polygon vertices)
        const T delta_max{ GLSL::max(aabb.max - aabb.min) };
        const VEC mid{ (aabb.max + aabb.min) / static_cast<T>(2.0) };
        const VEC p0(mid.x - super_triangle_margin * delta_max, mid.y - delta_max);
        const VEC p1(mid.x, mid.y + super_triangle_margin * delta_max);
        const VEC p2(mid.x + super_triangle_margin * delta_max, mid.y - delta_max);
        tris.emplace_back(p0);
        tris.emplace_back(p1);
        tris.emplace_back(p2);

        // Bowyer-Watson algorithm
        for (InputIt p{ first }; p != last; ++p) {
            const VEC point{ *p };

            // check if point is within any triangle circumcircle
            std::vector<edge_t> edges;
            std::vector<VEC> temporary_triangles;
            for (std::size_t i{}, len{ tris.size() }; i < len - 2; i += 3) {
                const VEC t0{ tris[i] };
                const VEC t1{ tris[i + 1] };
                const VEC t2{ tris[i + 2] };

                if (const circle_t circumcircle{ Algorithms2D::Internals::get_circumcircle(t0, t1, t2) };
                    GLSL::dot(point - circumcircle.center) <= circumcircle.radius_squared) {
                    edges.emplace_back(edge_t{ t0, t1 });
                    edges.emplace_back(edge_t{ t1, t2 });
                    edges.emplace_back(edge_t{ t2, t0 });
                }
                else {
                    temporary_triangles.emplace_back(t0);
                    temporary_triangles.emplace_back(t1);
                    temporary_triangles.emplace_back(t2);
                }
            }

            // remove duplicate edges
            std::set<std::size_t> duplicate_edges;
            for (std::size_t i{}, len{ edges.size() }; i < len; ++i) {
                const edge_t edge_i{ edges[i] };

                for (std::size_t j{ i + 1 }; j < len; ++j) {
                    const edge_t edge_j{ edges[j] };

                    bool identical{ Extra::are_vectors_identical(edge_i.first, edge_j.first) &&
                                    Extra::are_vectors_identical(edge_i.second, edge_j.second) };
                    identical |= Extra::are_vectors_identical(edge_i.first, edge_j.second) &&
                                 Extra::are_vectors_identical(edge_i.second, edge_j.first);
                    if (identical) {
                        duplicate_edges.insert(i);
                        duplicate_edges.insert(j);
                    }
                }
            }
            Algoithms::remove(edges, std::vector<std::size_t>(duplicate_edges.begin(), duplicate_edges.end()));

            // add triangles
            for (const edge_t& e : edges) {
                temporary_triangles.emplace_back(e.first);
                temporary_triangles.emplace_back(e.second);
                temporary_triangles.emplace_back(point);
            }

            // update triangulation
            tris = temporary_triangles;
        }
        
        // remove triangles which include super-triangle vertex
        std::set<std::size_t> super_triangles_vertices;
        for (std::size_t i{}, len{ tris.size() }; i < len - 2; i += 3) {
            Utilities::static_for<0, 1, 3>([&tris, &super_triangles_vertices, &p0, &p1, &p2, i](std::size_t j) {
                if (Extra::are_vectors_identical(tris[i + j], p0) ||
                    Extra::are_vectors_identical(tris[i + j], p1) ||
                    Extra::are_vectors_identical(tris[i + j], p2)) {
                    super_triangles_vertices.insert(i);
                    super_triangles_vertices.insert(i + 1);
                    super_triangles_vertices.insert(i + 2);
                }
            });
        }
        Algoithms::remove(tris, std::vector<std::size_t>(super_triangles_vertices.begin(), super_triangles_vertices.end()));

        // output
        assert(tris.size() % 3 == 0);
        return tris;
    }

    /**
    * \brief given a polygon, triangulate it using "Delaunay" method.
    *        notice that this function uses Bowyer-Watson algorithm, which means its O(n*long(n)) and might fail
    *        in cases of polygons with degenerate segments.
    *
    * @param {forward_iterator,     in}  iterator to polygon first point
    * @param {forward_iterator,     in}  iterator to polygon last point
    * @param {point_cloud_aabb,     in}  point cloud axis aligned bounding box (type signature is of AxisLignedBoundingBox::point_cloud_aabb)
    * @param {vector<IFixedVector>, out} vector of vertices of triangles, every three consecutive iterators define a triangle.
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>,
             class aabb_t = decltype(AxisLignedBoundingBox::point_cloud_aabb<InputIt>(std::declval<InputIt>(), std::declval<InputIt>()))>
        requires(GLSL::is_fixed_vector_v<VEC> && VEC::length() == 2)
    constexpr std::vector<VEC> triangulate_polygon_delaunay(const InputIt first, const InputIt last, const aabb_t aabb) {
        using T = typename VEC::value_type;

        // triangulate vertices
        std::vector<VEC> triangles{ Algorithms2D::triangulate_points_delaunay(FWD(first), FWD(last), FWD(aabb)) };

        // remove out-of-polygon triangles
        std::vector<std::size_t> outside;
        outside.reserve(triangles.size());
        for (std::size_t i{}; i < triangles.size(); i += 3) {
            const VEC centroid{ (triangles[i] + triangles[i + 1] + triangles[i + 2]) / static_cast<T>(3.0) };
            if (!Algorithms2D::is_point_inside_polygon(first, last, centroid)) {
                outside.emplace_back(i);
                outside.emplace_back(i + 1);
                outside.emplace_back(i + 2);
            }

        }
        Algoithms::remove(triangles, outside);

        // output
        return triangles;
    }

    /**
    * \brief given a point cloud, return the closest pair.
    *        complexity is O(n) due to using Rabin & Lipton method.
    *
    * @param {forward_iterator, in}  iterator to first point in point collection
    * @param {forward_iterator, in}  iterator to last point in point collection
    * @param {{forward_iterator, forward_iterator}, out} iterators to first and second points in pair whose distance is minimal
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr auto get_closest_pair(const InputIt first, const InputIt last) {
        using T = typename VEC::value_type;
        using vec3 = next_vector_type<VEC>::vector_type;
        using out_t = struct { InputIt p0; InputIt p1; };

        // housekeeping
        out_t pair;
        const std::size_t n{ static_cast<std::size_t>(std::distance(first, last)) };
        T md{ std::numeric_limits<T>::max() };

        // Select n/100 pairs of points uniformly at random, with replacement, and let 'md' be the minimum distance of the selected pairs.
        for (std::size_t i{}; i < n / 100; ++i) {
            static std::mt19937_64 rng1(std::chrono::steady_clock::now().time_since_epoch().count());
            static std::mt19937_64 rng2(std::chrono::steady_clock::now().time_since_epoch().count());
            const std::size_t A{ static_cast<std::size_t>(rng1()) % n };
            const std::size_t B{ static_cast<std::size_t>(rng2()) % n };
            if (A != B) {
                md = Numerics::min(md, GLSL::dot(*(first + A) - *(first + B)));
                if (md == T{}) {
                    pair.p0 = first + A;
                    pair.p1 = first + B;
                    return pair;
                }
            }
        }

        // Round the input points to a square grid of points whose size (the separation between adjacent grid points) is 'md'
        // and use a hash table to collect together pairs of input points that round to the same grid point.
        std::map<T, std::vector<std::size_t>> neighbors;
        md = std::ceil(std::sqrt(md));
        for (std::size_t i{}; i < n; ++i) {
            const VEC p{ *(first + i) / md };
            const T hash{ Hash::sample_white_noise_over_2D_domain(p.x, p.y) };
            neighbors[hash].push_back(i);
        }

        // For each input point, compute the distance to all other inputs that either round to the same grid point
        // or to another grid point within the Moore neighborhood of 3^k-1 surrounding grid points.
        std::size_t a{};
        std::size_t b{ 1 };
        md = GLSL::dot(*(first + a) - *(first + b));
        for (const auto& [p, id] : neighbors) {
            for (const std::int32_t dx : {-1, 0, 1}) {
                for (const std::int32_t dy : {-1, 0, 1}) {
                    const VEC pp{ p + VEC(static_cast<T>(dx), static_cast<T>(dy)) };
                    const T hash_pp{ Hash::sample_white_noise_over_2D_domain(pp.x, pp.y) };
                    if (!neighbors.count(hash_pp)) {
                        continue;
                    }

                    for (const std::size_t i : neighbors[hash_pp]) {
                        for (const std::size_t j : id) {
                            if (j == i) {
                                break;
                            }

                            if (const T cur{ GLSL::dot(*(first + i) - *(first + j)) };
                                cur < md) {
                                md = cur;
                                a = i;
                                b = j;
                            }
                        }
                    }
                }
            }
        }

        // output
        pair.p0 = first + a;
        pair.p1 = first + b;
        return pair;
    }

    /**
    * \brief given a polygon (as a collection of points) and its Delaunay triangulation return the largest inscribed circle (bounded circle).
    *        notice that Delaunay triangulation can be calculated using "Algorithms2d::triangulate_points_delaunay".
    * 
    * @param {forward_iterator,     in}        iterator to polygon first point
    * @param {forward_iterator,     in}        iterator to polygon last point
    * @param {vector<IFixedVector>, in}        vector of vertices of triangles (Delaunay triangulation),
    *                                          every three consecutive iterators define a triangle.
    * @param {{IFixedVector, value_type}, out} {largest inscribed circle center, largest inscribed circle squared radius}
    */
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr auto get_maximal_inscribed_circle(const InputIt first, const InputIt last, const std::vector<VEC>& delaunay) {
        using T = typename VEC::value_type;
        using out_t = struct { VEC center; T radius; };

        // housekeeping
        assert(delaunay.size() % 3 == 0);
        assert(Algorithms2D::Internals::is_simple(first, last));

        // get Voronoi points
        std::vector<VEC> voronoi;
        voronoi.reserve(delaunay.size() / 3);
        for (std::size_t i{}; i < delaunay.size(); i += 3) {
            voronoi.emplace_back((delaunay[i] + delaunay[i + 1] + delaunay[i + 2]) / static_cast<T>(3.0));
        }
        
        // find Voronoi point furthest from polygon edges
        out_t inscribed{ voronoi[0], T{} };
        for (const VEC& v : voronoi) {
            if (const T minimalDistance{ PointDistance::sdf_to_polygon(first, last, v) };
                minimalDistance > inscribed.radius) {
                inscribed.center = v;
                inscribed.radius = minimalDistance;
            }
        }

        // output
        return inscribed;
    }

    /**
    * \brief given a polygon (as a collection of points ordered in counter clockwise manner) and an infinite line (defined by origin and direction)
    *        return the polygon clipped by this half space.
    *
    * @param {forward_iterator,     in}  iterator to polygon first point (polygon is ordered in counter clockwise manner)
    * @param {forward_iterator,     in}  iterator to polygon last point (polygon is ordered in counter clockwise manner)
    * @param {IFixedVector,         in}  infinite line origin
    * @param {IFixedVector,         in}  infinite line direction (does not need to be normalized)
    * @param {vector<IFixedVector>, out} clipped polygon
    */
    template<std::forward_iterator It, class VEC = typename std::decay_t<decltype(*std::declval<It>())>>
    requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr std::vector<VEC> clip_polygon_by_infinte_line(const It first, const It last, const VEC& origin, const VEC& normal) {
        using T = typename VEC::value_type;

        assert(Algorithms2D::get_polygon_winding(first, last) == Algorithms2D::Winding::CounterClockWise);

        // housekeeping
        std::vector<VEC> clipped;
        clipped.reserve(static_cast<std::size_t>(std::distance(first, last)));

        // Determine state of last point
        VEC e1{ *(last - 1) };
        T prev_num{ GLSL::dot(origin - e1, normal) };
        bool prev_inside{ prev_num < T{} };

        // iterate over all vertices
        for (It j{ first }; j != last; ++j) {
            // is second point inside?
            VEC e2{ *j };
            const T num{ GLSL::dot(origin - e2, normal) };
            bool cur_inside{ num < T{} };

            // In -> Out or Out -> In: Add point on clipping plane
            if (cur_inside != prev_inside) {
                // Solve: dot(X - origin, normal) = 0 where X = e1 + t * (e2 - e1)
                const VEC e12{ e2 - e1 };
                if (const T denom{ GLSL::dot(e12, normal) };
                    !Numerics::areEquals(denom, T{})) {
                    clipped.emplace_back(e1 + (prev_num / denom) * e12);
                } // polygon edge is parallel to half space
                else {
                    cur_inside = prev_inside;
                }
            }

            // Point inside, add it
            if (cur_inside) {
                clipped.emplace_back(e2);
            }

            // Update previous state
            prev_num = num;
            prev_inside = cur_inside;
            e1 = e2;
        }

        return clipped;
    }

    /**
    * \brief given a simple polygon (as a collection of ordered points) return points and junctions along its approximated medial axis.
    *        notice that the medial axis is approximated and given as points, not lines.
    *
    * @param {forward_iterator,                   in}  iterator to polygon first point (polygon is ordered in counter clockwise manner)
    * @param {forward_iterator,                   in}  iterator to polygon last point (polygon is ordered in counter clockwise manner)
    * @param {value_type,                         in}  amount of distance "step" used when calculating medial axis points and merging medial axis points
    * @param {vector<{IFixedVector, value_type}>, out} collection of points and junctions along polygon approximated medial axis.
    *                                                  the information in collection index 'i' gives:
    *                                                  { medial axis point, medial axis point squared distance from polygon vertex at index 'i'}
    */
    template<std::forward_iterator It,
             class VEC = typename std::decay_t<decltype(*std::declval<It>())>,
             class T = typename VEC::value_type>
        requires(GLSL::is_fixed_vector_v<VEC>&& VEC::length() == 2)
    constexpr auto get_approximated_medial_axis(const It first, const It last, const T step) {
        using mat_t = struct { VEC point; T squared_distance; };

        assert(Algorithms2D::Internals::is_simple(first, last));

        // housekeeping
        T sign{ static_cast<T>(1.0) };
        std::vector<mat_t> medial_axis_points;
        medial_axis_points.reserve(static_cast<std::size_t>(std::distance(first, last)));

        // lambda to calculate the normal of a vertex given its neighboring (previous and next) vertices, in clock wise ordered polygon
        const auto get_normal_at_vertex = [](const VEC& prev, const VEC& vertex, const VEC& next) -> VEC {
            // edges
            const VEC p0{ vertex - prev };
            const VEC p1{ next - vertex };

            // edges normal (assume clock wise winding)
            VEC n0(-p0.y, p0.x);
            VEC n1(-p1.y, p1.x);

            // vertex normal
            return GLSL::normalize((n0 + n1) /  static_cast<T>(2.0) );
        };

        // lambda to approximate closest medial axis point to given polygon vertex ('vertex') given its neighbors ('prev' and 'next')
        const auto approximate_closest_mat_point = [&first, &last, step, sign, &get_normal_at_vertex]
                                                   (const VEC& prev, const VEC& vertex, const VEC& next) -> mat_t {
            // direction of advance
            const VEC normal{ get_normal_at_vertex(prev, vertex, next) };

            // find largest inscribed circle along 'n' which is tangential to 'p'
            VEC center(vertex);
            T distance_squared{ std::numeric_limits<T>::min() };
            T distance_squared_prev{};
            while (distance_squared > distance_squared_prev) {
                distance_squared_prev = distance_squared;
                center += sign * step * normal;
                distance_squared = PointDistance::squared_udf_to_polygon(first, last, center);
            }

            return mat_t{ center, distance_squared };
        };

        // determine normal direction to be inward
        if (const VEC normal{ get_normal_at_vertex(*(last - 1), *first, *(first + 1)) }; 
            !Algorithms2D::is_point_inside_polygon(first, last, *first + sign * step * normal)) {
            sign *= static_cast<T>(-1.0);
            assert(Algorithms2D::is_point_inside_polygon(first, last, *first + sign * step * normal));
        }

        // get medial axis points
        medial_axis_points.emplace_back(approximate_closest_mat_point(*(last - 1), *first, *(first + 1)));
        for (auto prev{ first }, curr{ first + 1 }, next{ first + 2 }; next != last; ++prev, ++curr, ++next) {
            medial_axis_points.emplace_back(approximate_closest_mat_point(*prev, *curr, *next));
        }
        medial_axis_points.emplace_back(approximate_closest_mat_point(*(last - 2), *(last - 1), *first));

        // output
        return medial_axis_points;
    }
}
