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

//
// triangle related functions
//
namespace Triangle {

    /**
    * \brief test if triangle is valid
    * @param {Vector2|Vector3, in}  triangle vertex #0
    * @param {Vector2|Vector3, in}  triangle vertex #1
    * @param {Vector2|Vector3, in}  triangle vertex #2
    * @param {bool,            out} true if triangle is valid, false otherwise
    **/
    template<GLSL::IFixedVector VEC>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr bool is_valid(const VEC& v0, const VEC& v1, const VEC& v2) noexcept {
        using T = typename VEC::value_type;

        // triangle sides length, sorted in ascending order
        const VEC v10{ v1 - v0 };
        const VEC v21{ v2 - v1 };
        const VEC v20{ v2 - v0 };
        std::array<T, 3> sides{ { GLSL::length(v1 - v0),
                                  GLSL::length(v2 - v1),
                                  GLSL::length(v2 - v0) } };
        if (sides[0] > sides[2]) { Utilities::swap(sides[0], sides[2]); }
        if (sides[0] > sides[1]) { Utilities::swap(sides[0], sides[1]); }
        if (sides[1] > sides[2]) { Utilities::swap(sides[1], sides[2]); }

        return ((sides[0] - (sides[2] - sides[1])) > T{});
    }

    /**
    * \brief calculate the barycentric coordinates for a cartesian coordinate point in a triangle defined by vertices
    * @param {Vector2|Vector3, in}  point
    * @param {Vector2|Vector3, in}  triangle vertex #0
    * @param {Vector2|Vector3, in}  triangle vertex #1
    * @param {Vector2|Vector3, in}  triangle vertex #2
    * @param {Vector3,         out} barycentric coordinates of a given point relative to triangle
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr GLSL::Vector3<T> get_point_in_barycentric(const VEC& p, const VEC& a, const VEC& b, const VEC& c) {
        assert(Triangle::is_valid(a, b, c));

        const VEC v0{ b - a };
        const VEC v1{ c - a };
        const VEC v2{ p - a };

        const T d00{ GLSL::dot(v0) };
        const T d01{ GLSL::dot(v0, v1) };
        const T d11{ GLSL::dot(v1) };
        const T denom{ d00 * d11 - d01 * d01 };
        if (denom <= T{}) [[unlikely]] {
            return GLSL::Vector3<T>(static_cast<T>(-1));
        }

        const T d20{ GLSL::dot(v2, v0) };
        const T d21{ GLSL::dot(v2, v1) };

        [[assume(denom > T{})]]
        const T v{ (d11 * d20 - d01 * d21) / denom };
        const T w{ (d00 * d21 - d01 * d20) / denom };
        return GLSL::Vector3<T>(static_cast<T>(1) - v - w, v, w);
    }

    /**
    * \brief check if a point is contained within a triangle
    *        uses Thomas Müller algorithm. faster than calculating triangle sign distance field at point.
    * @param {Vector2|Vector3, in}  point
    * @param {Vector2|Vector3, in}  triangle vertex #0
    * @param {Vector2|Vector3, in}  triangle vertex #1
    * @param {Vector2|Vector3, in}  triangle vertex #2
    * @param {bool,            out} true if point is inside triangle, false otherwise
    **/
    template<GLSL::IFixedVector VEC>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr bool is_point_within_triangle(const VEC& p, const VEC& a, const VEC& b, const VEC& c) {
        using T = typename VEC::value_type;
        constexpr std::size_t N{ VEC::length() };
        assert(Triangle::is_valid(a, b, c));

        const VEC local_a{ a - p };
        const VEC local_b{ b - p };
        const VEC local_c{ c - p };

        const auto u = GLSL::cross(local_b, local_c);
        const auto v = GLSL::cross(local_c, local_a);
        const auto w = GLSL::cross(local_a, local_b);

        if constexpr (N == 2) {
            return (u * v >= T{}) && (u * w >= T{});
        }
        else {
            return (GLSL::dot(u, v) >= T{}) && (GLSL::dot(u, w) >= T{});
        }
    }

    /**
    * \brief given triangle (by its vertices) - return its barycentric coordinates
    * @param {Vector2|Vector3, in}  triangle vertex #0
    * @param {Vector2|Vector3, in}  triangle vertex #1
    * @param {Vector2|Vector3, in}  triangle vertex #2
    * @param {Vector3,         out} barycentric coordinates
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr GLSL::Vector3<T> barycentric_from_cartesian(const VEC& a, const VEC& b, const VEC& c) {
        assert(Triangle::is_valid(a, b, c));

        const T daa{ GLSL::dot(a, a) };
        const T dab{ GLSL::dot(a, b) };
        const T dbb{ GLSL::dot(b, b) };
        const T dca{ GLSL::dot(c, a) };
        const T dcb{ GLSL::dot(c, b) };
        const T denom{ daa * dbb - dab * dab };
        if (denom <= T{}) [[unlikely]] {
            return GLSL::Vector3<T>(static_cast<T>(-1));
        }

        [[assume(denom != T{})]]
        const T y{ (dbb * dca - dab * dcb) / denom };
        const T z{ (daa * dcb - dab * dca) / denom };
        return GLSL::Vector3<T>(static_cast<T>(1) - y - z, y, z);
    }

    /**
    * \brief given a point and triangle, return point on triangle closest to the point
    * @param {Vector2|Vector3,  in} point
    * @param {Vector2|Vector3, in}  triangle vertex #0
    * @param {Vector2|Vector3, in}  triangle vertex #1
    * @param {Vector2|Vector3, in}  triangle vertex #2
    * @param {Vector2|Vector3, out} closest corner
    **/
    template<GLSL::IFixedVector VEC>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr VEC closest_point(const VEC& p, const VEC& a, const VEC& b, const VEC& c) noexcept {
        using T = typename VEC::value_type;
        assert(Triangle::is_valid(a, b, c));

        const VEC ab{ b - a };
        const VEC ac{ c - a };
        const VEC ap{ p - a };

        const T d1{ GLSL::dot(ab, ap) };
        const T d2{ GLSL::dot(ac, ap) };
        if (d1 <= T{} && d2 <= T{}) {
            return a;
        }

        const VEC bp{ p - b };
        const T d3{ GLSL::dot(ab, bp) };
        const T d4{ GLSL::dot(ac, bp) };
        if (d3 >= T{} && d4 <= d3) {
            return b;
        }

        const T vc{ d1 * d4 - d3 * d2 };
        if (vc <= T{} && d1 >= T{} && d3 <= T{}) {
            const T den{ d1 - d3 };
            [[assume(den > T{})]];
            return a + (d1 / den) * ab;
        }

        const VEC cp{ p - c };
        const T d5{ GLSL::dot(ab, cp) };
        const T d6{ GLSL::dot(ac, cp) };
        if (d6 >= T{} && d5 <= d6) {
            return c;
        }

        const T vb{ d5 * d2 - d1 * d6 };
        if (vb <= T{} && d2 >= T{} && d6 <= T{}) {
            const T den{ d2 - d6 };
            [[assume(den > T{})]];
            return a + (d2 / den) * ac;
        }

        const T va{ d3 * d6 - d5 * d4 };
        if (va <= T{} && (d4 - d3) >= T{} && (d5 - d6) >= T{}) {
            const T den{ (d4 - d3) + (d5 - d6) };
            [[assume(den > T{})]];
            return b + ((d4 - d3) / den) * (c - b);
        }

        const T den{ va + vb + vc };
        [[assume(den > T{})]];
        const T denom{ static_cast<T>(1) / den };
        return a + ab * vb * denom + ac * vc * denom;
    }

    /**
    * \brief orientation predicate for a point and a triangle
    * @param {Vector3,    in}  point
    * @param {Vector3,    in}  triangle vertex #0
    * @param {Vector3,    in}  triangle vertex #1
    * @param {Vector3,    in}  triangle vertex #2
    * @param {value_type, in}  tolerance for point to be on triangle
    * @param {int32_t,    out} -1/0/1 - point behind triangle / point on triangle / point in front of triangle
    **/
    template<class T>
        requires(std::is_floating_point_v<T>)
    constexpr std::int32_t point_triangle_orientation(const GLSL::Vector3<T>& p, const GLSL::Vector3<T>& a, const GLSL::Vector3<T>& b,
                                                      const GLSL::Vector3<T>& c, const T eps = static_cast<T>(1e-10)) noexcept {
        const GLSL::Matrix4<T> mat(GLSL::Vector4<T>(a, static_cast<T>(1)),
                                   GLSL::Vector4<T>(b, static_cast<T>(1)),
                                   GLSL::Vector4<T>(c, static_cast<T>(1)),
                                   GLSL::Vector4<T>(p, static_cast<T>(1)));
        const T det{ GLSL::determinant(mat) };
        if (det < -eps) {
            return -1;
        }
        else if (det > eps) {
            return 1;
        }
        else {
            return 0;
        }
    }

    /**
    * \brief given two non-degenerate non-coplanar triangles - check if they intersect and return intersection point/segment
    * @param {Vector3,            in}  triangle #1 vertex #0
    * @param {Vector3,            in}  triangle #1 vertex #1
    * @param {Vector3,            in}  triangle #1 vertex #2
    * @param {Vector3,            in}  triangle #2 vertex #0
    * @param {Vector3,            in}  triangle #2 vertex #1
    * @param {Vector3,            in}  triangle #2 vertex #2
    * @param {{Vector3, Vector3}, out} {point #1 of intersection segment, point #2 if intersection segment}
    *                                  if no intersectio occures, both point #1 and point #2 will be infinity
    **/
    template<class T>
        requires(std::is_floating_point_v<T>)
    constexpr auto check_triangles_intersection(const GLSL::Vector3<T>& t1a, const GLSL::Vector3<T>& t1b, const GLSL::Vector3<T>& t1c,
                                                const GLSL::Vector3<T>& t2a, const GLSL::Vector3<T>& t2b, const GLSL::Vector3<T>& t2c) noexcept {
        using vec_t = GLSL::Vector3<T>;
        using out_t = struct { vec_t p0; vec_t p1; };
        constexpr T eps{ static_cast<T>(1e-10) };

        assert(Triangle::is_valid(t1a, t1b, t1c));
        assert(Triangle::is_valid(t2a, t2b, t2c));

        // housekeeping
        out_t out{
            .p0 = vec_t(std::numeric_limits<T>::max()),
            .p1 = vec_t(std::numeric_limits<T>::max())
        };

        // lambda which permutes triangle vertices to the left
        const auto permute_left = [](vec_t& a, vec_t& b, vec_t& c) {
            vec_t temp(a);
            a = MOV(b);
            b = MOV(c);
            c = MOV(temp);
        };

        // lambda which permutes triangle vertices to the right
        const auto permute_right = [](vec_t& a, vec_t& b, vec_t& c) {
            vec_t temp(c);
            c = MOV(b);
            b = MOV(a);
            a = MOV(temp);
        };

        // lambda which permutes triangle vertex 'a' such that it would "lie on its side"
        // (oa/ob/oc = relative position to vertices a/b/c)
        const auto make_a_alone = [&permute_left, &permute_right](vec_t& a, vec_t& b, vec_t& c,
                                     std::int32_t oa, std::int32_t ob, std::int32_t oc) {
            // Permute a, b, c so that a is alone on its side
            if (oa == ob) {
                // c is alone, permute right so c becomes a
                permute_right(a, b, c);
            } // b is alone, permute so b becomes a
            else if (oa == oc) {
                permute_left(a, b, c);
            } else if (ob != oc) {
              // In case a, b, c have different orientation, put a on positive side
              if (ob > 0) {
                  permute_left(a, b, c);
              } else if (oc > 0) {
                  permute_right(a, b, c);
              }
            }
        };
        
        // lambda to permute triangle 2 so its 'a' vertex lies in front of triangle 1
        const auto make_a_positive = [](const vec_t& a1, const vec_t& a2, vec_t& b2, vec_t& c2) {
            const std::int32_t o{Triangle::point_triangle_orientation(a1, a2, b2, c2) };
            if (o < 0) {
                vec_t temp{ c2 };
                c2 = MOV(b2);
                b2 = MOV(temp);
            }
        };

        // lambda to calculate intersection point between segment and a plane (defined by point and normal)
        const auto get_segment_plane_intersect_point = [](const vec_t& a, const vec_t& b,
                                                          const vec_t& p, const vec_t& n,
                                                          vec_t& intersection) {
            const vec_t u{ b - a };
            const vec_t v{ a - p };
            const T dot1{ GLSL::dot(n, u) };
            const T dot2{ GLSL::dot(n, v) };
            [[assume(dot1 > T{})]];
            intersection = a + (-dot2 / dot1) * u;
        };

        // check rekative position of triangle #1 vertices against triangle #2
        const std::int32_t o1a{ Triangle::point_triangle_orientation(t1a, t2a, t2b, t2c) };
        const std::int32_t o1b{ Triangle::point_triangle_orientation(t1b, t2a, t2b, t2c) };
        const std::int32_t o1c{ Triangle::point_triangle_orientation(t1c, t2a, t2b, t2c) };

        // are vertices at same orientation?
        if ((o1a == o1b) && (o1a == o1c)) {
            return out;
        }

        // check rekative position of triangle #2 vertices against triangle #1
        const std::int32_t o2a{ Triangle::point_triangle_orientation(t2a, t1a, t1b, t1c) };
        const std::int32_t o2b{ Triangle::point_triangle_orientation(t2b, t1a, t1b, t1c) };
        const std::int32_t o2c{ Triangle::point_triangle_orientation(t2c, t1a, t1b, t1c) };

        // are vertices at same orientation?
        if ((o2a == o2b) && (o2a == o2c)) {
            return out;
        }

        // permute vertices
        vec_t _t1a(t1a);
        vec_t _t1b(t1b);
        vec_t _t1c(t1c);
        vec_t _t2a(t2a);
        vec_t _t2b(t2b);
        vec_t _t2c(t2c);
        make_a_alone(_t1a, _t1b, _t1c, o1a, o1b, o1c);
        make_a_alone(_t2a, _t2b, _t2c, o2a, o2b, o2c);

        // swap vertices
        make_a_positive(_t2a, _t1a, _t1b, _t1c);
        make_a_positive(_t1a, _t2a, _t2b, _t2c);

        // triangle 2 relative orientation after permutation
        std::int32_t o1{ Triangle::point_triangle_orientation(_t2b, _t1a, _t1b, _t2a) };
        std::int32_t o2{ Triangle::point_triangle_orientation(_t2a, _t1a, _t1c, _t2c) };
        if (o1 > 0 && o2 > 0) {
            return out;
        }

        // get triangle 2 'a' vertex relative to t1 triangle
        o1 = Triangle::point_triangle_orientation(_t2a, _t1a, _t1c, _t2b);
        o2 = Triangle::point_triangle_orientation(_t2a, _t1a, _t1b, _t2c);

        // triangle normals
        const vec_t n1{ GLSL::normalize(GLSL::cross(_t1c - _t1b, _t1a - _t1b)) };
        const vec_t n2{ GLSL::normalize(GLSL::cross(_t2c - _t2b, _t2a - _t2b)) };

        // calculate intersection points
        vec_t i0;
        vec_t i1;
        if (o1 > 0) {
            if (o2 > 0) {
                // Intersection: k i l j
                get_segment_plane_intersect_point(_t1a, _t1c, _t2a, n2, i0); // i
                get_segment_plane_intersect_point(_t2a, _t2c, _t1a, n1, i1); // l
            }
            else {
                // Intersection: k i j l
                get_segment_plane_intersect_point(_t1a, _t1c, _t2a, n2, i0); // i
                get_segment_plane_intersect_point(_t1a, _t1b, _t2a, n2, i1); // j
            }
        }
        else {
            if (o2 > 0) {
                // Intersection: i k l j
                get_segment_plane_intersect_point(_t2a, _t2b, _t1a, n1, i0); // k
                get_segment_plane_intersect_point(_t2a, _t2c, _t1a, n1, i1); // l
            }
            else {
                // Intersection: i k j l
                get_segment_plane_intersect_point(_t2a, _t2b, _t1a, n1, i0); // i
                get_segment_plane_intersect_point(_t1a, _t1b, _t2a, n2, i1); // k
            }
        }

        // output
        out.p0 = i0;
        out.p1 = i1;
        return out;
    }
}
