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
// euclidean unsigned/signed distance of a point from a primitive.
// see https://iquilezles.org/articles/distfunctions2d/ and https://iquilezles.org/articles/distfunctions/.
//
namespace PointDistance {

    /**
    * \brief return the unsigned distance of a point from segment
    * @param {IFixedVector, in}  point
    * @param {IFixedVector, in}  segment point #1
    * @param {IFixedVector, in}  segment point #2
    * @param {value_type,   out} unsigned distance, negative value if segment is a point
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr T udf_to_segment(const VEC& p, const VEC& a, const VEC& b) {
        const VEC pa{ p - a };
        const VEC ba{ b - a };
        const T dot{ GLSL::dot(ba) };
        if (Numerics::areEquals(dot, T{})) {
            return T{};
        }
        [[assume(dot > T{})]];
        const T h{ Numerics::clamp<T{}, static_cast<T>(1)>(GLSL::dot(pa, ba) / dot) };
        return GLSL::length(pa - ba * h);
    }

    /**
    * \brief return the unsigned distance of a point from segment
    * @param {IFixedVector, in}  point
    * @param {IFixedVector, in}  segment point #1
    * @param {IFixedVector, in}  segment point #2
    * @param {value_type,   out} unsigned squared distance, negative value if segment is a point
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr T squared_udf_to_segment(const VEC& p, const VEC& a, const VEC& b) {
        const VEC pa{ p - a };
        const VEC ba{ b - a };
        const T dot{ GLSL::dot(ba) };
        if (Numerics::areEquals(dot, T{})) {
            return T{};
        }
        [[assume(dot > T{})]];
        const T h{ Numerics::clamp<T{}, static_cast<T>(1)>(GLSL::dot(pa, ba) / dot) };
        return GLSL::dot(pa - ba * h);
    }

    /**
    * \brief return the signed distance of a point from circle/sphere
    * @param {IFixedVector, in}  point
    * @param {IFixedVector, in}  circle/sphere center
    * @param {value_type,   in}  circle/sphere radius
    * @param {value_type,   out} signed distance
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_floating_point_v<T> && (VEC::length() <= 3))
    constexpr T sdf_to_sphere(const VEC& p, const VEC& c, const T r) noexcept {
        return (GLSL::length(p - c) - r);
    }

    /**
    * \brief return the signed distance of a point from rectangle/box at center
    * @param {IFixedVector, in}  point
    * @param {IFixedVector, in}  rectangle/box extents
    * @param {value_type,   out} signed distance
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_floating_point_v<T> && (VEC::length() <= 3))
    constexpr T sdf_to_box_at_center(const VEC& p, const VEC& b) noexcept {
        assert(b.x > T{});
        assert(b.y > T{});
        if constexpr (VEC::length() == 3) {
            assert(b.z > T{});
        }

        const VEC d{ GLSL::abs(p) - b };
        const VEC zero;
        return GLSL::length(GLSL::max(d, zero)) + Numerics::min(GLSL::max(d), T{});
    }

    /**
    * \brief return the signed distance of a point from oriented rectangle/box
    * @param {Vector2,        in}  point
    * @param {Vector2,        in}  rectangle center
    * @param {Vector2,        in}  rectangle/box half extents
    * @param {Matrix2,        in}  rectangle/box orientation matrix
    * @param {floating_point, out} signed distance
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr T sdf_to_oriented_box_at_center(const GLSL::Vector2<T>& p, const GLSL::Vector2<T>& c, const GLSL::Vector2<T>& he, const GLSL::Matrix2<T> rot) noexcept {
        const GLSL::Vector2<T> point{ rot * (p - c) };
        return sdf_to_box_at_center(point, he);
    }

    /**
    * \brief return the signed distance of a point from triangle
    * @param {IFixedVector, in}  point
    * @param {IFixedVector, in}  triangle vertex #0
    * @param {IFixedVector, in}  triangle vertex #1
    * @param {IFixedVector, in}  triangle vertex #2
    * @param {value_type,   out} signed distance
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_floating_point_v<T> && (VEC::length() == 2))
    constexpr T sdf_to_triangle(const VEC& p, const VEC& p0, const VEC& p1, const VEC& p2) {
        constexpr T one{ static_cast<T>(1) };
        const VEC e0{ p1 - p0 };
        const VEC e1{ p2 - p1 };
        const VEC e2{ p0 - p2 };
        const T dot0{ GLSL::dot(e0) };
        const T dot1{ GLSL::dot(e1) };
        const T dot2{ GLSL::dot(e2) };
        assert(dot0 != T{});
        assert(dot0 != T{});
        assert(dot0 != T{});

        const VEC v0{ p - p0 };
        const VEC v1{ p - p1 };
        const VEC v2{ p - p2 };
        const VEC pq0{ v0 - e0 * Numerics::clamp<T{}, one>(GLSL::dot(v0, e0) / dot0) };
        const VEC pq1{ v1 - e1 * Numerics::clamp<T{}, one>(GLSL::dot(v1, e1) / dot1) };
        const VEC pq2{ v2 - e2 * Numerics::clamp<T{}, one>(GLSL::dot(v2, e2) / dot2) };

        const T s{ Numerics::sign(GLSL::cross(e0, e2)) };
        const VEC d{ GLSL::min(GLSL::min(VEC(GLSL::dot(pq0), s * GLSL::cross(v0, e0)),
                                         VEC(GLSL::dot(pq1), s * GLSL::cross(v1, e1))),
                                         VEC(GLSL::dot(pq2), s * GLSL::cross(v2, e2))) };
        assert(d.x >= T{});
        return -std::sqrt(d.x) * Numerics::sign(d.y);
    }

    /**
    * \brief return the signed distance of closed polygon
    * @param {array<IFixedVector>, in}  polygon points
    * @param {IFixedVector,        in}  point
    * @param {value_type,          out} signed distance
    **/
    template<std::size_t N, GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_floating_point_v<T> && (VEC::length() == 2))
    constexpr T sdf_to_polygon(const std::array<VEC, N>& v, const VEC& p) {
        constexpr T one{ static_cast<T>(1) };
        T d{ GLSL::dot(p - v[0]) };
        T s{ one };

        for (std::size_t i{}, j{ N - 1 }; i < N; j = i, i++) {
            const VEC e{ v[j] - v[i] };
            const VEC w{ p - v[i] };
            const T dot{ GLSL::dot(e) };
            assert(!Numerics::areEquals(dot, T{}));

            const VEC b{ w - e * Numerics::clamp<T{}, one> (GLSL::dot(w, e) / dot) };
            d = Numerics::min(d, GLSL::dot(b));
            if (p.y >= v[i].y && p.y < v[j].y && e.x * w.y > e.y * w.x) {
                s *= static_cast<T>(-1);
            }
        }

        [[assume(d >= T{})]];
        return (s * std::sqrt(d));
    }

    /**
    * \brief return the signed distance of closed polygon
    * @param {vector<IFixedVector>, in}  polygon points
    * @param {IFixedVector,         in}  point
    * @param {value_type,           out} signed distance
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_floating_point_v<T> && (VEC::length() == 2))
    constexpr T sdf_to_polygon(const std::vector<VEC>& v, const VEC& p) {
        constexpr T one{ static_cast<T>(1) };
        T d{ GLSL::dot(p - v[0]) };
        T s{ one };
        const std::size_t N{ v.size() };
        for (std::size_t i{}, j{ N - 1 }; i < N; j = i, i++) {
            const VEC e{ v[j] - v[i] };
            const VEC w{ p - v[i] };
            const T dot{ GLSL::dot(e) };
            assert(!Numerics::areEquals(dot, T{}));

            const VEC b{ w - e * Numerics::clamp<T{}, one> (GLSL::dot(w, e) / dot) };
            d = Numerics::min(d, GLSL::dot(b));
            if (p.y >= v[i].y && p.y < v[j].y && e.x * w.y > e.y * w.x) {
                s *= static_cast<T>(-1);
            }
        }

        [[assume(d >= T{})]];
        return (s * std::sqrt(d));
    }

    /**
    * \brief return the signed distance of n-star polygon located around center
    * @param {IFixedVector, in}  point
    * @param {value_type,   in}  polygon radius
    * @param {integral,     in}  polygon number of vertices
    * @param {value_type,   out} signed distance
    **/
    template<GLSL::IFixedVector VEC, typename U, class T = typename VEC::value_type>
        requires(std::is_integral_v<U> && std::is_floating_point_v<T> && (VEC::length() == 2))
    constexpr T sdf_to_regular_poygon(VEC p, const T r, const U n) {
        assert(n > 0);
        // these 4 lines can be precomputed for a given shape
        const T an{ std::numbers::pi_v<T> / static_cast<T>(n) };
        const VEC cs(std::cos(an), std::sin(an));

        // reduce to first sector
        const T bn = [&]() {
            if constexpr (std::is_floating_point_v<T>) {
                return std::fmod(std::atan2(p.x, p.y), static_cast<T>(2) * an) - an;
            }
            else {
                return (std::atan2(p.x, p.y) % static_cast<T>(2) * an) - an;
            }
        }();

        p = GLSL::length(p) * VEC(std::cos(bn), std::abs(std::sin(bn)));

        // line sdf
        p -= r * cs;
        p.y += Numerics::clamp(-p.y, T{}, r * cs.y);
        return Numerics::sign(p.x) * GLSL::length(p);
    }

    /**
    * \brief return the signed distance of ellipse located at center
    * @param {IFixedVector, in}  point
    * @param {IFixedVector, in}  {ellipse radii along x, ellipse radii along y}
    * @param {value_type,   out} signed distance
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_floating_point_v<T> && (VEC::length() == 2))
    constexpr T sdf_to_ellipse(const VEC p, const VEC ab) {
        VEC _p{ GLSL::abs(p) };
        VEC _ab{ ab };
        if (_p.x > _p.y) {
            _p = _p.yx;
            _ab = _ab.yx;
        }

        const T co = [&_ab, &_p]() {
            const T l{ _ab.y * _ab.y - _ab.x * _ab.x };
            assert(!Numerics::areEquals(l, T{}));
            const T m{ _ab.x * _p.x / l };
            const T m2{ m * m };
            const T n{ _ab.y * _p.y / l };
            const T n2{ n * n };
            const T c{ (m2 + n2 - static_cast<T>(1)) / static_cast<T>(3) };
            const T c3{ c * c * c };
            const T q{ c3 + static_cast<T>(2) * m2 * n2 };
            const T d{ c3 + m2 * n2 };
            const T g{ m + m * n2 };

            if (d < T{}) {
                assert(!Numerics::areEquals(c3, T{}));
                const T h{ std::acos(Numerics::clamp<static_cast<T>(-1), static_cast<T>(1)>(q / c3)) / static_cast<T>(3) };
                const T s{ std::cos(h) };
                const T t{ std::sin(h) * std::sqrt(static_cast<T>(3)) };
                const T squaredx{ -c * (s + t + static_cast<T>(2)) + m2 };
                const T squaredy{ -c * (s - t + static_cast<T>(2)) + m2 };
                assert(squaredx >= T{});
                assert(squaredy >= T{});
                const T rx{ std::sqrt(squaredx) };
                const T ry{ std::sqrt(squaredy) };
                const T rxry{ rx + ry };

                assert(rxry > T{});
                return (ry + std::copysign(rx, l) + std::abs(g) / rxry - m) / static_cast<T>(2);
            }
            else {
                [[assume(d >= T{})]];
                const T h{ static_cast<T>(2) * m * n * std::sqrt(d) };
                const T s{ std::copysign(std::pow(std::abs(q + h), static_cast<T>(1.0 / 3.0)), q + h) };
                const T u{ std::copysign(std::pow(std::abs(q - h), static_cast<T>(1.0 / 3.0)), q - h) };
                const T rx{ -s - u - static_cast<T>(4) * c + static_cast<T>(2) * m2 };
                const T ry{ (s - u) * std::sqrt(static_cast<T>(3)) };
                const T rxry{ rx * rx + ry * ry };
                assert(rxry >= T{});
                const T rm{ std::sqrt(rxry) };
                assert(rm > T{});
                const T rmrx{ rm - rx };
                assert(rmrx >= T{});
                return ((ry / std::sqrt(rmrx) + static_cast<T>(2) * g / rm - m) / static_cast<T>(2));
            }
            }();

        const T coo{ static_cast<T>(1) - co * co };
        assert(coo >= T{});
        const VEC cocoo{{ co, std::sqrt(coo) }};
        const VEC r{ _ab * cocoo };
        return std::copysign(GLSL::length(r - _p), _p.y - r.y);
    }

    /**
    * \brief return the signed distance of bounded plane
    * @param {Vector3,        in}  point
    * @param {Vector4,        in}  plane (nx, ny, nz, d)
    * @param {floating_point, out} signed distance
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr T sdf_to_plane(const GLSL::Vector3<T>& p, const GLSL::Vector4<T>& plane) {
        const GLSL::Vector3<T> n{ plane.xyz };
        assert(Extra::is_normalized(n));
        return GLSL::dot(p, n) + plane.w;
    }

    /**
    * \brief return the signed distance of capsule
    * @param {IFixedVector, in}  point
    * @param {IFixedVector, in}  capsule point #1
    * @param {IFixedVector, in}  capsule point #2
    * @param {value_type,   in}  capsule radius
    * @param {value_type,   out} signed distance
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_floating_point_v<T> && (VEC::length() == 3))
    constexpr T sdf_to_capsule(const VEC& p, const VEC& a, const VEC& b, const T r) {
        const VEC pa(p - a);
        const VEC ba(b - a);
        const T dot{ GLSL::dot(ba)};
        assert(dot > T{});
        const T h{ Numerics::clamp<T{}, static_cast<T>(1)>(GLSL::dot(pa, ba) / dot)};
        return GLSL::length(pa - ba * h) - r;
    }

    /**
    * \brief return the signed distance of capped cylinder
    * @param {IFixedVector, in}  point
    * @param {IFixedVector, in}  capped cylinder point #1
    * @param {IFixedVector, in}  capped cylinder point #2
    * @param {value_type,   in}  capped cylinder radius
    * @param {value_type,   out} signed distance
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_floating_point_v<T> && (VEC::length() == 3))
    constexpr T sdf_to_capped_cylinder(const VEC& p, const VEC& a, const VEC& b, const T r) {
        const VEC ba(b - a);
        const VEC pa(p - a);
        const T baba{ GLSL::dot(ba, ba) };
        assert(baba > T{});
        const T paba{ GLSL::dot(pa, ba) };
        const T x{ GLSL::length(pa * baba - ba * paba) - r * baba };
        const T y{ std::abs(paba - static_cast<T>(0.5) * baba) - static_cast<T>(0.5) * baba };
        const T x2{ x * x };
        const T y2{ y * y * baba };
        const T d{ (Numerics::max(x, y) < T{}) ?
                    -Numerics::min(x2, y2) :
                    (((x > T{}) ? x2 : T{}) + ((y > T{}) ? y2 : T{})) };

        return (Numerics::sign(d) * std::sqrt(std::abs(d)) / baba);
    }

    /**
    * \brief return the signed distance of bound ellipsoid located at center.
    *        this calculation is not exact, but is useful for checking if points is inside ellipsoied bounding box.
    * @param {IFixedVector, in}  point
    * @param {IFixedVector, in}  ellipsoid radii
    * @param {value_type,   out} signed distance
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_floating_point_v<T> && (VEC::length() == 3))
    constexpr T sdf_to_bound_ellipsoied(const VEC& p, const VEC& r) {
        assert(r.x > T{});
        assert(r.y > T{});
        assert(r.z > T{});

        const T k0{ GLSL::length(p / r) };
        const T k1{ GLSL::length(p / (r * r)) };
        
        if (k1 > 0) {
            return k0 * (k0 - static_cast<T>(1)) / k1;
        }
        else [[unlikely]] {
            return -GLSL::min(r);
        }
    }

    /**
    * \brief return the unsigned distance of triangle. point inside the triange will be returned as zero.
    * @param {IFixedVector, in}  point
    * @param {IFixedVector, in}  vertex #0
    * @param {IFixedVector, in}  vertex #1
    * @param {IFixedVector, in}  vertex #2
    * @param {value_type,   out} signed distance
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_floating_point_v<T> && (VEC::length() == 3))
    constexpr auto udf_to_triangle(const VEC& p, const VEC& a, const VEC& b, const VEC& c) {
        constexpr T one{ static_cast<T>(1) };
        const VEC ba(b - a);
        const VEC pa(p - a);
        const VEC cb(c - b);
        const VEC pb(p - b);
        const VEC ac(a - c);
        const VEC pc(p - c);
        const VEC nor(cross(ba, ac));

        // is point "outside" the triangle?
        if (Numerics::sign(GLSL::dot(GLSL::cross(ba, nor), pa)) +
            Numerics::sign(GLSL::dot(GLSL::cross(cb, nor), pb)) +
            Numerics::sign(GLSL::dot(GLSL::cross(ac, nor), pc)) < static_cast<T>(2.0)) {
            const T dot_ba{ GLSL::dot(ba) };
            const T dot_cb{ GLSL::dot(cb) };
            const T dot_ac{ GLSL::dot(ac) };
            assert(dot_ba > T{});
            assert(dot_cb > T{});
            assert(dot_ac > T{});

            const T squared{ Numerics::min(
                GLSL::dot(ba * Numerics::clamp < T{}, one > (GLSL::dot(ba, pa) / dot_ba) - pa),
                GLSL::dot(cb * Numerics::clamp < T{}, one > (GLSL::dot(cb, pb) / dot_cb) - pb),
                GLSL::dot(ac * Numerics::clamp < T{}, one > (GLSL::dot(ac, pc) / dot_ac) - pc)) };
            assert(squared >= T{});
            return std::sqrt(squared);
        } // point "inside" the triangle
        else {
            const T dot{ GLSL::dot(nor) };
            assert(dot > T{});

            const T squared{ GLSL::dot(nor, pa) * GLSL::dot(nor, pa) / dot };
            assert(squared >= T{});
            return std::sqrt(squared);
        }
    }

    /**
    * \brief return the unsigned distance of quad. point inside the quad will be returned as zero.
    * @param {IFixedVector, in}  point
    * @param {IFixedVector, in}  vertex #0
    * @param {IFixedVector, in}  vertex #1
    * @param {IFixedVector, in}  vertex #2
    * @param {IFixedVector, in}  vertex #3
    * @param {value_type,   out} signed distance
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_floating_point_v<T> && (VEC::length() == 3))
    constexpr auto udf_to_quad(const VEC& p, const VEC& a, const VEC& b, const VEC& c, const VEC& d) {
        constexpr T one{ static_cast<T>(1) };
        const VEC ba(b - a);
        const VEC pa(p - a);
        const VEC cb(c - b);
        const VEC pb(p - b);
        const VEC dc(d - c);
        const VEC pc(p - c);
        const VEC ad(a - d);
        const VEC pd(p - d);
        const VEC nor(GLSL::cross(ba, ad));

        // point "outside" the quad?
        if (Numerics::sign(GLSL::dot(GLSL::cross(ba, nor), pa)) +
            Numerics::sign(GLSL::dot(GLSL::cross(cb, nor), pb)) +
            Numerics::sign(GLSL::dot(GLSL::cross(dc, nor), pc)) +
            Numerics::sign(GLSL::dot(cross(ad, nor), pd)) < static_cast<T>(3.0)) {
            const T do_ba{ GLSL::dot(ba) };
            const T do_cb{ GLSL::dot(cb) };
            const T do_dc{ GLSL::dot(dc) };
            const T do_ad{ GLSL::dot(ad) };
            assert(do_ba > T{});
            assert(do_cb > T{});
            assert(do_dc > T{});
            assert(do_ad > T{});

            const T squared{ Numerics::min(
                GLSL::dot(ba * Numerics::clamp < T{}, one > (GLSL::dot(ba, pa) / do_ba) - pa),
                GLSL::dot(cb * Numerics::clamp < T{}, one > (GLSL::dot(cb, pb) / do_cb) - pb),
                GLSL::dot(dc * Numerics::clamp < T{}, one > (GLSL::dot(dc, pc) / do_dc) - pc),
                GLSL::dot(ad * Numerics::clamp < T{}, one > (GLSL::dot(ad, pd) / do_ad) - pd)) };
            assert(squared >= T{});
            return std::sqrt(squared);
        } // point "inside" the quad
        else {
            const T dot{ GLSL::dot(nor) };
            assert(dot > T{});

            const T squared{ GLSL::dot(nor, pa) * GLSL::dot(nor, pa) / dot };
            assert(squared >= T{});
            return std::sqrt(squared);
        }
    }
}
