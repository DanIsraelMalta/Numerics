#pragma once
#include "Glsl.h"

//
// euclidean unsigned/signed distance of a point from a primitive.
// see https://iquilezles.org/articles/distfunctions2d/ and https://iquilezles.org/articles/distfunctions/.
//
namespace PointDistance {

    /**
    * \brief return the unsigned distance between point and segment
    * @param {Vector2|Vector3, in}  point
    * @param {Vector2|Vector3, in}  segment point #0
    * @param {Vector2|Vector3, in}  segment point #1
    * @param {floating_point,  out} distance
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr T udf_to_segment(const VEC& p, const VEC& a, const VEC& b) {
        const VEC ba{ b - a };
        const T dot{ GLSL::dot(ba) };
        assert(!Numerics::areEquals(dot, T{}));
        const T k{ GLSL::dot(p - a, ba) / dot };
        return GLSL::distance(p, GLSL::mix(a, b, Numerics::clamp(k, T{}, static_cast<T>(1))));
    }

    /**
    * \brief return the signed distance of a point from segment
    * @param {Vector2,        in}  point
    * @param {Vector2,        in}  segment point #1
    * @param {Vector2,        in}  segment point #2
    * @param {floating_point, out} signed distance
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr T sdf_to_segment(const VEC& p, const VEC& a, const VEC& b) {
        const VEC pa{ p - a };
        const VEC ba{ b - a };
        const T dot{ GLSL::dot(ba) };
        assert(!Numerics::areEquals(dot, T{}));
        const T h{ Numerics::clamp<T{}, static_cast<T>(1)>(GLSL::dot(pa, ba) / dot) };
        return GLSL::length(pa - ba * h);
    }

    /**
    * \brief return the signed distance of a point from circle/sphere
    * @param {Vector2|Vector3, in}  point
    * @param {Vector2|Vector3, in}  circle/sphere center
    * @param {floating_point,  in}  circle/sphere radius
    * @param {floating_point,  out} signed distance
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_floating_point_v<T> && (VEC::length() <= 3))
    constexpr T sdf_to_sphere(const VEC& p, const VEC& c, const T r) noexcept {
        return (GLSL::length(p - c) - r);
    }

    /**
    * \brief return the signed distance of a point from rectangle/box at center
    * @param {Vector2,        in}  point
    * @param {Vector2,        in}  rectangle/box extents
    * @param {floating_point, out} signed distance
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
    * \brief return the signed distance of a point from triangle
    * @param {Vector2,        in}  point
    * @param {Vector2,        in}  triangle vertex #0
    * @param {Vector2,        in}  triangle vertex #1
    * @param {Vector2,        in}  triangle vertex #2
    * @param {floating_point, out} signed distance
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto sdf_to_triangle(const GLSL::Vector2<T>& p, const GLSL::Vector2<T>& p0, const GLSL::Vector2<T>& p1, const GLSL::Vector2<T>& p2) {
        constexpr T one{ static_cast<T>(1) };
        const GLSL::Vector2<T> e0{ p1 - p0 };
        const GLSL::Vector2<T> e1{ p2 - p1 };
        const GLSL::Vector2<T> e2{ p0 - p2 };
        const T dot0{ GLSL::dot(e0) };
        const T dot1{ GLSL::dot(e1) };
        const T dot2{ GLSL::dot(e2) };
        assert(dot0 != T{});
        assert(dot0 != T{});
        assert(dot0 != T{});

        const GLSL::Vector2<T> v0{ p - p0 };
        const GLSL::Vector2<T> v1{ p - p1 };
        const GLSL::Vector2<T> v2{ p - p2 };
        const GLSL::Vector2<T> pq0{ v0 - e0 * Numerics::clamp<T{}, one>(GLSL::dot(v0, e0) / dot0) };
        const GLSL::Vector2<T> pq1{ v1 - e1 * Numerics::clamp<T{}, one>(GLSL::dot(v1, e1) / dot1) };
        const GLSL::Vector2<T> pq2{ v2 - e2 * Numerics::clamp<T{}, one>(GLSL::dot(v2, e2) / dot2) };

        const T s{ Numerics::sign(GLSL::cross(e0, e2)) };
        const GLSL::Vector2<T> d{ GLSL::min(GLSL::min(GLSL::Vector2<T>(GLSL::dot(pq0), s * GLSL::cross(v0, e0)),
                                                      GLSL::Vector2<T>(GLSL::dot(pq1), s * GLSL::cross(v1, e1))),
                                                      GLSL::Vector2<T>(GLSL::dot(pq2), s * GLSL::cross(v2, e2))) };
        assert(d.x >= T{});
        return -std::sqrt(d.x) * Numerics::sign(d.y);
    }

    /**
    * \brief return the signed distance of closed polygon
    * @param {array<Vector2>, in}  polygon points
    * @param {Vector2,        in}  point
    * @param {floating_point, out} signed distance
    **/
    template<std::size_t N, typename T>
        requires(std::is_floating_point_v<T>)
    constexpr T sdf_to_polygon(const std::array<GLSL::Vector2<T>, N>& v, const GLSL::Vector2<T>& p) {
        constexpr T one{ static_cast<T>(1) };
        T d{ GLSL::dot(p - v[0]) };
        T s{ one };

        for (std::size_t i{}, j{ N - 1 }; i < N; j = i, i++) {
            const GLSL::Vector2<T> e{ v[j] - v[i] };
            const GLSL::Vector2<T> w{ p - v[i] };
            const T dot{ GLSL::dot(e) };
            assert(!Numerics::areEquals(dot, T{}));

            const GLSL::Vector2<T> b{ w - e * Numerics::clamp < T{}, one > (GLSL::dot(w, e) / dot) };
            d = Numerics::min(d, GLSL::dot(b));
            const GLSL::Vector3<bool> c(p.y >= v[i].y,
                p.y < v[j].y,
                e.x * w.y > e.y * w.x);
            if (GLSL::all(c) || GLSL::all(GLSL::glsl_not(c))) {
                s *= static_cast<T>(-1);
            }
        }

        [[assume(d >= T{})]];
        return (s * std::sqrt(d));
    }

    /**
    * \brief return the signed distance of n-star polygon located around center
    * @param {Vector2,        in}  point
    * @param {floating_point, in}  polygon radius
    * @param {integral,       in}  polygon number of vertices
    * @param {floating_point, out} signed distance
    **/
    template<typename U, typename T>
        requires(std::is_integral_v<U>&& std::is_floating_point_v<T>)
    constexpr T sdf_to_regular_poygon(GLSL::Vector2<T> p, const T r, const U n) {
        assert(n > 0);
        // these 4 lines can be precomputed for a given shape
        const T an{ std::numbers::pi_v<T> / static_cast<T>(n) };
        const GLSL::Vector2<T> cs(std::cos(an), std::sin(an));

        // reduce to first sector
        const T bn = [&]() {
            if constexpr (std::is_floating_point_v<T>) {
                return std::fmod(std::atan2(p.x, p.y), static_cast<T>(2) * an) - an;
            }
            else {
                return (std::atan2(p.x, p.y) % static_cast<T>(2) * an) - an;
            }
            }();

        p = GLSL::length(p) * GLSL::Vector2<T>(std::cos(bn), std::abs(std::sin(bn)));

        // line sdf
        p -= r * cs;
        p.y += Numerics::clamp(-p.y, T{}, r * cs.y);
        return Numerics::sign(p.x) * GLSL::length(p);
    }

    /**
    * \brief return the signed distance of ellipse located at center
    * @param {Vector2,        in}  point
    * @param {Vector2,        in}  {ellipse radii along x, ellipse radii along y}
    * @param {floating_point, out} signed distance
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr T sdf_to_ellipse(const GLSL::Vector2<T> p, const GLSL::Vector2<T> ab) {
        GLSL::Vector2<T> _p{ GLSL::abs(p) };
        GLSL::Vector2<T> _ab{ ab };
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
                const T h{ std::acos(q / c3) / static_cast<T>(3) };
                const T s{ std::cos(h) };
                const T t{ std::sin(h) * std::sqrt(static_cast<T>(3)) };
                const T squaredx{ -c * (s + t + static_cast<T>(2)) + m2 };
                const T squaredy{ -c * (s - t + static_cast<T>(2)) + m2 };
                assert(squaredx >= T{});
                assert(squaredy >= T{});
                const T rx{ std::sqrt(squaredx) };
                const T ry{ std::sqrt(squaredy) };
                const T rxry{ rx + ry };

                assert(!Numerics::areEquals(rxry, T{}));
                return (ry + std::copysign(rx, l) + std::abs(g) / rxry - m) / static_cast<T>(2);
            }
            else {
                [[assume(d >= T{})]];
                const T h{ static_cast<T>(2) * m * n * std::sqrt(d) };
                const T s{ std::copysign(std::pow(std::abs(q + h), static_cast<T>(1 / 3)), q + h) };
                const T u{ std::copysign(std::pow(std::abs(q - h), static_cast<T>(1 / 3)), q - h) };
                const T rx{ -s - u - static_cast<T>(4) * c + static_cast<T>(2) * m2 };
                const T ry{ (s - u) * std::sqrt(static_cast<T>(3)) };
                const T rxry{ rx * rx + ry * ry };
                assert(rxry >= T{});
                const T rm{ std::sqrt(rxry) };
                assert(!Numerics::areEquals(rm, T{}));
                const T rmrx{ rm - rx };
                assert(rmrx >= T{});
                return ((ry / std::sqrt(rmrx) + static_cast<T>(2) * g / rm - m) / static_cast<T>(2));
            }
            }();

        const T coo{ static_cast<T>(1) - co * co };
        assert(coo >= T{});
        GLSL::Vector2<T> cocoo{{ co, std::sqrt(coo) }};
        const GLSL::Vector2<T> r{ _ab * cocoo };
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
        assert(Numerics::areEquals(GLSL::length(n), static_cast<T>(1)));
        return GLSL::dot(p, n) + plane.w;
    }

    /**
    * \brief return the signed distance of capsule
    * @param {Vector3,        in}  point
    * @param {Vector3,        in}  capsule point #1
    * @param {Vector3,        in}  capsule point #2
    * @param {floating_point, in}  capsule radius
    * @param {floating_point, out} signed distance
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr T sdf_to_capsule(const GLSL::Vector3<T>& p, const GLSL::Vector3<T>& a, const GLSL::Vector3<T>& b, const T r) {
        const GLSL::Vector3<T> pa(p - a);
        const GLSL::Vector3<T> ba(b - a);
        const T dot{ GLSL::dot(ba)};
        assert(!Numerics::areEquals(dot, T{}));
        const T h{ Numerics::clamp<T{}, static_cast<T>(1)>(GLSL::dot(pa, ba) / dot)};
        return GLSL::length(pa - ba * h) - r;
    }

    /**
    * \brief return the signed distance of capped cylinder
    * @param {Vector3,        in}  point
    * @param {Vector3,        in}  capped cylinder point #1
    * @param {Vector3,        in}  capped cylinder point #2
    * @param {floating_point, in}  capped cylinder radius
    * @param {floating_point, out} signed distance
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr T sdf_to_capped_cylinder(const GLSL::Vector3<T>& p, const GLSL::Vector3<T>& a, const GLSL::Vector3<T>& b, const T r) {
        const GLSL::Vector3<T> ba(b - a);
        const GLSL::Vector3<T> pa(p - a);
        const T baba{ GLSL::dot(ba, ba) };
        assert(!Numerics::areEquals(baba, T{}));
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
    * @param {Vector3,        in}  point
    * @param {Vector3,        in}  ellipsoid radii
    * @param {floating_point, out} signed distance
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr T sdf_to_bound_ellipsoied(const GLSL::Vector3<T>& p, const GLSL::Vector3<T>& r) {
        assert(!Numerics::areEquals(r.x, T{}));
        assert(!Numerics::areEquals(r.y, T{}));
        assert(!Numerics::areEquals(r.z, T{}));

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
    * @param {Vector3,        in}  point
    * @param {Vector3,        in}  vertex #0
    * @param {Vector3,        in}  vertex #1
    * @param {Vector3,        in}  vertex #2
    * @param {floating_point, out} signed distance
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto udf_to_triangle(const GLSL::Vector3<T>& p, const GLSL::Vector3<T>& a, const GLSL::Vector3<T>& b, const GLSL::Vector3<T>& c) {
        constexpr T one{ static_cast<T>(1) };
        const GLSL::Vector3<T> ba(b - a);
        const GLSL::Vector3<T> pa(p - a);
        const GLSL::Vector3<T> cb(c - b);
        const GLSL::Vector3<T> pb(p - b);
        const GLSL::Vector3<T> ac(a - c);
        const GLSL::Vector3<T> pc(p - c);
        const GLSL::Vector3<T> nor(cross(ba, ac));

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
    * @param {Vector3,        in}  point
    * @param {Vector3,        in}  vertex #0
    * @param {Vector3,        in}  vertex #1
    * @param {Vector3,        in}  vertex #2
    * @param {Vector3,        in}  vertex #3
    * @param {floating_point, out} signed distance
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto udf_to_quad(const GLSL::Vector3<T>& p, const GLSL::Vector3<T>& a, const GLSL::Vector3<T>& b, const GLSL::Vector3<T>& c, const GLSL::Vector3<T>& d) {
        constexpr T one{ static_cast<T>(1) };
        const GLSL::Vector3<T> ba(b - a);
        const GLSL::Vector3<T> pa(p - a);
        const GLSL::Vector3<T> cb(c - b);
        const GLSL::Vector3<T> pb(p - b);
        const GLSL::Vector3<T> dc(d - c);
        const GLSL::Vector3<T> pc(p - c);
        const GLSL::Vector3<T> ad(a - d);
        const GLSL::Vector3<T> pd(p - d);
        const GLSL::Vector3<T> nor(GLSL::cross(ba, ad));

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
