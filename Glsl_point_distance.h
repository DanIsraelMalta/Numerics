#pragma once
#include "Glsl.h"

//
// euclidean unsigned/signed distance of a point from a primitive.
// All primitives are centered at the origin
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
        return GLSL::distance(p, GLSL::mix(a, b, GLSL::clamp(k, T{}, static_cast<T>(1))));
    }

    //
    // 2D udf/sdf
    // (see https://iquilezles.org/articles/distfunctions2d/)
    //
    namespace TwoD {

        /**
        * \brief return the signed distance of a point from circle at center
        * @param {Vector2,        in}  point
        * @param {floating_point, in}  circle radius
        * @param {floating_point, out} signed distance
        **/
        template<typename T>
            requires(std::is_floating_point_v<T>)
        constexpr T circle_sdf(const GLSL::Vector2<T>& p, const T r) noexcept {
            return (GLSL::length(p) - r);
        }

        /**
        * \brief return the signed distance of a point from rectangle at center
        * @param {Vector2,        in}  point
        * @param {Vector2,        in}  rectangle extents
        * @param {floating_point, out} signed distance
        **/
        template<typename T>
            requires(std::is_floating_point_v<T>)
        constexpr T rectangle_sdf(const GLSL::Vector2<T>& p, const GLSL::Vector2<T>& b) noexcept {
            const GLSL::Vector2<T> d{ GLSL::abs(p) - b };
            return GLSL::length(Numerics::max(d.x, d.y, T{})) + Numerics::min(GLSL::max(d), T{});
        }

        /**
        * \brief return the signed distance of a point from segment
        * @param {Vector2,        in}  point
        * @param {Vector2,        in}  segment point #1
        * @param {Vector2,        in}  segment point #2
        * @param {floating_point, out} signed distance
        **/
        template<typename T>
            requires(std::is_floating_point_v<T>)
        constexpr T segment_sdf(const GLSL::Vector2<T>& p, const GLSL::Vector2<T>& a, const GLSL::Vector2<T>& b) {
            const GLSL::Vector2<T> pa{ p - a };
            const GLSL::Vector2<T> ba{ b - a };
            assert(!Numerics::areEquals(GLSL::dot(ba), T{}));
            const T h{ Numerics::min(Numerics::max(GLSL::dot(pa, ba) / GLSL::dot(ba), T{}), static_cast<T>(1)) };
            return GLSL::length(pa - ba * h);
        }

        /**
        * \brief return the signed distance of a point from equilateral triangle centered at center
        * @param {Vector2,        in}  point
        * @param {floating_point, in}  equilateral triangle ...
        * @param {floating_point, out} signed distance
        **/
        template<typename T>
            requires(std::is_floating_point_v<T>)
        constexpr T sdf_equilateral_triangle(GLSL::Vector2<T> p, const T r) noexcept {
            constexpr T k{ std::sqrt(static_cast<T>(3)) };
            p.x = std::abs(p.x) - r;
            p.y += r / k;

            if (p.x + k * p.y > T{}) {
                p = GLSL::Vector2<T>((p.x - k * p.y) / static_cast<T>(2), (-k * p.x - p.y) / static_cast<T>(2));
            }

            p.x -= GLSL::clamp(p.x, static_cast<T>(-2), T{});
            return std::copysign(-GLSL::length(p), p.y);
        }

        /**
        * \brief return the signed distance of a point from isosceles triangle centered at center
        * @param {Vector2,        in}  point
        * @param {Vector2,        in}  equilateral triangle {width, height}
        * @param {floating_point, out} signed distance
        **/
        template<typename T>
            requires(std::is_floating_point_v<T>)
        constexpr T sdf_isosceles_triangle(GLSL::Vector2<T> p, const GLSL::Vector2<T>& q) {
            constexpr T one{ static_cast<T>(1) };
            assert(!Numerics::areEquals(GLSL::dot(q), T{}));
            assert(!Numerics::areEquals(q.x, T{}));

            p.x = std::abs(p.x);
            const GLSL::Vector2<T> a{ p - q * GLSL::clamp(GLSL::dot(p, q) / GLSL::dot(q), T{}, one) };
            const GLSL::Vector2<T> b{ p - q * GLSL::Vector2<T>(GLSL::clamp(p.x / q.x, T{}, one), one) };
            const T s{ Numerics::sign(-q.y) };
            const GLSL::Vector2<T> d(Numerics::min(GLSL::dot(a), s * (p.x * q.y - p.y * q.x)),
                                     Numerics::min(GLSL::dot(b), s * (p.y - q.y)));
            return -std::sqrt(d.x) * Numerics::sign(d.y);
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
        constexpr auto sdf_triangle(const GLSL::Vector2<T>& p, const GLSL::Vector2<T>& p0, const GLSL::Vector2<T>& p1, const GLSL::Vector2<T>& p2) {
            constexpr T one{ static_cast<T>(1) };
            const GLSL::Vector2<T> e0{ p1 - p0 };
            const GLSL::Vector2<T> e1{ p2 - p1 };
            const GLSL::Vector2<T> e2{ p0 - p2 };
            assert(!Numerics::areEquals(GLSL::dot(e0), T{}));
            assert(!Numerics::areEquals(GLSL::dot(e1), T{}));
            assert(!Numerics::areEquals(GLSL::dot(e2), T{}));

            const GLSL::Vector2<T> v0{ p - p0 };
            const GLSL::Vector2<T> v1{ p - p1 };
            const GLSL::Vector2<T> v2{ p - p2 };
            const GLSL::Vector2<T> pq0{ v0 - e0 * GLSL::clamp(GLSL::dot(v0, e0) / GLSL::dot(e0), T{}, one) };
            const GLSL::Vector2<T> pq1{ v1 - e1 * GLSL::clamp(GLSL::dot(v1, e1) / GLSL::dot(e1), T{}, one) };
            const GLSL::Vector2<T> pq2{ v2 - e2 * GLSL::clamp(GLSL::dot(v2, e2) / GLSL::dot(e2), T{}, one) };

            const T s{ Numerics::sign(GLSL::cross(e0, e2)) };
            const GLSL::Vector2<T> d{ GLSL::min(GLSL::min(GLSL::Vector2<T>(GLSL::dot(pq0), s * GLSL::cross(v0, e0)),
                                                          GLSL::Vector2<T>(GLSL::dot(pq1), s * GLSL::cross(v1, e1))),
                                                          GLSL::Vector2<T>(GLSL::dot(pq2), s * GLSL::cross(v2, e2))) };
            return -std::sqrt(d.x) * Numerics::sign(d.y);
        }

        /**
        * \brief return the signed distance of pie/sector
        * @param {Vector2,        in}  point
        * @param {Vector2,        in}  pie/sector {sine of aperture, cosine of aperture}
        * @param {floating_point, in}  pie/sector radius
        * @param {floating_point, out} signed distance
        **/
        template<typename T>
            requires(std::is_floating_point_v<T>)
        constexpr T sdf_sector(GLSL::Vector2<T> p, const GLSL::Vector2<T>& c, const T r) noexcept {
            p.x = std::abs(p.x);
            const T l{ GLSL::length(p) - r };
            const T m{ GLSL::length(p - c * GLSL::clamp(GLSL::dot(p, c), T{}, r)) };
            return Numerics::max(l, std::copysign(m, c.y * p.x - c.x * p.y));
        }

        /**
        * \brief return the signed distance of arc
        * @param {Vector2,        in}  point
        * @param {Vector2,        in}  arc {sine of aperture, cosine of aperture}
        * @param {floating_point, in}  arc center radius
        * @param {floating_point, in}  arc width around radius
        * @param {floating_point, out} signed distance
        **/
        template<typename T>
            requires(std::is_floating_point_v<T>)
        constexpr T sdf_arc(GLSL::Vector2<T> p, const GLSL::Vector2<T>& sc, const T ra, const T rb) noexcept {
            p.x = std::abs(p.x);
            return ((sc.y * p.x > sc.x * p.y) ? GLSL::length(p - sc * ra) : std::abs(GLSL::length(p) - ra)) - rb;
        }

        /**
        * \brief return the signed distance of closed polygon
        * @param {array<Vector2>, in}  polygon points
        * @param {Vector2,        in}  point
        * @param {floating_point, out} signed distance
        **/
        template<std::size_t N, typename T>
            requires(std::is_floating_point_v<T>)
        constexpr T sdf_polygon(const std::array<GLSL::Vector2<T>, N>& v, const GLSL::Vector2<T>& p) {
            T d{ GLSL::dot(p - v[0]) };
            const T s{ static_cast<T>(1) };

            for (std::size_t i{}, j{ N - 1 }; i < N; j = i, i++) {
                const GLSL::Vector2<T> e{ v[j] - v[i] };
                assert(!Numerics::areEquals(GLSL::dot(e), T{}));
                const GLSL::Vector2<T> w{ p - v[i] };

                const GLSL::Vector2<T> b{ w - e * GLSL::clamp(dot(w, e) / GLSL::dot(e), T{}, static_cast<T>(1)) };
                d = Numerics::min(d, dot(b));
                const GLSL::Vector3<bool> c(p.y >= v[i].y,
                                            p.y < v[j].y,
                                            e.x * w.y > e.y * w.x);
                if (GLSL::all(c) || GLSL::all(GLSL::glsl_not(c))) {
                    s *= static_cast<T>(-1);
                }
            }

            [[assume(d > T{})]];
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
            requires(std::is_integral_v<U> && std::is_floating_point_v<T>)
        constexpr T sdf_to_regular_poygon(GLSL::Vector2<T> p, const T r, const U n) {
            assert(n > 0);
            // these 4 lines can be precomputed for a given shape
            const T an{ std::numbers::pi_v<T> / static_cast<T>(n) };
            const GLSL::Vector2<T> cs(std::cos(an), std::sin(an));

            // reduce to first sector
            const T bn = [&]() {
                if constexpr (std::is_floating_point_v<T>) {
                    return std::fmod(std::atan(p.x, p.y), static_cast<T>(2) * an) - an;
                }
                else {
                    return (std::atan(p.x, p.y) % static_cast<T>(2) * an) - an;
                }
            }();
            p = GLSL::length(p) * GLSL::Vector2<T>(std::cos(bn), std::abs(std::sin(bn)));

            // line sdf
            p -= r * cs;
            p.y += GLSL::clamp(-p.y, T{}, r * cs.y);
            return Numerics::sign(p.x) * GLSL::length(p);
        }

        template<std::size_t n, typename T>
            requires((n > 0)  && std::is_floating_point_v<T>)
        constexpr T sdf_to_regular_poygon(GLSL::Vector2<T> p, const T r) noexcept {
            assert(n > 0);
            // these 4 lines can be precomputed for a given shape
            constexpr T an{ std::numbers::pi_v<T> / static_cast<T>(n) };
            constexpr GLSL::Vector2<T> cs(std::cos(an), std::sin(an));

            // reduce to first sector
            const T bn = [&]() {
                if constexpr (std::is_floating_point_v<T>) {
                    return std::fmod(std::atan(p.x, p.y), static_cast<T>(2) * an) - an;
                }
                else {
                    return (std::atan(p.x, p.y) % static_cast<T>(2) * an) - an;
                }
            }();
            p = GLSL::length(p) * GLSL::Vector2<T>(std::cos(bn), std::abs(std::sin(bn)));

            // line sdf
            p -= r * cs;
            p.y += GLSL::clamp(-p.y, T{}, r * cs.y);
            return Numerics::sign(p.x) * GLSL::length(p);
        }

        /**
        * \brief return the signed distance of ellipse located at center
        * @param {Vector2,        in}  point
        * @param {Vector2,        in}  {ellipse radii alon x, ellipse radii alon y}
        * @param {floating_point, out} signed distance
        **/
        template<typename T>
            requires(std::is_floating_point_v<T>)
        constexpr T sdf_ellipse(GLSL::Vector2<T> p, GLSL::Vector2<T> ab) {
            constexpr T third{ static_cast<T>(1 / 3) };
            constexpr T one{ static_cast<T>(1) };
            constexpr T two{ static_cast<T>(2) };
            constexpr T three{ static_cast<T>(3) };
            constexpr T threeSqrt{ std::sqrt(static_cast<T>(3)) };
            constexpr T four{ static_cast<T>(4) };

            p = std::abs(p);
            if (p.x > p.y) {
                p = p.yx;
                ab = ab.yx;
            }

            const T co = [&ab, &p]() {
                const T l{ ab.y * ab.y - ab.x * ab.x };
                assert(!Numerics::areEquals(l, T{}));
                const T m{ ab.x * p.x / l };
                const T m2{ m * m };
                const T n{ ab.y * p.y / l };
                const T n2{ n * n };
                const T c{ (m2 + n2 - one) / three };
                const T c3{ c * c * c };
                const T q{ c3 + two * m2 * n2 };
                const T d{ c3 + m2 * n2 };
                const T g{ m + m * n2 };

                if (d < T{}) {
                    const T h{ std::acos(q / c3) / three };
                    const T s{ std::cos(h) };
                    const T t{ std::sin(h) * threeSqrt };
                    assert(-c * (s + t + two) + m2 > T{});
                    assert(-c * (s - t + two) + m2 > T{});
                    const T rx{ std::sqrt(-c * (s + t + two) + m2) };
                    const T ry{ std::sqrt(-c * (s - t + two) + m2) };

                    assert(!Numerics::areEquals(rx * ry - m, T{}));
                    return (ry + std::copysign(rx, l) + std::abs(g) / (rx * ry) - m) / two;
                }
                else {
                    const T h{ two * m * n * std::sqrt(d) };
                    const T s{ std::copysign(std::pow(std::abs(q + h), third), q + h) };
                    const T u{ std::copysign(std::pow(std::abs(q - h), third), q - h) };
                    const T rx{ -s - u - four * c + two * m2 };
                    const T ry{ (s - u) * threeSqrt };
                    assert(rx * rx + ry * ry > T{});
                    const T rm{ std::sqrt(rx * rx + ry * ry) };

                    assert(!Numerics::areEquals(rm, T{}));
                    assert(rm - rx > T{});
                    return ((ry / std::sqrt(rm - rx) + two * g / rm - m) / two);
                }
            }();

            assert(one - co * co > T{});
            const GLSL::Vector2<T> r{ ab * GLSL::Vector2<T>(co, std::sqrt(one - co * co)) };
            return GLSL::length(r - p) * Numerics::sign(p.y - r.y);
        }
    }

    //
    // 3D udf/sdf
    // (see https://iquilezles.org/articles/distfunctions/)
    //
    namespace ThreeD {

        /**
        * \brief return the signed distance of sphere located at center
        * @param {Vector3,        in}  point
        * @param {floating_point, in}  sphere radius
        * @param {floating_point, out} signed distance
        **/
        template<typename T>
            requires(std::is_floating_point_v<T>)
        constexpr T sdf_sphere(const GLSL::Vector3<T>& p, const T s) noexcept{
            return (GLSL::length(p) - s);
        }

        /**
        * \brief return the signed distance of box located at center
        * @param {Vector3,        in}  point
        * @param {Vector3,        in}  box extents
        * @param {floating_point, out} signed distance
        **/
        template<typename T>
            requires(std::is_floating_point_v<T>)
        constexpr T sdf_box(const GLSL::Vector3<T>& p, const GLSL::Vector3<T>& b) noexcept {
            const GLSL::Vector3<T> q{ GLSL::abs(p) - b };
            return (GLSL::length(Numerics::max(q.x, q.y, q.z, T{})) + Numerics::min(max(q), T{}));
        }

        /**
        * \brief return the signed distance of torus located at center
        * @param {Vector3,        in}  point
        * @param {Vector2,        in}  {torus outside radius, torus width from outside radius inwards}
        * @param {floating_point, out} signed distance
        **/
        template<typename T>
            requires(std::is_floating_point_v<T>)
        constexpr T sdf_to_torus(const GLSL::Vector3<T>& p, const GLSL::Vector2<T>& t) noexcept {
            const GLSL::Vector2<T> q(GLSL::length(p.xz) - t.x, p.y);
            return (GLSL::length(q) - t.y);
        }

        /**
        * \brief return the signed distance of cone located at center
        * @param {Vector3,        in}  point
        * @param {Vector2,        in}  {sine fo cone angle, cosine of cone angle}
        * @param {floating_point, in}  cone height
        * @param {floating_point, out} signed distance
        **/
        template<typename T>
            requires(std::is_floating_point_v<T>)
        constexpr T sdf_to_cone(const GLSL::Vector3<T>& p, const GLSL::Vector2<T>& c, const T h) {
            constexpr T one{ static_cast<T>(1) };
            assert(!Numerics::areEquals(c.y, T{}));
            const GLSL::Vector2<T> q(h * GLSL::Vector2<T>(c.x / c.y, static_cast<T>(-1.0)));
            assert(!Numerics::areEquals(GLSL::dot(q), T{}) && !Numerics::areEquals(q.x, T{}));

            const GLSL::Vector2<T> w(GLSL::length(p.xz), p.y);
            const GLSL::Vector2<T> a(w - q * GLSL::clamp(GLSL::dot(w, q) / GLSL::dot(q), T{}, one));
            const GLSL::Vector2<T> b(w - q * GLSL::Vector2<T>(GLSL::clamp(w.x / q.x, T{}, one), one));
            const T k{ Numerics::sign(q.y) };
            const T d{ Numerics::min(GLSL::dot(a), GLSL::dot(b)) };
            const T s{ Numerics::max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y)) };

            [[assume(d > T{})]];
            return Numerics::sign(s) * std::sqrt(d);
        }

        /**
        * \brief return the signed distance of bounded cone located at center (approximate calculation)
        * @param {Vector3,        in}  point
        * @param {Vector2,        in}  {sine fo cone angle, cosine of cone angle}
        * @param {floating_point, in}  cone height
        * @param {floating_point, out} signed distance
        **/
        template<typename T>
            requires(std::is_floating_point_v<T>)
        constexpr T sdf_to_bounded_cone(const GLSL::Vector3<T>& p, const GLSL::Vector2<T>& c, const T h) noexcept {
            const T q{ GLSL::length(p.xz) };
            return Numerics::max(GLSL::dot(c.xy, GLSL::Vector2<T>(q, p.y)), -h - p.y);
        }

        /**
        * \brief return the signed distance of bounded plane
        * @param {Vector3,        in}  point
        * @param {Vector2,        in}  plane normal (must be normalized)
        * @param {floating_point, in}  plane distance
        * @param {floating_point, out} signed distance
        **/
        template<typename T>
            requires(std::is_floating_point_v<T>)
        constexpr T sdf_to_plane(const GLSL::Vector3<T>& p, const GLSL::Vector3<T>& n, const T h) noexcept {
            assert(Numerics::areEquals(GLSL::length(n), static_cast<T>(1)));
            return GLSL::dot(p, n) + h;
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
            assert(!Numerics::areEquals(GLSL::dot(ba), T{}));
            const T h{ Numerics::clamp(GLSL::dot(pa, ba) / GLSL::dot(ba), T{}, static_cast<T>(1)) };
            return GLSL::length(pa - ba * h) - r;
        }

        /**
        * \brief return the signed distance of vertical capsule
        * @param {Vector3,        in}  point
        * @param {floating_point, in}  capsule height
        * @param {floating_point, in}  capsule radius
        * @param {floating_point, out} signed distance
        **/
        template<typename T>
            requires(std::is_floating_point_v<T>)
        constexpr T sdf_to_vertical_capsule(GLSL::Vector3<T>& p, const T h, const T r) noexcept {
            p.y -= GLSL::clamp(p.y, T{}, h);
            return GLSL::length(p) - r;
        }

        /**
        * \brief return the signed distance of vertical capped cylinder
        * @param {Vector3,        in}  point
        * @param {floating_point, in}  capped cylinder height
        * @param {floating_point, in}  capped cylinder radius
        * @param {floating_point, out} signed distance
        **/
        template<typename T>
            requires(std::is_floating_point_v<T>)
        constexpr T sdf_to_vertical_capped_cylinder(const GLSL::Vector3<T>& p, const T h, const T r) noexcept {
            const GLSL::Vector2<T> d(GLSL::abs(GLSL::Vector2<T>(GLSL::length(p.xz), p.y)) - GLSL::Vector2<T>(r, h));
            return (Numerics::min(Numerics::max(d.x, d.y), T{}) + GLSL::length(Numerics::max(d, T{})));
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
        * \brief return the signed distance of bound ellipsoid located at center (not exact)
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
            assert(!Numerics::areEquals(k1, T{}));
            return (k0 * (k0 - static_cast<T>(1)) / k1);
        }

        /**
        * \brief return the unsigned distance of triangle
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

            return std::sqrt(
                (Numerics::sign(GLSL::dot(GLSL::cross(ba, nor), pa)) +
                    Numerics::sign(GLSL::dot(GLSL::cross(cb, nor), pb)) +
                    Numerics::sign(GLSL::dot(GLSL::cross(ac, nor), pc)) < static_cast<T>(2.0))
                ?
                Numerics::min(
                    GLSL::dot(ba * GLSL::clamp(GLSL::dot(ba, pa) / GLSL::dot(ba), T{}, one) - pa),
                    GLSL::dot(cb * GLSL::clamp(GLSL::dot(cb, pb) / GLSL::dot(cb), T{}, one) - pb),
                    GLSL::dot(ac * GLSL::clamp(GLSL::dot(ac, pc) / GLSL::dot(ac), T{}, one) - pc))
                :
                GLSL::dot(nor, pa) * GLSL::dot(nor, pa) / GLSL::dot(nor));
        }

        /**
        * \brief return the unsigned distance of quad
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

            return std::sqrt(
                (Numerics::sign(GLSL::dot(GLSL::cross(ba, nor), pa)) +
                    Numerics::sign(GLSL::dot(GLSL::cross(cb, nor), pb)) +
                    Numerics::sign(GLSL::dot(GLSL::cross(dc, nor), pc)) +
                    Numerics::sign(GLSL::dot(cross(ad, nor), pd)) < static_cast<T>(3.0))
                ?
                Numerics::min(
                    GLSL::dot(ba * GLSL::clamp(GLSL::dot(ba, pa) / GLSL::dot(ba), T{}, one) - pa),
                    GLSL::dot(cb * GLSL::clamp(GLSL::dot(cb, pb) / GLSL::dot(cb), T{}, one) - pb),
                    GLSL::dot(dc * GLSL::clamp(GLSL::dot(dc, pc) / GLSL::dot(dc), T{}, one) - pc),
                    GLSL::dot(ad * GLSL::clamp(GLSL::dot(ad, pd) / GLSL::dot(ad), T{}, one) - pd))
                :
                GLSL::dot(nor, pa) * GLSL::dot(nor, pa) / GLSL::dot(nor));
        }
    }
}
