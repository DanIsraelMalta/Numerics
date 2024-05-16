#pragma once
#include "Glsl.h"

//
// ray-primitive intersection functions
// (see https://www.shadertoy.com/playlist/l3dXRf)
//

namespace RayIntersections {

    /**
    * \brief return the intersection between ray and plane
    * @param {Vector3,        in}  ray origin
    * @param {Vector3,        in}  ray direction (should be normalized; not tested for normalization)
    * @param {floating_point, in}  plane {x, y, z, d} - p.xyz must be normalized
    * @param {floating_point, out} distance to closest intersection point, along ray, from ray origin
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto plane_intersect(const GLSL::Vector3<T>& ro, const GLSL::Vector3<T>& rd, const GLSL::Vector4<T>& p) {
        assert(Numerics::areEquals(GLSL::length(rd), static_cast<T>(1)));
        assert(Numerics::areEquals(GLSL::length(p.xyz), static_cast<T>(1)));

        return -(GLSL::dot(ro, p.xyz) + p.w) / GLSL::dot(rd, p.xyz);
    }

    /**
    * \brief return the intersection between ray and disk
    * @param {Vector3,        in}  ray origin
    * @param {Vector3,        in}  ray direction (should be normalized; not tested for normalization)
    * @param {Vector3,        in}  disk center
    * @param {Vector3,        in}  disk normal (should be normalized
    * @param {floating_point, in}  disk radius
    * @param {floating_point, out} distance to closest intersection point, along ray, from ray origin
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto disk_intersect(const GLSL::Vector3<T>& ro, const GLSL::Vector3<T>& rd, const GLSL::Vector3<T>& c, const GLSL::Vector3<T>& n, T r) {
        assert(Numerics::areEquals(GLSL::length(rd), static_cast<T>(1)));
        assert(Numerics::areEquals(GLSL::length(n), static_cast<T>(1)));

        const GLSL::Vector3<T> o{ ro - c };
        const T t{ -GLSL::dot(n, o) / GLSL::dot(rd, n) };
        const GLSL::Vector3<T> q{ o + rd * t };
        return (GLSL::dot(q) < r * r) ? t : static_cast<T>(-1);
    }

    /**
    * \brief return the intersection between ray and triangle
    * @param {Vector3,        in}  ray origin
    * @param {Vector3,        in}  ray direction (should be normalized; not tested for normalization)
    * @param {Vector3,        in}  triangle vertex #0
    * @param {Vector3,        in}  triangle vertex #1
    * @param {Vector3,        in}  triangle vertex #2
    * @param {Vector3, out} intersection point
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto triangle_intersect(const GLSL::Vector3<T>& ro, const GLSL::Vector3<T>& rd, const GLSL::Vector3<T>& v0, const GLSL::Vector3<T>& v1, const GLSL::Vector3<T>& v2) {
        assert(Numerics::areEquals(GLSL::length(rd), static_cast<T>(1)));

        const GLSL::Vector3<T> v1v0{ v1 - v0 };
        const GLSL::Vector3<T> v2v0{ v2 - v0 };
        const GLSL::Vector3<T> rov0{ ro - v0 };
        const GLSL::Vector3<T> n{ GLSL::cross(v1v0, v2v0) };
        const GLSL::Vector3<T> q{ GLSL::cross(rov0, rd) };
        const T d{ static_cast<T>(1.0) / GLSL::dot(rd, n) };
        const T u{ d * GLSL::dot(-q, v2v0) };
        const T v{ d * GLSL::dot(q, v1v0) };
        const T t = [&n, &rov0, u, v, d]() {
            if ((u < T{}) ||
                (v < T{}) ||
                (u + v > static_cast<T>(1.0))) {
                return static_cast<T>(-1.0);
            }
            else {
                return d * GLSL::dot(-n, rov0);
            }
        }();

        return GLSL::Vector3<T>(t, u, v);
    }

    /**
    * \brief return the intersection between ray and ellipse
    * @param {Vector3, in}  ray origin
    * @param {Vector3, in}  ray direction (should be normalized; not tested for normalization)
    * @param {Vector3, in}  ellipse center
    * @param {Vector3, in}  ellipse radii #1 ('u')
    * @param {Vector3, in}  ellipse radii #1 ('v')
    * @param {Vector3, out} intersection point
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto ellipse_intersect(const GLSL::Vector3<T>& ro, const GLSL::Vector3<T>& rd, const GLSL::Vector3<T>& c, const GLSL::Vector3<T>& u, const GLSL::Vector3<T>& v) {
        assert(Numerics::areEquals(GLSL::length(rd), static_cast<T>(1)));

        const GLSL::Vector3<T> q{ ro - c };
        const GLSL::Vector3<T> n{ GLSL::cross(u, v) };
        const T t{ -GLSL::dot(n, q) / GLSL::dot(rd, n) };
        const T qrdt{ q + rd * t };
        const T r{ GLSL::dot(u, qrdt) };
        const T s{ GLSL::dot(v, qrdt) };

        if (r * r + s * s > static_cast<T>(1)) {
            return GLSL::Vector3<T>(static_cast<T>(-1));
        }
        return GLSL::Vector3<T>(t, s, r);
    }

    /**
    * \brief return the intersection between ray and sphere
    * @param {Vector3, in}  ray origin
    * @param {Vector3, in}  ray direction (should be normalized; not tested for normalization)
    * @param {Vector4, in}  sphere {center x, center y, cernter z, radius}
    * @param {Vector2, out} {distance to closest intersection point, along ray, from ray origin,
    *                        distance to furthest intersection point, along ray, from ray origin}
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto sphere_intersect(const GLSL::Vector3<T>& ro, const GLSL::Vector3<T>& rd, const GLSL::Vector4<T>& sph) {
        assert(Numerics::areEquals(GLSL::length(rd), static_cast<T>(1)));

        const GLSL::Vector3<T> oc = ro - sph.xyz;
        const T b{ GLSL::dot(oc, rd) };

        const GLSL::Vector3<T> qc{ oc - b * rd };
        T h{ sph.w * sph.w - GLSL::dot(qc) };
        if (h < T{}) {
            return GLSL::Vector2<T>(static_cast<T>(-1));
        }

        [[assume(h > T{})]];
        h = std::sqrt(h);
        return GLSL::Vector2<T>(-b - h, -b + h);
    }

    /**
    * \brief return the intersection between ray and rotated box
    * @param {Vector3, in}  ray origin
    * @param {Vector3, in}  ray direction (should be normalized; not tested for normalization)
    * @param {Matrix4, in}  world to box transformation matrix
    * @param {Vector3, in}  box half length
    * @param {Vector2, out} {distance to closest intersection point, along ray, from ray origin,
    *                        distance to furthest intersection point, along ray, from ray origin}
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto box_intersect(const GLSL::Vector3<T>& row, const GLSL::Vector3<T>& rdw, const GLSL::Matrix4<T>& txx, const GLSL::Vector3<T>& rad) {
        constexpr T one{ static_cast<T>(1) };
        assert(Numerics::areEquals(GLSL::length(rdw), static_cast<T>(1)));

        // convert from world to box space
        const GLSL::Vector3<T> rd(txx * rdw);
        const GLSL::Vector3<T> ro(txx * row);
        assert(!Numerics::areEquals(rd.x, T{}) && !Numerics::areEquals(rd.y, T{}) && !Numerics::areEquals(rd.z, T{}));

        // ray-box intersection in box space
        const GLSL::Vector3<T> s(Numerics::sign(-rd.x),
                                 Numerics::sign(-rd.y),
                                 Numerics::sign(-rd.z));
        const GLSL::Vector3<T> srd{ s * rad };
        const GLSL::Vector3<T> t1((-ro + srd) / rd);
        const GLSL::Vector3<T> t2((-ro - srd) / rd);

        return GLSL::Vector2<T>(GLSL::max(t1), GLSL::min(t2));
    }

    /**
    * \brief return the intersection between ray and capsule
    * @param {Vector3,        in}  ray origin
    * @param {Vector3,        in}  ray direction (should be normalized; not tested for normalization)
    * @param {Vector3,        in}  capsule extreme #1
    * @param {Vector3,        in}  capsule extreme #2
    * @param {floating_point, in}  capsule radius
    * @param {floating_point, out} distance to closest intersection point, along ray, from ray origin
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto capsule_intersect(const GLSL::Vector3<T>& ro, const GLSL::Vector3<T>& rd, const GLSL::Vector3<T>& pa, const GLSL::Vector3<T>& pb, const T ra) {
        assert(Numerics::areEquals(GLSL::length(rd), static_cast<T>(1)));

        const GLSL::Vector3<T> ba{ pb - pa };
        const GLSL::Vector3<T> oa{ ro - pa };

        const T baba{ GLSL::dot(ba) };
        const T bard{ GLSL::dot(ba, rd) };
        const T baoa{ GLSL::dot(ba, oa) };
        const T rdoa{ GLSL::dot(rd, oa) };
        const T oaoa{ GLSL::dot(oa) };
        const T a{ baba - bard * bard };
        T b{ baba * rdoa - baoa * bard };
        T c{ baba * oaoa - baoa * baoa - ra * ra * baba };

        if (T h{ b * b - a * c }; h >= T{}) {
            assert(!Numerics::areEquals(a, T{}));
            [[assume(h >= T{})]];
            const T t{ (-b - std::sqrt(h)) / a };
            const T y{ baoa + t * bard };

            // body
            if ((y > T{}) && (y < baba)) {
                return t;
            }

            // caps
            const GLSL::Vector3<T> oc{ (y <= T{}) ? oa : ro - pb };
            b = GLSL::dot(rd, oc);
            c = GLSL::dot(oc) - ra * ra;
            h = b * b - c;

            if (h > T{}) {
                [[assume(h >= T{})]];
                return (-b - std::sqrt(h));
            }
        }

        return static_cast<T>(-1.0);
    }

    /**
    * \brief return the intersection between ray and capped cylinder
    * @param {Vector3,        in}  ray origin
    * @param {Vector3,        in}  ray direction (should be normalized; not tested for normalization)
    * @param {Vector3,        in}  capped cylinder extreme #1
    * @param {Vector3,        in}  capped cylinder extreme #2
    * @param {floating_point, in}  capped cylinder radius
    * @param {vector4,        out} {distance to closest intersection point, along ray, from ray origin,
    *                               intersection point x, intersction point y, intersection point z}
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto capped_cylinder_intersect(const GLSL::Vector3<T>& ro, const GLSL::Vector3<T>& rd, const GLSL::Vector3<T>& a, const GLSL::Vector3<T>& b, const T ra) {
        assert(Numerics::areEquals(GLSL::length(rd), static_cast<T>(1)));

        const GLSL::Vector3<T> ba{ b - a };
        const GLSL::Vector3<T> oc{ ro - a };

        const T baba{ GLSL::dot(ba) };
        const T bard{ GLSL::dot(ba, rd) };
        const T baoc{ GLSL::dot(ba, oc) };
        const T k2{ baba - bard * bard };
        const T k1{ baba * GLSL::dot(oc, rd) - baoc * bard };
        const T k0{ baba * GLSL::dot(oc) - baoc * baoc - ra * ra * baba };
        T h{ k1 * k1 - k2 * k0 };

        if (h < T{}) {
            return GLSL::Vector4<T>(static_cast<T>(-1.0));
        }

        assert(!Numerics::areEquals(k2, T{}));
        [[assume(h >= T{})]];
        h = std::sqrt(h);
        T t{ (-k1 - h) / k2 };
        const T y{ baoc + t * bard };

        // body
        if ((y > T{}) && (y < baba)) {
            assert(!Numerics::areEquals(baba, T{}));
            assert(!Numerics::areEquals(ra, T{}));
            const GLSL::Vector3<T> yzw{ (oc + t * rd - ba * y / baba) / ra };
            return GLSL::Vector4<T>(t, yzw);
        }

        // caps
        assert(!Numerics::areEquals(bard, T{}));
        t = (((y < T{}) ? T{} : baba) - baoc) / bard;
        if (std::abs(k1 + k2 * t) < h) {
            assert(baba >= T{});
            const GLSL::Vector3<T> yzw{ ba * Numerics::sign(y) / std::sqrt(baba) };
            return GLSL::Vector4<T>(t, yzw);
        }

        return GLSL::Vector4<T>(static_cast<T>(-1.0));
    }

    /**
    * \brief return the intersection between ray and infinite cylinder
    * @param {Vector3,        in}  ray origin
    * @param {Vector3,        in}  ray direction (should be normalized; not tested for normalization)
    * @param {Vector3,        in}  infinite cylinder base point #1
    * @param {Vector3,        in}  infinite cylinder direction/axis (should be normalized)
    * @param {floating_point, in}  infinite cylinder radius
    * @param {Vector2,        out} {distance to closest intersection point, along ray, from ray origin,
    *                               distance to furthest intersection point, along ray, from ray origin}
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto infinte_cylinder_intersection(const GLSL::Vector3<T>& ro, const GLSL::Vector3<T>& rd, const GLSL::Vector3<T>& cb, const GLSL::Vector3<T>& ca, const T cr) {
        assert(Numerics::areEquals(GLSL::length(rd), static_cast<T>(1)));
        assert(Numerics::areEquals(GLSL::length(ca), static_cast<T>(1)));

        const Vector3<T> oc{ ro - cb };

        const T card{ GLSL::dot(ca, rd) };
        const T caoc{ GLSL::dot(ca, oc) };
        const T a{ static_cast<T>(-1.0) - card * card };
        const T b{ GLSL::dot(oc, rd) - caoc * card };
        const T c{ GLSL::dot(oc) - caoc * caoc - cr * cr };

        T h{ b * b - a * c };
        if (h < T{}) {
            return GLSL::Vector2<T>(static_cast<T>(-1.0));
        }

        assert(!Numerics::areEquals(a, T{}));
        [[assume(h >= T{})]];
        h = std::sqrt(h);
        return GLSL::Vector2<T>((-b - h) / a, (-b + h) / a);
    }

    /**
    * \brief return the intersection between ray and capped cone
    * @param {Vector3,        in}  ray origin
    * @param {Vector3,        in}  ray direction (should be normalized; not tested for normalization)
    * @param {Vector3,        in}  capped cone extreme #1
    * @param {Vector3,        in}  capped cone extreme #2
    * @param {floating_point, in}  capped cone radius at extreme #1
    * @param {floating_point, in}  capped cone radius at extreme #2
    * @param {vector4,        out} {distance to closest intersection point, along ray, from ray origin,
    *                               intersection point x, intersction point y, intersection point z}
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto cone_intersect(const GLSL::Vector3<T>& ro, const GLSL::Vector3<T>& rd, const GLSL::Vector3<T>& pa, const GLSL::Vector3<T>& pb, const T ra, const T rb) {
        assert(Numerics::areEquals(GLSL::length(rd), static_cast<T>(1)));

        const GLSL::Vector3<T> ba{ pb - pa };
        const GLSL::Vector3<T> oa{ ro - pa };
        const GLSL::Vector3<T> ob{ ro - pb };

        const T m0{ GLSL::dot(ba) };
        const T m1{ GLSL::dot(oa, ba) };
        const T m2{ GLSL::dot(rd, ba) };
        const T m3{ GLSL::dot(rd, oa) };
        const T m5{ GLSL::dot(oa) };
        const T m9{ GLSL::dot(ob, ba) };

        // caps
        if (m1 < T{}) {
            const GLSL::Vector3<T> b{ oa * m2 - rd * m1 };
            if (GLSL::dot(b) < ra * ra * m2 * m2) {
                assert(!Numerics::areEquals(m2, T{}));
                assert(m0 >= T{});
                return GLSL::Vector4<T>(-m1 / m2, -ba / std::sqrt(m0));
            }
        }
        else if (m9 > T{}) {
            const T t{ -m9 / m2 };
            const GLSL::Vector3<T> b{ ob + rd * t };
            if (GLSL::dot(b) < rb * rb) {
                assert(m0 >= T{});
                return GLSL::Vector4<T>(t, ba / std::sqrt(m0));
            }
        }

        // body
        const T rr{ ra - rb };
        const T hy{ m0 + rr * rr };
        const T k2{ m0 * m0 - m2 * m2 * hy };
        const T k1{ m0 * m0 * m3 - m1 * m2 * hy + m0 * ra * (rr * m2) };
        const T k0{ m0 * m0 * m5 - m1 * m1 * hy + m0 * ra * (rr * static_cast<T>(2) * m1 - m0 * ra) };
        const T h{ k1 * k1 - k2 * k0 };

        if (h < T{}) {
            return GLSL::Vector4<T>(static_cast<T>(-1));
        }

        [[assume(h >= T{})]];
        assert(!Numerics::areEquals(k2, T{}));
        const T t{ (-k1 - std::sqrt(h)) / k2 };
        const T y{ m1 + t * m2 };
        if ((y < T{}) || (y > m0)) {
            return GLSL::Vector4<T>(static_cast<T>(-1));
        }

        return GLSL::Vector4<T>(t, GLSL::normalize(m0 * (m0 * (oa + t * rd) + rr * ba * ra) - ba * hy * y));
    }

    /**
    * \brief return the intersection between ray and ellipsoid centered at center
    * @param {Vector3, in}  ray origin
    * @param {Vector3, in}  ray direction (should be normalized; not tested for normalization)
    * @param {Vector3, in}  ellipsoid radii
    * @param {Vector2, out} {distance to closest intersection point, along ray, from ray origin,
    *                        distance to furthest intersection point, along ray, from ray origin}
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto ellipsoid_intersection(const GLSL::Vector3<T>& ro, const GLSL::Vector3<T>& rd, const GLSL::Vector3<T>& ra) {
        assert(Numerics::areEquals(GLSL::length(rd), static_cast<T>(1)));
        assert(!Numerics::areEquals(ra.x, T{}) && !Numerics::areEquals(ra.y, T{}) && !Numerics::areEquals(ra.z, T{}));
        const GLSL::Vector3<T> ocn{ ro / ra };
        const GLSL::Vector3<T> rdn{ rd / ra };

        const T a{ GLSL::dot(rdn) };
        const T b{ GLSL::dot(ocn, rdn) };
        const T c{ GLSL::dot(ocn) };

        T h{ b * b - a * (c - static_cast<T>(1)) };
        if (h < T{}) {
            return GLSL::Vector2<T>(static_cast<T>(-1));
        }

        assert(!Numerics::areEquals(a, T{}));
        [[assume(h >= T{})]];
        h = std::sqrt(h);
        return GLSL::Vector2<T>((-b - h) / a, (-b + h) / a);
    }

    /**
    * \brief return the intersection between ray and torus centered at center
    * @param {Vector3,        in}  ray origin
    * @param {Vector3,        in}  ray direction (should be normalized; not tested for normalization)
    * @param {Vector2,        in}  {torus outside radius, torus width from outside radius inwards}
    * @param {floating_point, out} distance to closest intersection point, along ray, from ray origin
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto torus_intersect(const GLSL::Vector3<T>& ro, const GLSL::Vector3<T>& rd, const GLSL::Vector2<T>& tor) {
        constexpr T third{ static_cast<T>(1 / 3) };
        constexpr T half{ static_cast<T>(0.5) };
        constexpr T one{ static_cast<T>(1) };
        constexpr T two{ static_cast<T>(2) };
        constexpr T three{ static_cast<T>(3) };
        constexpr T four{ static_cast<T>(4) };
        assert(Numerics::areEquals(GLSL::length(rd), one));

        const T Ra2{ tor.x * tor.x };
        const T ra2{ tor.y * tor.y };
        const T m{ GLSL::dot(ro) };
        const T n{ GLSL::dot(ro, rd) };
        const T k{ (m + Ra2 - ra2) / two };

        T k3{ n };
        T k2{ n * n - Ra2 * GLSL::dot(rd.xy) + k };
        T k1{ n * k - Ra2 * GLSL::dot(rd.xy, ro.xy) };
        T k0{ k * k - Ra2 * GLSL::dot(ro.xy) };
        const T po{ one };
        if (std::abs(k3 * (k3 * k3 - k2) + k1) < static_cast<T>(0.01)) {
            po = static_cast<T>(-1);
            T tmp{ k1 };
            k1 = k3;
            k3 = tmp;

            assert(!Numerics::areEquals(k0, T{}));
            k0 = one / k0;
            k1 = k1 * k0;
            k2 = k2 * k0;
            k3 = k3 * k0;
        }

        T c2{ two * k2 - two * k3 * k3 };
        T c1{ k3 * (k3 * k3 - k2) + k1 };
        T c0{ k3 * (k3 * (c2 + two * k2) - static_cast<T>(8) * k1) + four * k0 };
        c2 *= third;
        c1 *= two;
        c0 *= third;
        const T Q{ c2 * c2 + c0 };
        const T R{ c2 * c2 * c2 - three * c2 * c0 + c1 * c1 };

        if (T h{ R * R - Q * Q * Q }; h >= T{}) {
            [[assume(h >= T{})]];
            h = std::sqrt(h);

            const T v{ Numerics::sign(R + h) * std::pow(std::abs(R + h), third) }; // cube root
            const T u{ Numerics::sign(R - h) * std::pow(std::abs(R - h), third) }; // cube root

            const GLSL::Vector2<T> s((v + u) + four * c2, (v - u) * std::sqrt(three));
            const T y{ std::sqrt(half * (GLSL::length(s) + s.x)) };
            assert(!Numerics::areEquals(y, T{}));
            const T x{ half * s.y / y };
            const T r{ two * c1 / (x * x + y * y) };

            T t1{ x - r - k3 };
            assert(!Numerics::areEquals(t1, T{}));
            t1 = (po < T{}) ? two / t1 : t1;

            T t2{ -x - r - k3 };
            assert(!Numerics::areEquals(t2, T{}));
            t2 = (po < T{}) ? two / t2 : t2;

            const T t{ static_cast<T>(1e20) };
            if (t1 > T{}) {
                t = t1;
            }
            if (t2 > T{}) {
                t = Numerics::min(t, t2);
            }

            return t;
        }

        const T sQ{ std::sqrt(Q) };
        const T w{ sQ * std::cos(std::acos(-R / (sQ * Q)) * third) };
        const T d2{ -(w + c2) };
        if (d2 < T{}) {
            return static_cast<T>(-1);
        }

        [[assume(d2 >= T{})]];
        const T d1{ std::sqrt(d2) };
        [[assume(d1 >= T{})]];
        const T h1{ std::sqrt(w - two * c2 + c1 / d1) };
        const T h2{ std::sqrt(w - two * c2 - c1 / d1) };
        T t1{ -d1 - h1 - k3 };
        t1 = (po < T{}) ? two / t1 : t1;

        T t2{ -d1 + h1 - k3 };
        t2 = (po < T{}) ? two / t2 : t2;

        T t3{ d1 - h2 - k3 };
        t3 = (po < T{}) ? two / t3 : t3;

        T t4{ d1 + h2 - k3 };
        t4 = (po < T{}) ? two / t4 : t4;

        T t{ static_cast<T>(1e20) };
        if (t1 > T{}) {
            t = t1;
        }
        if (t2 > T{}) {
            t = Numerics::min(t, t2);
        }
        if (t3 > T{}) {
            t = Numerics::min(t, t3);
        }
        if (t4 > T{}) {
            t = Numerics::min(t, t4);
        }

        return t;
    }
}