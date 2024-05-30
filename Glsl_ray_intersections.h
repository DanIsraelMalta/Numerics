#pragma once
#include "Glsl.h"
#include "Glsl_extra.h"

//
// ray-primitive intersection functions
// (see https://www.shadertoy.com/playlist/l3dXRf)
//

namespace RayIntersections {

    /**
    * \brief return the distance to closest intersection point between ray and plane, otherwise return -1
    *        notice that intersection is calculated only along the positive side of the plane (according to the normal)
    * @param {Vector3,        in}  ray origin
    * @param {Vector3,        in}  ray direction (should be normalized; not tested for normalization)
    * @param {floating_point, in}  plane {normal x, normal y, normal z, d}
    * @param {floating_point, out} distance to closest intersection point, along ray, from ray origin
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr T plane_intersect(const GLSL::Vector3<T>& ro, const GLSL::Vector3<T>& rd, const GLSL::Vector4<T>& p) {
        assert(Extra::is_normalized(rd));

        const GLSL::Vector3<T> normal(p.xyz);
        assert(Extra::is_normalized(normal));

        const T dot{ GLSL::dot(rd, normal) };
        return (dot > T{}) ? -(GLSL::dot(ro, normal) + p.w) / dot : static_cast<T>(-1);
    }

    /**
    * \brief return distance to closest intersection point betweeb ray and disk, otherwise return -1
    *        notice that intersection is calculated only along the positive side of the disk (according to the normal)
    * @param {Vector3,        in}  ray origin
    * @param {Vector3,        in}  ray direction (should be normalized; not tested for normalization)
    * @param {Vector3,        in}  disk center
    * @param {Vector3,        in}  disk normal (should be normalized
    * @param {floating_point, in}  disk radius
    * @param {floating_point, out} distance to closest intersection point, along ray, from ray origin
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr T disk_intersect(const GLSL::Vector3<T>& ro, const GLSL::Vector3<T>& rd, const GLSL::Vector3<T>& c, const GLSL::Vector3<T>& n, T r) {
        assert(Extra::is_normalized(rd));
        assert(Extra::is_normalized(n));

        const GLSL::Vector3<T> o{ ro - c };
        const T dot{ GLSL::dot(rd, n) };
        if (dot > T{}) {
            return static_cast<T>(-1);
        }

        const T t{ -GLSL::dot(n, o) / dot };
        const GLSL::Vector3<T> q{ o + rd * t };
        return (GLSL::dot(q) < r * r) ? t : static_cast<T>(-1);
    }

    /**
    * \brief return the intersection between ray and triangle
    * @param {Vector3, in}  ray origin
    * @param {Vector3, in}  ray direction (should be normalized; not tested for normalization)
    * @param {Vector3, in}  triangle vertex #0
    * @param {Vector3, in}  triangle vertex #1
    * @param {Vector3, in}  triangle vertex #2
    * @param {Vector3, out} intersection point (in cartesian coordinate), if no intersection return vector filled with -1
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr GLSL::Vector3<T> triangle_intersect_cartesian(const GLSL::Vector3<T>& ro, const GLSL::Vector3<T>& rd, const GLSL::Vector3<T>& v0, const GLSL::Vector3<T>& v1, const GLSL::Vector3<T>& v2) {
        assert(Extra::is_normalized(rd));

        const GLSL::Vector3<T> v10{ v1 - v0 };
        const GLSL::Vector3<T> v20{ v2 - v0 };
        const GLSL::Vector3<T> n{ GLSL::cross(v10, v20) };
        const GLSL::Vector3<T> normal{ GLSL::normalize(n) };
        const T dot{ GLSL::dot(normal, rd) };
        if (Numerics::areEquals(dot, T{})) {
            return GLSL::Vector3<T>(static_cast<T>(-1.0));
        }

        const T t{ GLSL::dot(-ro, normal) / dot };
        return GLSL::Vector3<T>(ro + rd * t);
    }

    /**
    * \brief return the intersection between ray and triangle
    * @param {Vector3, in}  ray origin
    * @param {Vector3, in}  ray direction (should be normalized; not tested for normalization)
    * @param {Vector3, in}  triangle vertex #0
    * @param {Vector3, in}  triangle vertex #1
    * @param {Vector3, in}  triangle vertex #2
    * @param {Vector3, out} intersection point (distance to intersection, intersection point in barycentric coordinate), if no intersection return vector filled with -1
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr GLSL::Vector3<T> triangle_intersect_barycentric(const GLSL::Vector3<T>& ro, const GLSL::Vector3<T>& rd, const GLSL::Vector3<T>& v0, const GLSL::Vector3<T>& v1, const GLSL::Vector3<T>& v2) {
        assert(Extra::is_normalized(rd));

        const GLSL::Vector3<T> v1v0{ v1 - v0 };
        const GLSL::Vector3<T> v2v0{ v2 - v0 };
        const GLSL::Vector3<T> rov0{ ro - v0 };
        const GLSL::Vector3<T> n{ GLSL::cross(v1v0, v2v0) };
        const GLSL::Vector3<T> q{ GLSL::cross(rov0, rd) };
        const T dot{ GLSL::dot(rd, n) };
        if (Numerics::areEquals(dot, T{})) {
            return GLSL::Vector3<T>(static_cast<T>(-1.0));
        }

        const T d{ static_cast<T>(1.0) / dot };
        const T u{ d * GLSL::dot(-q, v2v0) };
        const T v{ d * GLSL::dot(q, v1v0) };
        if ((Numerics::min(u, v) < T{}) || (u + v > static_cast<T>(1.0))) {
            return GLSL::Vector3<T>(static_cast<T>(-1.0));
        }
        return GLSL::Vector3<T>(d* GLSL::dot(-n, rov0), u, v);
    }

    /**
    * \brief return the intersection between ray and ellipse
    * @param {Vector3, in}  ray origin
    * @param {Vector3, in}  ray direction (should be normalized; not tested for normalization)
    * @param {Vector3, in}  ellipse center
    * @param {Vector3, in}  ellipse radii #1 ('u')
    * @param {Vector3, in}  ellipse radii #1 ('v')
    * @param {Vector3, out} intersection point (distance to intersection point, intersection point in uv coordinate), if no intersection return vector filled with -1
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto ellipse_intersect(const GLSL::Vector3<T>& ro, const GLSL::Vector3<T>& rd, const GLSL::Vector3<T>& c, const GLSL::Vector3<T>& u, const GLSL::Vector3<T>& v) {
        assert(Extra::is_normalized(rd));

        const GLSL::Vector3<T> q{ ro - c };
        const GLSL::Vector3<T> n{ GLSL::cross(u, v) };
        const T dot{ GLSL::dot(rd, n) };
        if (Numerics::areEquals(dot, T{})) {
            return GLSL::Vector3<T>(static_cast<T>(-1.0));
        }

        const T t{ -GLSL::dot(n, q) / dot };
        const GLSL::Vector3<T> qrdt{ q + rd * t };
        const T r{ GLSL::dot(u, qrdt) };
        const T s{ GLSL::dot(v, qrdt) };

        if (r * r + s * s > static_cast<T>(1)) {
            return GLSL::Vector3<T>(static_cast<T>(-1));
        }
        return GLSL::Vector3<T>(t, s, r);
    }

    /**
    * \brief return the intersection between ray and sphere.
    *        notice that based on function output:
    *        if y coordinate is negative - ray does not intersect sphere
    *        else if x coordinate is negative - origin is inside sphere and y coordinate is intersection distance
    *        else origin is outside sphere and x coordinate is intersection distance.
    * @param {Vector3, in}  ray origin
    * @param {Vector3, in}  ray direction (should be normalized; not tested for normalization)
    * @param {Vector4, in}  sphere {center x, center y, cernter z, radius}
    * @param {Vector2, out} {distance to closest intersection point, along ray, from ray origin,
    *                        distance to furthest intersection point, along ray, from ray origin}
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr GLSL::Vector2<T> sphere_intersect(const GLSL::Vector3<T>& ro, const GLSL::Vector3<T>& rd, const GLSL::Vector4<T>& sph) {
        assert(Extra::is_normalized(rd));

        const GLSL::Vector3<T> c(sph.xyz);
        const GLSL::Vector3<T> oc{ ro - c };
        const T b{ GLSL::dot(oc, rd) };
        const GLSL::Vector3<T> qc{ oc - b * rd };
        T h{ sph.w * sph.w - GLSL::dot(qc) };
        if (h < T{}) {
            return GLSL::Vector2<T>(static_cast<T>(-1));
        }

        [[assume(h >= T{})]];
        h = std::sqrt(h);
        return GLSL::Vector2<T>(-b - h, -b + h);
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
        assert(Extra::is_normalized(rd));
        if (Numerics::areEquals(ra.x, T{}) ||
            Numerics::areEquals(ra.y, T{}) ||
            Numerics::areEquals(ra.z, T{})) {
            return GLSL::Vector2<T>(static_cast<T>(-1));
        }

        const GLSL::Vector3<T> ocn{ ro / ra };
        const GLSL::Vector3<T> rdn{ rd / ra };
        const T a{ GLSL::dot(rdn) };
        const T b{ GLSL::dot(ocn, rdn) };
        const T c{ GLSL::dot(ocn) };

        T h{ b * b - a * (c - static_cast<T>(1)) };
        if (h < T{} || Numerics::areEquals(a, T{})) {
            return GLSL::Vector2<T>(static_cast<T>(-1));
        }

        [[assume(h >= T{})]];
        [[assume(a != T{})]];
        h = std::sqrt(h);
        return GLSL::Vector2<T>((-b - h) / a, (-b + h) / a);
    }
}
