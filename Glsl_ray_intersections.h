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
    * @param IFixedVector, in}  ray origin
    * @param IFixedVector, in}  ray direction (should be normalized; not tested for normalization)
    * @param IFixedVector, in}  disk center
    * @param IFixedVector, in}  disk normal (should be normalized
    * @param {value_type,  in}  disk radius
    * @param {value_type,  out} distance to closest intersection point, along ray, from ray origin
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_floating_point_v<T> && (VEC::length() == 3))
    constexpr T disk_intersect(const VEC& ro, const VEC& rd, const VEC& c, const VEC& n, T r) {
        assert(Extra::is_normalized(rd));
        assert(Extra::is_normalized(n));

        const VEC o{ ro - c };
        const T dot{ GLSL::dot(rd, n) };
        if (dot > T{}) {
            return static_cast<T>(-1);
        }

        const T t{ -GLSL::dot(n, o) / dot };
        const VEC q{ o + rd * t };
        return (GLSL::dot(q) < r * r) ? t : static_cast<T>(-1);
    }

    /**
    * \brief return the intersection between ray and triangle
    * @param {IFixedVector, in}  ray origin
    * @param {IFixedVector, in}  ray direction (should be normalized; not tested for normalization)
    * @param {IFixedVector, in}  triangle vertex #0
    * @param {IFixedVector, in}  triangle vertex #1
    * @param {IFixedVector, in}  triangle vertex #2
    * @param {IFixedVector, out} intersection point (in cartesian coordinate), if no intersection return vector filled with -1
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_floating_point_v<T> && (VEC::length() == 3))
    constexpr VEC triangle_intersect_cartesian(const VEC& ro, const VEC& rd, const VEC& v0, const VEC& v1, const VEC& v2) {
        assert(Extra::is_normalized(rd));

        const VEC v10{ v1 - v0 };
        const VEC v20{ v2 - v0 };
        const VEC n{ GLSL::cross(v10, v20) };
        const VEC normal{ GLSL::normalize(n) };
        const T dot{ GLSL::dot(normal, rd) };
        if (Numerics::areEquals(dot, T{})) {
            return VEC(static_cast<T>(-1.0));
        }

        const T t{ GLSL::dot(-ro, normal) / dot };
        return VEC(ro + rd * t);
    }

    /**
    * \brief return the intersection between ray and triangle
    * @param {IFixedVector, in}  ray origin
    * @param {IFixedVector, in}  ray direction (should be normalized; not tested for normalization)
    * @param {IFixedVector, in}  triangle vertex #0
    * @param {IFixedVector, in}  triangle vertex #1
    * @param {IFixedVector, in}  triangle vertex #2
    * @param {IFixedVector, out} intersection point (distance to intersection, intersection point in barycentric coordinate), if no intersection return vector filled with -1
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_floating_point_v<T> && (VEC::length() == 3))
    constexpr VEC triangle_intersect_barycentric(const VEC& ro, const VEC& rd, const VEC& v0, const VEC& v1, const VEC& v2) {
        assert(Extra::is_normalized(rd));

        const VEC v1v0{ v1 - v0 };
        const VEC v2v0{ v2 - v0 };
        const VEC rov0{ ro - v0 };
        const VEC n{ GLSL::cross(v1v0, v2v0) };
        const VEC q{ GLSL::cross(rov0, rd) };
        const T dot{ GLSL::dot(rd, n) };
        if (Numerics::areEquals(dot, T{})) {
            return VEC(static_cast<T>(-1.0));
        }

        const T d{ static_cast<T>(1.0) / dot };
        const T u{ d * GLSL::dot(-q, v2v0) };
        const T v{ d * GLSL::dot(q, v1v0) };
        if ((Numerics::min(u, v) < T{}) || (u + v > static_cast<T>(1.0))) {
            return VEC(static_cast<T>(-1.0));
        }
        return VEC(d* GLSL::dot(-n, rov0), u, v);
    }

    /**
    * \brief return the intersection between ray and ellipse
    * @param {IFixedVector, in}  ray origin
    * @param {IFixedVector, in}  ray direction (should be normalized; not tested for normalization)
    * @param {IFixedVector, in}  ellipse center
    * @param {IFixedVector, in}  ellipse radii #1 ('u')
    * @param {IFixedVector, in}  ellipse radii #1 ('v')
    * @param {IFixedVector, out} intersection point (distance to intersection point, intersection point in uv coordinate), if no intersection return vector filled with -1
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_floating_point_v<T> && (VEC::length() == 3))
    constexpr VEC ellipse_intersect(const VEC& ro, const VEC& rd, const VEC& c, const VEC& u, const VEC& v) {
        assert(Extra::is_normalized(rd));

        const VEC q{ ro - c };
        const VEC n{ GLSL::cross(u, v) };
        const T dot{ GLSL::dot(rd, n) };
        if (Numerics::areEquals(dot, T{})) {
            return VEC(static_cast<T>(-1.0));
        }

        const T t{ -GLSL::dot(n, q) / dot };
        const VEC qrdt{ q + rd * t };
        const T r{ GLSL::dot(u, qrdt) };
        const T s{ GLSL::dot(v, qrdt) };

        if (r * r + s * s > static_cast<T>(1)) {
            return VEC(static_cast<T>(-1));
        }
        return VEC(t, s, r);
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
    constexpr GLSL::Vector2<T> ellipsoid_intersection(const GLSL::Vector3<T>& ro, const GLSL::Vector3<T>& rd, const GLSL::Vector3<T>& ra) {
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
