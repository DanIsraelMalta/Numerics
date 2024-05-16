#pragma once
#include "Glsl.h"

//
// primitives axis aligned bounding boxes
// (https://iquilezles.org/articles/diskbbox/, https://iquilezles.org/articles/ellipses/)
//

namespace AxisLignedBoundingBox {

    /**
    * \brief return the axis aligned bounding box of a disk
    * @param {Vector3,            in}  disk center
    * @param {Vector3,            in}  disk normal (should be normalized)
    * @param {floating_point,     in}  disk radius
    * @param {[Vector3, Vector3], out} [aabb min, aabb max]
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto disk_aabb(const GLSL::Vector3<T> c, const GLSL::Vector3<T> n, const T rad) {
        constexpr GLSL::Vector3<T> one(static_cast<T>(1));
        assert(Numerics::areEquals(GLSL::length(n), static_cast<T>(1)));

        const GLSL::Vector3<T> n2{ n * n };
        assert(!Numerics::areEquals(n2.x, static_cast<T>(1)));
        assert(!Numerics::areEquals(n2.y, static_cast<T>(1)));
        assert(!Numerics::areEquals(n2.z, static_cast<T>(1)));
        const GLSL::Vector3<T> e{ rad * GLSL::sqrt(one - n2) };
        return std::array<GLSL::Vector3<T>, 2>{ {c - e, c + e} };
    }

    /**
    * \brief return the axis aligned bounding box of a cylinder
    * @param {Vector3,            in}  cylinder edge point #1
    * @param {Vector3,            in}  cylinder edge point #2
    * @param {floating_point,     in}  cylinder radius
    * @param {[Vector3, Vector3], out} [aabb min, aabb max]
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto cylinder_aabb(const GLSL::Vector3<T> pa, const GLSL::Vector3<T> pb, const T ra) {
        constexpr GLSL::Vector3<T> one(static_cast<T>(1));
        const GLSL::Vector3<T> a{ pb - pa };
        assert(GLSL::dot(a) > T{});
        const GLSL::Vector3<T> e{ ra * sqrt(one - a * a / GLSL::dot(a)) };
        return std::array<GLSL::Vector3<T>, 2>{ { GLSL::min(pa - e, pb - e),
                                                  GLSL::max(pa + e, pb + e) } };
    }

    /**
    * \brief return the axis aligned bounding box of a cone
    * @param {Vector3,            in}  cone edge point #1
    * @param {Vector3,            in}  cone edge point #2
    * @param {floating_point,     in}  cone radius at point #1
    * @param {floating_point,     in}  cone radius at point #2
    * @param {[Vector3, Vector3], out} [aabb min, aabb max]
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto cone_aabb(const GLSL::Vector3<T> pa, const GLSL::Vector3<T> pb, const T ra, const T rb) {
        constexpr GLSL::Vector3<T> one(static_cast<T>(1));
        const GLSL::Vector3<T> a{ pb - pa };
        assert(GLSL::dot(a) > T{});
        const GLSL::Vector3<T> e{ GLSL::sqrt(one - a * a / GLSL::dot(a)) };
        const T era{ e * ra };
        const T erb{ e * rb };
        return std::array<GLSL::Vector3<T>, 2>{ { GLSL::min(pa - era, pb - erb),
                                                  GLSL::max(pa + era, pb + erb) } };
    }

    /**
    * \brief return the axis aligned bounding box of a planar/flat ellipse
    * @param {Vector3,            in}  ellipse center
    * @param {Vector3,            in}  ellipse first axis
    * @param {floating_point,     in}  ellipse second axis
    * @param {[Vector3, Vector3], out} [aabb min, aabb max]
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto ellipse_aabb(const GLSL::Vector3<T> c, const GLSL::Vector3<T> u, const GLSL::Vector3<T> v) {
        const GLSL::Vector3<T> e{ GLSL::sqrt(u * u + v * v) };
        return std::array<GLSL::Vector3<T>, 2>{ { c - e, c + e }};
    }
}