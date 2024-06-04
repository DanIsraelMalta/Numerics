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
        using out_t = struct { GLSL::Vector3<T> min; GLSL::Vector3<T> max; };
        constexpr GLSL::Vector3<T> one(static_cast<T>(1));
        assert(Numerics::areEquals(GLSL::length(n), static_cast<T>(1)));

        const GLSL::Vector3<T> one_minus_n2{ one - n * n };
        assert(one_minus_n2.x >= T{});
        assert(one_minus_n2.y >= T{});
        assert(one_minus_n2.z >= T{});
        const GLSL::Vector3<T> e{ rad * GLSL::sqrt(one_minus_n2) };

        return out_t{ c - e , c + e };
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
        using out_t = struct { GLSL::Vector3<T> min; GLSL::Vector3<T> max; };
        constexpr GLSL::Vector3<T> one(static_cast<T>(1));

        const GLSL::Vector3<T> a{ pb - pa };
        const T dot{ GLSL::dot(a) };
        assert(dot > T{});
        const GLSL::Vector3<T> squared{ one - a * a / dot };
        assert(squared.x >= T{});
        assert(squared.y >= T{});
        assert(squared.z >= T{});
        const GLSL::Vector3<T> e{ ra * GLSL::sqrt(one - a * a / dot) };

        return out_t{ GLSL::min(pa - e, pb - e), GLSL::max(pa + e, pb + e) };
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
        using out_t = struct { GLSL::Vector3<T> min; GLSL::Vector3<T> max; };
        constexpr GLSL::Vector3<T> one(static_cast<T>(1));

        const GLSL::Vector3<T> a{ pb - pa };
        const T dot{ GLSL::dot(a) };
        assert(dot > T{});
        const GLSL::Vector3<T> squared{ one - a * a / dot };
        assert(squared.x >= T{});
        assert(squared.y >= T{});
        assert(squared.z >= T{});
        const GLSL::Vector3<T> e{ GLSL::sqrt(squared) };
        const GLSL::Vector3<T> era{ e * ra };
        const GLSL::Vector3<T> erb{ e * rb };

        return out_t{ GLSL::min(pa - era, pb - erb), GLSL::max(pa + era, pb + erb) };
    }

    /**
    * \brief return the axis aligned bounding box of a planar/flat ellipse.
    *        the ellipse plane is defined by its axes.
    * @param {Vector3,            in}  ellipse center
    * @param {Vector3,            in}  vector connecting ellipse edge points along long axis
    * @param {Vector3,            in}  vector connecting ellipse edge points along small axis
    * @param {[Vector3, Vector3], out} [aabb min, aabb max]
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto ellipse_aabb(const GLSL::Vector3<T> c, const GLSL::Vector3<T> u, const GLSL::Vector3<T> v) {
        using out_t = struct { GLSL::Vector3<T> min; GLSL::Vector3<T> max; };

        const GLSL::Vector3<T> squared{ u * u + v * v };
        assert(squared.x >= T{});
        assert(squared.y >= T{});
        assert(squared.z >= T{});
        const GLSL::Vector3<T> e{ GLSL::sqrt(squared) };

        return out_t{ c - e, c + e };
    }

    /**
	* \brief given point cloud - return its axis aligned bounding box
	* @param {forward_iterator,             in}  iterator to collection first point
	* @param {forward_iterator,             in}  iterator to collection last point
	* @param {{IFixedVector, IFixedVector}, out} {aabb min, aabb max}
	**/
	template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
		requires(GLSL::is_fixed_vector_v<VEC>)
	constexpr auto point_cloud_aabb(const InputIt first, const InputIt last) noexcept {
        using T = typename VEC::value_type;
		using out_t = struct { VEC min; VEC max; };

		VEC min(std::numeric_limits<T>::max());
		VEC maxNegative(std::numeric_limits<T>::max());
        for (auto it{ first }; it != last; ++it) {
			min = GLSL::min(min, *it);
			maxNegative = GLSL::min(maxNegative, -*it);
		}

		return out_t{ min, -maxNegative };
	}
}
