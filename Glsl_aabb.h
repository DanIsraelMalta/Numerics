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
// primitives axis aligned bounding boxes
// (https://iquilezles.org/articles/diskbbox/, https://iquilezles.org/articles/ellipses/)
//

namespace AxisLignedBoundingBox {

    /**
    * \brief return the axis aligned bounding box of a disk
    * @param {IFixedVector,                 in}  disk center
    * @param {IFixedVector,                 in}  disk normal (should be normalized)
    * @param {value_type,                   in}  disk radius
    * @param {[IFixedVector, IFixedVector], out} {aabb min, aabb max}
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(VEC::length() == 3)
    constexpr auto disk_aabb(const VEC& c, const VEC& n, const T rad) {
        using out_t = struct { VEC min; VEC max; };
        constexpr VEC one(static_cast<T>(1));
        assert(Numerics::areEquals(GLSL::length(n), static_cast<T>(1)));

        const VEC one_minus_n2{ one - n * n };
        assert(one_minus_n2.x >= T{});
        assert(one_minus_n2.y >= T{});
        assert(one_minus_n2.z >= T{});
        [[assume(one_minus_n2.x >= T{})]];
        [[assume(one_minus_n2.y >= T{})]];
        [[assume(one_minus_n2.z >= T{})]];
        const VEC e{ rad * GLSL::sqrt(one_minus_n2) };

        return out_t{ c - e , c + e };
    }

    /**
    * \brief return the axis aligned bounding box of a cylinder
    * @param {IFixedVector,                 in}  cylinder edge point #1
    * @param {IFixedVector,                 in}  cylinder edge point #2
    * @param {value_type,                   in}  cylinder radius
    * @param {[IFixedVector, IFixedVector], out} {aabb min, aabb max}
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(VEC::length() == 3)
    constexpr auto cylinder_aabb(const VEC& pa, const VEC& pb, const T ra) {
        using out_t = struct { VEC min; VEC max; };
        constexpr VEC one(static_cast<T>(1));

        const VEC a{ pb - pa };
        const T dot{ GLSL::dot(a) };
        assert(dot > T{});
        const VEC squared{ one - a * a / dot };
        [[assume(squared.x >= T{})]];
        [[assume(squared.y >= T{})]];
        [[assume(squared.z >= T{})]];
        const VEC e{ ra * GLSL::sqrt(one - a * a / dot) };

        return out_t{ GLSL::min(pa - e, pb - e), GLSL::max(pa + e, pb + e) };
    }

    /**
    * \brief return the axis aligned bounding box of a cone
    * @param {IFixedVector,                 in}  cone edge point #1
    * @param {IFixedVector,                 in}  cone edge point #2
    * @param {value_type,                   in}  cone radius at point #1
    * @param {value_type,                   in}  cone radius at point #2
    * @param {[IFixedVector, IFixedVector], out} {aabb min, aabb max}
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(VEC::length() == 3)
    constexpr auto cone_aabb(const VEC& pa, const VEC& pb, const T ra, const T rb) {
        using out_t = struct { VEC min; VEC max; };
        constexpr VEC one(static_cast<T>(1));

        const VEC a{ pb - pa };
        const T dot{ GLSL::dot(a) };
        assert(dot > T{});
        const VEC squared{ one - a * a / dot };
        [[assume(squared.x >= T{})]];
        [[assume(squared.y >= T{})]];
        [[assume(squared.z >= T{})]];
        const VEC e{ GLSL::sqrt(squared) };
        const VEC era{ e * ra };
        const VEC erb{ e * rb };

        return out_t{ GLSL::min(pa - era, pb - erb), GLSL::max(pa + era, pb + erb) };
    }

    /**
    * \brief return the axis aligned bounding box of a planar/flat ellipse.
    *        the ellipse plane is defined by its axes.
    * @param {IFixedVector,                 in}  ellipse center
    * @param {IFixedVector,                 in}  vector connecting ellipse edge points along long axis
    * @param {IFixedVector,                 in}  vector connecting ellipse edge points along small axis
    * @param {[IFixedVector, IFixedVector], out} {aabb min, aabb max}
    **/
    template<GLSL::IFixedVector VEC>
        requires(VEC::length() == 3)
    constexpr auto ellipse_aabb(const VEC& c, const VEC& u, const VEC& v) {
        using out_t = struct { VEC min; VEC max; };

        const VEC squared{ u * u + v * v };
        [[assume(squared.x >= T{})]];
        [[assume(squared.y >= T{})]];
        [[assume(squared.z >= T{})]];
        const VEC e{ GLSL::sqrt(squared) };

        return out_t{ c - e, c + e };
    }

    /**
    * \brief return the axis aligned bounding box of a sphere.
    * @param {IFixedVector,                 in}  sphere center
    * @param {value_type,                   in}  sphere radius
    * @param {[IFixedVector, IFixedVector], out} {aabb min, aabb max}
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(VEC::length() == 3)
    constexpr auto sphere_aabb(const VEC& center, const T radius) {
        using out_t = struct { GLSL::Vector3<T> min; GLSL::Vector3<T> max; };
        return out_t{ center - radius, center + radius };
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
