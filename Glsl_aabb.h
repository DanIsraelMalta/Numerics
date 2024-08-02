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
// axis aligned bounding box related functions
//
namespace Aabb {

    /**
    * \brief return aabb centroid
    * @param {Vector2|Vector3, in}  aabb min
    * @param {Vector2|Vector3, in}  aabb max
    * @param {Vector2|Vector3, out} aabb centroid
    **/
    template<GLSL::IFixedVector VEC>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr VEC centroid(const VEC& min, const VEC& max) noexcept {
        using T = typename VEC::value_type;
        return (min + max) / static_cast<T>(2);
    }

    /**
    * \brief return aabb diagnoal
    * @param {Vector2|Vector3, in}  aabb min
    * @param {Vector2|Vector3, in}  aabb max
    * @param {Vector2|Vector3, out} aabb diagnoal
    **/
    template<GLSL::IFixedVector VEC>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr VEC diagnonal(const VEC& min, const VEC& max) noexcept {
        return GLSL::abs(max - min);
    }

    /**
    * \brief test if point is inside aabb
    * @param {Vector2|Vector3, in}  point
    * @param {Vector2|Vector3, in}  aabb min
    * @param {Vector2|Vector3, in}  aabb max
    * @param {bool,            out} true if point is inside aabb, false otherwise
    **/
    template<GLSL::IFixedVector VEC>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr bool is_point_inside(const VEC& p, const VEC& min, const VEC& max) noexcept {
        return GLSL::lessThanEqual(p, max) && GLSL::lessThanEqual(min, p);
    }

    /**
    * \brief expand an aabb to include a given point
    * @param {Vector2|Vector3,                    in} point
    * @param {Vector2|Vector3,                    in} aabb min
    * @param {Vector2|Vector3,                    in} aabb max
    * @param {{Vector2|Vector3, Vector2|Vector3}, out} {expanded min, expanded max}
    **/
    template<GLSL::IFixedVector VEC>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr auto expand(const VEC& p, const VEC& min, const VEC& max) noexcept {
        using out_t = struct { VEC min; VEC max; };
        return out_t{ GLSL::min(min, p), GLSL::max(max, p) };
    }

    /**
    * \brief square an aabb using its longest side (square center is aabb center)
    * @param {Vector2|Vector3,                    in} aabb min
    * @param {Vector2|Vector3,                    in} aabb max
    * @param {{Vector2|Vector3, Vector2|Vector3}, out} {square min, square max}
    **/
    template<GLSL::IFixedVector VEC>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr auto square(const VEC& p, const VEC& min, const VEC& max) noexcept {
        using T = typename VEC::value_type;
        using out_t = struct { VEC min; VEC max; };
        
        const VEC diag{ Aabb::diagnonal(min, max) / static_cast<T>(2) };
        const VEC cntr{ min + diag };
        const T mmax{ GLSL::max(GLSL::abs(diag)) };

        return out_t{ cntr + mmax , cntr - mmax };
    }

    /**
    * \brief given a point and aabb, return point on aabb closest to the point
    * @param {Vector2|Vector3,  in} point
    * @param {Vector2|Vector3,  in} aabb min
    * @param {Vector2|Vector3,  in} aabb max
    * @param {Vector2|Vector3, out} closest corner
    **/
    template<GLSL::IFixedVector VEC>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr auto closest_point(const VEC& p, const VEC& min, const VEC& max) noexcept {
        using T = typename VEC::value_type;

        VEC corner;
        Utilities::static_for<0, 1, VEC::length()>([&corner, &p, &min, &max](std::size_t i) {
            T v{ p[i] };
            if (v < min[i]) {
                v = min[i];
            } else if (v > max[i]) {
                v = max[i];
            }
            corner[i] = v;
        });

        return corner;
    }
}
