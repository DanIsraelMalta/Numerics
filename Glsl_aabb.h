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
        return GLSL::all(GLSL::lessThanEqual(p, max)) && GLSL::all(GLSL::lessThan(min, p));
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
}
