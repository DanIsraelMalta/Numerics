#pragma once
#include "Glsl.h"

//
// triangle related functions
//
namespace Triangle {

    /**
    * \brief test if triangle is valid
    * @param {Vector2|Vector3, in}  triangle vertex #0
    * @param {Vector2|Vector3, in}  triangle vertex #1
    * @param {Vector2|Vector3, in}  triangle vertex #2
    * @param {bool,            out} true if triangle is valid, false otherwise
    **/
    template<GLSL::IFixedVector VEC>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr auto is_valid(const VEC& v0, const VEC& v1, const VEC& v2) noexcept {
        using T = typename VEC::value_type;

        // triangle sides length, sorted in ascending order
        std::array<T, 3> sides{ { GLSL::dot(v1 - v0),
                                  GLSL::dot(v2 - v1),
                                  GLSL::dot(v2 - v0) } };
        if (sides[0] > sides[2]) { Utilities::swap(sides[0], sides[2]); }
        if (sides[0] > sides[1]) { Utilities::swap(sides[0], sides[1]); }
        if (sides[1] > sides[2]) { Utilities::swap(sides[1], sides[2]); }

        return ((sides[0] - (sides[2] - sides[1])) > T{});
    }

    /**
    * \brief calculate the barycentric coordinates for a cartesian coordinate point in a triangle defined by vertices
    * @param {Vector2|Vector3, in}  point
    * @param {Vector2|Vector3, in}  triangle vertex #0
    * @param {Vector2|Vector3, in}  triangle vertex #1
    * @param {Vector2|Vector3, in}  triangle vertex #2
    * @param {Vector3,         out} barycentric coordinates of a given point relative to triangle
    **/
    template<GLSL::IFixedVector VEC>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr auto cartesian_to_barycentric(const VEC& p, const VEC& a, const VEC& b, const VEC& c) {
        using T = typename VEC::value_type;
        assert(Triangle::is_valid(a, b, c));

        const VEC v0{ b - a };
        const VEC v1{ c - a };
        const VEC v2{ p - a };

        const T d00{ GLSL::dot(v0) };
        const T d01{ GLSL::dot(v0, v1) };
        const T d11{ GLSL::dot(v1) };
        const T denom{ d00 * d11 - d01 * d01 };
        if (denom <= T{}) [[unlikely]] {
            return GLSL::Vector3<T>(static_cast<T>(-1));
        }
        [[assume(denom != T{})]];

        const T d20{ GLSL::dot(v2, v0) };
        const T d21{ GLSL::dot(v2, v1) };

        const T v{ (d11 * d20 - d01 * d21) / denom };
        const T w{ (d00 * d21 - d01 * d20) / denom };
        return GLSL::Vector3<T>(static_cast<T>(1) - v - w, v, w);
    }

    /**
    * \brief return the center of the incircle of a triangle
    * @param {Vector2|Vector3, in}  triangle vertex #0
    * @param {Vector2|Vector3, in}  triangle vertex #1
    * @param {Vector2|Vector3, in}  triangle vertex #2
    * @param {Vector2|Vector3, out} triangle incircle center
    **/
    template<GLSL::IFixedVector VEC>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr auto triangle_incenter(const VEC& v0, const VEC& v1, const VEC& v2) {
        using T = typename VEC::value_type;
        assert(Triangle::is_valid(v0, v1, v2));

        const T l0{ GLSL::length(v2 - v1) };
        const T l1{ GLSL::length(v0 - v2) };
        const T l2{ GLSL::length(v1 - v0) };
        const T sum{ l0 + l1 + l2 };
        [[assume(sum != T{})]];
        return (v0 * l0 + v1 * l1 + v2 * l2) / sum;
    }
}