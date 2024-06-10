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
    constexpr bool is_valid(const VEC& v0, const VEC& v1, const VEC& v2) noexcept {
        using T = typename VEC::value_type;

        // triangle sides length, sorted in ascending order
        const VEC v10{ v1 - v0 };
        const VEC v21{ v2 - v1 };
        const VEC v20{ v2 - v0 };
        std::array<T, 3> sides{ { GLSL::length(v1 - v0),
                                  GLSL::length(v2 - v1),
                                  GLSL::length(v2 - v0) } };
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
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr GLSL::Vector3<T> get_point_in_barycentric(const VEC& p, const VEC& a, const VEC& b, const VEC& c) {
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

        const T d20{ GLSL::dot(v2, v0) };
        const T d21{ GLSL::dot(v2, v1) };

        [[assume(denom > T{})]]
        const T v{ (d11 * d20 - d01 * d21) / denom };
        const T w{ (d00 * d21 - d01 * d20) / denom };
        return GLSL::Vector3<T>(static_cast<T>(1) - v - w, v, w);
    }

    /**
    * \brief check if a point is contained within a triangle
    *        uses Thomas Müller algorithm. faster than calculating triangle sign distance field at point.
    * @param {Vector2|Vector3, in}  point
    * @param {Vector2|Vector3, in}  triangle vertex #0
    * @param {Vector2|Vector3, in}  triangle vertex #1
    * @param {Vector2|Vector3, in}  triangle vertex #2
    * @param {bool,            out} true if point is inside triangle, false otherwise
    **/
    template<GLSL::IFixedVector VEC>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr bool is_point_within_triangle(const VEC& p, const VEC& a, const VEC& b, const VEC& c) {
        using T = typename VEC::value_type;
        constexpr std::size_t N{ VEC::length() };
        assert(Triangle::is_valid(a, b, c));

        const VEC local_a{ a - p };
        const VEC local_b{ b - p };
        const VEC local_c{ c - p };

        const auto u = GLSL::cross(local_b, local_c);
        const auto v = GLSL::cross(local_c, local_a);
        const auto w = GLSL::cross(local_a, local_b);

        if constexpr (N == 2) {
            return (u * v >= T{}) && (u * w >= T{});
        }
        else {
            return (GLSL::dot(u, v) >= T{}) && (GLSL::dot(u, w) >= T{});
        }
    }

    /**
    * \brief given triangle (by its vertices) - return its barycentric coordinates
    * @param {Vector2|Vector3, in}  triangle vertex #0
    * @param {Vector2|Vector3, in}  triangle vertex #1
    * @param {Vector2|Vector3, in}  triangle vertex #2
    * @param {Vector3,         out} barycentric coordinates
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr GLSL::Vector3<T> barycentric_from_cartesian(const VEC& a, const VEC& b, const VEC& c) {
        assert(Triangle::is_valid(a, b, c));

        const T daa{ GLSL::dot(a, a) };
        const T dab{ GLSL::dot(a, b) };
        const T dbb{ GLSL::dot(b, b) };
        const T dca{ GLSL::dot(c, a) };
        const T dcb{ GLSL::dot(c, b) };
        const T denom{ daa * dbb - dab * dab };
        if (denom <= T{}) [[unlikely]] {
            return GLSL::Vector3<T>(static_cast<T>(-1));
        }

        [[assume(denom != T{})]]
        const T y{ (dbb * dca - dab * dcb) / denom };
        const T z{ (daa * dcb - dab * dca) / denom };
        return GLSL::Vector3<T>(static_cast<T>(1) - y - z, y, z);
    }
}
