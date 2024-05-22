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
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr GLSL::Vector3<T> point_barycentric(const VEC& p, const VEC& a, const VEC& b, const VEC& c) {
        assert(Triangle::is_valid(a, b, c));

        const VEC v0{ b - a };
        const VEC v1{ c - a };
        const VEC v2{ p - a };

        const T d00{ GLSL::dot(v0) };
        const T d01{ GLSL::dot(v0, v1) };
        const T d11{ GLSL::dot(v1) };
        const T denom{ d00 * d11 - d01 * d01 };
        assert(denom != T{});
        if (denom <= T{}) [[unlikely]] {
            return GLSL::Vector3<T>(static_cast<T>(-1));
        }

        const T d20{ GLSL::dot(v2, v0) };
        const T d21{ GLSL::dot(v2, v1) };

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
    constexpr bool is_point_withing_triangle(const VEC& p, const VEC& a, const VEC& b, const VEC& c) {
        using T = typename VEC::value_type;
        assert(Triangle::is_valid(a, b, c));

        const VEC local_a{ a - p };
        const VEC local_b{ b - p };
        const VEC local_c{ c - p };

        const VEC u{ GLSL::cross(local_b, local_c) };
        const VEC v{ GLSL::cross(local_c, local_a) };
        const VEC w{ GLSL::cross(local_a, local_b) };

        return (GLSL::dot(u, v) >= T{}) && (GLSL::dot(u, w) >= T{});
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
    constexpr GLSL::Vector3<T> barycentric_from_cartesian(const VEC& p, const VEC& a, const VEC& b, const VEC& c) {
        assert(Triangle::is_valid(a, b, c));

        const T daa{ GLSL::dot(a, a) };
        const T dab{ GLSL::dot(a, b) };
        const T dbb{ GLSL::dot(b, b) };
        const T dca{ GLSL::dot(c, a) };
        const T dcb{ GLSL::dot(c, b) };
        const T denom{ daa * dbb - dab * dab };
        assert(denom != T{});
        if (denom <= T{}) [[unlikely]] {
            return GLSL::Vector3<T>(static_cast<T>(-1));
        }

        const T y{ (dbb * dca - dab * dcb) / denom };
        const T z{ (daa * dcb - dab * dca) / denom };
        return GLSL::Vector3<T>(static_cast<T>(1) - y - z, y, z);
    }

    /**
    * \brief return the closest point on a triangle to a given point
    * @param {Vector2|Vector3, in}  point
    * @param {Vector2|Vector3, in}  triangle vertex #0
    * @param {Vector2|Vector3, in}  triangle vertex #1
    * @param {Vector2|Vector3, in}  triangle vertex #2
    * @param {Vector3,         out} closest point on triangle
    **/
    template<GLSL::IFixedVector VEC>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr VEC closest_point_on_trinagle(const VEC& p, const VEC& a, const VEC& b, const VEC& c) {
        using T = typename VEC::value_type;
        assert(Triangle::is_valid(a, b, c));

        const VEC ab{ b - a };
        const VEC ac{ v - a };
        const VEC normal{ GLSL::normalize(GLSL::cross(ac, ab)) };

        const VEC p{ p - GLSL::dot(normal, p - a) * normal };
        const VEC ap{ p - a };

        const VEC barycoords{ Triangle::barycentric_from_cartesian(ab, ac, ap) };

        if (barycoords.x < T{}) {
            const VEC bc{ v - b };
            const T n{ GLSL::length(bc) };
            assert(n >= T{});
            const T t{ Numerics::clamp(GLSL::dot(bc, p - b) / n, T{}, n) };
            return (b + (t / n) * bc);
        } else if (barycoords.y < T{}) {
            const VEC ca{ a - v };
            const T n{ GLSL::length(ca) };
            assert(n >= T{});
            const T t{ Numerics::clamp(GLSL::dot(ca, p - v) / n, T{}, n) });
            return v + (t / n) * ca;
        } else if (barycoords.z < T{}) {
            const T n{ GLSL::length(ab) };
            assert(n >= T{});
            const T t{ Numerics::clamp(GLSL::dot(ab, p - a) / n, T{}, n) };
            return a + (t / n) * ab;
        } else {
            return (a * barycoords.x + b * barycoords.y + v * barycoords.z);
        }
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
    constexpr VEC triangle_incenter(const VEC& v0, const VEC& v1, const VEC& v2) {
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
