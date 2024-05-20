#pragma once
#include "Glsl.h"

//
// general numerical utilities for vectors and matrices
//

namespace Extra {

    /**
    * \brief efficient inverse of affine rigid transformation (assuming matrix is not singular)
    *        an affine rigid transformation matrix is a 4x4 matrix with first 3x3 portion holding
    *        rotation information and the fourth column holds the translation information.
    * @param {Matrix4, in}  affine rigid transformation
    * @param {Matrix4, out} invert of input matrix
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr GLSL::Matrix4<T> affine_rigid_inverse(const GLSL::Matrix4<T>& mat) noexcept {
        // transpose DCM
        GLSL::Matrix4<T> inv{ GLSL::transpose(mat) };

        // multiply translation by DCM transpose
        const GLSL::Vector4<T> x{ mat[0] };
        const GLSL::Vector4<T> y{ mat[1] };
        const GLSL::Vector4<T> z{ mat[2] };
        inv[3] = GLSL::Vector4<T>{ -GLSL::dot(z, x), -GLSL::dot(z, y), -GLSL::dot(z), static_cast<T>(1) };

        // outout
        return inv;
    }

    /**
    * \brief return the outer product of two vectors
    * @param  {Vector2|Vector3|Vector4, in}  x
    * @param  {Vector2|Vector3|Vector4, in}  y
    * @return {Matrix2|Matrix3|Matrix4, out} outer product between x and y
    **/
    template<GLSL::IFixedVector VEC>
        requires(VEC::length() <= 4)
    constexpr auto outer_product(const VEC& x, const VEC& y) noexcept {
        using T = typename VEC::value_type;
        constexpr std::size_t N{ VEC::length() };
        if constexpr (N == 2) {
            return GLSL::Matrix2<T>(x[0] * y[0], x[0] * y[1],
                                    x[1] * y[0], x[1] * y[1]);
        }
        else if constexpr (N == 3) {
            return GLSL::Matrix3<T>(x[0] * y[0], x[0] * y[1], x[0] * y[2],
                                    x[1] * y[0], x[1] * y[1], x[1] * y[2],
                                    x[2] * y[0], x[2] * y[1], x[2] * y[2]);
        }
        else {
            return GLSL::Matrix4<T>(x[0] * y[0], x[0] * y[1], x[0] * y[2], x[0] * y[3],
                                    x[1] * y[0], x[1] * y[1], x[1] * y[2], x[1] * y[3],
                                    x[2] * y[0], x[2] * y[1], x[2] * y[2], x[2] * y[3],
                                    x[3] * y[0], x[3] * y[1], x[3] * y[2], x[3] * y[3]);
        }
    }

    /**
    * \brief generate look-at matrix (3x3)
    * @param {Vector3,        in}  origin
    * @param {Vector3,        in}  target
    * @param {floating_point, in}  roll angle [rad]
    * @param {Matrix3,        out} look at matrix
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr GLSL::Matrix3<T> look_at_matrix(const GLSL::Vector3<T>& origin, const GLSL::Vector3<T>& target, const T roll) {
        const GLSL::Vector3<T> rr(std::sin(roll), std::cos(roll), T{});
        const GLSL::Vector3<T> ww{ GLSL::normalize(target - origin) };
        const GLSL::Vector3<T> uu{ GLSL::normalize(GLSL::cross(ww, rr)) };
        const GLSL::Vector3<T> vv{ GLSL::normalize(GLSL::cross(uu, ww)) };

        return GLSL::Matrix3<T>(uu, vv, ww);
    }

    /**
    * \brief generate look-at matrix (3x3 or 4x4)
    * @param {Vector3,         in}  origin/eye
    * @param {Vector3,         in}  target
    * @param {Vector3,         in}  up direction (normalized)
    * @param {Matrix3|Matrix4, out} look at matrix
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr GLSL::Matrix3<T> look_at_matrix(const GLSL::Vector3<T>& origin, const GLSL::Vector3<T>& target, const GLSL::Vector3<T>& up) {
        assert(Numerics::areEquals(GLSL::length(up), static_cast<T>(1)));

        const GLSL::Vector3<T> Z{ -GLSL::normalize(target - origin) };
        const GLSL::Vector3<T> X{ -GLSL::cross(up, Z) };
        const GLSL::Vector3<T> Y{  GLSL::cross(Z, X) };

        return GLSL::Matrix3<T>(X, Y, Z);
    }

    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr GLSL::Matrix4<T> look_at_matrix(const GLSL::Vector3<T>& origin, const GLSL::Vector3<T>& target, const GLSL::Vector3<T>& up) {
        assert(Numerics::areEquals(GLSL::length(up), static_cast<T>(1)));

        const GLSL::Vector3<T> Z{ -GLSL::normalize(target - origin) };
        const GLSL::Vector3<T> X{ -GLSL::cross(up, Z) };
        const GLSL::Vector3<T> Y{  GLSL::cross(Z, X) };

        return GLSL::Matrix4<T>(X.x, Y.x, Z.x,  GLSL::dot(X, origin),
                                X.y, Y.y, Z.y, -GLSL::dot(Y, origin),
                                X.z, Y.z, Z.z,  GLSL::dot(Z, origin),
                                T{}, T{}, T{},  static_cast<T>(1));
    }

    /**
    * \brief generate a DCM (direct cosine matrix) matrix from Euler angles
    * @param {Vector3, in}  Euler angles
    * @param {Matrix3, out} DC, matrix
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr GLSL::Matrix3<T> rotation_matrix_from_euler(const GLSL::Vector3<T>& euler) noexcept {
        const T a{ std::sin(euler.x) };
        const T c{ std::sin(euler.y) };
        const T e{ std::sin(euler.z) };
        const T b{ std::cos(euler.x) };
        const T d{ std::cos(euler.y) };
        const T f{ std::cos(euler.z) };
        const T ac{ a * c };
        const T bc{ b * c };
        return GLSL::Matrix3<T>(d * f, d * e, -c,
                                ac * f - b * e, ac * e + b * f, a * d,
                                bc * f + a * e, bc * e - a * f, b * d);
    }

    /**
    * \brief orient (rotate) a given vector towards a given direction
    * @param {Vector2|Vector3, in}  vector to orient
    * @param {Vector2|Vector3, in}  direction (must be normalized)
    * @param {Vector2|Vector3, out} vector oriented according to direction
    **/
    template<GLSL::IFixedVector VEC>
        requires((VEC::length() == 2) || (VEC::length() == 3))
    constexpr VEC orientate(const VEC& v, const VEC& dir) noexcept {
        using T = typename VEC::value_type;
        assert(Numerics::areEquals(GLSL::length(dir), static_cast<T>(1)));

        VEC res(v);
        if (const T kk{ GLSL::dot(dir, v) }; kk > T{}) {
            res -= static_cast<T>(2) * dir * kk;
        }

        return res;
    }

    /**
    * \brief create a plane going through 3 points
    * @param {Vector3, in}  point #0
    * @param {Vector3, in}  point #1
    * @param {Vector3, in}  point #2
    * @param {Vector4, out} plane {x, y, z, n}
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr GLSL::Vector4<T> create_plane(const GLSL::Vector3<T>& a, const GLSL::Vector3<T>& b, const GLSL::Vector3<T>& c) {
        assert(!Numerics::areEquals(GLSL::dot(b - a), T{}));
        assert(!Numerics::areEquals(GLSL::dot(c - a), T{}));
        const GLSL::Vector3<T> n(GLSL::normalize(GLSL::cross(b - a, c - a)));
        return GLSL::Vector4<T>(n, -GLSL::dot(a, n));
    }

    /**
    * \brief generate a DCM (direct cosine matrix) matrix from axis and rotation angle
    * @param {Vector3,        in}  axis
    * @param {floating_point, in}  angle [rad]
    * @param {Matrix3,        out} DCM (direct cosine matrix) matrix
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr GLSL::Matrix3<T> rotation_matrix_from_axis_angle(const GLSL::Vector3<T>& a, const T t) noexcept {
        const T sint{ std::sin(t) };
        const T cost{ std::cos(t) };
        const T icost{ static_cast<T>(1) - cost };

        return GLSL::Matrix3<T>(a.x * a.x * icost + cost, a.y * a.x * icost - sint * a.z, a.z * a.x * icost + sint * a.y,
                                a.x * a.y * icost + sint * a.z, a.y * a.y * icost + cost, a.z * a.y * icost - sint * a.x,
                                a.x * a.z * icost - sint * a.y, a.y * a.z * icost + sint * a.x, a.z * a.z * icost + cost);
    }

    /**
    * \brief returns a point rotated around an axis. the rotation is centered
    *        around the origin. (uses optimized version of rodrigues rotation formula)
    * @param {Vector3,        in}  point
    * @param {Vector3,        in}  axis (should be normalized)
    * @param {floating_point, in}  angle [rad]
    * @param {Vector3,        out} rotated point
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr GLSL::Vector3<T> rotate_point_around_axis(const GLSL::Vector3<T>& p, const GLSL::Vector3<T>& axis, const T angle) noexcept {
        assert(Numerics::areEquals(GLSL::length(axis), static_cast<T>(1)));
        return GLSL::mix(GLSL::dot(axis, p) * axis, p, std::cos(angle)) + GLSL::cross(axis, p) * std::sin(angle);
    }

    /**
    * \brief return a orthonormalized matrix (using Gram-Schmidt process)
    * @param  {IFixedCubicMatrix, in}  matrix
    * @return {IFixedCubicMatrix, out} orthonormalized matrix
    **/
    template<GLSL::IFixedCubicMatrix MAT>
    constexpr MAT orthonormalize(const MAT& mat) {
        using T = typename MAT::value_type;

        MAT out;
        Utilities::static_for<0, 1, MAT::columns()>([&out, &mat](std::size_t i) {
            out[i] = mat[i];

            for (std::size_t j{}; j < i; ++j) {
                assert(!Numerics::areEquals(GLSL::dot(out[j]), T{}));
                out[i] -= out[j] * (GLSL::dot(out[i], out[j]) / GLSL::dot(out[j]));
            }

            out[i] = GLSL::normalize(out[i]);
        });

        return out;
    }

    /**
    * \brief return an orthonormal basis for given input vector
    * @param {Vector3, in}  vector (normalized)
    * @param {Matrix3, out} orthonormal basis
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr GLSL::Matrix3<T> orthonomrmalBasis(const GLSL::Vector3<T>& u) noexcept {
        assert(Numerics::areEquals(GLSL::length(u), static_cast<T>(1)));

        GLSL::Vector3<T> v{ [&u]() {
            const GLSL::Vector3<T> t{ GLSL::abs(u) };

            // x <= y && x <= z
            if ((t.x <= t.y) && (t.x <= t.z)) {
                return GLSL::normalize(GLSL::Vector3<T>({ T{}, -u.z, u.y }));
            } // y <= x && y <= z
            else if ((t.y <= t.x) && (t.y <= t.z)) {
                return GLSL::normalize(GLSL::Vector3<T>({ -u.z, T{}, u.x }));
            } // z <= x && z <= y
            else {
                return GLSL::normalize(GLSL::Vector3<T>({ -u.y, u.x, T{} }));
            }
        }() };

        const GLSL::Vector3<T> w{ GLSL::cross(u, v) };

        return GLSL::Matrix3<T>(u, v, w);
    }
}
