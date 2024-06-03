#pragma once
#include "Glsl.h"
#include "Glsl_extra.h"

//
// spatial relatied operations using vectors and matrices
//

namespace Transformation {

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
    * \brief generate look-at matrix (4x4)
    * @param {Vector3,                in}  origin/eye
    * @param {Vector3,                in}  target/center
    * @param {Vector3|floating_point, in}  either world up direction (normalized) or roll angle [rad]
    * @param {Matrix3,                out} look at matrix
    **/
    template<typename T, typename U>
        requires(std::is_floating_point_v<T> && (std::is_same_v<T, U> || GLSL::is_fixed_vector_v<U>))
    constexpr GLSL::Matrix3<T> create_look_at_matrix(const GLSL::Vector3<T>& origin, const GLSL::Vector3<T>& target, const U roll_or_world_up) {
        const GLSL::Vector3<T> arbitrary = [roll_or_world_up]() {
            if constexpr (GLSL::is_fixed_vector_v<U>) {
				assert(Extra::is_normalized(roll_or_world_up));
                return roll_or_world_up;
            } else {
                return GLSL::Vector3<T>(std::sin(roll_or_world_up), std::cos(roll_or_world_up), T{});
            }
        }();

        const GLSL::Vector3<T> forward{ GLSL::normalize(origin - target) };
        const GLSL::Vector3<T> right{ GLSL::normalize(GLSL::cross(arbitrary, forward)) };
        const GLSL::Vector3<T> upper{ GLSL::cross(forward, right) };

        return GLSL::Matrix3<T>(right, upper, forward);

    }

    /**
    * \brief return rotation axis and rotation angle of a rotation matrix.
    *        only works for non symmetric rotation matrices.
    * @param {Matrix3,               in}  rotation matrix (not symmetric)
    * @param {{Vector3, arithmetic}, out} {rotation axis, rotation angle [rad] cosine }
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto get_axis_angle_from_rotation_matrix(const GLSL::Matrix3<T>& mat) {
        using out_t = struct { GLSL::Vector3<T> axis; T cosine; };

        assert(Extra::is_dcm_matrix(mat));
        assert(!Extra::is_symmetric(mat));

        const GLSL::Vector3<T> axis(mat(1, 2) - mat(2, 1), mat(2, 0) - mat(0, 2), mat(0, 1) - mat(1, 0));
        const T cosine{ (mat(0,0) + mat(1,1) + mat(2,2) - static_cast<T>(1)) / static_cast<T>(2) };
        return out_t{ GLSL::normalize(axis), cosine };

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
        assert(Extra::is_normalized(dir));

        VEC res(v);
        if (const T kk{ GLSL::dot(dir, v) }; kk > T{}) {
            res -= static_cast<T>(2) * dir * kk;
        }

        return res;
    }

    /**
    * \brief generate a DCM (direct cosine matrix) matrix from axis and rotation angle
    * @param {Vector3,        in}  axis
    * @param {floating_point, in}  angle [rad]
    * @param {Matrix3,        out} DCM (direct cosine matrix) matrix
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr GLSL::Matrix3<T> rotation_matrix_from_axis_angle(const GLSL::Vector3<T>& axis, const T angle) noexcept {
        const T sint{ std::sin(angle) };
        const T cost{ std::cos(angle) };
        const T icost{ static_cast<T>(1) - cost };
        const T axy{ axis.x * axis.y * icost };
        const T axz{ axis.x * axis.z * icost };
        const T ayz{ axis.y * axis.z * icost };
        const T sax{ sint * axis.x };
        const T say{ sint * axis.y };
        const T saz{ sint * axis.z };

        return GLSL::Matrix3<T>(axis.x * axis.x * icost + cost, axy - saz,                      axz + say,
                                axy + saz,                      axis.y * axis.y * icost + cost, ayz - sax,
                                axz - say,                      ayz + sax,                      axis.z * axis.z * icost + cost);
    }
    template<auto angle, class T = decltype(angle)>
        requires(std::is_floating_point_v<T>)
    constexpr GLSL::Matrix3<T> rotation_matrix_from_axis_angle(const GLSL::Vector3<T>& axis) noexcept {
        constexpr T sint{ std::sin(angle) };
        constexpr T cost{ std::cos(angle) };
        constexpr T icost{ static_cast<T>(1) - cost };
        const T axy{ axis.x * axis.y * icost };
        const T axz{ axis.x * axis.z * icost };
        const T ayz{ axis.y * axis.z * icost };
        const T sax{ sint * axis.x };
        const T say{ sint * axis.y };
        const T saz{ sint * axis.z };

        return GLSL::Matrix3<T>(axis.x * axis.x * icost + cost, axy - saz,                      axz + say,
                                axy + saz,                      axis.y * axis.y * icost + cost, ayz - sax,
                                axz - say,                      ayz + sax,                      axis.z * axis.z * icost + cost);
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
        assert(Extra::is_normalized(axis));
        return GLSL::mix(GLSL::dot(axis, p) * axis, p, std::cos(angle)) + GLSL::cross(axis, p) * std::sin(angle);
    }
    template<auto angle, class T = decltype(angle)>
        requires(std::is_floating_point_v<T>)
    constexpr GLSL::Vector3<T> rotate_point_around_axis(const GLSL::Vector3<T>& p, const GLSL::Vector3<T>& axis) noexcept {
        constexpr T sinAngle{ std::sin(angle) };
        constexpr T cosAngle{ std::cos(angle) };
        assert(Extra::is_normalized(axis));
        return GLSL::mix(GLSL::dot(axis, p) * axis, p, cosAngle) + GLSL::cross(axis, p) * sinAngle;
    }

}
