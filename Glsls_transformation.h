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
#include "Glsl_extra.h"

//
// spatial related operations using vectors and matrices
//

namespace Transformation {
   
    /**
    * \brief generate look-at matrix (3x3)
    * @param {IFixedVector,            in}  origin/eye
    * @param {IFixedVector,            in}  target/center
    * @param {IFixedVector|value_type, in}  either world up direction (normalized) or roll angle [rad]
    * @param {IFixedCubicMatrix,       out} look at matrix
    **/
    template<GLSL::IFixedVector VEC, class U, class T = typename VEC::value_type,
             class MAT = appropriate_matrix_type<VEC>::matrix_type>
        requires(VEC::length() <= 3 && std::is_floating_point_v<T> &&
                 (std::is_same_v<T, U> || GLSL::is_fixed_vector_v<U>))
    constexpr MAT create_look_at_matrix(const VEC& origin, const VEC& target, const U roll_or_world_up) {
        const VEC arbitrary = [roll_or_world_up]() {
            if constexpr (GLSL::is_fixed_vector_v<U>) {
                assert(Extra::is_normalized(roll_or_world_up));
                return roll_or_world_up;
            } else {
                return VEC(std::sin(roll_or_world_up), std::cos(roll_or_world_up), T{});
            }
        }();

        const VEC forward{ GLSL::normalize(origin - target) };
        const VEC right{ GLSL::normalize(GLSL::cross(arbitrary, forward)) };
        const VEC upper{ GLSL::cross(forward, right) };

        return MAT(right, upper, forward);
    }

    /**
    * \brief return rotation axis and rotation angle of a rotation matrix.
    *        only works for non symmetric rotation matrices.
    * @param {IFixedCubicMatrix,          in}  rotation matrix (not symmetric)
    * @param {{IFixedVector, value_type}, out} {rotation axis, rotation angle [rad] cosine }
    **/
    template<GLSL::IFixedCubicMatrix MAT>
        requires(MAT::columns() == 3)
    constexpr auto get_axis_angle_from_rotation_matrix(const MAT& mat) {
        using VEC = typename MAT::vector_type;
        using T = typename MAT::value_type;
        using out_t = struct { VEC axis; T cosine; };

        assert(Extra::is_orthonormal_matrix(mat));
        assert(!Extra::is_symmetric(mat));

        const VEC axis(mat(1, 2) - mat(2, 1), mat(2, 0) - mat(0, 2), mat(0, 1) - mat(1, 0));
        const T cosine{ (mat(0,0) + mat(1,1) + mat(2,2) - static_cast<T>(1)) / static_cast<T>(2) };
        return out_t{ GLSL::normalize(axis), cosine };

    }

    /**
    * \brief orient (rotate) a given vector towards a given direction
    * @param {IFixedVector, in}  vector to orient
    * @param {IFixedVector, in}  direction (must be normalized)
    * @param {IFixedVector, out} vector oriented according to direction
    **/
    template<GLSL::IFixedVector VEC>
        requires(VEC::length() <= 3)
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
    * @param {IFixedVector,      in}  axis
    * @param {value_type,        in}  angle [rad]
    * @param {IFixedCubicMatrix, out} DCM (direct cosine matrix) matrix
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type,
             class MAT = appropriate_matrix_type<VEC>::matrix_type>
        requires(VEC::length() <= 3 && std::is_floating_point_v<T>)
    constexpr MAT rotation_matrix_from_axis_angle(const VEC& axis, const T angle) noexcept {
        assert(Extra::is_normalized(axis));
        const T sint{ std::sin(angle) };
        const T cost{ std::cos(angle) };
        const T icost{ static_cast<T>(1) - cost };
        const T axy{ axis.x * axis.y * icost };
        const T axz{ axis.x * axis.z * icost };
        const T ayz{ axis.y * axis.z * icost };
        const T sax{ sint * axis.x };
        const T say{ sint * axis.y };
        const T saz{ sint * axis.z };

        return MAT(axis.x * axis.x * icost + cost, axy - saz,                      axz + say,
                   axy + saz,                      axis.y * axis.y * icost + cost, ayz - sax,
                   axz - say,                      ayz + sax,                      axis.z * axis.z * icost + cost);
    }
    template<auto angle, GLSL::IFixedVector VEC, class T = typename VEC::value_type,
             class MAT = appropriate_matrix_type<VEC>::matrix_type>
        requires(VEC::length() <= 3 && std::is_floating_point_v<T> && std::is_same_v<T, decltype(angle)>)
    constexpr MAT rotation_matrix_from_axis_angle(const VEC& axis) noexcept {
        assert(Extra::is_normalized(axis));
        constexpr T sint{ std::sin(angle) };
        constexpr T cost{ std::cos(angle) };
        constexpr T icost{ static_cast<T>(1) - cost };
        const T axy{ axis.x * axis.y * icost };
        const T axz{ axis.x * axis.z * icost };
        const T ayz{ axis.y * axis.z * icost };
        const T sax{ sint * axis.x };
        const T say{ sint * axis.y };
        const T saz{ sint * axis.z };

        return MAT(axis.x * axis.x * icost + cost, axy - saz,                      axz + say,
                   axy + saz,                      axis.y * axis.y * icost + cost, ayz - sax,
                   axz - say,                      ayz + sax,                      axis.z * axis.z * icost + cost);
    }

    /**
    * \brief given axis and angle, return quaternion
    * @param {IFixedVector, in}  axis (should be normalized)
    * @param {value_type,   in}  angle [rad]
    * @param {IFixedVector, out} quaternion
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(VEC::length() == 3)
    constexpr VEC rotate_point_around_axis(const VEC& p, const VEC& axis, const T angle) noexcept {
        assert(Extra::is_normalized(axis));
        return GLSL::mix(GLSL::dot(axis, p) * axis, p, std::cos(angle)) + GLSL::cross(axis, p) * std::sin(angle);
    }
    template<auto angle, GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(VEC::length() == 3 && std::is_floating_point_v<T> && std::is_same_v<T, decltype(angle)>)
    constexpr VEC rotate_point_around_axis(const VEC& p, const VEC& axis) noexcept {
        constexpr T sinAngle{ std::sin(angle) };
        constexpr T cosAngle{ std::cos(angle) };
        assert(Extra::is_normalized(axis));
        return GLSL::mix(GLSL::dot(axis, p) * axis, p, cosAngle) + GLSL::cross(axis, p) * sinAngle;
    }

    /**
    * \brief given axis and angle, return quaternion
    * @param {IFixedVector, in}  axis (should be normalized)
    * @param {value_type,   in}  angle [rad]
    * @param {IFixedVector, out} quaternion
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type,
             class out_t = next_vector_type<VEC>::vector_type>
        requires(VEC::length() == 3)
    constexpr out_t create_quaternion_from_axis_angle(const VEC& axis, const T angle) {
        assert(Extra::is_normalized(axis));
        const T halfAngle{ angle / static_cast<T>(2.0) };
        return out_t(std::sin(halfAngle) * axis, std::cos(halfAngle));
    }
    template<auto angle, GLSL::IFixedVector VEC, class T = typename VEC::value_type,
             class out_t = next_vector_type<VEC>::vector_type>
        requires(VEC::length() == 3 && std::is_floating_point_v<T> && std::is_same_v<T, decltype(angle)>)
    constexpr out_t create_quaternion_from_axis_angle(const VEC& axis) {
        assert(Extra::is_normalized(axis));
        constexpr T halfAngle{ angle / static_cast<T>(2.0) };
        constexpr T cosHalf{ std::cos(halfAngle) };
        constexpr T sinHalf{ std::sin(halfAngle) };
        return out_t(sinHalf * axis, cosHalf);
    }

    /**
    * \brief create a quaternion from DCM matrix
    * @param {IFixedCubicMatrix, in}  DCM matrix
    * @param {IFixedVector,      out} quaternion
    **/
    template<GLSL::IFixedCubicMatrix MAT, class VEC3 = typename MAT::vector_type,
             class VEC4 = next_vector_type<VEC3>::vector_type>
        requires(MAT::columns() == 3)
    constexpr VEC4 create_quaternion_from_rotation_matrix(const MAT& mat) {
        using T = typename MAT::value_type;
        assert(Extra::is_orthonormal_matrix(mat));

        const T tr{static_cast<T>(1) + mat(0,0) + mat(1,1) + mat(2,2)};

        // to avoid large numerical distortions (DCM trace shoule be positive)
        if (tr > T{}) {
            [[assume(tr > T{})]];
            const T S{ static_cast<T>(2) * std::sqrt(tr) };
            [[assume(S > T{})]];
            const T Sinv{ static_cast<T>(1) / S };

            return VEC4((mat(2,1) - mat(1,2)) * Sinv,
                        (mat(0,2) - mat(2,0)) * Sinv,
                        (mat(1,0) - mat(0,1)) * Sinv,
                        static_cast<T>(0.25) * S);
        } // DCM trace is zero, find largest diagonal element, and build quaternion using it
        else {
            // column 0
            if ((mat(0, 0) > mat(1, 1)) && (mat(0, 0) > mat(2, 2))) {
                const T S{ static_cast<T>(2) * std::sqrt(static_cast<T>(1) + mat(0, 0) - mat(1, 1) - mat(2, 2)) };
                [[assume(S > T{})]];
                const T Sinv{ static_cast<T>(1) / S };

                return VEC4(static_cast<T>(0.25) * S,
                            (mat(1, 0) + mat(0, 1)) * Sinv,
                            (mat(0, 2) + mat(2, 0)) * Sinv,
                            (mat(2, 1) - mat(1, 2)) * Sinv);
            }   // Column 1
            else if (mat(1, 1) > mat(2, 2))
            {
                const T S{ static_cast<T>(2) * std::sqrt(static_cast<T>(1) + mat(1, 1) - mat(0, 0) - mat(2, 2)) };
                [[assume(S > T{})]];
                const T Sinv{ static_cast<T>(1) / S };

                return VEC4((mat(1, 0) + mat(0, 1)) * Sinv,
                            static_cast<T>(0.25) * S,
                            (mat(2, 1) + mat(1, 2)) * Sinv,
                            (mat(0, 2) - mat(2, 0)) * Sinv);
            }
            else
            {   // Column 2
                const T S{ static_cast<T>(2) * std::sqrt(static_cast<T>(1) + mat(2, 2) - mat(0, 0) - mat(1, 1)) };
                [[assume(S > T{})]];
                const T Sinv{ static_cast<T>(1) / S };

                return VEC4((mat(0, 2) + mat(2, 0)) * Sinv,
                            (mat(2, 1) + mat(1, 2)) * Sinv,
                            static_cast<T>(0.25) * S,
                            (mat(1, 0) - mat(0, 1)) * Sinv);
            }
        }
    }

    /**
    * \brief create a quaternion from DCM matrix
    * @param {IFixedVector, in}  DCM matrix
    * @param {IFixedCubicMatrix, out} quaternion
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type,
             class VEC3 = prev_vector_type<VEC>::vector_type,
             class MAT = appropriate_matrix_type<VEC3>::matrix_type>
        requires(VEC::length() == 4 && std::is_floating_point_v<T>)
    constexpr MAT create_rotation_matrix_from_quaternion(const VEC& quat) {
        constexpr T one{ static_cast<T>(1) };
        constexpr T two{ static_cast<T>(2) };
        assert(Extra::is_normalized(quat));
        const T q0q1{ quat.x * quat.y };
        const T q0q2{ quat.x * quat.z };
        const T q1q2{ quat.y * quat.z };
        const T q3q0{ quat.w * quat.x };
        const T q3q1{ quat.w * quat.y };
        const T q3q2{ quat.w * quat.z };
        const T q0Sqr{ quat.x * quat.x };
        const T q1Sqr{ quat.y * quat.y };
        const T q2Sqr{ quat.z * quat.z };
        return MAT(one - two * (q2Sqr + q1Sqr), two * (q0q1 - q3q2),         two * (q0q2 + q3q1),
                   two * (q0q1 + q3q2),         one - two * (q0Sqr + q2Sqr), two * (q1q2 - q3q0),
                   two * (q0q2 - q3q1),         two * (q1q2 + q3q0),         one - two * (q0Sqr + q1Sqr) );
    }

    /**
    * \brief rotate a point using normalized quaternion
    * @param {IFixedVector, in}  quaternion (normalized)
    * @param {IFixedVector, in}  point to rotate
    * @param {IFixedVector, out} rotate point
    **/
    template<GLSL::IFixedVector VEC4, GLSL::IFixedVector VEC3, class T = typename VEC3::value_type>
        requires((VEC4::length() == 4) && (VEC3::length() == 3) && std::is_floating_point_v<T> &&
                 std::is_same_v<T, typename VEC4::value_type>)
    constexpr VEC3 rotate_point_using_quaternion(const VEC4& quat, const VEC3& point) {
        const VEC3 axis{ quat.xyz };
        const VEC3 temp{ static_cast<T>(2) * GLSL::cross(axis, point) };
        return (point + quat.w * temp + GLSL::cross(axis, temp));
    }
}
