#pragma once
#include "Glsl.h"

//
// general numerical utilities for vectors and matrices
//

namespace Extra {
    /**
    * \brief transform a matrix to identity matrix
    * @param {IFixedCubicMatrix, in|out} matrix which will be an identity matrix
    **/
    template<GLSL::IFixedCubicMatrix MAT>
    constexpr void make_identity(MAT& mat) noexcept {
        using T = typename MAT::value_type;
        constexpr std::size_t N{ MAT::columns() };

        mat = std::array<T, N * N>{{ T{} }};
        Utilities::static_for<0, 1, N>([&mat](std::size_t i) {
            mat(i, i) = static_cast<T>(1);
        });
    }

    /**
    * \brief transform a matrix to Van-Der-Monde matrix
    * @param {IFixedCubicMatrix, in|out} matrix which will be Van-Der-Monde matrix
    * @param {IFixedVector,      in}     vector holding base of powers of Van-Der-Monde matrix elements
    **/
    template<GLSL::IFixedCubicMatrix MAT, class VEC = MAT::vector_type>
    constexpr void make_van_der_monde(MAT& mat, const VEC& vec) noexcept {
        using T = typename MAT::value_type;
        constexpr std::size_t N{ MAT::columns() };

        Utilities::static_for<0, 1, N>([&mat](std::size_t i) {
            Utilities::static_for<0, 1, N>([&mat, i](std::size_t j) {
                const T power{static_cast<T>(N - j - i)};
                mat(i, j) = static_cast<T>(std::pow(vec[i], power));
            });
        });
    }

    /**
    * \brief transform a matrix to Toeplitz matrix
    * @param {IFixedCubicMatrix, in|out} matrix which will be Toeplitz matrix
    * @param {IFixedVector,      in}     vector holding values to fill matrix
    **/
    template<GLSL::IFixedCubicMatrix MAT, class VEC = MAT::vector_type>
    constexpr void make_toeplitz(MAT& mat, const VEC& vec) noexcept {
        using T = typename MAT::value_type;
        constexpr std::size_t N{ MAT::columns() };

        Utilities::static_for<0, 1, N>([&mat](std::size_t i) {
            Utilities::static_for<0, 1, N>([&mat, i](std::size_t j) {
                mat(i, j) = (i >= j) ? vec[i - j] : vec[j - i];
            });
        });
    }

    /**
    * \brief fill a vector with positive random values in the range of [0, static_cast<T>(std::numeric_limits<std::uint32_t>::max())]
    * @param {IFixedVector, in} vector to be filled with random values
    **/
    template<GLSL::IFixedVector VEC>
    constexpr void make_random(VEC& vec) noexcept {
        using T = typename VEC::value_type;
        constexpr std::uint32_t value = (__TIME__[7] - '0') * 1u +
                                        (__TIME__[6] - '0') * 10u +
                                        (__TIME__[4] - '0') * 60u +
                                        (__TIME__[3] - '0') * 600u +
                                        (__TIME__[1] - '0') * 3600u +
                                        (__TIME__[0] - '0') * 36000u;

        vec[0] = static_cast<T>(Hash::pcg(value));
        Utilities::static_for<1, 1, VEC::length()>([&vec](std::size_t i) {
            vec[i] = static_cast<T>(Hash::pcg(static_cast<std::uint32_t>(vec[i-1])));
        });
    }

    /**
    * \brief test if vector is normalized
    * @param {IFixedVector, in}  vector
    * @param {value_type,   in}  allowed tolerance for vector L2 norm from 1 (default is 1e-5)
    * @param {bool,         out} true if vector L2 norm is 1
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
    constexpr bool is_normalized(const VEC& vec, const T tol = static_cast<T>(1e-5)) noexcept {
        return std::abs(static_cast<T>(1) - GLSL::dot(vec)) <= tol;
    }

    /**
    * \brief tests if a given matrix is symmetric
    * @param {IFixedCubicMatrix, in}  matrix which will be testd for symmetry
    * @param {bool,              out} true if matrix is symmetrical
    **/
    template<GLSL::IFixedCubicMatrix MAT, class T = typename MAT::value_type>
    constexpr bool is_symmetric(const MAT& mat, const T tol = static_cast<T>(1e-5)) noexcept {
        bool symmetric{ true };
        Utilities::static_for<0, 1, 3>([&mat, &symmetric, tol](std::size_t i) {
            Utilities::static_for<0, 1, 3>([&mat, &symmetric, tol, i](std::size_t j) {
                symmetric &= mat(i, j) - mat(j, i) <= tol;
            });
        });
        return symmetric;
    }

    /**
    * \brief tests if a given 3x3 matrix is direct cosine matrix (DCM)
    * @param {Matrix3, in}  tested matrix
    * @param {bool,    out} true if matrix is DCM
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr bool is_dcm_matrix(const GLSL::Matrix3<T>& mat) noexcept {
        bool is_dcm{ true };
        is_dcm &= Extra::is_normalized(mat[0]);
        is_dcm &= Extra::is_normalized(mat[1]);
        is_dcm &= Extra::is_normalized(mat[2]);
        is_dcm &= Extra::is_normalized(GLSL::cross(mat[0], mat[1]));
        is_dcm &= Extra::is_normalized(GLSL::cross(mat[0], mat[2]));
        is_dcm &= Extra::is_normalized(GLSL::cross(mat[1], mat[2]));
        return is_dcm;
    }

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
    * \brief create a plane going through 3 points
    * @param {Vector3, in}  point #0
    * @param {Vector3, in}  point #1
    * @param {Vector3, in}  point #2
    * @param {Vector4, out} plane {normal x, normal y, normal z, distance}
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr GLSL::Vector4<T> create_plane(const GLSL::Vector3<T>& a, const GLSL::Vector3<T>& b, const GLSL::Vector3<T>& c) {
        const GLSL::Vector3<T> ba{ b - a };
        const GLSL::Vector3<T> ca{ c - a };
        assert(GLSL::dot(ba) > T{});
        assert(GLSL::dot(ca) > T{});
        const GLSL::Vector3<T> n(GLSL::normalize(GLSL::cross(ba, ca)));
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
        assert(Extra::is_normalized(u));

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

    /**
    * \brief return Manhattan distance between two vectors
    * @param  {IFixedVector, in}  x
    * @param  {IFixedVector, in}  y
    * @return {arithmetic,   out} Manhattan distance between x and y
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
    constexpr T manhattan_distance(const VEC& x, const VEC& y) noexcept {
        return GLSL::sum(GLSL::abs(x - y));
    }

    /**
    * \brief return Chebyshev distance between two vectors
    * @param  {IFixedVector, in}  x
    * @param  {IFixedVector, in}  y
    * @return {arithmetic,   out} Chebyshev distance between x and y
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
    constexpr T chebyshev_distance(const VEC& x, const VEC& y) noexcept {
        return GLSL::max(GLSL::abs(x - y));
    }

    /**
    * \brief return inverse Chebyshev distance between two vectors
    * @param  {IFixedVector, in}  x
    * @param  {IFixedVector, in}  y
    * @return {arithmetic,   out} inverse Chebyshev distance between x and y
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
    constexpr T inverse_chebyshev_distance(const VEC& x, const VEC& y) noexcept {
        return GLSL::min(GLSL::abs(x - y));
    }

    /**
    * \brief calculate "left looking" dot product between two vectors
    * @param {VEC,        in}  x
    * @param {VEC,        in}  y
    * @param {value_type, out} dot product between x and y
    **/
    template<std::size_t N, GLSL::IFixedVector VEC>
        requires(N <= VEC::length())
    constexpr VEC::value_type left_dot(const VEC& x, const VEC& y) noexcept {
        using T = typename VEC::value_type;
        T dot{};

        if constexpr (std::is_floating_point_v<T>) {
            Utilities::static_for<0, 1, N>([&dot, &x, &y](std::size_t i) {
                dot = std::fma(x[i], y[i], dot);
            });
        } else {
            Utilities::static_for<0, 1, N>([&dot, &x, &y](std::size_t i) {
                dot += x[i] * y[i];
            });
        }

        return dot;
    }

    /**
    * \brief calculate "left looking" dot product of vector
    * @param {VEC,        in}  vector
    * @param {value_type, out} dot product of vector
    **/
    template<std::size_t N, GLSL::IFixedVector VEC>
        requires(N <= VEC::length())
    constexpr VEC::value_type left_dot(const VEC& x) noexcept {
        using T = typename VEC::value_type;
        T dot{};

        if constexpr (std::is_floating_point_v<T>) {
            Utilities::static_for<0, 1, N>([&dot, &x](std::size_t i) {
                dot = std::fma(x[i], x[i], dot);
            });
        }
        else {
            Utilities::static_for<0, 1, N>([&dot, &x](std::size_t i) {
                dot += x[i] * x[i];
            });
        }

        return dot;
    }
}
