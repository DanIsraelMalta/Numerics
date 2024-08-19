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
    * \brief given a vector - return its companion matrix
    * @param {IFixedVector,      in}     vector 
    * @param {IFixedCubicMatrix, in|out} vectorc ompanion matrix
    **/
    template<GLSL::IFixedCubicMatrix MAT, class VEC = MAT::vector_type>
    constexpr void make_companion(MAT& mat, const VEC& vec) noexcept {
        using T = typename MAT::value_type;
        constexpr std::size_t N{ MAT::columns() };

        Utilities::static_for<0, 1, N - 1>([&mat](std::size_t i) {
            mat(i, i + 1) = static_cast<T>(1);
        });
        mat[N - 1] = -vec;
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
    * \brief test if two vector are numerically equal
    * @param {IFixedVector, in}  vector #1
    * @param {IFixedVector, in}  vector #2
    * @param {value_type,   in}  allowed tolerance for vector elements (default is 1e-5)
    * @param {bool,         out} true if vectors are identical
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
    constexpr bool are_vectors_identical(const VEC& a, const VEC& b, const T tol = static_cast<T>(1e-5)) noexcept {
        bool identical{ true };
        Utilities::static_for<0, 1, VEC::length()>([&a, &b, &identical, tol](std::size_t i) {
            identical &= std::abs(a[i] - b[i]) <= tol;
        });
        return identical;
    }

    /**
    * \brief tests if a given matrix is symmetric
    * @param {IFixedCubicMatrix, in}  matrix which will be testd for symmetry
    * @param {bool,              out} true if matrix is symmetrical
    **/
    template<GLSL::IFixedCubicMatrix MAT, class T = typename MAT::value_type>
    constexpr bool is_symmetric(const MAT& mat, const T tol = static_cast<T>(1e-5)) noexcept {
        bool symmetric{ true };
        Utilities::static_for<0, 1, MAT::columns()>([&mat, &symmetric, tol](std::size_t i) {
            Utilities::static_for<0, 1, MAT::columns()>([&mat, &symmetric, tol, i](std::size_t j) {
                symmetric &= mat(i, j) - mat(j, i) <= tol;
            });
        });
        return symmetric;
    }

    /**
    * \brief tests if a given 3x3 matrix is direct cosine matrix (DCM)
    * @param {IFixedCubicMatrix, in}  3x3 matrix
    * @param {bool,              out} true if matrix is DCM
    **/
    template<GLSL::IFixedCubicMatrix MAT, class T = typename MAT::value_type>
        requires(MAT::columns() == 3)
    constexpr bool is_dcm_matrix(const MAT& mat) noexcept {
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
    * @param {value_type, out} left dot product between x and y
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
    * @param {value_type, out} left dot product of vector
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

    /**
    * \brief given quaternion, return its rotation angle
    * @param {Vector4,    in}  quaternion
    * @param {value_type, out} quaternion angle
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr T get_quaternion_angle(const GLSL::Vector4<T>& quat) {
        assert(Extra::is_normalized(quat));
        return static_cast<T>(2) * std::acos(quat.w);
    }

    /**
    * \brief given quaternion (normalized), return its rotation angle
    * @param {Vector4, in}  quaternion (normalized)
    * @param {Vector3, out} quaternion axis
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr GLSL::Vector3<T> get_quaternion_axis(const GLSL::Vector4<T>& quat) {
        assert(Extra::is_normalized(quat));
        return quat.xyz;
    }

    /**
    * \brief return the conjugate of a given quaternion (since quaternion is normalized, its also the inverse)
    * @param {Vector4, in}  quaternion (normalized)
    * @param {Vector4, out} quaternion conjugate/inverse
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr GLSL::Vector4<T> get_quaternion_conjugate(const GLSL::Vector4<T>& quat) {
        return GLSL::Vector4<T>(-quat.x, -quat.y, -quat.z, quat.w);
    }

    /**
    * \brief return the cross product of two vectors or quaternions.
    *        2D operator is based on wedge operator from geometric algebra.
    * @param {Vector4, in}  quaternion
    * @param {Vector4, in}  quaternion
    * @param {Vector4, out} product of two quaternions
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto multiply_quaternions(const GLSL::Vector4<T>& x, const GLSL::Vector4<T>& y) noexcept {
        return GLSL::Vector4<T>(x[0] * y[0] - x[1] * y[1] - x[2] * y[2] - x[3] * y[3],
                                x[0] * y[1] + x[1] * y[0] - x[2] * y[3] + x[3] * y[2],
                                x[0] * y[2] + x[1] * y[3] + x[2] * y[0] - x[3] * y[1],
                                x[0] * y[3] - x[1] * y[2] + x[2] * y[1] + x[3] * y[0]);
    }
}
