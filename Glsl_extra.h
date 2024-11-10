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

        Utilities::static_for<0, 1, N>([&mat, &vec](std::size_t i) {
            Utilities::static_for<0, 1, N>([&&mat, &vec, i](std::size_t j) {
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

        Utilities::static_for<0, 1, N>([&mat, &vec](std::size_t i) {
            Utilities::static_for<0, 1, N>([&mat, &vec, i](std::size_t j) {
                mat(i, j) = (i >= j) ? vec[i - j] : vec[j - i];
            });
        });
    }

    /**
    * \brief given a vector - return its companion matrix
    * @param {IFixedVector,      in}     vector 
    * @param {IFixedCubicMatrix, in|out} vector companion matrix
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
    * @param {IFixedCubicMatrix, in}  matrix which will be tested for symmetry
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
    * \brief tests if a given matrix is orthonormal
    * @param {IFixedCubicMatrix, in}  matrix
    * @param {bool,              out} true if matrix is orthonormal
    **/
    template<GLSL::IFixedCubicMatrix MAT, class T = typename MAT::value_type>
    constexpr bool is_orthonormal_matrix(const MAT& mat) noexcept {
        Utilities::static_for<0, 1, MAT::columns() - 1>([&mat](std::size_t i) {
            if (!Extra::is_normalized(mat[i]) ||
                !Extra::is_normalized(mat[i + 1]) ||
                !Numerics::areEquals(T{}, GLSL::dot(GLSL::normalize(mat[i]), GLSL::normalize(mat[i + 1])))) {
                return false;
            }
        });
        return true;
    }

    /**
    * \brief return the outer product of two vectors
    * @param  {IFixedVector,      in}  x
    * @param  {IFixedVector,      in}  y
    * @return {IFixedCubicMatrix, out} outer product between x and y
    **/
    template<GLSL::IFixedVector VEC, class MAT = appropriate_matrix_type<VEC>::matrix_type>
    constexpr MAT outer_product(const VEC& x, const VEC& y) noexcept {
        using T = typename VEC::value_type;
        constexpr std::size_t N{ VEC::length() };
        if constexpr (N == 2) {
            return MAT(x[0] * y[0], x[0] * y[1],
                       x[1] * y[0], x[1] * y[1]);
        }
        else if constexpr (N == 3) {
            return MAT(x[0] * y[0], x[0] * y[1], x[0] * y[2],
                       x[1] * y[0], x[1] * y[1], x[1] * y[2],
                       x[2] * y[0], x[2] * y[1], x[2] * y[2]);
        }
        else if constexpr (N == 4) {
            return MAT(x[0] * y[0], x[0] * y[1], x[0] * y[2], x[0] * y[3],
                       x[1] * y[0], x[1] * y[1], x[1] * y[2], x[1] * y[3],
                       x[2] * y[0], x[2] * y[1], x[2] * y[2], x[2] * y[3],
                       x[3] * y[0], x[3] * y[1], x[3] * y[2], x[3] * y[3]);
        }
        else {
            MAT out;

            Utilities::static_for<0, 1, N>([&out, &x, &y](std::size_t i) {
                Utilities::static_for<0, 1, N>([&out, &x, &y, i](std::size_t j) {
                    out(i, j) = x[j] * y[i];
                });
            });

            return out;
        }
    }

    /**
    * \brief return householder matrix of a given vector
    * @param  {IFixedVector,      in}  vector (normalized)
    * @return {IFixedCubicMatrix, out} householder matrix (reflection matrix about hyperplane with unit normal vector 'vec')
    **/
    template<GLSL::IFixedVector VEC>
    constexpr auto Householder(const VEC& vec) noexcept {
        using T = typename VEC::value_type;
        constexpr std::size_t N{ VEC::length() };
        constexpr T one{ static_cast<T>(1) };
        constexpr T two{ static_cast<T>(-2) };

        assert(Extra::is_normalized(vec));

        if constexpr (N == 2) {
            return GLSL::Matrix2<T>(one + two * vec[0] * vec[0],       two * vec[0] * vec[1],
                                          two * vec[1] * vec[0], one + two * vec[1] * vec[1]);
        }
        else if constexpr (N == 3) {
            return GLSL::Matrix3<T>(one + two * vec[0] * vec[0],       two * vec[0] * vec[1],       two * vec[0] * vec[2],
                                          two * vec[1] * vec[0], one + two * vec[1] * vec[1],       two * vec[1] * vec[2],
                                          two * vec[2] * vec[0],       two * vec[2] * vec[1], one + two * vec[2] * vec[2]);
        }
        else if constexpr (N == 4) {
            return GLSL::Matrix4<T>(one + two * vec[0] * vec[0],       two * vec[0] * vec[1],       two * vec[0] * vec[2],       two * vec[0] * vec[3],
                                          two * vec[1] * vec[0], one + two * vec[1] * vec[1],       two * vec[1] * vec[2],       two * vec[1] * vec[3],
                                          two * vec[2] * vec[0],       two * vec[2] * vec[1], one + two * vec[2] * vec[2],       two * vec[2] * vec[3],
                                          two * vec[3] * vec[0],       two * vec[3] * vec[1],       two * vec[3] * vec[2], one + two * vec[3] * vec[3]);
        }
        else {
            GLSL::MatrixN<T, N> out;
            Extra::make_identity(out);
            const GLSL::MatrixN<T, N> reflection_matrix{ Extra::outer_product(vec, vec) };
            return (out + reflection_matrix * two);
        }
    }

    /**
    * \brief perform generalized vector addition, i.e. - return alp?a * x + y
    *        where alpha is a scalar, x and y are vectors.
    *        this is BLAS level 1 function.
    * @param{value_type,        in}  alpha
    * @param{IFixedVector,      in}  x
    * @param{IFixedVector,      in}  y
    * @param{IFixedVector,      out} alp?a * x + y
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
    constexpr VEC axpy(const T alpha, const VEC& x, const VEC& y) {
        constexpr std::size_t N{ VEC::length() };

        // perform generalized vector addition
        VEC out;
        Utilities::static_for<0, 1, N>([&out, &x, &y, alpha](std::size_t i) {
            out[i] = std::fma(alpha, x[i], y[i]);
        });
        return out;
    }

    /**
    * \brief perform generalized matrix-vector multiplication, i.e. - return alp?a * M * x + beta * y
    *        where alph and beta are scalars, x and y are vectors and M is a matrix.
    *        this is BLAS level 2 function.
    * @param{value_type,        in}  alpha
    * @param{IFixedCubicMatrix, in}  M
    * @param{IFixedVector,      in}  x
    * @param{value_type,        in}  beta
    * @param{IFixedVector,      in}  y
    * @param{IFixedVector,      out} alp?a * M * x + beta * y
    **/
    template<GLSL::IFixedCubicMatrix MAT, class VEC = typename MAT::value_type, class T = typename MAT::value_type>
    constexpr VEC gemv(const T alpha, const MAT& A, const VEC& x, const T beta, const VEC& y) {
        constexpr std::size_t N{ MAT::columns() };

        // transpose matrix A so we calculate x*M' instead of M*x
        const MAT AT(GLSL::transpose(A));

        // perform generalized matrix-vector multiplication
        VEC out;
        Utilities::static_for<0, 1, N>([&out, &AT, &x, &y, alpha, beta](std::size_t i) {
            out[i] = std::fma(alpha, GLSL::dot(x, AT[i]),  beta * y[i]);
        });
        return out;
    }

    /**
    * \brief perform generalized matrix-matrix multiplication, i.e. - return alp?a * A * B + beta * C
    *        where alph and beta are scalars, A and B and C are matrices.
    *        this is BLAS level 3 function.
    * @param{value_type,        in}  alpha
    * @param{IFixedCubicMatrix, in}  A
    * @param{IFixedCubicMatrix, in}  B
    * @param{value_type,        in}  beta
    * @param{IFixedCubicMatrix, in}  C
    * @param{IFixedVector,      out} alpha * A * B + beta * C
    **/
    template<GLSL::IFixedCubicMatrix MAT, class T = typename MAT::value_type>
    constexpr MAT gemm(const T alpha, const MAT& A, const MAT& B, const T beta, const MAT& C) {
        using VEC = typename MAT::value_type;
        constexpr std::size_t N{ MAT::columns() };

        // transpose matrix A so we calculate B*A instead of A*B
        const MAT AT(GLSL::transpose(A));

        // perform generalized matrix-matrix multiplication
        MAT out;
        Utilities::static_for<0, 1, N>([&out, &B, &AT, &C, alpha, beta](std::size_t i) {
            Utilities::static_for<0, 1, N>([&out, &B, &AT, &C, alpha, beta, i](std::size_t j) {
                out(i, j) = std::fma(alpha, GLSL::dot(B[i], AT[j]), beta * C(i, j));
            });
        });
        return out;
    }

    /**
    * \brief return the convolution between two vectors.
    * @param {IFixedVector, in}  first vector
    * @param {IFixedVector, in}  second vector
    * @param {IFixedVector, out} vector holding the convolution between first and second collections
    **/
    template<GLSL::IFixedVector VEC1, GLSL::IFixedVector VEC2>
        requires(std::is_same_v<typename VEC1::value_type, typename VEC1::value_type>)
    constexpr auto conv(const VEC1& u, const VEC2& v) {
        constexpr std::size_t N1{ VEC1::length() };
        constexpr std::size_t N2{ VEC2::length() };
        constexpr std::size_t M{ N1 + N2 - 1 };
        using T = typename VEC1::value_type;
        using out_t = GLSL::VectorN<T, M>;

        out_t out;
        for (std::size_t i{}; i < M; ++i) {
            T sum{};
            std::size_t iter{ i };
            for (std::size_t j{}; j <= i; ++j) {
                if ((j < N1) && (iter < N2)) {
                    sum += u[j] * v[iter];
                }
                --iter;
            }
            out[i] = sum;
        }

        return out;
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

        const GLSL::Vector3<T> v{ [&u]() {
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
    * \brief return the mean and standard deviation of a collection of vectors.
    * @param {forward_iterator,         in}  iterator to first value
    * @param {forward_iterator,         in}  iterator to last value
    * @param {{arithmetic, arithmetic}, out} {mean, standard devialtion}
    **/
    template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(GLSL::is_fixed_vector_v<VEC>)
    constexpr auto mean_and_std(InputIt first, const InputIt last) {
        using T = typename VEC::value_type;
        using out_t = struct { VEC mean; VEC std; };
        VEC mean{};
        VEC std{};
        T count{};
        for (InputIt it{ first }; it != last; ++it) {
            ++count;
            const VEC delta{ *it - mean };
            mean += delta / count;
            const VEC delta2{ *it - mean };
            std += delta * delta2;
        }
        return out_t{ mean, std / count };
    }

    /**
    * \brief given an N-dimensional vector as coordinate in discretized world, world dimensions and cell size in world,
    *        return the index of that coordinate in a flatten vector which stores the world in row-major style.
    *        see 'index_to_vector' for the apposite operation.
    * @param {IFixedVector, in}  position in N dimensional space
    * @param {IFixedVector, in}  size of world along each coordinate (amount of cells along each coordinate)
    * @param {value_type,   in}  cell size in N dimensional space (1 by default)
    * @param {size_t,       out} index of position in a flatten world stored in row major style.
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
    constexpr std::size_t vector_to_index(const VEC& position, const VEC& world, const T cellSize = static_cast<T>(1)) {
        std::size_t index{};
        std::size_t mul{ 1 };
        Utilities::static_for<0, 1, VEC::length()>([&position, &world, &index, &mul, cellSize](std::size_t i) {
            const T pos{ position[i] / cellSize };
            [[assume(pos >= T{})]];
            index += static_cast<std::size_t>(pos) * mul;
            const T worldi{ world[i] };
            [[assume(worldi >= T{})]]
            mul *= static_cast<std::size_t>(worldi);
        });
        assert(index < GLSL::prod(world));
        return index;
    }

    /**
    * \brief given an index in N-dimensional vector flatten in row-major style, return its world coordinate.
    *        see 'vector_to_index' for the apposite operation.
    * @param {size_t,       in}  index of position in a flatten world stored in row major style.
    * @param {IFixedVector, in}  size of world along each coordinate (amount of cells along each coordinate)
    * @param {value_type,   in}  cell size in N dimensional space (1 by default)
    * @param {IFixedVector, out} position in N dimensional space
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
    constexpr VEC index_to_vector(std::size_t index, const VEC& world) {
        assert(index < GLSL::prod(world));
        VEC res;
        const T prod{ GLSL::prod(world) };
        [[assume(prod > T{})]];
        std::size_t mul{ static_cast<std::size_t>(prod) };
        Utilities::static_for<1, 1, VEC::length() + 1>([&world, &mul, &res, &index](std::size_t i) {
            const std::size_t j{ VEC::length() - i };
            const T world_j{ world[j] };
            [[assume(world_j > T{})]];
            mul /= static_cast<std::size_t>(world_j);
            res[j] = static_cast<T>(index / mul);
            assert(res[j] < world_j);
            index -= static_cast<std::size_t>(res[j]) * mul;
        });
        return res;
    }

    /**
    * \brief given quaternion, return its rotation angle
    * @param {IFixedVector,    in}  quaternion
    * @param {value_type, out} quaternion angle
    **/
    template<GLSL::IFixedVector VEC, class T = VEC::value_type>
        requires(VEC::length() == 4)
    constexpr T get_quaternion_angle(const VEC& quat) {
        assert(Extra::is_normalized(quat));
        return static_cast<T>(2) * std::acos(quat.w);
    }

    /**
    * \brief given quaternion (normalized), return its rotation angle
    * @param {IFixedVector, in}  quaternion (normalized)
    * @param {IFixedVector, out} quaternion axis
    **/
    template<GLSL::IFixedVector VEC, class out = prev_vector_type<VEC>::vector_type>
        requires(VEC::length() == 4)
    constexpr out get_quaternion_axis(const VEC& quat) {
        using T = typename VEC::value_type;
        assert(Extra::is_normalized(quat));
        const T num{ static_cast<T>(1) - quat.w * quat.w };
        [[assume(num > T{})]];
        const T s{ std::sqrt(num) };
        if (Numerics::areEquals(s, T{})) [[unlikely]] {
            return quat.xyz;
        }
        return (quat.xyz / s);
    }

    /**
    * \brief return the conjugate of a given quaternion (since quaternion is normalized, its also the inverse)
    * @param {IFixedVector, in}  quaternion (normalized)
    * @param {IFixedVector, out} quaternion conjugate/inverse
    **/
    template<GLSL::IFixedVector VEC>
        requires(VEC::length() == 4)
    constexpr VEC get_quaternion_conjugate(const VEC& quat) {
        return VEC(-quat.x, -quat.y, -quat.z, quat.w);
    }

    /**
    * \brief return the cross product of two vectors or quaternions.
    *        2D operator is based on wedge operator from geometric algebra.
    * @param {IFixedVector, in}  quaternion
    * @param {IFixedVector, in}  quaternion
    * @param {IFixedVector, out} product of two quaternions
    **/
    template<GLSL::IFixedVector VEC>
        requires(VEC::length() == 4)
    constexpr VEC multiply_quaternions(const VEC& x, const VEC& y) noexcept {
        return VEC(x[0] * y[0] - x[1] * y[1] - x[2] * y[2] - x[3] * y[3],
                   x[0] * y[1] + x[1] * y[0] - x[2] * y[3] + x[3] * y[2],
                   x[0] * y[2] + x[1] * y[3] + x[2] * y[0] - x[3] * y[1],
                   x[0] * y[3] - x[1] * y[2] + x[2] * y[1] + x[3] * y[0]);
    }
}
