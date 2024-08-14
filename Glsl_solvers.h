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
// matrix decompositions and eigenvalue related operations
// 

namespace Decomposition {

    /**
    * \brief return the eigenvalues of a 2x2 ot 3x3 matrix
    *
    * @param {IFixedCubicMatrix, in}  input matrix
    * @param {IFixedVector,      out} vector holding matrix eigenvalues
    **/
    template<GLSL::IFixedCubicMatrix MAT, class VEC = typename MAT::vector_type>
        requires(MAT::columns() <= 3)
    constexpr VEC eigenvalues(const MAT& mat) {
        using T = typename MAT::value_type;
        constexpr std::size_t N{ MAT::columns() };

        if constexpr (N == 2) {
            const T diff{ mat(0, 0) - mat(1, 1) };
            const T center{ mat(0, 0) + mat(1, 1) };
            const T deltaSquared{ diff * diff + static_cast<T>(4) * mat(1, 0) * mat(0, 1) };
            assert(deltaSquared >= T{});
            const T delta{ std::sqrt(deltaSquared) };

            return VEC( (center + delta) / static_cast<T>(2), (center - delta) / static_cast<T>(2) );
        }
        else {
            const T tr{ mat(0, 0) + mat(1, 1) + mat(2, 2) };
            const T det{ GLSL::determinant(mat) };
            const T cofSum{ mat(1, 1) * mat(2, 2) - mat(2, 1) * mat(1, 2) +
                            mat(0, 0) * mat(2, 2) - mat(2, 0) * mat(0, 2) +
                            mat(0, 0) * mat(1, 1) - mat(1, 0) * mat(0, 1) };
            const std::array<T, 6> roots{ Numerics::SolveCubic(-tr, cofSum, -det) };
            return VEC(roots[0], roots[2], roots[4]);
        }
    }

    /**
    * \brief perform QR decomposition using gram-schmidt process.
    *         numerically less accurate than QR_GivensRotation.
    *
    * @param {IFixedCubicMatrix,                      in}  input matrix
    * @param {{IFixedCubicMatrix, IFixedCubicMatrix}, out} {Q matrix (orthogonal matrix with orthogonal columns, i.e. - Q*Q^T = I),  R matrix (upper triangular matrix) }
    **/
    template<GLSL::IFixedCubicMatrix MAT>
    constexpr auto QR_GramSchmidt(const MAT& mat) {
        using out_t = struct { MAT Q; MAT R; };

        const MAT Q(Extra::orthonormalize(mat));

        MAT R;
        Utilities::static_for<0, 1, MAT::columns()>([&Q, &R, &mat](std::size_t i) {
            for (std::size_t j{ i }; j < MAT::columns(); ++j) {
                R(j, i) = GLSL::dot(mat[j], Q[i]);
            }
        });

        return out_t{ Q, R };
    }

    /**
    * \brief perform QR decomposition using "givens rotation".
    *        numerically more accurate than QR_GramSchmidt.
    *
    * @param {IFixedCubicMatrix, in}     matrix
    * @param {{IFixedCubicMatrix, IFixedCubicMatrix}, out} {Q matrix (orthogonal matrix with orthogonal columns, i.e. - Q*Q^T = I),  R matrix (upper triangular matrix) }
    **/
    template<GLSL::IFixedCubicMatrix MAT>
    constexpr auto QR_GivensRotation(const MAT& mat) {
        using out_t = struct { MAT Q; MAT R; };
        using T = typename MAT::value_type;
        constexpr std::size_t N{ MAT::columns() };
        constexpr T tol{ Numerics::equality_precision<T>() };

        // "givens rotation" - return {cosine, sine, radius}
        const auto givensRotation = [](const T a, const T b) {
            if (std::abs(a) <= tol) {
                return std::array<T, 3>{{T{}, Numerics::sign(b), std::abs(b)}};
            }
            else if (std::abs(b) <= tol) {
                return std::array<T, 3>{{Numerics::sign(a), T{}, std::abs(a)}};
            }
            else if (std::abs(a) > std::abs(b)) {
                [[assume(a != T{})]];
                const T t{ b / a };
                const T u{ Numerics::sign(a) * std::sqrt(t * t + static_cast<T>(1)) };
                [[assume(u != T{})]];
                const T c{ static_cast<T>(1) / u };
                return std::array<T, 3>{{c, t* c, a* u}};
            }
            else {
                [[assume(b != T{})]];
                const T t{ a / b };
                const T u{ Numerics::sign(b) * std::sqrt(t * t + static_cast<T>(1)) };
                [[assume(u != T{})]];
                const T s{ static_cast<T>(1) / u };
                return std::array<T, 3>{{t* s, s, b* u}};
            }
        };

        // decomposition
        MAT R(mat);
        MAT Q;
        Extra::make_identity(Q);
        for (std::size_t j{}; j < N; ++j) {
            for (std::size_t i{ N - 1 }; i >= j + 1; --i) {
                const GLSL::Vector3<T> CSR(givensRotation(R(j, i - 1), R(j, i)));

                // R' = G * R
                for (std::size_t x{}; x < N; ++x) {
                    const T temp1{ R(x, i - 1) };
                    const T temp2{ R(x, i) };
                    R(x, i - 1) = temp1 * CSR[0] + temp2 * CSR[1];
                    R(x, i) = -temp1 * CSR[1] + temp2 * CSR[0];
                }
                R(j, i - 1) = CSR[2];
                R(j, i) = T{};

                // Q' = Q * G^
                for (std::size_t x{}; x < N; ++x) {
                    const T temp1{ Q(i - 1, x) };
                    const T temp2{ Q(i,     x) };
                    Q(i - 1, x) = temp1 * CSR[0] + temp2 * CSR[1];
                    Q(i, x) = -temp1 * CSR[1] + temp2 * CSR[0];
                }
            }
        }

        return out_t{ Q, R };
    }

    /**
    * \brief given symmetric positive definite matrix A, constructs a lower triangular matrix L such that L*L' = A.
    *        (it's roughly TWICE as efficient as the LU decomposition)
    *
    * @param {IFixedCubicMatrix, in}     matrix
    * @param {IFixedCubicMatrix, in|out} lower triangular matrix
    **/
    template<GLSL::IFixedCubicMatrix MAT>
    constexpr MAT Cholesky(const MAT& mat) {
        using T = typename MAT::value_type;

        MAT lower;
        for (std::size_t j{}; j < MAT::columns(); ++j) {
            T d{};

            for (std::size_t k{}; k < j; ++k) {
                T s{};

                for (std::size_t i{}; i < k; ++i) {
                    s += lower(k, i) * lower(j, i);
                }

                assert(!Numerics::areEquals(lower(k, k), T{}));
                lower(j, k) = s = (mat(j, k) - s) / lower(k, k);
                d += s * s;
            }

            d = mat(j, j) - d;
            lower(j, j) = (d > T{}) ? std::sqrt(d) : T{};
        }

        return lower;
    }

    /**
    * \brief perform LU decomposition using using Doolittle algorithm.
    *        i.e. - given matrix decompose it to L*P*U, where L is lower triangular with unit diagonal,
    *               U is an upper triangular and P is a diagonal pivot matrix (given as a vector holding its diagonal)
    *        this implementation is geared towards easy usage in linear system solutions.
    * @param {IFixedCubicMatrix,                   in}  matrix to decompose
    * @param {{IFixedCubicMatrix, array, int32_t}, out} {LU matrix (decomposed matrix; upper triangular is U, lower triangular is L, diagonal is part of U), decomposition pivot vector, pivot sign}
    **/
    template<GLSL::IFixedCubicMatrix MAT>
    constexpr auto LU(const MAT& mat) {
        constexpr std::size_t N{ MAT::columns() };
        using T = typename MAT::value_type;
        using VEC = typename MAT::vector_type;
        using out_t = struct { MAT LU; VEC Pivot; std::int32_t Sign; bool Degenerate; };

        // housekeeping
        constexpr T tol{ static_cast<T>(10) * Numerics::equality_precision<T>() };
        MAT LU(mat);
        VEC Pivot;
        Utilities::static_for<0, 1, N>([&Pivot](std::size_t i) {
            Pivot[i] = static_cast<T>(i);
        });
        std::int32_t Sign{ 1 };

        for (std::size_t c{}; c < N; ++c) {
            // find pivot
            std::size_t _pivot{ c };
            for (std::size_t r{ c + 1 }; r < N; ++r) {
                if (std::abs(LU(c, r)) > std::abs(LU(c, _pivot))) {
                    _pivot = r;
                }
            }

            // exchange pivot
            if (_pivot != c) {
                for (std::size_t cc{}; cc < N; ++cc) {
                    Utilities::swap(LU(cc, _pivot), LU(cc, c));
                }
                Utilities::swap(Pivot[_pivot], Pivot[c]);
                Sign = -Sign;
            }

            // calculate multipliers and eliminate c-th column.
            if (!Numerics::areEquals(LU(c, c), T{})) {
                [[assume(LU(c, c) != T{})]];
                for (std::size_t r{ c + 1 }; r < N; ++r) {
                    LU(c, r) /= LU(c, c);

                    for (std::size_t cc{ c + 1 }; cc < N; ++cc) {
                        LU(cc, r) -= LU(c, r) * LU(cc, c);
                    }
                }
            }
        }

        return out_t{ LU, Pivot, Sign };
    }

    /**
    * \brief using Schur decomposition - return eigenvector and eigenvalues of given matrix.
    * @param {IFixedCubicMatrix,                     in}  matrix to decompose
    * @param {size_t,                                in}  maximal number of iterations (default is 10)
    * @param {value_type,                            in}  minimal error in iteration to stop calculation (default is 1e-5)
    * @param {IFixedCubicMatrix, IFixedCubicMatrix}, out} {matrix whose columns are eigenvectors, upper triangular matrix whose diagonal holds eigenvalues }
    **/
    template<GLSL::IFixedCubicMatrix MAT, class T = typename MAT::value_type>
    constexpr auto Schur(const MAT& mat, const std::size_t iter = 10, const T tol = static_cast<T>(1e-5)) {
        using qr_t = decltype(Decomposition::QR_GivensRotation(mat));
        using VEC = typename MAT::vector_type;
        using out_t = struct { MAT eigenvectors; MAT schur; };

        MAT A(mat);
        qr_t QR;
        T err{ static_cast<T>(10) * tol };
        std::size_t i{};
        while ((i < iter) && (err > tol)) {
            const VEC eigenvalues0{ GLSL::trace(A) };

            QR = Decomposition::QR_GivensRotation(A);
            A = QR.R * QR.Q;

            err = GLSL::max(GLSL::abs(GLSL::trace(A) - eigenvalues0));
            ++i;
        }

        return out_t{ QR.Q, A };
    }

    template<std::size_t N, GLSL::IFixedCubicMatrix MAT>
    constexpr auto Schur(const MAT& mat) {
        using qr_t = decltype(Decomposition::QR_GivensRotation(mat));
        using out_t = struct { MAT eigenvectors; MAT schur; };

        MAT A(mat);
        qr_t QR;
        Utilities::static_for<0, 1, N>([&A, &QR](std::size_t i) {
            QR = Decomposition::QR_GivensRotation(A);
            A = QR.R * QR.Q;
        });

        return out_t{ QR.Q, A };
    }

    /**
    * \brief using power iteration method - approximate the spectral radius (absolute value of largest eigenvalue) and appropriate eigenvector.
    *        user supplies two stoppage criteria's:
    *        1. maximal amount of iterations (10 by default)
    *        2. minimal value between two consecutive eigenvalue approximation (1e-5 by default).
    * @param {IFixedCubicMatrix, in}  matrix
    * @param {size_t,            in}  maximal number of iterations (default is 10)
    * @param {value_type,        in}  minimal error in iteration to stop calculation (default is 1e-5)
    * @param {value_type,        out} spectral radius
    **/
    template<GLSL::IFixedCubicMatrix MAT, class T = typename MAT::value_type>
    constexpr T spectral_radius(const MAT& mat, const std::size_t iter = 10, const T tol = static_cast<T>(1e-5)) {
        using VEC = typename MAT::vector_type;

        // initialize random "eigenvector"
        VEC eigenvector;
        Extra::make_random(eigenvector);

        // eigenvector calculation via power iteration
        VEC eigenvector_next;
        std::size_t i{};
        T eigenvalue{ static_cast<T>(10) * tol };
        T eigenvaluePrev{};
        while ((i < iter) && (std::abs(eigenvalue - eigenvaluePrev) > tol)) {
            eigenvaluePrev = eigenvalue;
            eigenvector_next = eigenvector * mat;

            const T max{ GLSL::max(eigenvector_next) };
            assert(!Numerics::areEquals(max, T{}));
            eigenvector /= max;

            eigenvalue = GLSL::dot(eigenvector_next, eigenvector);
            ++i;
        }

        // output
        const T dot{ GLSL::dot(eigenvector) };
        assert(!Numerics::areEquals(dot, T{}));
        [[assume(dot > T{})]];
        return eigenvalue / dot;
    }

    template<std::size_t N, GLSL::IFixedCubicMatrix MAT, class T = typename MAT::value_type>
    constexpr T spectral_radius(const MAT& mat) {
        using VEC = typename MAT::vector_type;

        // initialize random "eigenvector"
        VEC eigenvector;
        Extra::make_random(eigenvector);

        // eigenvector calculation via power iteration
        VEC eigenvector_next;
        Utilities::static_for<0, 1, N>([&mat, &eigenvector, &eigenvector_next](std::size_t i) {
            eigenvector_next = eigenvector * mat;

            const T max{ GLSL::max(eigenvector_next) };
            assert(!Numerics::areEquals(max, T{}));
            eigenvector /= max;
        });

        // output
        const T dot{ GLSL::dot(eigenvector) };
        assert(!Numerics::areEquals(dot, T{}));
        [[assume(dot > T{})]];
        return GLSL::dot(eigenvector_next, eigenvector) / dot;
    }

    /**
    * \brief calculate matrix determinant using LU decomposition
    * @param {MAT,        in}  matrix
    * @param {value_type, out} matrix determinant
    **/
    template<GLSL::IFixedCubicMatrix MAT, class T = typename MAT::value_type>
    constexpr T determinant_using_lu(const MAT& mat) {
        // LU decomposition
        auto lowerUpper = Decomposition::LU(mat);
        T det{ static_cast<T>(lowerUpper.Sign) };

        // determinant calculation
        Utilities::static_for<0, 1, MAT::columns()>([&det, &lowerUpper](std::size_t i) {
            det *= lowerUpper.LU(i, i);
        });

        // output
        return det;
    }

    /**
    * \brief calculate matrix determinant using QR decomposition
    * @param {MAT,        in}  matrix
    * @param {value_type, out} matrix determinant
    **/
    template<GLSL::IFixedCubicMatrix MAT, class T = typename MAT::value_type>
    constexpr T determinant_using_qr(const MAT& mat) {
        // QR decomposition
        auto qr = Decomposition::QR_GivensRotation(mat);
        T det{ static_cast<T>(1) };

        // determinant calculation
        Utilities::static_for<0, 1, MAT::columns()>([&det, &qr](std::size_t i) {
            det *= qr.R(i, i);
        });

        // output
        return det;
    }

    /**
    * \brief invert a matrix using LU decomposition
    * @param {MAT, in}  matrix
    * @param {MAT, out} matrix inverse
    **/
    template<GLSL::IFixedCubicMatrix MAT>
    constexpr MAT inverse_using_lu(const MAT& mat) {
        using T = typename MAT::value_type;
        using VEC = typename MAT::vector_type;
        constexpr std::size_t N{ MAT::columns() };

        // LU decomposition
        auto lowerUpper = Decomposition::LU(mat);

        // inverted matrix
        MAT out;
        for (std::size_t j{}; j < N; ++j) {
            // columns
            for (std::size_t i{}; i < N; ++i) {
                out(j, i) = (static_cast<std::size_t>(lowerUpper.Pivot[i]) == j) ? static_cast<T>(1) : T{};

                for (std::size_t k{}; k < i; ++k) {
                    out(j, i) -= lowerUpper.LU(k, i) * out(j, k);
                }
            }

            // rows
            for (std::int32_t i{ N - 1 }; i >= 0; i--) {
                for (std::int32_t k{ i + 1 }; k < N; ++k) {
                    out(j, i) -= lowerUpper.LU(k, i) * out(j, k);
                }

                assert(!Numerics::areEquals(lowerUpper.LU(i, i), T{}));
                out(j, i) /= lowerUpper.LU(i, i);
            }
        }

        // output
        return out;
    }

    /**
    * \brief given non singular matrix, return the rotation matrix of its polar decomposition.
    *        in general, polar decomposition decompose a matrix to R*P where:
    *        > R is an orthogonal unitary matrix representing rotation.
    *        > P is a positive semi definite symmetric matrix represents deformation/scaling.
    *          (P might have negative sign for small magnitude singular values)
    *        here, we only return R. P can be calculated by the user (MAT * Rinv)
    * @param {IFixedCubicMatrix, in}  matrix to decompose
    * @param {size_t,            in}  maximal number of iterations (default is 10)
    * @param {value_type,        in}  minimal rotation matrix squared Frobenius norm for calculation to halt.
    *                                 since orthogonal matrix Frobenius norm is 1, the tolerance should be larger than 1.
    *                                 default is 1.1 - meaning operation will stop when squared Frobenius norm will be smaller than 1.1.
    * @param {IFixedCubicMatrix, out} rotation matrix of polar decomposition
    **/
    template<GLSL::IFixedCubicMatrix MAT, class T = typename MAT::value_type>
    constexpr MAT PD_rotation(const MAT& mat, const std::size_t iter = 10, const T tol = static_cast<T>(1.1)) {
        if constexpr (MAT::columns() == 2) {
            const T x0{ mat(0, 0) + mat(1, 1) };
            const T x1{ mat(0, 1) - mat(1, 0) };
            const T den{ std::sqrt(x0 * x0 + x1 * x1) };
            assert(den > T{});
            const T c{  x0 / den };
            const T s{ -x1 / den };

            return MAT( c, s,
                       -s, c);
        }
        else {
            MAT R(mat);
            std::size_t i{};
            T frobSquared{ std::numeric_limits<T>::max() };
            while ((i < iter) && (frobSquared > tol)) {
                const MAT Rinv(GLSL::transpose(Decomposition::inverse_using_lu(R)));
                R = (R + Rinv) / static_cast<T>(2);

                Utilities::static_for<0, 1, MAT::columns()>([&R, &frobSquared](std::size_t i) {
                    frobSquared += dot(R[i]);
                });

                ++i;
            }

            // output
            return R;
        }
    }
    template<std::size_t N, GLSL::IFixedCubicMatrix MAT>
    constexpr MAT PD_rotation(const MAT& mat) {
        using T = typename MAT::value_type;

        if constexpr (MAT::columns() == 2) {
            const T x0{ mat(0, 0) + mat(1, 1) };
            const T x1{ mat(0, 1) - mat(1, 0) };
            const T den{ std::sqrt(x0 * x0 + x1 * x1) };
            assert(den > T{});
            const T c{ x0 / den };
            const T s{ -x1 / den };

            return MAT( c, s,
                       -s, c);
        }
        else {
            MAT R(mat);
            Utilities::static_for<0, 1, N>([&R](std::size_t i) {
                const MAT Rinv(GLSL::transpose(Decomposition::inverse_using_lu(R)));
                R = (R + Rinv) / static_cast<T>(2);
            });

            // output
            return R;
        }
    }

    /**
    * \brief calculate the eigenvalues and eigenvectors of cubic 3x3 symmetric matrix.
    *        notice that this calculation is correct only in cases where the eigenvalues are real and "well separated".
    * @param {IFixedCubicMatrix,                 in}  matrix
    * @param {IFixedCubicMatrix, IFixedVector}, out} {matrix whose columns are eigenvectors, vector whose elements are eigenvalues }
    **/
    template<GLSL::IFixedCubicMatrix MAT>
        requires(MAT::columns() == 3)
    constexpr auto EigenSymmetric3x3(const MAT& mat) {
        using VEC = typename MAT::vector_type;
        using T = typename MAT::value_type;
        using out_t = struct { MAT eigenvectors; VEC eigenvalues; };

        assert(Extra::is_symmetric(mat));
        
        // eigenvalues
        const VEC eigenvalues{ Decomposition::eigenvalues(mat) };
        
        // eigenvectors
        MAT eigenvectors(T{});
        Utilities::static_for<0, 1, 3>([&mat, &eigenvectors, eigenvalues](std::size_t i) {
            const VEC r1(mat(0, 0) - eigenvalues[i], mat(0, 1),                  mat(0, 2));
            const VEC r2(mat(0, 1),                  mat(1, 1) - eigenvalues[i], mat(1, 2));
            const VEC r3(mat(0, 2),                  mat(1, 2),                  mat(2, 2) - eigenvalues[i]);
            const VEC e1{ GLSL::cross(r1, r2) };
            VEC e2{ GLSL::cross(r2, r3) };
            VEC e3{ GLSL::cross(r3, r1) };

            // make e2 and e2 point in the direction of e1
            if (GLSL::dot(e1, e2) < T{}) {
                e2 *= static_cast<T>(-1);
            }
            if (GLSL::dot(e1, e3) < T{}) {
                e3 *= static_cast<T>(-1);
            }

            eigenvectors[i] = GLSL::normalize(e1 + e2 + e3);
        });

        // output
        return out_t{ eigenvectors, eigenvalues };
    }
};

//
// linear equation system solvers using decompositions
// 

namespace Solvers {

    /**
    * \brief solve linear system A*x=b using LU decomposition
    *
    * @param {IFixedCubicMatrix, in}  A
    * @param {IFixedVector,      in}  b
    * @param {IFixedVector,      out} x
    **/
    template<GLSL::IFixedCubicMatrix MAT, GLSL::IFixedVector VEC>
        requires(std::is_same_v<typename MAT::value_type, typename VEC::value_type> && (MAT::columns() == VEC::length()))
    constexpr VEC SolveLU(const MAT& mat, const VEC& b) {
        using T = typename MAT::value_type;
        constexpr std::size_t N{ MAT::columns() };

        // LU decomposition
        auto lowerUpper = Decomposition::LU(mat);

        // x is the permuted copy of B as piv
        VEC x;
        Utilities::static_for<0, 1, N>([&x, &lowerUpper, &b](std::size_t i) {
            x[i] = b[static_cast<std::size_t>(lowerUpper.Pivot[i])];
        });

        // Solve L*Y = b(pivoted)
        for (std::size_t k{}; k < N; ++k) {
            for (std::size_t i{ k + 1 }; i < N; ++i) {
                x[i] -= x[k] * lowerUpper.LU(k, i);
            }
        }

        // Solve U*X = Y
        for (std::int64_t k{ N - 1 }; k >= 0; k--) {
            assert(!Numerics::areEquals(lowerUpper.LU(k, k), T{}));
            x[k] /= lowerUpper.LU(k, k);

            for (std::size_t i{}; i < static_cast<std::size_t>(k); ++i) {
                x[i] -= x[k] * lowerUpper.LU(k, i);
            }
        }

        // output
        return x;
    }

    /**
    * \brief solve linear system A*x=b using Cholesky decomposition. 'A' must be positive definite.
    *
    * @param {IFixedCubicMatrix, in}  A (CUBIC and positive definite)
    * @param {IFixedVector,      in}  b (column vector)
    * @param {IFixedVector,      out} x (column vector)
    **/
    template<GLSL::IFixedCubicMatrix MAT, GLSL::IFixedVector VEC>
        requires(std::is_same_v<typename MAT::value_type, typename VEC::value_type> && (MAT::columns() == VEC::length()))
    constexpr VEC SolveCholesky(const MAT& A, const VEC& b) {
        using T = typename MAT::value_type;
        constexpr std::size_t N{ MAT::columns() };

        // Cholesky decomposition
        const MAT L(Decomposition::Cholesky(A));

        // Solve L*y = b;
        VEC x(b);
        for (std::size_t k{}; k < N; ++k) {
            for (std::size_t i{}; i < k; ++i) {
                x[k] -= x[i] * L(k, i);
            }

            assert(!Numerics::areEquals(L(k, k), T{}));
            x[k] /= L(k, k);
        }

        // Solve L'*X = Y;
        for (std::int64_t k{ N - 1 }; k >= 0; --k) {
            const std::size_t _k{ static_cast<std::size_t>(k) };
            for (std::size_t i{ _k + 1 }; i < N; ++i) {
                x[_k] -= x[i] * L(i, _k);
            }

            assert(!Numerics::areEquals(L(_k, _k), T{}));
            x[_k] /= L(_k, _k);
        }

        // output
        return x;
    }

    /**
    * \brief solve linear system A*x=b (using internally both QR & LU decomposition)
    *
    *        Notice that QR decomposition is used and not:
    *        > pseudo-inverse - to avoid increasing the output matrix condition number (happens when multiplying the matrix by its transpose),
    *        > SVD - high running time complexity.
    *
    * @param {IFixedCubicMatrix, in}  A (ROWxCOL, ROW >= COL)
    * @param {IFixedVector,      in}  B (column vector, ROWx1)
    * @param {IFixedVector,      out} X (column vector, COL, 1)
    **/
    template<GLSL::IFixedCubicMatrix MAT, GLSL::IFixedVector VEC>
        requires(std::is_same_v<typename MAT::value_type, typename VEC::value_type> && (MAT::columns() == VEC::length()))
    constexpr VEC SolveQR(const MAT& A, const VEC& b) {
        // QR decomposition
        const auto qr = Decomposition::QR_GivensRotation(A);

        // C = Q * b
        const VEC C(b * qr.Q);

        // R*x = C
        auto lowerUpper = Decomposition::LU(qr.R);

        // output
        return SolveLU(lowerUpper.LU, C);
    }
};
