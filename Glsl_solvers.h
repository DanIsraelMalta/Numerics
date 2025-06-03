//-------------------------------------------------------------------------------
//
// Copyright (c) 2025, Dan Israel Malta <malta.dan@gmail.com>
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
// matrix decompositions and eigenvalue related and based operations
// 

namespace Decomposition {

    // QR decomposition algorithm type
    enum class QR_DEOMPOSITION_TYPE : std::uint8_t {
        GramSchmidt        = 0, // use gram-schmidt process. very fast. numerically unstable.
        SchwarzRutishauser = 1, // use Schwarz-Rutishauser algorithm. better numerical accuracy than gram-schmidt and faster yet slower.
        GivensRotation     = 2  // use "givens rotation" algorithm. numerically precise. slower than other options.
    };

    /**
    * \brief return a balanced form of a given matrix (matrix whose rows and columns normal are similar while eigenvalues are identical).
    *        background:
    *        > The sensitivity of eigenvalues to rounding errors can be reduced by the procedure of balancing.
    *          Since errors in eigensystem found numerically are generally proportional to the
    *          Euclidean norm of the matrix (the square root of the sum of the squares of the elements) - the
    *          idea of balancing is to use similarity transformations to make corresponding rows and columns 
    *          have comparable norms, thus reducing the overall norm of the matrix while leaving the eigenvalues unchanged.
    *        > Notice that a symmetric matrix is already balanced.
    *        > balancing is done using "Parlett and Reinsch" algorithm to balance the L1 norm of input matrix.
    *
    * @param {IFixedCubicMatrix, in}  matrix to balance
    * @param {IFixedCubicMatrix, out} balanced matrix.
    **/
    template<GLSL::IFixedCubicMatrix MAT, class T = typename MAT::value_type>
    constexpr MAT balance_matrix(const MAT& mat) {
        using VEC = typename MAT::vector_type;
        constexpr std::size_t N{ MAT::columns() };

        constexpr T radix{ std::numeric_limits<T>::radix };
        constexpr T radix_squared{ radix * radix };
        constexpr T norm_ratio{ static_cast<T>(0.95) };

        MAT out(mat);
        bool converged{ false };
        while (!converged) {
            converged = true;
            
            for (std::size_t i{}; i < N; ++i) {
                // calculate row and column norms
                T r{};
                T c{};
                Utilities::static_for<0, 1, N>([&out, &c, &r, i](std::size_t j) {
                    c += std::abs(out(i, j));
                    r += std::abs(out(j, i));
                });
                c -= std::abs(out(i, i));
                r -= std::abs(out(i, i));

                // if any norm is zero, skip balancing
                if (Numerics::areEquals(c, T{}) || Numerics::areEquals(r, T{})) {
                    continue;
                }

                // find integer power of machine radix that is closest to balancing the matrix
                T g{ r / radix };
                T f{ static_cast<T>(1) };
                const T s{ c + r };
                while (c < g) {
                    f *= radix;
                    c *= radix_squared;
                }

                g = r * radix;
                while (c > g) {
                    f /= radix;
                    c /= radix;
                }

                // perform similarity transformation
                [[assume(f > T{})]];
                if ((c + r) / f < norm_ratio * s) {
                    converged = false;
                    g = static_cast<T>(1) / f;
                    Utilities::static_for<0, 1, N>([&out, i, g, f](std::size_t j) {
                        out(j, i) *= g;
                        out(i, j) *= f;
                    });
                }
            }
        }

        return out;
    }

    /**
    * \brief perform QR decomposition.
    *
    * @param {IFixedCubicMatrix,                      in}  input matrix
    * @param {{IFixedCubicMatrix, IFixedCubicMatrix}, out} {Q matrix (orthogonal matrix with orthogonal columns, i.e. - Q*Q^T = I),  R matrix (upper triangular matrix) }
    **/
    template<QR_DEOMPOSITION_TYPE TYPE = QR_DEOMPOSITION_TYPE::GivensRotation, GLSL::IFixedCubicMatrix MAT>
    constexpr auto QR(const MAT& mat) {
        using out_t = struct { MAT Q; MAT R; };

        if constexpr (TYPE == QR_DEOMPOSITION_TYPE::GramSchmidt) {
            const MAT Q(Extra::orthonormalize(mat));

            MAT R;
            Utilities::static_for<0, 1, MAT::columns()>([&Q, &R, &mat](std::size_t i) {
                for (std::size_t j{ i }; j < MAT::columns(); ++j) {
                    R(j, i) = GLSL::dot(mat[j], Q[i]);
                }
            });

            return out_t{ Q, R };
        }
        else if constexpr (TYPE == QR_DEOMPOSITION_TYPE::SchwarzRutishauser) {
            MAT Q(mat);
            MAT R;

            for (std::size_t k{}; k < MAT::columns(); ++k) {
                for (std::size_t i{}; i < k; ++i) {
                    R(k, i) = GLSL::dot(Q[i], Q[k]);
                    Q[k] -= R(k, i) * Q[i];
                }

                R(k, k) = GLSL::length(Q[k]);
                Q[k] /= R(k, k);
            }

            return out_t{ -Q, -R };
        }
        else if constexpr (TYPE == QR_DEOMPOSITION_TYPE::GivensRotation) {
            using T = typename MAT::value_type;
            constexpr std::size_t N{ MAT::columns() };
            constexpr T tol{ Numerics::equality_precision<T>() };

            // "givens rotation" - return {cosine, sine, radius}
            const auto givensRotation = [](const T a, const T b) noexcept -> GLSL::Vector3<T> {
                if (std::abs(a) <= tol) {
                    return GLSL::Vector3<T>(T{}, Numerics::sign(b), std::abs(b));
                }
                else if (std::abs(b) <= tol) {
                    return GLSL::Vector3<T>(Numerics::sign(a), T{}, std::abs(a));
                }
                else if (std::abs(a) > std::abs(b)) {
                    [[assume(a != T{})]];
                    const T t{ b / a };
                    const T squared{ std::fma(t, t, static_cast<T>(1.0)) };
                    [[assume(squared >= T{})]];
                    const T u{ Numerics::sign(a) * std::sqrt(squared) };
                    [[assume(u != T{})]];
                    const T c{ static_cast<T>(1) / u };
                    return GLSL::Vector3<T>(c, t * c, a * u);
                }
                else {
                    [[assume(b != T{})]];
                    const T t{ a / b };
                    const T squared{ std::fma(t, t, static_cast<T>(1.0)) };
                    [[assume(squared >= T{})]];
                    const T u{ Numerics::sign(b) * std::sqrt(squared) };
                    [[assume(u != T{})]];
                    const T s{ static_cast<T>(1) / u };
                    return GLSL::Vector3<T>(t * s, s, b * u);
                }
            };

            // housekeeping
            MAT R(mat);
            MAT Q;
            Extra::make_identity(Q);

            // perform decomposition
            for (std::size_t j{}; j < N; ++j) {
                for (std::size_t i{ N - 1 }; i > j; --i) {
                    const GLSL::Vector3<T> CSR(givensRotation(R(j, i - 1), R(j, i)));
                    for (std::size_t x{}; x < N; ++x) {
                        // R' = G * R
                        T temp1{ R(x, i - 1) };
                        T temp2{ R(x, i) };
                        R(x, i - 1) = temp1 * CSR[0] + temp2 * CSR[1];
                        R(x, i)     = Numerics::diff_of_products(temp2, CSR[0], temp1, CSR[1]);

                        // Q' = Q * G^
                        temp1 = Q(i - 1, x);
                        temp2 = Q(i,     x);
                        Q(i - 1, x) = temp1 * CSR[0] + temp2 * CSR[1];
                        Q(i, x)     = Numerics::diff_of_products(temp2, CSR[0], temp1, CSR[1]);
                    }
                    R(j, i - 1) = CSR[2];
                    R(j, i) = T{};
                }
            }

            return out_t{ Q, R };
        }
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
    * \brief using Schur decomposition - return eigenvector and eigenvalues of given matrix.
    * @param {IFixedCubicMatrix,                     in}  matrix to decompose
    * @param {bool,                                  in}  should input matrix be balanced? (default is false)
    * @param {size_t,                                in}  maximal number of iterations (default is 10)
    * @param {value_type,                            in}  minimal error in iteration to stop calculation (default is 1e-5)
    * @param {IFixedCubicMatrix, IFixedCubicMatrix}, out} {matrix whose columns are eigenvectors, upper triangular matrix whose diagonal holds eigenvalues }
    **/
    template<GLSL::IFixedCubicMatrix MAT, class T = typename MAT::value_type>
    constexpr auto Schur(const MAT& mat, const bool balance_input = false, const std::size_t iter = 10, const T tol = static_cast<T>(1e-5)) {
        using qr_t = decltype(Decomposition::QR(mat));
        using VEC = typename MAT::vector_type;
        using out_t = struct { MAT eigenvectors; MAT schur; };

        MAT A(balance_input ? Decomposition::balance_matrix(mat)  : mat);
        MAT Q;
        Extra::make_identity(Q);
        qr_t QR;
        T err{ static_cast<T>(10) * tol };
        std::size_t i{};
        while ((i < iter) && (err > tol)) {
            const VEC eigenvalues0{ GLSL::trace(A) };

            QR = Decomposition::QR(A);
            A = QR.R * QR.Q;
            Q *= QR.Q;

            err = GLSL::max(GLSL::abs(GLSL::trace(A) - eigenvalues0));
            ++i;
        }

        return out_t{ Q, A };
    }
    
    template<std::size_t N, GLSL::IFixedCubicMatrix MAT>
    constexpr auto Schur(const MAT& mat, const bool balance_input = false) {
        using qr_t = decltype(Decomposition::QR(mat));
        using out_t = struct { MAT eigenvectors; MAT schur; };

        MAT A(balance_input ? Decomposition::balance_matrix(mat) : mat);
        MAT Q;
        Extra::make_identity(Q);
        qr_t QR;
        Utilities::static_for<0, 1, N>([&A, &Q, &QR]() {
            QR = Decomposition::QR(A);
            A = QR.R * QR.Q;
            Q *= QR.Q;
        });

        return out_t{ Q, A };
    }

    /**
    * \brief return eigenvalues of given matrix (uses QR decomposition for matrices with more than 3 columns).
    * @param {IFixedCubicMatrix, in}  matrix to decompose
    * @param {bool,              in}  should input matrix be balanced? (default is false)
    * @param {size_t,            in}  maximal number of iterations (default is 10)
    * @param {value_type,        in}  minimal error in iteration to stop calculation (default is 1e-5)
    * @param {IFixedVector,      out} vector holding matrix eigenvalues
    **/
    template<GLSL::IFixedCubicMatrix MAT, class VEC = typename MAT::vector_type, class T = typename MAT::value_type>
    constexpr VEC eig(const MAT& mat, const bool balance_input = false, const std::size_t iter = 10, const T tol = static_cast<T>(1e-5)) {
        using qr_t = decltype(Decomposition::QR(mat));

        if constexpr (MAT::columns() == 2) {
            const T diff{ mat(0, 0) - mat(1, 1) };
            const T center{ mat(0, 0) + mat(1, 1) };
            const T deltaSquared{ std::fma(static_cast<T>(4.0), mat(1, 0) * mat(0, 1), diff * diff) };
            assert(deltaSquared >= T{});
            const T delta{ std::sqrt(deltaSquared) };

            return VEC((center + delta) / static_cast<T>(2), (center - delta) / static_cast<T>(2));
        }
        else if constexpr (MAT::columns() == 3) {
            const T tr{ mat(0, 0) + mat(1, 1) + mat(2, 2) };
            const T det{ GLSL::determinant(mat) };
            const T cofSum{ Numerics::diff_of_products(mat(1, 1), mat(2, 2), mat(2, 1), mat(1, 2)) +
                            Numerics::diff_of_products(mat(0, 0), mat(2, 2), mat(2, 0), mat(0, 2)) +
                            Numerics::diff_of_products(mat(0, 0), mat(1, 1), mat(1, 0), mat(0, 1)) };
            const std::array<T, 6> roots{ Numerics::SolveCubic(-tr, cofSum, -det) };
            return VEC(roots[0], roots[2], roots[4]);
        }
        else {
            MAT A(balance_input ? Decomposition::balance_matrix(mat) : mat);
            qr_t QR;
            T err{ static_cast<T>(10) * tol };
            std::size_t i{};
            while ((i < iter) && (err > tol)) {
                const VEC eigenvalues0{ GLSL::trace(A) };

                QR = Decomposition::QR(A);
                A = QR.R * QR.Q;

                err = GLSL::max(GLSL::abs(GLSL::trace(A) - eigenvalues0));
                ++i;
            }

            return GLSL::trace(A);
        }
    }
    
    template<std::size_t N, GLSL::IFixedCubicMatrix MAT, class VEC = typename MAT::vector_type>
    constexpr VEC eig(const MAT& mat, const bool balance_input = false) {
        using T = typename MAT::value_type;
        using qr_t = decltype(Decomposition::QR(mat));

        if constexpr (MAT::columns() == 2) {
            const T diff{ mat(0, 0) - mat(1, 1) };
            const T center{ mat(0, 0) + mat(1, 1) };
            const T deltaSquared{ std::fma(static_cast<T>(4.0), mat(1, 0) * mat(0, 1), diff * diff) };
            assert(deltaSquared >= T{});
            const T delta{ std::sqrt(deltaSquared) };

            return VEC((center + delta) / static_cast<T>(2), (center - delta) / static_cast<T>(2));
        }
        else if constexpr (MAT::columns() == 3) {
            const T tr{ mat(0, 0) + mat(1, 1) + mat(2, 2) };
            const T det{ GLSL::determinant(mat) };
            const T cofSum{ Numerics::diff_of_products(mat(1, 1), mat(2, 2), mat(2, 1), mat(1, 2)) +
                            Numerics::diff_of_products(mat(0, 0), mat(2, 2), mat(2, 0), mat(0, 2)) +
                            Numerics::diff_of_products(mat(0, 0), mat(1, 1), mat(1, 0), mat(0, 1)) };
            const std::array<T, 6> roots{ Numerics::SolveCubic(-tr, cofSum, -det) };
            return VEC(roots[0], roots[2], roots[4]);
        }
        else {
            MAT A(balance_input ? Decomposition::balance_matrix(mat) : mat);
            qr_t QR;
            Utilities::static_for<0, 1, N>([&A, &QR]() {
                QR = Decomposition::QR(A);
                A = QR.R * QR.Q;
            });

            return GLSL::trace(A);
        }
    }

    /**
    * \brief calculate SVD (singular value decomposition) via QR algorithm
    * @param {IFixedCubicMatrix,                                   in}  matrix to decompose
    * @param {size_t,                                              in}  maximal number of iterations (default is 15)
    * @param {value_type,                                          in}  minimal error in iteration to stop calculation (default is 1e-5)
    * @param {IFixedVector, IFixedCubicMatrix, IFixedCubicMatrix}, out} {vector holding singular values, matrix holding eigenvectors of mat*trnanspose(mat), matrix holding eigenvectors of trnanspose(mat)*mat }
    **/
    template<GLSL::IFixedCubicMatrix MAT, class T = typename MAT::value_type>
    constexpr auto SVD(const MAT& mat, const std::size_t iter = 15, const T tol = static_cast<T>(1e-5)) {
        using qr_t = decltype(Decomposition::QR(mat));
        using VEC = typename MAT::vector_type;
        using out_t = struct { VEC S; MAT U; MAT V; };

        // housekeeping
        qr_t qr1;
        qr_t qr2;
        MAT U;
        MAT V;
        Extra::make_identity(U);
        Extra::make_identity(V);

        // lambda for one iteration in SVD calculation
        const auto iterative_step = [&qr1, &qr2, &U, &V](const MAT& m) {
            qr1 = Decomposition::QR(m);
            qr2 = Decomposition::QR(GLSL::transpose(qr1.R));
            qr1.R = GLSL::transpose(qr2.R);
            V *= qr1.Q;
            U *= qr2.Q;
        };

        // calculate SVD
        iterative_step(mat);
        T err{ static_cast<T>(10) * tol };
        std::size_t i{ 1 };
        while ((i < iter) && (err > tol)) {
            const VEC singulars0{ GLSL::trace(qr1.R) };

            iterative_step(qr1.R);

            err = GLSL::max(GLSL::abs(GLSL::trace(qr1.R) - singulars0));
            ++i;
        }

        return out_t{ GLSL::trace(qr1.R), U, V };
    }

    template<std::size_t N, GLSL::IFixedCubicMatrix MAT>
    constexpr auto SVD(const MAT& mat) {
        using qr_t = decltype(Decomposition::QR(mat));
        using VEC = typename MAT::vector_type;
        using out_t = struct { VEC S; MAT U; MAT V; };

        // housekeeping
        qr_t qr1;
        qr_t qr2;
        MAT U;
        MAT V;
        Extra::make_identity(U);
        Extra::make_identity(V);

        // lambda for one iteration in SVD calculation
        const auto iterative_step = [&qr1, &qr2, &U, &V](const MAT& m) {
            qr1 = Decomposition::QR(m);
            qr2 = Decomposition::QR(GLSL::transpose(qr1.R));
            qr1.R = GLSL::transpose(qr2.R);
            V *= qr1.Q;
            U *= qr2.Q;
        };

        // calculate SVD
        iterative_step(mat);
        Utilities::static_for<1, 1, N>([&qr1, &iterative_step]() {
            iterative_step(qr1.R);
        });

        return out_t{ GLSL::trace(qr1.R), U, V };
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
        Utilities::static_for<0, 1, N>([&mat, &eigenvector, &eigenvector_next]() {
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
    * \brief calculate matrix determinant using QR decomposition
    * @param {MAT,        in}  matrix
    * @param {value_type, out} matrix determinant
    **/
    template<GLSL::IFixedCubicMatrix MAT, class T = typename MAT::value_type>
    constexpr T determinant_using_qr(const MAT& mat) {
        using qr_t = decltype(Decomposition::QR(mat));
        constexpr std::size_t N{ MAT::columns() };

        // QR decomposition
        const qr_t qr{ Decomposition::QR(mat) };
        T det{ static_cast<T>(1) };

        // determinant calculation
        Utilities::static_for<0, 1, N>([&det, &qr](std::size_t i) {
            det *= qr.R(i, i);
        });

        // output
        return det;
    }

    /**
    * \brief invert a matrix using QR decomposition
    * @param {MAT,        in}  matrix
    * @param {value_type, out} matrix determinant
    **/
    template<GLSL::IFixedCubicMatrix MAT>
    constexpr MAT invert_using_qr(const MAT& mat) {
        using qr_t = decltype(Decomposition::QR(mat));
        constexpr std::size_t N{ MAT::columns() };

        // QR decomposition
        const qr_t qr{ Decomposition::QR(mat) };

        // invert R matrix (using back substation)
        MAT rInv;
        Extra::make_identity(rInv);
        for (std::size_t k{}; k < N; ++k) {
            for (std::size_t j{}; j < N; ++j) {
                for (std::size_t i{}; i < k; ++i) {
                    rInv(k, j) -= rInv(i, j) * qr.R(k, i);
                }
                rInv(k, j) /= qr.R(k, k);
            }
        }

        // output
        return (rInv * GLSL::transpose(qr.Q));
    }

    /**
    * \brief calculate the eigenvalues and eigenvectors of symmetric matrix (suitable for matrices with less than 10 columns/rows).
    *        notice that this calculation is correct only in cases where the eigenvalues are real and "well separated".
    * @param {IFixedCubicMatrix,                 in}  matrix
    * @param {IFixedCubicMatrix, IFixedVector}, out} {matrix whose columns are eigenvectors, vector whose elements are eigenvalues }
    **/
    template<GLSL::IFixedCubicMatrix MAT>
        requires(MAT::columns() <= 10)
    constexpr auto EigenSymmetric(const MAT& mat) {
        using VEC = typename MAT::vector_type;
        using T = typename MAT::value_type;
        using out_t = struct { MAT eigenvectors; VEC eigenvalues; };
        constexpr std::size_t N{ MAT::columns() };

        assert(Extra::is_symmetric(mat));
        
        if constexpr (N == 3) {
            // eigenvalues
            const VEC eigenvalues{ Decomposition::eig(mat) };

            // eigenvectors
            MAT eigenvectors(T{});
            VEC r1(mat[0]);
            VEC r2(mat[1]);
            VEC r3(mat[2]);
            Utilities::static_for<0, 1, 3>([&mat, &eigenvectors, &r1, &r2, &r3, eigenvalues](std::size_t i) {
                const T value{ eigenvalues[i] };
                r1[0] = mat(0, 0) - value;
                r2[1] = mat(1, 1) - value;
                r3[2] = mat(2, 2) - value;
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
#ifdef _DEBUG
            Utilities::static_for<0, 1, N>([&mat, &eigenvectors, &eigenvalues](const std::size_t i) {
                assert(Extra::is_normalized(eigenvectors[i]));
                const VEC lhs(mat * eigenvectors[i]);
                const VEC rhs(eigenvalues[i] * eigenvectors[i]);
                assert(std::abs(GLSL::length(lhs) - GLSL::length(rhs)) < 1e-2);
            });
#endif
            return out_t{ eigenvectors, eigenvalues };
        }
        else {
            // housekeeping 
            constexpr std::size_t max_sweeps{ 50 }; // Maximum number of matrix sweeps
            VEC eigenvalues(GLSL::trace(mat));
            MAT a(mat);
            VEC b(eigenvalues);
            VEC z(T{});
            MAT eigenvectors;
            Extra::make_identity(eigenvectors);

            // lambda to perform givens rotation
            const auto rotate = [](MAT& mat, std::size_t i, std::size_t j, std::size_t k, std::size_t l, T s, T tau) {
                const T g{ mat(i, j) };
                const T h{ mat(k, l) };
                mat(i, j) = std::fma(-s, (h + g * tau), g);
                mat(k, l) = std::fma(s, (g - h * tau), h);
            };

            // lambda to calculate 'a' matrix "average sum of off-diagonal elements"
            const auto average_off_diagonal_sum = [N, &a]() -> T {
                T sm{};
                for (std::size_t i{}; i < N - 1; ++i) {
                    for (std::size_t j{ i + 1 }; j < N; ++j) {
                        sm += std::abs(a(i, j));
                    }
                }
                return (sm / (N * N));
            };

            std::size_t sweeps{};
            T average_sum{ average_off_diagonal_sum() };
            while (sweeps < max_sweeps && average_sum > std::numeric_limits<T>::min()) {
                average_sum = average_off_diagonal_sum();

                // choose a fraction of 'off diagonal elements threshold' for first 3 iterations.
                const T tol{ sweeps < 4 ? static_cast<T>(0.2) * average_sum : std::numeric_limits<T>::min() };

                // sweep
                for (std::size_t p{}; p < N - 1; ++p) {
                    for (std::size_t q{ p + 1 }; q < N; ++q) {
                        T& a_pq{ a(p, q) };
                        T& eig_at_p{ eigenvalues[p] };
                        T& eig_at_q{ eigenvalues[q] };

                        const T abs_a_pq{ std::abs(a_pq) };
                        const T g{ static_cast<T>(100.0) * abs_a_pq };

                        // After four sweeps, skip the rotation if the off-diagonal element is small
                        if (sweeps > 4 &&
                            Numerics::areEquals(std::abs(eig_at_p) + g, std::abs(eig_at_p)) &&
                            Numerics::areEquals(std::abs(eig_at_q) + g, std::abs(eig_at_q))) {
                            a_pq = T{};
                        }
                        else if (abs_a_pq > tol) {
                            T h{ eig_at_q - eig_at_p };
                            const T abs_h{ std::abs(h) };
                            T t{};
                            if (Numerics::areEquals(abs_h + g, abs_h)) {
                                [[assume(h != T{})]];
                                t = a_pq / h;
                            }
                            else {
                                const T theta{ static_cast<T>(0.5) * h / a_pq };
                                t = static_cast<T>(1.0) / (std::abs(theta) + std::sqrt(static_cast<T>(1.0) + theta * theta));
                                if (theta < T{}) {
                                    t = -t;
                                }
                            }

                            const T c{ static_cast<T>(1.0) / std::sqrt(static_cast<T>(1.0) + t * t) };
                            const T s{ t * c };
                            [[assume(c > T{})]];
                            const T tau{ s / (static_cast<T>(1.0) + c) };
                            h = t * a_pq;

                            a_pq = T{};

                            z[p] -= h;
                            z[q] += h;

                            eig_at_p -= h;
                            eig_at_q += h;

                            // rotate matrix
                            for (std::size_t j{}; j < p; ++j) {
                                rotate(a, j, p, j, q, s, tau);
                            }
                            for (std::size_t j{ p + 1 }; j < q; ++j) {
                                rotate(a, p, j, j, q, s, tau);
                            }
                            for (std::size_t j{ q + 1 }; j < N; ++j) {
                                rotate(a, p, j, q, j, s, tau);
                            }
                            for (std::size_t j{}; j < N; ++j) {
                                rotate(eigenvectors, j, p, j, q, s, tau);
                            }
                        }
                    }
                }

                // Update eigenvalues
                Utilities::static_for<0, 1, N>([&b, &z, &eigenvalues](const std::size_t i) {
                    b[i] += z[i];
                    eigenvalues[i] = b[i];
                    z[i] = T{};
                });

                // update iterations
                ++sweeps;
            }

            // output
            eigenvectors = GLSL::transpose(eigenvectors);
#ifdef _DEBUG
            Utilities::static_for<0, 1, N>([&mat, &eigenvectors, &eigenvalues](const std::size_t i) {
                assert(Extra::is_normalized(eigenvectors[i]));
                const VEC lhs(mat * eigenvectors[i]);
                const VEC rhs(eigenvalues[i] * eigenvectors[i]);
                assert(std::abs(GLSL::length(lhs) - GLSL::length(rhs)) < 1e-2);
            });
#endif
            return out_t{ eigenvectors, eigenvalues };
        }
    }
};

//
// linear equation system solvers using decompositions
// 

namespace Solvers {

    /**
    * \brief solve linear system A*x=b using QR decomposition.
    *        Notice that QR decomposition is used and not:
    *        > pseudo-inverse - to avoid increasing the output matrix condition number (happens when multiplying the matrix by its transpose),
    *        > SVD - high running time complexity.
    *
    * @param {IFixedCubicMatrix, in}  A
    * @param {IFixedVector,      in}  B
    * @param {IFixedVector,      out} X
    **/
    template<GLSL::IFixedCubicMatrix MAT, GLSL::IFixedVector VEC>
        requires(std::is_same_v<typename MAT::value_type, typename VEC::value_type> && (MAT::columns() == VEC::length()))
    constexpr VEC SolveQR(const MAT& A, const VEC& b) {
        using T = typename MAT::value_type;
        using qr_t = decltype(Decomposition::QR(A));
        constexpr std::size_t N{ MAT::columns() };

        // QR decomposition
        const qr_t qr{ Decomposition::QR(A) };

        // A * x = b -> {A = R * Q} -> R * x = b * Q'
        VEC x(GLSL::transpose(qr.Q) * b);

        // lambda to calculate x at index i
        const auto calculate_x = [&qr, &x](const std::size_t i) {
            T sum{ x[i] };
            for (std::size_t j{ i + 1 }; j < N; ++j) {
                sum -= qr.R(j, i) * x[j];
            }
            x[i] = sum / qr.R(i, i);
        };

        // calculate X
        for (std::size_t i{ N - 1 }; i > 0; --i) {
            calculate_x(i);
        }
        calculate_x(0);

        // output
        return x;
    }

    /**
    * \brief solve linear system A*x=b using Cholesky decomposition. 'A' must be symmetric positive definite.
    *
    * @param {IFixedCubicMatrix, in}  A
    * @param {IFixedVector,      in}  b
    * @param {IFixedVector,      out} x
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
        // lambda to calculate x at index k
        const auto calculate_x = [&L, &x](const std::size_t k) {
            for (std::size_t i{ k + 1 }; i < N; ++i) {
                x[k] -= x[i] * L(i, k);
            }

            assert(!Numerics::areEquals(L(k, k), T{}));
            x[k] /= L(k, k);
        };
        for (std::size_t i{ N - 1 }; i > 0; --i) {
            calculate_x(i);
        }
        calculate_x(0);

        // output
        return x;
    }
};
