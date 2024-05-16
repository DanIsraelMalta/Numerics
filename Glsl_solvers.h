#pragma once
#include "Glsl.h"
#include "Glsl_extra.h"

//
// matrix decompositions
// 

namespace Decomposition {
    /**
    * \brief perform QR decomposition using gram-schmidt process.
    *         numerically less accureate than QR_GivensRotation.
    *
    * @param {IFixedCubicMatrix,                      in}  input matrix
    * @param {{IFixedCubicMatrix, IFixedCubicMatrix}, out} {Q matrix (orthogonal matrix with orthogonal columns, i.e. - Q*Q^T = I),  R matrix (upper triangular matrix) }
    **/
    template<GLSL::IFixedCubicMatrix MAT>
    constexpr auto QR_GramSchmidt(const MAT& mat) noexcept {
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
    *        numerically more accureate than QR_GramSchmidt.
    *
    * @param {IFixedCubicMatrix, in}     matrix
    * @param {{IFixedCubicMatrix, IFixedCubicMatrix}, out} {Q matrix (orthogonal matrix with orthogonal columns, i.e. - Q*Q^T = I),  R matrix (upper triangular matrix) }
    **/
    template<GLSL::IFixedCubicMatrix MAT>
    constexpr auto QR_GivensRotation(const MAT& mat) noexcept {
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
                const T u{ Numerics::sign(a) * std::sqrt(t * t + static_cast<T>(1) ) };
                [[assume(u != T{})]];
                const T c{ static_cast<T>(1) / u };
                return std::array<T, 3>{{c, t * c, a * u}};
            }
            else {
                [[assume(b != T{})]];
                const T t{ a / b};
                const T u{ Numerics::sign(b) * std::sqrt(t * t + static_cast<T>(1)) };
                [[assume(u != T{})]];
                const T s{ static_cast<T>(1) / u };
                return std::array<T, 3>{{t * s, s, b * u}};
            }
        };

        // decomposition
        MAT R(mat);
        MAT Q;
        Utilities::static_for<0, 1, N>([&Q](std::size_t i) {
            Q(i, i) = static_cast<T>(1);
        });
        for (std::size_t j{}; j < N; ++j) {
            for (std::size_t i{ N - 1 }; i >= j + 1; --i) {
                const GLSL::Vector3<T> CSR(givensRotation(R(j, i - 1), R(j, i)));

                // R' = G * R
                for (std::size_t x{}; x < N; ++x) {
                    const T temp1{ R(x, i - 1) };
                    const T temp2{ R(x, i) };
                    R(x, i - 1) =  temp1 * CSR[0] + temp2 * CSR[1];
                    R(x, i)     = -temp1 * CSR[1] + temp2 * CSR[0];
                }
                R(j, i - 1) = CSR[2];
                R(j, i) = T{};

                // Q' = Q * G^
                for (std::size_t x{}; x < N; ++x) {
                    const T temp1{ Q(i - 1, x) };
                    const T temp2{ Q(i,     x) };
                    Q(i - 1, x) =  temp1 * CSR[0] + temp2 * CSR[1];
                    Q(i, x)     = -temp1 * CSR[1] + temp2 * CSR[0];
                }
            }
        }

        return out_t{Q, R};
    }

    /**
    * \brief given positive definite matrix A, constructs a lower triangular matrix L such that L*L' = A.
    *        (it's roughly TWICE as efficient as the LU decomposition)
    *
    * @param {IFixedCubicMatrix, in}     matrix
    * @param {IFixedCubicMatrix, in|out} lower triangular matrix
    **/
    template<GLSL::IFixedCubicMatrix MAT>
    constexpr MAT Cholesky(const MAT& mat) noexcept {
        using T = typename MAT::value_type;

        MAT lower;
        for (std::size_t j{}; j < MAT::columns(); ++j) {
            T d{};

            for (std::size_t k{}; k < j; ++k) {
                T s{};

                for (std::size_t i{}; i < k; ++i) {
                    s += lower(i, k) * lower(i, j);
                }

                assert(!Numerics::areEquals(lower(k, k), T{}));
                lower(k, j) = s = (mat(k, j) - s) / lower(k, k);
                d += s * s;
            }

            d = mat(j, j) - d;
            lower(j, j) = (d > T{}) ? std::sqrt(d) : T{};
        }

        return lower;
    }

    /**
    * \brief perform LU decomposition using using Doolittle algorithm.
    *        i.e. - given matrix decompose it to L*P*U, where L is lower traingular with unit diagonal,
    *               U is an upper triangular and P is a diagonal pivot matrix (given as a vector holding its diagonal)
    *        this implementatoin is geared towrds easy usage in linear system solutions.
    * @param {IFixedCubicMatrix,                         in}  matrix to decomopse
    * @param {{IFixedCubicMatrix, array, int32_t}, out} {LU matrix (decomposed matrix; upper triangular is U, lower triangular is L, diagnoal is part of U), decomposition pivot vector, pivot sign}
    **/
    template<GLSL::IFixedCubicMatrix MAT>
    [[nodiscard]] constexpr auto LU(const MAT& mat) noexcept {
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
};

//
// linear equation system solvers
// 

namespace Solvers {
    /**
    * \brief calculate matrix determinant using LU decomposition for matrixes larger than 4x4)
    * @param {MAT,        in}  matrix
    * @param {value_type, out} matrix determinant
    **/
    template<GLSL::IFixedCubicMatrix MAT, class T = typename MAT::value_type>
    constexpr T determinant(const MAT& mat) noexcept {
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
    * \brief invert a matrix
    * @param {MAT, in}  matrix
    * @param {MAT, out} matrix inverse
    **/
    template<GLSL::IFixedCubicMatrix MAT>
    constexpr MAT inverse(const MAT& mat) {
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
    * \brief solve linear system A*x=b using LU decomposition
    *
    * @param {IFixedCubicMatrix, in}  A
    * @param {IFixedVector,      in}  b
    * @param {IFixedVector,      out} x
    **/
    template<GLSL::IFixedCubicMatrix MAT, GLSL::IFixedVector VEC>
        requires(std::is_same_v<typename MAT::value_type, typename VEC::value_type> && (MAT::columns() == VEC::length()))
    constexpr VEC SolveLU(const MAT& mat, const VEC& b) noexcept {
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
    constexpr VEC SolveCholesky(const MAT& A, const VEC& b) noexcept {
        using T = typename MAT::value_type;
        constexpr std::size_t N{ MAT::columns() };

        // Cholesky decomposition
        const MAT L(Decomposition::Cholesky(A));

        // Solve L*y = b;
        VEC x(b);
        for (std::size_t k{}; k < N; ++k) {
            for (std::size_t i{}; i < k; ++i) {
                x[k] -= x[i] * L(i, k);
            }

            assert(!Numerics::areEquals(L(k, k), T{}));
            x[k] /= L(k, k);
        }

        // Solve L'*X = Y;
        for (int64_t k{ N - 1 }; k >= 0; --k) {
            for (std::size_t i{ k + 1 }; i < N; ++i) {
                x[k] -= x[i] * L(k, i);
            }

            assert(!Numerics::areEquals(L(k, k), T{}));
            x[k] /= L(k, k);
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
    constexpr VEC SolveQR(const MAT& A, const VEC& b) noexcept {
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

    