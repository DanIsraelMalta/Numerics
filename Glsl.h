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
#include "Utilities.h"
#include "Numerics.h"
#include "Variadic.h"

/**
* glsl constructs and operations
**/
namespace GLSL {

    /**
    * concept of a fixed sized arithmetic vector.
    **/
    template<typename VEC, typename...Args>
    concept IFixedVector = std::is_integral_v<decltype(VEC::length())> &&                                         // IFixedVector amount of elements
                           std::is_arithmetic_v<typename VEC::value_type> &&                                      // IFixedVector arithmetics underlying type
                           std::is_constructible_v<VEC, typename VEC::value_type> &&                              // IFixedVector is constructible from a single scalar element
                           (std::is_arithmetic_v<Args> && ...) &&                                                 // arguments are of arithmetic type...
                           std::is_constructible_v<VEC, Args...>&&                                                // IFixedVector is constructible from variadic amount of arithmetics (amount identical to VEC::length)
                           std::is_constructible_v<VEC, std::array<typename VEC::value_type, VEC::length()>&&>&&  // IFixedVector is constructible from a moveable array
                           std::is_assignable_v<VEC, std::array<typename VEC::value_type, VEC::length()>&&>&&     // IFixedVector is assignable from a moveable array
        requires(VEC vec, std::size_t i) {
            { vec[i] } -> std::same_as<typename VEC::value_type&>;      // IFixedVector elements can be accessed randomly
    };

    // trait to check that an argument is of IFixedVector concept
    template<typename T> constexpr bool is_fixed_vector_v = IFixedVector<T>;

    /**
    * concept of a fixed size arithmetic column major cubic matrix.
    **/
    template<typename MAT, typename...Args>
    concept IFixedCubicMatrix = std::is_integral_v<decltype(MAT::length())> &&                                        // IFixedMatrix amount of elements
                                std::is_integral_v<decltype(MAT::columns())> &&                                       // IFixedMatrix amount of columns
                                std::is_arithmetic_v<typename MAT::value_type> &&                                     // IFixedMatrix arithmetics underlying type
                                IFixedVector<typename MAT::vector_type> &&                                            // IFixedMatrix arithmetics underlying column type is an IFixedVector
                                std::is_constructible_v<MAT, typename MAT::value_type> &&                             // IFixedMatrix is constructible from a single scalar element
                                (std::is_arithmetic_v<Args> && ...) &&                                                // arguments are of arithmetic type...
                                std::is_constructible_v<MAT, Args...>&&                                               // IFixedMatrix is constructible from variadic amount of arithmetics (amount identical to VEC::length)
                                std::is_constructible_v<MAT, std::array<typename MAT::value_type, MAT::length()>&&>&& // IFixedMatrix is constructible from a moveable array
                                std::is_assignable_v<MAT, std::array<typename MAT::value_type, MAT::length()>&&>&&    // IFixedMatrix is assignable from a moveable array
        requires(MAT mat, std::size_t i) {
            { mat[i]    } -> std::same_as<typename MAT::vector_type&>;  // IFixedMatrix columns can be accessed randomly
            { mat(i, i) } -> std::same_as<typename MAT::value_type&>;   // IFixedMatrix elements can be accessed randomly (col, row)
    };

    //
    // IFixedVector operations and functions
    // 

    // unary arithmetic operator overload for IFixedVector
    template<IFixedVector VEC>
    constexpr auto operator - (const VEC& xi_vec) {
        VEC v(xi_vec);
        v *= static_cast<typename VEC::value_type>(-1.0);
        return v;
    }

    // compound operator overload for IFixedVector
#define M_OPERATOR(OP)                                                                     \
    template<IFixedVector VEC>                                                             \
    constexpr VEC& operator OP (VEC& lhs, const VEC& rhs) {                                \
        Utilities::static_for<0, 1, VEC::length()>([&lhs, &rhs](std::size_t i) {           \
            lhs[i] OP rhs[i];                                                              \
        });                                                                                \
        return lhs;                                                                        \
    }                                                                                      \
    template<IFixedVector VEC>                                                             \
    constexpr VEC& operator OP (VEC& lhs, const typename VEC::value_type rhs) {            \
        Utilities::static_for<0, 1, VEC::length()>([&lhs, &rhs](std::size_t i) {           \
            lhs[i] OP rhs;                                                                 \
        });                                                                                \
        return lhs;                                                                        \
    }                                                                                      \
    template<IFixedVector VEC>                                                             \
    constexpr VEC& operator OP (VEC& lhs, VEC&& rhs) {                                     \
        Utilities::static_for<0, 1, VEC::length()>([&lhs, r = MOV(rhs)](std::size_t i) {   \
            lhs[i] OP MOV(r[i]);                                                           \
        });                                                                                \
        return lhs;                                                                        \
    }

    M_OPERATOR(-= );
    M_OPERATOR(+= );
    M_OPERATOR(*= );
    M_OPERATOR(/= );
    M_OPERATOR(&= );
    M_OPERATOR(|= );
    M_OPERATOR(^= );
    M_OPERATOR(>>= );
    M_OPERATOR(<<= );

#undef M_OPERATOR

    // binary arithmetic operator overload for IFixedVector
#define M_OPERATOR(OP, AOP)                                                 \
    template<IFixedVector VEC>                                              \
    constexpr VEC operator OP (VEC lhs, const VEC& rhs) {                   \
        return (lhs AOP rhs);                                               \
    }                                                                       \
    template<IFixedVector VEC>                                              \
    constexpr VEC operator OP (VEC lhs, VEC&& rhs) {                        \
        return (lhs AOP FWD(rhs));                                          \
    }                                                                       \
    template<IFixedVector VEC>                                              \
    constexpr VEC operator OP (VEC lhs, typename VEC::value_type rhs) {     \
        return (lhs AOP rhs);                                               \
    }                                                                       \
    template<IFixedVector VEC>                                              \
    constexpr VEC operator OP (typename VEC::value_type rhs, VEC lhs) {     \
        return (lhs AOP rhs);                                               \
    }                                                                       \

    M_OPERATOR(+, +=);
    M_OPERATOR(-, -=);
    M_OPERATOR(*, *=);
    M_OPERATOR(/ , /=);
    M_OPERATOR(&, &=);
    M_OPERATOR(| , |=);
    M_OPERATOR(^, ^=);
    M_OPERATOR(>> , >>=);
    M_OPERATOR(<< , <<=);

#undef M_OPERATOR

    // standard element wise unary functions for IFixedVector
#define M_UNARY_FUNCTION(NAME, FUNC)                                           \
    template<IFixedVector VEC>                                                 \
    constexpr VEC NAME(const VEC& x) {                                         \
        VEC out{};                                                             \
        Utilities::static_for<0, 1, VEC::length()>([&x, &out](std::size_t i) { \
            out[i] = FUNC(x[i]);                                               \
        });                                                                    \
        return out;                                                            \
    }

    M_UNARY_FUNCTION(abs, std::abs);
    M_UNARY_FUNCTION(floor, std::floor);
    M_UNARY_FUNCTION(ceil, std::ceil);
    M_UNARY_FUNCTION(trunc, std::trunc);
    M_UNARY_FUNCTION(round, std::round);
    M_UNARY_FUNCTION(exp, std::exp);
    M_UNARY_FUNCTION(exp2, std::exp2);
    M_UNARY_FUNCTION(log, std::log);
    M_UNARY_FUNCTION(log2, std::log2);
    M_UNARY_FUNCTION(sqrt, std::sqrt);
    M_UNARY_FUNCTION(sin, std::sin);
    M_UNARY_FUNCTION(cos, std::cos);
    M_UNARY_FUNCTION(tan, std::tan);
    M_UNARY_FUNCTION(asin, std::asin);
    M_UNARY_FUNCTION(acos, std::acos);
    M_UNARY_FUNCTION(atan, std::atan);
    M_UNARY_FUNCTION(sinh, std::sinh);
    M_UNARY_FUNCTION(cosh, std::cosh);
    M_UNARY_FUNCTION(tanh, std::tanh);
    M_UNARY_FUNCTION(asinh, std::asinh);
    M_UNARY_FUNCTION(acosh, std::acosh);
    M_UNARY_FUNCTION(atanh, std::atanh);

#undef M_UNARY_FUNCTION

    // standard element wise binary functions for IFixedVector
#define M_BINARY_FUNCTION(NAME, FUNC)                                              \
    template<IFixedVector VEC>                                                     \
    constexpr VEC NAME(const VEC& x,typename VEC::value_type y) {                  \
        VEC out{};                                                                 \
        Utilities::static_for<0, 1, VEC::length()>([&x, y, &out](std::size_t i) {  \
            out[i] = FUNC(x[i], y);                                                \
        });                                                                        \
        return out;                                                                \
    }

    M_BINARY_FUNCTION(pow, std::pow);
    M_BINARY_FUNCTION(atan2, std::atan2);
    M_BINARY_FUNCTION(mod, std::fmod);

#undef M_BINARY_FUNCTION

    //
    // GLSL special operations for IFixedVector
    //

    // named relational operations
#define M_RELATIONAL_FUNCTION(NAME, OP)                                                \
    template<IFixedVector VEC>                                                         \
    constexpr bool NAME(const VEC& x, const VEC& y) {                                  \
        bool result{ true };                                                           \
        Utilities::static_for<0, 1, VEC::length()>([&x, &y, &result](std::size_t i) {  \
            result &= x[i] OP y[i];                                                    \
        });                                                                            \
        return result;                                                                 \
    }

    M_RELATIONAL_FUNCTION(equal, == );
    M_RELATIONAL_FUNCTION(notEqual, != );
    M_RELATIONAL_FUNCTION(lessThan, < );
    M_RELATIONAL_FUNCTION(lessThanEqual, <= );
    M_RELATIONAL_FUNCTION(greaterThan, > );
    M_RELATIONAL_FUNCTION(greaterThanEqual, >= );

#undef M_RELATIONAL_FUNCTION

    //
    // GLSL boolean operations for IFixedVector
    //

    /**
    * \brief returns true if ALL/ANY elements are true
    * @param {VEC,  in}  vector to invert
    * @param {bool, out} true if ALL elements are true
    **/
#define M_BOOL_FUNCTION(NAME, OP)                                                  \
    template<IFixedVector VEC, class T = typename VEC::value_type>                 \
        requires(std::is_same_v<T, bool>)                                          \
    constexpr bool NAME(const VEC& x) {                                            \
        bool result{ true };                                                       \
        Utilities::static_for<0, 1, VEC::length()>([&x, &result](std::size_t i) {  \
            result OP x[i];                                                        \
        });                                                                        \
        return result;                                                             \
    }

    M_BOOL_FUNCTION(all, &= );
    M_BOOL_FUNCTION(any, != );

#undef M_BOOL_FUNCTION

    /**
    * \brief inverts elements in logical vector
    * @param {VEC, in}  vector to invert
    * @param {VEC, out} inverted vector
    **/
    template<IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_same_v<T, bool>)
    constexpr VEC glsl_not(const VEC& x) noexcept {
        VEC result;
        Utilities::static_for<0, 1, VEC::length()>([&x, &result](std::size_t i) {
            result[i] = !x[i];
        });

        return result;
    }

    //
    // GLSL generic operations
    //

    /**
    * \brief returns minimal element in vector
    * @param {VEC,        in}  vector
    * @param {value_type, out} minimal element
    **/
    template<IFixedVector VEC, class T = typename VEC::value_type>
    constexpr T min(const VEC& x) noexcept {
        T _min{ x[0] };
        Utilities::static_for<0, 1, VEC::length()>([&x, &_min](std::size_t i) {
            _min = Numerics::min(_min, x[i]);
        });
        return _min;
    }

    /**
    * \brief returns vector filled with minimal values from two different vectors
    * @param {VEC, in}  vector 1
    * @param {VEC, in}  vector 2
    * @param {VEC, out} vector holding minimal elements of input arguments
    **/
    template<IFixedVector VEC>
    constexpr VEC min(const VEC& x, const VEC& y) noexcept {
        VEC out;
        Utilities::static_for<0, 1, VEC::length()>([&x, &y, &out](std::size_t i) {
            out[i] = Numerics::min(x[i], y[i]);
        });
        return out;
    }

    /**
    * \brief returns maximal element in vector
    * @param {VEC,        in}  vector
    * @param {value_type, out} maximal element
    **/
    template<IFixedVector VEC, class T = typename VEC::value_type>
    constexpr T max(const VEC& x) noexcept {
        T _max{ x[0] };
        Utilities::static_for<0, 1, VEC::length()>([&x, &_max](std::size_t i) {
            _max = Numerics::max(_max, x[i]);
        });
        return _max;
    }

    /**
    * \brief returns vector filled with maximal values from two different vectors
    * @param {VEC, in}  vector 1
    * @param {VEC, in}  vector 2
    * @param {VEC, out} vector holding maximal elements of input arguments
    **/
    template<IFixedVector VEC>
    constexpr VEC max(const VEC& x, const VEC& y) noexcept {
        VEC out;
        Utilities::static_for<0, 1, VEC::length()>([&x, &y, &out](std::size_t i) {
            out[i] = Numerics::max(x[i], y[i]);
        });
        return out;
    }

    /**
    * \brief clamp vector elements to a given range
    * @param {VEC|value_type, in}  vector to clamp
    * @param {VEC|value_type, in}  minimal value
    * @param {VEC|value_type, in}  maximal value
    * @param {VEC|value_type, out} vector with clamped elements
    **/
    template<IFixedVector VEC, class T = typename VEC::value_type>
    constexpr VEC clamp(const VEC& x, const T minVal, const T maxVal) noexcept {
        VEC out{};
        Utilities::static_for<0, 1, VEC::length()>([&x, &out, minVal, maxVal](std::size_t i) {
            out[i] = Numerics::clamp(x[i], minVal, maxVal);
        });
        return out;
    }

    template<IFixedVector VEC, class T = typename VEC::value_type>
    constexpr VEC clamp(const VEC& x, const VEC& minVal, const VEC& maxVal) noexcept {
        VEC out{};
        Utilities::static_for<0, 1, VEC::length()>([&x, &out, &minVal, &maxVal](std::size_t i) {
            out[i] = Numerics::clamp(x[i], minVal[i], maxVal[i]);
        });
        return out;
    }

    template<auto minVal, auto maxVal, IFixedVector VEC, class T = decltype(minVal)>
        requires(std::is_same_v<T, decltype(maxVal)> && std::is_same_v<T, typename VEC::value_type> && (minVal < maxVal))
    constexpr VEC clamp(const VEC& x) noexcept {
        VEC out{};
        Utilities::static_for<0, 1, VEC::length()>([&x, &out](std::size_t i) {
            out[i] = Numerics::clamp<minVal, maxVal>(x[i]);
        });
        return out;
    }

    /**
    * \brief returns -1 if x is less than 0 and 1 if x is greater or equal to 0.
    * @param {VEC, in}  vector
    * @param {VEC, out} vector with values -1|+1 according to input vector elements
    **/
    template<IFixedVector VEC>
    constexpr VEC sign(const VEC& x) noexcept {
        VEC out{};
        Utilities::static_for<0, 1, VEC::length()>([&out, &x](std::size_t i) {
            out[i] = Numerics::sign(x[i]);
        });
        return out;
    }

    /**
    * \brief mix two vectors
    * @param {VEC,        in}  vector #1 (x)
    * @param {VEC,        in}  vector #2 (y)
    * @param {value_type, in}  mixing parameter [0, 1] (value isn checked for range only if template parameter) (a)
    * @param {VEC,        out} vector whose elements are - (a-1) * x + a * y
    **/
    template<IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_floating_point_v<T>)
    constexpr VEC mix(const VEC& x, const VEC& y, const T a) noexcept {
        const T am1{ static_cast<T>(1) - a };
        VEC out{};
        Utilities::static_for<0, 1, VEC::length()>([&out, &x, &y, am1, a](std::size_t i) {
            out[i] = am1 * x[i] + a * y[i];
        });

        return out;
    }
    template<auto a, IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_same_v<T, decltype(a)> && std::is_floating_point_v<T> && (a >= T{}) && (a <= static_cast<T>(1)))
    constexpr VEC mix(const VEC& x, const VEC& y) noexcept {
        constexpr T am1{ static_cast<T>(1) - a };
        VEC out{};
        Utilities::static_for<0, 1, VEC::length()>([&out, &x, &y](std::size_t i) {
            out[i] = am1 * x[i] + a * y[i];
        });

        return out;
    }

    /**
    * \brief return the inverse of square root of vector
    * @param {VEC, in}  vector
    * @param {VEC, out} inverse of square root of vector
    **/
    template<IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_floating_point_v<T>)
    constexpr VEC inversesqrt(const VEC& x) {
        constexpr T one{ static_cast<T>(1) };

        VEC out{};
        Utilities::static_for<0, 1, VEC::length()>([&out, &x](std::size_t i) {
            assert(!Numerics::areEquals(x[i], T{}));
            out[i] = one / static_cast<T>(std::sqrt(x[i]));
        });
        return out;
    }

    /**
    * \brief generate a step vector by comparing two vectors/values
    * @param {VEC|value_type, in}  edge
    * @param {VEC,            in}  vector
    * @param {VEC,            out} element i is 0 if it is smaller than edge, otherwise its 1
    **/
    template<IFixedVector VEC>
    constexpr VEC step(const typename VEC::value_type edge, const VEC& x) noexcept {
        using T = typename VEC::value_type;

        VEC out{};
        Utilities::static_for<0, 1, VEC::length()>([&out, &x, edge](std::size_t i) {
            out[i] = (x[i] < edge) ? T{} : static_cast<T>(1);
        });

        return out;
    }
    template<auto edge, IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_same_v<T, decltype(edge)>)
    constexpr VEC step(const VEC& x) noexcept {
        VEC out{};
        Utilities::static_for<0, 1, VEC::length()>([&out, &x](std::size_t i) {
            out[i] = (x[i] < edge) ? T{} : static_cast<T>(1);
        });

        return out;
    }
    template<IFixedVector VEC>
    constexpr VEC step(const VEC& edge, const VEC& x) noexcept {
        using T = typename VEC::value_type;

        VEC out{};
        Utilities::static_for<0, 1, VEC::length()>([&out, &x, &edge](std::size_t i) {
            out[i] = (x[i] < edge[i]) ? T{} : static_cast<T>(1);
        });

        return out;
    }

    /**
    * \brief perform Hermite interpolation between two values
    * @param {VEC|value_type, in}  lower edge of interpolation
    * @param {VEC|value_type, in}  upper edge of interpolation
    * @param {VEC,            in}  vector
    * @param {VEC,            out} element i is smooth Hermite interpolation between 0 and 1 when edge0 < x[i] < edge1
    **/
    template<IFixedVector VEC, class T = typename VEC::value_type>
    constexpr VEC smoothstep(const T edge0, const T edge1, const VEC& x) {
        VEC out{};
        const T den{ edge1 - edge0 };
        assert(edge1 > edge0);

        Utilities::static_for<0, 1, VEC::length()>([&out, &x, edge0, den](std::size_t i) {
            const T t{ clamp((x - edge0) / den, T{}, static_cast<T>(1)) };
            out[i] = t * t * (static_cast<T>(3.0) - static_cast<T>(2.0) * t);
        });

        return out;
    }
    template<auto edge0, auto edge1, IFixedVector VEC, class T = typename VEC::value_type>
        requires(std::is_arithmetic_v<decltype(edge0)> && std::is_same_v<decltype(edge0), decltype(edge1)> && std::is_same_v<decltype(edge0), T> && (edge1 > edge0))
    constexpr VEC smoothstep(const VEC& x) noexcept {
        constexpr T den{ edge1 - edge0 };
        [[assume(den > T)]];

        VEC out{};
        Utilities::static_for<0, 1, VEC::length()>([&out, &x](std::size_t i) {
            const T t{ clamp((x - edge0) / den, T{}, static_cast<T>(1)) };
            out[i] = t * t * (static_cast<T>(3.0) - static_cast<T>(2.0) * t);
        });

        return out;
    }
    template<IFixedVector VEC>
    constexpr VEC smoothstep(const VEC edge0, const VEC edge1, const VEC& x) {
        using T = typename VEC::value_type;
        VEC out{};
        Utilities::static_for<0, 1, VEC::length()>([&out, &x, &edge0, &edge1](std::size_t i) {
            assert(!Numerics::areEquals(edge1[i], edge0[i]));
            const T t{ clamp((x[i] - edge0[i]) / (edge1[i] - edge0[i]), T{}, static_cast<T>(1)) };
            out[i] = t * t * (static_cast<T>(3.0) - static_cast<T>(2.0) * t);
        });

        return out;
    }

    /**
    * \brief calculate dot product between two vectors
    * @param {VEC,        in}  x
    * @param {VEC,        in}  y
    * @param {value_type, out} dot product between x and y
    **/
    template<IFixedVector VEC>
    constexpr VEC::value_type dot(const VEC& x, const VEC& y) noexcept {
        using T = typename VEC::value_type;
        T dot{};

        if constexpr (std::is_floating_point_v<T>) {
            Utilities::static_for<0, 1, VEC::length()>([&dot, &x, &y](std::size_t i) {
                dot = std::fma(x[i], y[i], dot);
            });
        } else {
            Utilities::static_for<0, 1, VEC::length()>([&dot, &x, &y](std::size_t i) {
                dot += x[i] * y[i];
            });
        }

        return dot;
    }

    /**
    * \brief calculate dot product of vector
    * @param {VEC,        in}  vector
    * @param {value_type, out} dot product of vector
    **/
    template<IFixedVector VEC>
    constexpr VEC::value_type dot(const VEC& x) noexcept {
        using T = typename VEC::value_type;
        T dot{};

        if constexpr (std::is_floating_point_v<T>) {
            Utilities::static_for<0, 1, VEC::length()>([&dot, &x](std::size_t i) {
                dot = std::fma(x[i], x[i], dot);
            });
        }
        else {
            Utilities::static_for<0, 1, VEC::length()>([&dot, &x](std::size_t i) {
                dot += x[i] * x[i];
            });
        }

        return dot;
    }

    /**
    * \brief calculate product of vector elements
    * @param {VEC,        in}  vector
    * @param {value_type, out} product of vector elements
    **/
    template<IFixedVector VEC, class T = typename VEC::value_type>
    constexpr T prod(const VEC& x) noexcept {
        T prod{ static_cast<T>(1) };
        Utilities::static_for<0, 1, VEC::length()>([&prod, &x](std::size_t i) {
            prod *= x[i];
        });
        return prod;
    }

    /**
    * \brief calculate sum (simple arithmetic, not floating point correct) of vector elements
    * @param {VEC,        in}  vector
    * @param {value_type, out} sum of vector elements
    **/
    template<IFixedVector VEC, class T = typename VEC::value_type>
    constexpr T sum(const VEC& x) noexcept {
        T sum{};
        Utilities::static_for<0, 1, VEC::length()>([&sum, &x](std::size_t i) {
            sum += x[i];
        });
        return sum;
    }

    /**
    * \brief project one vector on another
    * @param {VEC, in}  vector to project
    * @param {VEC, in}  vectoer to be projected on
    * @param {VEC, out} projected vector
    **/
    template<IFixedVector VEC>
    constexpr VEC project(const VEC& to, const VEC& on) {
        using T = typename VEC::value_type;
        const T d{ dot(on) };
        assert(d >= T{});
        [[assume(d >= T{})]];
        return (dot(to, on) / d) * on;
    }

    /**
    * \brief return the cross product of two vectors
    *        2D operator is based on wedge operator from geometric algebra.
    * @param {VEC,            in}  x (2d or 3d vector)
    * @param {VEC,            in}  y (2d or 3d vector)
    * @param {VEC|value_type, out} cross product between x and y (vector in 3D case, value in 2D case)
    **/
    template<IFixedVector VEC>
        requires(VEC::length() == 2 || VEC::length() == 3)
    constexpr auto cross(const VEC& x, const VEC& y) noexcept {
        if constexpr (VEC::length() == 2) {
            return (x[0] * y[1] - x[1] * y[0]);
        }
        else {
            return VEC(x[1] * y[2] - y[1] * x[2],
                       x[2] * y[0] - y[2] * x[0],
                       x[0] * y[1] - y[0] * x[1]);
        }
    }

    /**
    * \brief calculate Euclidean/L2 norm of vector
    * @param {VEC,        in}  vector
    * @param {value_type, out} euclidean/L2 norm of vector
    **/
    template<IFixedVector VEC, class T = typename VEC::value_type>
    constexpr T length(const VEC& x) {
        const T d{ dot(x) };
        assert(d >= T{});
        [[assume(d >= T{})]];
        return static_cast<T>(std::sqrt(d));
    }

    /**
    * \brief calculate Euclidean/L2 distance between two vectors
    * @param {VEC,        in}  a
    * @param {VEC,        in}  b
    * @param {value_type, out} euclidean distance between a & b
    **/
    template<IFixedVector VEC, class T = typename VEC::value_type>
    constexpr T distance(const VEC& p0, const VEC& p1) {
        return length(p0 - p1);
    }

    /**
    * \brief return a normalized vector (might throw exception if 'x' is nullified)
    * @param {VEC, in}  vector
    * @param {VEC, out} normalized vector
    **/
    template<IFixedVector VEC>
    constexpr VEC normalize(const VEC& x) {
        using T = typename VEC::value_type;
        const T l{ GLSL::length(x) };
        assert(l >= T{});
        [[assume(l >= T{})]];
        return (x / l);
    }

    /**
    * \brief return a vector pointing in the same direction as another
    * @param {VEC, in}  N (vector orientation)
    * @param {VEC, in}  I (incident vector)
    * @param {VEC, in}  Nref (reference vector)
    * @param {VEC, out} If dot(Nref, I) < 0 returns N, otherwise it returns -N
    **/
    template<IFixedVector VEC>
    constexpr VEC faceforward(const VEC& N, const VEC& I, const VEC& Nref) noexcept {
        using T = typename VEC::value_type;
        assert(Numerics::areEquals(length(N), static_cast<T>(1)));
        return (dot(Nref, I) < T{}) ? N : (-N);
    }

    /**
    * \brief calculate the reflection direction for an incident vector
    * @param {VEC, in}  I (incident vector)
    * @param {VEC, in}  N (normal vector; should be normalized)
    * @param {VEC, out} For a given incident vector I and surface normal N reflect returns the reflection direction
    **/
    template<IFixedVector VEC>
    constexpr VEC reflect(const VEC& I, const VEC& N) noexcept {
        using T = typename VEC::value_type;
        assert(Numerics::areEquals(length(N), static_cast<T>(1)));
        return (I - static_cast<T>(2) * dot(I, N) * N);
    }

    /**
    * \brief calculate the refraction direction for an incident vector
    * @param {VEC,        in}  I (incident vector)
    * @param {VEC,        in}  N (normal vector; should be normalized)
    * @param {value_type, in}  eta (ratio of indices of refraction)
    * @param {VEC,        out} r a given incident vector I, surface normal N and ratio of indices of refraction, eta, refract returns the refraction vector
    **/
    template<IFixedVector VEC, class T = typename VEC::value_type>
    constexpr VEC refract(const VEC& I, const VEC& N, const T eta) {
        assert(Numerics::areEquals(length(N), static_cast<T>(1)));

        const T _dot{ dot(N, I) };
        const T k{ static_cast<T>(1) - eta * eta * (static_cast<T>(1) - _dot * _dot) };
        assert(k > T{});

        return (eta * I - (eta * _dot + std::sqrt(k)) * N);
    }

    //
    // IFixedCubicMatrix operations and functions
    //

    // standard element wise unary functions for IFixedCubicMatrix
#define M_UNARY_FUNCTION(NAME, FUNC)                                            \
    template<IFixedCubicMatrix MAT>                                             \
    constexpr MAT NAME(const MAT& x) {                                          \
        MAT out{};                                                              \
        Utilities::static_for<0, 1, MAT::columns()>([&x, &out](std::size_t i) { \
            out[i] = FUNC(x[i]);                                                \
        });                                                                     \
        return out;                                                             \
    }

    M_UNARY_FUNCTION(abs, std::abs);
    M_UNARY_FUNCTION(floor, std::floor);
    M_UNARY_FUNCTION(ceil, std::ceil);
    M_UNARY_FUNCTION(trunc, std::trunc);
    M_UNARY_FUNCTION(round, std::round);
    M_UNARY_FUNCTION(exp, std::exp);
    M_UNARY_FUNCTION(exp2, std::exp2);
    M_UNARY_FUNCTION(log, std::log);
    M_UNARY_FUNCTION(log2, std::log2);
    M_UNARY_FUNCTION(sqrt, std::sqrt);
    M_UNARY_FUNCTION(sin, std::sin);
    M_UNARY_FUNCTION(cos, std::cos);
    M_UNARY_FUNCTION(tan, std::tan);
    M_UNARY_FUNCTION(asin, std::asin);
    M_UNARY_FUNCTION(acos, std::acos);
    M_UNARY_FUNCTION(atan, std::atan);
    M_UNARY_FUNCTION(sinh, std::sinh);
    M_UNARY_FUNCTION(cosh, std::cosh);
    M_UNARY_FUNCTION(tanh, std::tanh);
    M_UNARY_FUNCTION(asinh, std::asinh);
    M_UNARY_FUNCTION(acosh, std::acosh);
    M_UNARY_FUNCTION(atanh, std::atanh);

#undef M_UNARY_FUNCTION

    // standard element wise binary functions for IFixedCubicMatrix
#define M_BINARY_FUNCTION(NAME, FUNC)                                               \
    template<IFixedCubicMatrix MAT>                                                 \
    constexpr MAT NAME(const MAT& x,typename MAT::value_type y) {                   \
        MAT out{};                                                                  \
        Utilities::static_for<0, 1, MAT::columns()>([&x, &y, &out](std::size_t i) { \
            out[i] = FUNC(x[i], y);                                                 \
        });                                                                         \
        return out;                                                                 \
    }

    M_BINARY_FUNCTION(pow, std::pow);
    M_BINARY_FUNCTION(atan2, std::atan2);
    M_BINARY_FUNCTION(mod, std::fmod);

#undef M_BINARY_FUNCTION

    // named relational operations
#define M_RELATIONAL_FUNCTION(NAME, OP)                                                       \
    template<IFixedCubicMatrix MAT>                                                           \
    constexpr auto NAME(const MAT& x, const MAT& y) {                                         \
        bool result{ true };                                                                  \
        Utilities::static_for<0, 1, MAT::columns()>([&x, &y, &result](std::size_t i) {        \
            Utilities::static_for<0, 1, MAT::columns()>([i, &x, &y, &result](std::size_t j) { \
                result &= x(i, j) OP y(i, j);                                                 \
            });                                                                               \
        });                                                                                   \
        return result;                                                                        \
    }

    M_RELATIONAL_FUNCTION(equal, == );
    M_RELATIONAL_FUNCTION(notEqual, != );
    M_RELATIONAL_FUNCTION(lessThan, < );
    M_RELATIONAL_FUNCTION(lessThanEqual, <= );
    M_RELATIONAL_FUNCTION(greaterThan, > );
    M_RELATIONAL_FUNCTION(greaterThanEqual, >= );

#undef M_RELATIONAL_FUNCTION

    // matrix-matrix/scalar compound arithmetic operator overload (all are component wise), without multiplication
#define M_OPERATOR(OP)                                                                     \
    template<IFixedCubicMatrix MAT>                                                        \
    constexpr MAT& operator OP (MAT& lhs, const MAT& rhs) {                                \
        Utilities::static_for<0, 1, MAT::columns()>([&lhs, &rhs](std::size_t i) {          \
            lhs[i] OP rhs[i];                                                              \
        });                                                                                \
        return lhs;                                                                        \
    }                                                                                      \
    template<IFixedCubicMatrix MAT>                                                        \
    constexpr MAT& operator OP (MAT& lhs, MAT&& rhs) {                                     \
        Utilities::static_for<0, 1, MAT::columns()>([&lhs, r = MOV(rhs)](std::size_t i) {  \
            lhs[i] OP MOV(r[i]);                                                           \
        });                                                                                \
        return lhs;                                                                        \
    }                                                                                      \
    template<IFixedCubicMatrix MAT>                                                        \
    constexpr MAT& operator OP (MAT& lhs, const typename MAT::value_type rhs) {            \
        Utilities::static_for<0, 1, MAT::columns()>([&lhs, &rhs](std::size_t i) {          \
            lhs[i] OP rhs;                                                                 \
        });                                                                                \
        return lhs;                                                                        \
    }

    M_OPERATOR(-= );
    M_OPERATOR(+= );
    M_OPERATOR(/= );
    M_OPERATOR(&= );
    M_OPERATOR(|= );
    M_OPERATOR(^= );
    M_OPERATOR(>>= );
    M_OPERATOR(<<= );

#undef M_OPERATOR

    /**
    * \brief return a given row of a matrix
    * @param {MAT,         in}  cubic matrix
    * @param {size_t,      in}  row number
    * @param {vector_type, out} row
    **/
    template<IFixedCubicMatrix MAT, class VEC = typename MAT::vector_type>
    constexpr VEC row(const MAT& mat, const std::size_t i) noexcept {
        assert(i < MAT::columns());

        VEC r;
        Utilities::static_for<0, 1, MAT::columns()>([&r, &mat, i](std::size_t j) {
            r[j] = mat[j][i];
        });

        return r;
    }

    // overload unary arithmetic operator overload for IFixedCubicMatrix
    template<IFixedCubicMatrix MAT>
    constexpr MAT operator - (const MAT& mat) {
        MAT m(mat);
        m *= static_cast<typename MAT::value_type>(-1.0);
        return m;
    }

    /**
    * \brief perform matrix-matrix multiplication
    * @param {MAT,            in}  matrix #1
    * @param {MAT|value_type, in}  matrix #2 | scalar
    * @param {MAT,            out} matrix #1 * matrix #2
    **/
    template<IFixedCubicMatrix MAT>
    constexpr MAT& operator *= (const MAT& lhs, const MAT& rhs) {
        using VEC = typename MAT::vector_type;
        constexpr std::size_t N{ MAT::columns() };

        MAT out;
        Utilities::static_for<0, 1, N>([&lhs, &rhs, &out](std::size_t i) {
            VEC r(row(lhs, i));
            Utilities::static_for<0, 1, N>([&r, &rhs, &out, i](std::size_t j) {
                out(j, i) = dot(r, rhs[j]);
            });
        });

        return out;
    }
    template<IFixedCubicMatrix MAT, class T = typename MAT::value_type>
    constexpr MAT& operator *= (const MAT& lhs, const T rhs) {
        constexpr std::size_t N{ MAT::columns() };

        MAT out(lhs);
        Utilities::static_for<0, 1, N>([&out, rhs](std::size_t i) {
            out[i] *= rhs;
        });

        return out;
    }

    template<IFixedCubicMatrix MAT>
    constexpr MAT& operator *= (MAT& lhs, const MAT& rhs) {
        using VEC = typename MAT::vector_type;
        constexpr std::size_t N{ MAT::columns() };

        Utilities::static_for<0, 1, N>([&lhs, &rhs](std::size_t i) {
            VEC r(row(lhs, i));
            Utilities::static_for<0, 1, N>([&r, &rhs, &lhs, i](std::size_t j) {
                lhs(j, i) = dot(r, rhs[j]);
            });
        });

        return lhs;
    }
    template<IFixedCubicMatrix MAT, class T = typename MAT::value_type>
    constexpr MAT& operator *= (MAT& lhs, const T rhs) {
        constexpr std::size_t N{ MAT::columns() };

        Utilities::static_for<0, 1, N>([&lhs, rhs](std::size_t i) {
            lhs[i] *= rhs;
        });

        return lhs;
    }

    template<IFixedCubicMatrix MAT>
    constexpr MAT& operator *= (MAT& lhs, MAT&& rhs) {
        using VEC = typename MAT::vector_type;
        constexpr std::size_t N{ MAT::columns() };

        Utilities::static_for<0, 1, N>([&lhs, &rhs](std::size_t i) {
            VEC r(row(lhs, i));
            Utilities::static_for<0, 1, N>([&r, _rhs = MOV(rhs), &lhs, i](std::size_t j) {
                lhs(j, i) = dot(r, _rhs[j]);
            });
        });

        return lhs;
    }

    // matrix-matrix arithmetic operator overload
#define M_OPERATOR(OP, AOP)                                                \
    template<IFixedCubicMatrix MAT>                                        \
    constexpr MAT operator OP (MAT lhs, const MAT& rhs) {                  \
        return (lhs AOP rhs);                                              \
    }                                                                      \
    template<IFixedCubicMatrix MAT>                                        \
    constexpr MAT operator OP (MAT lhs, typename MAT::value_type rhs) {    \
        return (lhs AOP rhs);                                              \
    }                                                                      \
    template<IFixedCubicMatrix MAT>                                        \
    constexpr MAT operator OP (MAT lhs, MAT&& rhs) {                       \
        return (lhs AOP FWD(rhs));                                         \
    }

    M_OPERATOR(+, +=);
    M_OPERATOR(-, -=);
    M_OPERATOR(*, *=);
    M_OPERATOR(/ , /=);
    M_OPERATOR(&, &=);
    M_OPERATOR(| , |=);
    M_OPERATOR(^, ^=);
    M_OPERATOR(>> , >>=);
    M_OPERATOR(<< , <<=);

#undef M_OPERATOR

    /**
    * \brief perform matrix-matrix component wise multiplication
    * @param {MAT, in}  matrix #1
    * @param {MAT, in}  matrix #2
    * @param {MAT, out} matrix #1 * matrix #2 (component wise)
    **/
    template<IFixedCubicMatrix MAT>
    constexpr MAT matrixCompMult(const MAT& lhs, const MAT& rhs) noexcept {
        MAT m;

        Utilities::static_for<0, 1, MAT::columns()>([&lhs, &rhs, &m](std::size_t i) {
            Utilities::static_for<0, 1, MAT::columns()>([&lhs, &rhs, &m, i](std::size_t j) {
                m[i][j] = lhs(i, j) * rhs(i, j);
            });
        });

        return m;
    }

    /**
    * \brief perform matrix-vector multiplication (right multiplication)
    *        this is slower than vector-matrix multiplicaton (left multiplication)
    * @param {MAT, in}  matrix
    * @param {VEC, in}  vector
    * @param {MAT, out} matrix * vector
    **/
    template<IFixedCubicMatrix MAT, IFixedVector VEC>
        requires(std::is_same_v<typename MAT::vector_type, VEC> &&
                 std::is_same_v<typename MAT::value_type, typename VEC::value_type> &&
                 MAT::columns() == VEC::length())
    constexpr VEC operator * (MAT& lhs, const VEC& rhs) {
        VEC out;
        Utilities::static_for<0, 1, MAT::columns()>([&lhs, &rhs, &out](std::size_t i) {
            out[i] = dot(row(lhs, i), rhs);
        });
        return out;
    }
    template<IFixedCubicMatrix MAT, IFixedVector VEC>
        requires(std::is_same_v<typename MAT::vector_type, VEC> &&
                 std::is_same_v<typename MAT::value_type, typename VEC::value_type> &&
                 MAT::columns() == VEC::length())
    constexpr VEC operator * (const MAT& lhs, const VEC& rhs) {
        VEC out;
        Utilities::static_for<0, 1, MAT::columns()>([&lhs, &rhs, &out](std::size_t i) {
            out[i] = dot(row(lhs, i), rhs);
        });
        return out;
    }

    /**
    * \brief perform vector-matrix multiplication (left multiplication)
    *        this is faster than matrix-vector multiplicaton (right multiplication)
    * @param {VEC, in}  vector
    * @param {MAT, in}  matrix
    * @param {MAT, out} vector * matrix
    **/
    template<IFixedCubicMatrix MAT, IFixedVector VEC>
        requires(std::is_same_v<typename MAT::vector_type, VEC> &&
                 std::is_same_v<typename MAT::value_type, typename VEC::value_type> &&
                 MAT::columns() == VEC::length())
    constexpr VEC operator * (VEC& lhs, const MAT& rhs) {
        const VEC _lhs(lhs);
        Utilities::static_for<0, 1, MAT::columns()>([&lhs, &rhs, &_lhs](std::size_t i) {
            lhs[i] = dot(_lhs, rhs[i]);
        });
        return lhs;
    }
    template<IFixedCubicMatrix MAT, IFixedVector VEC>
        requires(std::is_same_v<typename MAT::vector_type, VEC> &&
                 std::is_same_v<typename MAT::value_type, typename VEC::value_type> &&
                 MAT::columns() == VEC::length())
    constexpr VEC operator * (const VEC& lhs, const MAT& rhs) {
        VEC out;
        Utilities::static_for<0, 1, MAT::columns()>([&lhs, &rhs, &out](std::size_t i) {
            out[i] = dot(lhs, rhs[i]);
        });
        return out;
    }

    /**
    * \brief calculate matrix determinant
    *        (to calculate determinant of larger matrices - see 'Glsl_solvers.h' - Decomposition::determinant_using_lu and Decomposition::determinant_using_qr
    * @param {MAT,        in}  matrix
    * @param {value_type, out} matrix determinant
    **/
    template<IFixedCubicMatrix MAT, class T = typename MAT::value_type>
        requires(MAT::columns() <= 4)
    constexpr T determinant(const MAT& mat) noexcept {
        using VEC = typename MAT::vector_type;
        constexpr std::size_t N{ MAT::columns() };

        if constexpr (N == 2) {
            return (mat(0,0) * mat(1,1) - mat(0,1) * mat(1,0));
        }
        else if constexpr (N == 3) {
            const VEC& x(mat[0]);
            const VEC& y(mat[1]);
            const VEC& z(mat[2]);
            return (x.x * (z.z * y.y - y.z * z.y) +
                    x.y * (y.z * z.x - z.z * y.x) +
                    x.z * (z.y * y.x - y.y * z.x));
        }
        else if constexpr (N == 4) {
            const VEC& x(mat[0]);
            const VEC& y(mat[1]);
            const VEC& z(mat[2]);
            const VEC& w(mat[3]);

            const T b00{ x.x * y.y - x.y * y.x };
            const T b01{ x.x * y.z - x.z * y.x };
            const T b02{ x.x * y.w - x.w * y.x };
            const T b03{ x.y * y.z - x.z * y.y };
            const T b04{ x.y * y.w - x.w * y.y };
            const T b05{ x.z * y.w - x.w * y.z };
            const T b06{ z.x * w.y - z.y * w.x };
            const T b07{ z.x * w.z - z.z * w.x };
            const T b08{ z.x * w.w - z.w * w.x };
            const T b09{ z.y * w.z - z.z * w.y };
            const T b10{ z.y * w.w - z.w * w.y };
            const T b11{ z.z * w.w - z.w * w.z };

            return (b00 * b11 -
                    b01 * b10 +
                    b02 * b09 +
                    b03 * b08 -
                    b04 * b07 +
                    b05 * b06);
        }
    }

    /**
    * \brief transpose a matrix
    * @param {MAT, in}  matrix
    * @param {MAT, out} transposed matrix
    **/
    template<IFixedCubicMatrix MAT>
    constexpr MAT transpose(const MAT& mat) noexcept {
        constexpr std::size_t N{ MAT::columns() };

        if constexpr (N == 2) {
            return MAT(mat(0, 0), mat(1, 0),
                       mat(0, 1), mat(1, 1));
        }
        else if constexpr (N == 3) {
            return MAT(mat(0, 0), mat(1, 0), mat(2, 0),
                       mat(0, 1), mat(1, 1), mat(2, 1),
                       mat(0, 2), mat(1, 2), mat(2, 2));
        }
        else if constexpr (N == 4) {
            return MAT(mat(0, 0), mat(1, 0), mat(2, 0), mat(3, 0),
                       mat(0, 1), mat(1, 1), mat(2, 1), mat(3, 1),
                       mat(0, 2), mat(1, 2), mat(2, 2), mat(3, 2),
                       mat(0, 3), mat(1, 3), mat(2, 3), mat(3, 3));
        }
        else {
            MAT out(mat);
            Utilities::static_for<0, 1, MAT::columns()>([&out](std::size_t i) {
                for (std::size_t j{ i + 1 }; j < MAT::columns(); ++j) {
                    Utilities::swap(out(i, j), out(j, i));
                }
            });
            return out;
        }
    }

    /**
    * \brief invert a matrix (to invert larger matrices - see 'Glsl_solvers.h' - Decomposition::inverse_using_lu)
    * @param {MAT, in}  matrix
    * @param {MAT, out} matrix inverse
    **/
    template<IFixedCubicMatrix MAT>
        requires(MAT::columns() <= 4)
    constexpr MAT inverse(const MAT& mat) {
        using T = typename MAT::value_type;
        using VEC = typename MAT::vector_type;
        constexpr std::size_t N{ MAT::columns() };

        if constexpr (N == 2) {
            const VEC x(mat[0]);
            const VEC y(mat[1]);
            const T det{ x.x * y.y - x.y * y.x };
            assert(!Numerics::areEquals(det, T{}));

            return MAT( y.y / det, -x.y / det,
                       -y.x / det,  x.x / det);
        }
        else if constexpr (N == 3) {
            const VEC x(mat[0]);
            const VEC y(mat[1]);
            const VEC z(mat[2]);

            const T a00{ x.x };
            const T a01{ x.y };
            const T a02{ x.z };
            const T a10{ y.x };
            const T a11{ y.y };
            const T a12{ y.z };
            const T a20{ z.x };
            const T a21{ z.y };
            const T a22{ z.z };

            const T b01{  a22 * a11 - a12 * a21 };
            const T b11{ -a22 * a10 + a12 * a20 };
            const T b21{  a21 * a10 - a11 * a20 };

            const T det{ a00 * b01 + a01 * b11 + a02 * b21 };
            assert(!Numerics::areEquals(det, T{}));

            return MAT(b01 / det, (-a22 * a01 + a02 * a21) / det, ( a12 * a01 - a02 * a11) / det,
                       b11 / det, ( a22 * a00 - a02 * a20) / det, (-a12 * a00 + a02 * a10) / det,
                       b21 / det, (-a21 * a00 + a01 * a20) / det, ( a11 * a00 - a01 * a10) / det);
        }
        else if constexpr (N == 4) {
            const VEC x(mat[0]);
            const VEC y(mat[1]);
            const VEC z(mat[2]);
            const VEC w(mat[3]);

            const T a00{ x.x };
            const T a01{ x.y };
            const T a02{ x.z };
            const T a03{ x.w };
            const T a10{ y.x };
            const T a11{ y.y };
            const T a12{ y.z };
            const T a13{ y.w };
            const T a20{ z.x };
            const T a21{ z.y };
            const T a22{ z.z };
            const T a23{ z.w };
            const T a30{ w.x };
            const T a31{ w.y };
            const T a32{ w.z };
            const T a33{ w.w };

            const T b00{ a00 * a11 - a01 * a10 };
            const T b01{ a00 * a12 - a02 * a10 };
            const T b02{ a00 * a13 - a03 * a10 };
            const T b03{ a01 * a12 - a02 * a11 };
            const T b04{ a01 * a13 - a03 * a11 };
            const T b05{ a02 * a13 - a03 * a12 };
            const T b06{ a20 * a31 - a21 * a30 };
            const T b07{ a20 * a32 - a22 * a30 };
            const T b08{ a20 * a33 - a23 * a30 };
            const T b09{ a21 * a32 - a22 * a31 };
            const T b10{ a21 * a33 - a23 * a31 };
            const T b11{ a22 * a33 - a23 * a32 };

            const T det{ b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06 };
            assert(!Numerics::areEquals(det, T{}));

            return MAT((a11 * b11 - a12 * b10 + a13 * b09) / det,
                       (a02 * b10 - a01 * b11 - a03 * b09) / det,
                       (a31 * b05 - a32 * b04 + a33 * b03) / det,
                       (a22 * b04 - a21 * b05 - a23 * b03) / det,
                       (a12 * b08 - a10 * b11 - a13 * b07) / det,
                       (a00 * b11 - a02 * b08 + a03 * b07) / det,
                       (a32 * b02 - a30 * b05 - a33 * b01) / det,
                       (a20 * b05 - a22 * b02 + a23 * b01) / det,
                       (a10 * b10 - a11 * b08 + a13 * b06) / det,
                       (a01 * b08 - a00 * b10 - a03 * b06) / det,
                       (a30 * b04 - a31 * b02 + a33 * b00) / det,
                       (a21 * b02 - a20 * b04 - a23 * b00) / det,
                       (a11 * b07 - a10 * b09 - a12 * b06) / det,
                       (a00 * b09 - a01 * b07 + a02 * b06) / det,
                       (a31 * b01 - a30 * b03 - a32 * b00) / det,
                       (a20 * b03 - a21 * b01 + a22 * b00) / det);
        }
    }

    /**
    * \brief return Frobenius norm of a matrix
    * @param {MAT,        in}  matrix
    * @param {value_type, out} Frobenius norm
    **/
    template<IFixedCubicMatrix MAT>
    constexpr auto frobenius_norm(const MAT& mat) noexcept {
        typename MAT::value_type frob{};
        Utilities::static_for<0, 1, MAT::columns()>([&mat, &frob](std::size_t i) {
            frob += dot(mat[i]);
        });

        [[assume(frob >= T{})]];
        return std::sqrt(frob);
    }

    /**
    * \brief return the trace (diagonal) of a matrix
    * @param {MAT, in}  matrix
    * @param {VEC, out} trace
    **/
    template<IFixedCubicMatrix MAT, class VEC = typename MAT::vector_type>
    constexpr VEC trace(const MAT& mat) noexcept {
        VEC out;
        Utilities::static_for<0, 1, MAT::columns()>([&mat, &out](std::size_t i) {
            out[i] = mat(i, i);
        });
        return out;
    }

    /**
    * \brief returns minimal element in matrix
    * @param {MAT,        in}  matrix
    * @param {value_type, out} maximal element
    **/
    template<IFixedCubicMatrix MAT, class VEC = typename MAT::vector_type, class T = typename MAT::value_type>
    constexpr T min(const MAT& mat) noexcept {
        VEC minColumns;
        Utilities::static_for<0, 1, MAT::columns()>([&minColumns, &mat](std::size_t i) {
            minColumns[i] = GLSL::min(mat[i]);
        });
        return GLSL::min(minColumns);
    }

    /**
    * \brief returns maximal element in matrix
    * @param {MAT,        in}  matrix
    * @param {value_type, out} maximal element
    **/
    template<IFixedCubicMatrix MAT, class VEC = typename MAT::vector_type, class T = typename MAT::value_type>
    constexpr T max(const MAT& mat) noexcept {
        VEC maxColumns;
        Utilities::static_for<0, 1, MAT::columns()>([&maxColumns, &mat](std::size_t i) {
            maxColumns[i] = GLSL::max(mat[i]);
        });
        return GLSL::max(maxColumns);
    }

    //
    // IFixedVector constructs
    //

    /**
    * \brief 'swizzling' class (implements IFixedVector concept)
    *
    * @param {arithmetic, in} swizzled element underlying type
    * @param {size_t,     in} swizzled element length, i.e. - VectorBase<..,N>
    * @param {Indexes..., in} elements to swizzle, given as indices's in array
    *                         their length should be equal to the underlying VectorBase, i.e. sizeof...(Indexes) = N
    **/
    template<typename T, std::size_t N, std::size_t... Indexes>
        requires(std::is_arithmetic_v<T> && (N > 0) && (N == sizeof...(Indexes)))
    class Swizzle final {
        // internals
        private:
            AlignedStorage(T) std::array<T, N> data{};

        // API
        public:
            //
            // constructors, assignments and casting
            //

            // underlying type and amount of elements
            using value_type = T;
            static constexpr std::integral_constant<std::size_t, N> length = {};

            // default constructors
            constexpr Swizzle() noexcept = default;

            // construct from a single value
            explicit constexpr Swizzle(const T value) noexcept {
                Utilities::static_for<0, 1, N>([this, value](std::size_t i) {
                    data[i] = value;
                });
            }

            // construct from a moveable array
            explicit constexpr Swizzle(std::array<T, length>&& _data) noexcept : data(Utilities::exchange(_data, std::array<T, length>{})) {}

            // construct from a parameter pack
            template<typename...TS>
                requires(std::is_same_v<T, TS> && ...)
            explicit constexpr Swizzle(TS&&... values) noexcept : data{ FWD(values)... } {}

            // assign a moveable array
            constexpr Swizzle& operator=(std::array<T, length>&& _data) noexcept {
                data = Utilities::exchange(_data, std::array<T, length>{});
                return *this;
            };

            // copy semantics
            template<std::size_t... OtherIndexes>
                requires(N == sizeof...(OtherIndexes))
            explicit(false) Swizzle(const Swizzle<T, N, OtherIndexes...>& other) {
                static_assert(Variadic::lowerThan<N>(Indexes...));
                static_assert(Variadic::lowerThan<N>(OtherIndexes...));
                constexpr std::array<std::size_t, N> indexesLhs{ Indexes... };
                constexpr std::array<std::size_t, N> indexesRhs{ OtherIndexes... };

                Utilities::static_for<0, 1, N>([this, &other, indexesLhs, indexesRhs](std::size_t i) {
                    data[indexesLhs[i]] = other[indexesRhs[i]];
                });
            }
            template<std::size_t... OtherIndexes>
                requires(N == sizeof...(OtherIndexes))
            Swizzle& operator=(const Swizzle<T, N, OtherIndexes...>& other) {
                static_assert(Variadic::lowerThan<N>(Indexes...));
                static_assert(Variadic::lowerThan<N>(OtherIndexes...));
                constexpr std::array<std::size_t, N> indexesLhs{ Indexes... };
                constexpr std::array<std::size_t, N> indexesRhs{ OtherIndexes... };

                Utilities::static_for<0, 1, N>([this, &other, indexesLhs, indexesRhs](std::size_t i) {
                    data[indexesLhs[i]] = other[indexesRhs[i]];
                });

                return *this;
            }

            // move semantics
            template<std::size_t... OtherIndexes>
                requires(N == sizeof...(OtherIndexes))
            explicit(false) Swizzle(Swizzle<T, N, OtherIndexes...>&& other) noexcept {
                static_assert(Variadic::lowerThan<N>(Indexes...));
                static_assert(Variadic::lowerThan<N>(OtherIndexes...));
                constexpr std::array<std::size_t, N> indexesLhs{ Indexes... };
                constexpr std::array<std::size_t, N> indexesRhs{ OtherIndexes... };

                Utilities::static_for<0, 1, N>([this, &other, indexesLhs, indexesRhs](std::size_t i) {
                    data[indexesLhs[i]] = MOV(other[indexesRhs[i]]);
                });
            }
            template<std::size_t... OtherIndexes>
                requires(N == sizeof...(OtherIndexes))
            Swizzle& operator=(Swizzle<T, N, OtherIndexes...>&& other) noexcept {
                static_assert(Variadic::lowerThan<N>(Indexes...));
                static_assert(Variadic::lowerThan<N>(OtherIndexes...));
                constexpr std::array<std::size_t, N> indexesLhs{ Indexes... };
                constexpr std::array<std::size_t, N> indexesRhs{ OtherIndexes... };

                Utilities::static_for<0, 1, N>([this, &other, indexesLhs, indexesRhs](std::size_t i) {
                    data[indexesLhs[i]] = MOV(other[indexesRhs[i]]);
                });

                return *this;
            }

            // cast as IFixedVector
            template<IFixedVector VEC> explicit(false) operator VEC() const {
                constexpr std::array<std::size_t, N> indexes{ Indexes... };

                VEC pack;
                Utilities::static_for<0, 1, N>([this, &pack, indexes](std::size_t i) {
                    assert(indexes[i] < N);
                    pack[i] = data.at(indexes[i]);
                });

                return pack;
            }

            // overload stream '<<' operator
            friend std::ostream& operator<<(std::ostream& xio_stream, const Swizzle& swizzle) {
                xio_stream << "{";
                Utilities::static_for<0, 1, N - 1>([&xio_stream, &swizzle](std::size_t i) {
                    xio_stream << swizzle[i] << ", ";
                });
                xio_stream << swizzle[N-1] << "}";

                return xio_stream;
            }

            //
            // operator overloading
            //

            // overload operator '[]' for element access
            constexpr T  operator[](const std::size_t i) const { constexpr std::array<std::size_t, N> indexes{ Indexes... }; assert(i < N); assert(indexes[i] < N); return data.at(indexes[i]); }
            constexpr T& operator[](const std::size_t i) { constexpr std::array<std::size_t, N> indexes{ Indexes... }; assert(i < N); assert(indexes[i] < N); return data[indexes[i]]; }

            // compound arithmetic operator overload
#define M_OPERATOR(OP)                                                                                    \
            template<std::size_t... OtherIndexes>                                                         \
                requires(N == sizeof...(OtherIndexes))                                                    \
            Swizzle& operator OP (const Swizzle<T, N, OtherIndexes...>& other) {                          \
                static_assert(Variadic::lowerThan<N>(Indexes...));                                        \
                static_assert(Variadic::lowerThan<N>(OtherIndexes...));                                   \
                constexpr std::array<std::size_t, N> indexesLhs{ Indexes... };                            \
                constexpr std::array<std::size_t, N> indexesRhs{ OtherIndexes... };                       \
                Utilities::static_for<0, 1, N>([this, &other, indexesLhs, indexesRhs](std::size_t i) {    \
                    data[indexesLhs[i]] OP other[indexesRhs[i]];                                          \
                });                                                                                       \
                return *this;                                                                             \
            }                                                                                             \
            template<std::size_t... OtherIndexes>                                                         \
                requires(N == sizeof...(OtherIndexes))                                                    \
            Swizzle& operator OP (Swizzle<T, N, OtherIndexes...>&& other) {                               \
                static_assert(Variadic::lowerThan<N>(Indexes...));                                        \
                static_assert(Variadic::lowerThan<N>(OtherIndexes...));                                   \
                constexpr std::array<std::size_t, N> indexesLhs{ Indexes... };                            \
                constexpr std::array<std::size_t, N> indexesRhs{ OtherIndexes... };                       \
                Utilities::static_for<0, 1, N>([this, &other, indexesLhs, indexesRhs](std::size_t i) {    \
                    data[indexesLhs[i]] OP MOV(other[indexesRhs[i]]);                                     \
                });                                                                                       \
                return *this;                                                                             \
            }                                                                                             \
            Swizzle& operator OP (T other) {                                                              \
                Utilities::static_for<0, 1, N>([this, &other](std::size_t i) {                            \
                    data[i] OP other;                                                                     \
                });                                                                                       \
                return *this;                                                                             \
            }

        M_OPERATOR(-= );
        M_OPERATOR(+= );
        M_OPERATOR(*= );
        M_OPERATOR(/= );
        M_OPERATOR(&= );
        M_OPERATOR(|= );
        M_OPERATOR(^= );
        M_OPERATOR(>>= );
        M_OPERATOR(<<= );

#undef M_OPERATOR

            // arithmetic operator overload
#define M_OPERATOR(OP, AOP)                                                                                            \
            template<std::size_t... OtherIndexes>                                                                      \
            friend Swizzle operator OP (Swizzle<T, N, Indexes...> lhs, const Swizzle<T, N, OtherIndexes...>& rhs) {    \
                return (lhs AOP rhs);                                                                                  \
            }                                                                                                          \
            template<std::size_t... OtherIndexes>                                                                      \
            friend Swizzle operator OP (Swizzle<T, N, Indexes...> lhs, Swizzle<T, N, OtherIndexes...>&& rhs) {         \
                return (lhs AOP FWD(rhs));                                                                             \
            }                                                                                                          \
            friend Swizzle operator OP (Swizzle<T, N, Indexes...> lhs, T rhs) {                                        \
                return (lhs AOP rhs);                                                                                  \
            }

        M_OPERATOR(+, +=);
        M_OPERATOR(-, -=);
        M_OPERATOR(*, *=);
        M_OPERATOR(/ , /=);
        M_OPERATOR(&, &=);
        M_OPERATOR(| , |=);
        M_OPERATOR(^, ^=);
        M_OPERATOR(>> , >>=);
        M_OPERATOR(<< , <<=);

#undef M_OPERATOR     

            bool operator==(const Swizzle& other) const {
                return (data == other.data);
            }
    };

    // swizzle traits and concept
    template<typename>                                          struct is_swizzle : public std::false_type {};
    template<typename T, std::size_t N, std::size_t... Indexes> struct is_swizzle<Swizzle<T, N, Indexes...>> : public std::true_type {};
    template<typename T> constexpr bool is_swizzle_v = is_swizzle<T>::value;
    template<typename T> concept ISwizzle = is_swizzle_v<T>;

    // overload unary arithmetic operator overload for Swizzle
    template<ISwizzle SWZ>
    constexpr auto operator - (const SWZ& swizzle) {
        SWZ s(swizzle);

        Utilities::static_for<0, 1, SWZ::length()>([&s](std::size_t i) {
            s[i] *= static_cast<typename SWZ::value_type>(-1.0);
        });

        return s;
    }

    /**
    * \brief 1x2 numerical vector implementation of IFixedVector concept
    * @param {arithmetic, in} elements underlying type
    **/
    template<typename T>
        requires(std::is_arithmetic_v<T>)
    struct Vector2 final {
        static constexpr std::integral_constant<std::size_t, 2> length = {};
        using value_type = T;

        // accessors
        union {
            // field access
            struct AlignedStorage(T) { T x, y; };

            // array
            AlignedStorage(T) std::array<T, 2> data{};

            // swizzle
            Swizzle<T, 2, 0, 0> xx;
            Swizzle<T, 2, 0, 1> xy;
            Swizzle<T, 2, 1, 0> yx;
            Swizzle<T, 2, 1, 1> yy;
        };

        // construct from a single value
        constexpr explicit Vector2(const T value) noexcept : x(value), y(value) {}
        constexpr explicit Vector2(const T _x, const T _y) noexcept : x(_x), y(_y) {}

        // construct from a moveable array
        constexpr explicit Vector2(std::array<T, length>&& _data) noexcept : data(Utilities::exchange(_data, std::array<T, length>{})) {}

        // construct from a parameter pack
        template<typename...TS>
            requires(std::is_same_v<T, TS> && ...)
        constexpr explicit Vector2(TS&&... values) noexcept : data{ FWD(values)... } {}

        // constructo from a pointer
        constexpr explicit Vector2(const T* _data) {
            [[assume(_data != nullptr)]];
            AssumeAligned(T, _data);
            std::memcpy(data.data(), _data, length * sizeof(T));
        }

        // assign a moveable array
        constexpr Vector2& operator=(std::array<T, length>&& _data) noexcept {
            data = Utilities::exchange(_data, std::array<T, length>{});
            return *this;
        };

        // overload stream '<<' operator
        friend std::ostream& operator<<(std::ostream& xio_stream, const Vector2& vec) {
            return xio_stream << "{" << vec[0] << ", " << vec[1] << "}";
        }

        // overload operator '[]'
        constexpr T  operator[](const std::size_t i) const { assert(i < length); return data.at(i); }
        constexpr T& operator[](const std::size_t i) { assert(i < length); return data[i]; }
    };

    // Vector2 traits and concept
    template<typename>   struct is_vector2 : public std::false_type {};
    template<typename T> struct is_vector2<Vector2<T>> : public std::true_type {};
    template<typename T> constexpr bool is_vector2_v = is_vector2<T>::value;
    template<typename T> concept IVector2 = is_vector2_v<T>;

    /**
    * \brief 1x3 numerical vector implementation of IFixedVector concept
    * @param {arithmetic, in} elements underlying type
    **/
    template<typename T>
        requires(std::is_arithmetic_v<T>)
    struct Vector3 final {
        static constexpr std::integral_constant<std::size_t, 3> length = {};
        using value_type = T;


        // accessors
        union {
            // field access
            struct AlignedStorage(T) { T x, y, z; };

            // array
            AlignedStorage(T) std::array<T, 3> data{};

            // swizzle
            Swizzle<T, 2, 0, 0> xx;
            Swizzle<T, 2, 0, 1> xy;
            Swizzle<T, 2, 0, 2> xz;
            Swizzle<T, 2, 1, 0> yx;
            Swizzle<T, 2, 1, 1> yy;
            Swizzle<T, 2, 1, 2> yz;
            Swizzle<T, 2, 2, 0> zx;
            Swizzle<T, 2, 2, 1> zy;
            Swizzle<T, 2, 2, 2> zz;
            Swizzle<T, 3, 0, 0, 0> xxx;
            Swizzle<T, 3, 0, 0, 1> xxy;
            Swizzle<T, 3, 0, 0, 2> xxz;
            Swizzle<T, 3, 0, 1, 0> xyx;
            Swizzle<T, 3, 0, 1, 1> xyy;
            Swizzle<T, 3, 0, 1, 2> xyz;
            Swizzle<T, 3, 0, 2, 0> xzx;
            Swizzle<T, 3, 0, 2, 1> xzy;
            Swizzle<T, 3, 0, 2, 2> xzz;
            Swizzle<T, 3, 1, 0, 0> yxx;
            Swizzle<T, 3, 1, 0, 1> yxy;
            Swizzle<T, 3, 1, 0, 2> yxz;
            Swizzle<T, 3, 1, 1, 0> yyx;
            Swizzle<T, 3, 1, 1, 1> yyy;
            Swizzle<T, 3, 1, 1, 2> yyz;
            Swizzle<T, 3, 1, 2, 0> yzx;
            Swizzle<T, 3, 1, 2, 1> yzy;
            Swizzle<T, 3, 1, 2, 2> yzz;
            Swizzle<T, 3, 2, 0, 0> zxx;
            Swizzle<T, 3, 2, 0, 1> zxy;
            Swizzle<T, 3, 2, 0, 2> zxz;
            Swizzle<T, 3, 2, 1, 0> zyx;
            Swizzle<T, 3, 2, 1, 1> zyy;
            Swizzle<T, 3, 2, 1, 2> zyz;
            Swizzle<T, 3, 2, 2, 0> zzx;
            Swizzle<T, 3, 2, 2, 1> zzy;
            Swizzle<T, 3, 2, 2, 2> zzz;
        };

        // IFixedVector constraints
    public:

        // construct from a single/several value
        constexpr explicit Vector3(const T value) noexcept : x(value), y(value), z(value) {}
        constexpr explicit Vector3(const T _x, const T _y, const T _z) noexcept : x(_x), y(_y), z(_z) {}

        // construct from a moveable array
        constexpr explicit Vector3(std::array<T, length>&& _data) noexcept : data(Utilities::exchange(_data, std::array<T, length>{})) {}

        // construct from a parameter pack
        template<typename...TS>
            requires(std::is_same_v<T, TS> && ...)
        constexpr explicit Vector3(TS&&... values) noexcept : data{ FWD(values)... } {}

        // constructo from a pointer
        constexpr explicit Vector3(const T* _data) {
            [[assume(_data != nullptr)]];
            AssumeAligned(T, _data);
            std::memcpy(data.data(), _data, length * sizeof(T));
        }

        // assign a moveable array
        constexpr Vector3& operator=(std::array<T, length>&& _data) noexcept {
            data = Utilities::exchange(_data, std::array<T, length>{});
            return *this;
        };

        // overload stream '<<' operator
        friend std::ostream& operator<<(std::ostream& xio_stream, const Vector3& vec) {
            return xio_stream << "{" << vec[0] << ", " << vec[1] << ", " << vec[2] << "}";
        }

        // overload operator '[]'
        constexpr T  operator[](const std::size_t i) const { assert(i < length); return data.at(i); }
        constexpr T& operator[](const std::size_t i) { assert(i < length); return data[i]; }

        // public API
    public:

        // construct from scalar and vector2
        constexpr Vector3(const T x, const Vector2<T>& yz) noexcept : data{ x,     yz[0], yz[1] } {}
        constexpr Vector3(const Vector2<T>& xy, const T z = T{}) noexcept : data{ xy[0], xy[1], z } {}
    };

    // Vector3 traits and concept
    template<typename>   struct is_vector3 : public std::false_type {};
    template<typename T> struct is_vector3<Vector3<T>> : public std::true_type {};
    template<typename T> constexpr bool is_vector3_v = is_vector3<T>::value;
    template<typename T> concept IVector3 = is_vector3_v<T>;

    /**
    * \brief 1x4 numerical vector implementation of IFixedVector concept
    * @param {arithmetic, in} elements underlying type
    **/
    template<typename T>
        requires(std::is_arithmetic_v<T>)
    struct Vector4 final {
        static constexpr std::integral_constant<std::size_t, 4> length = {};
        using value_type = T;

        // accessors
        union {
            // field access
            struct AlignedStorage(T) { T x, y, z, w; };

            // array
            AlignedStorage(T) std::array<T, 4> data{};

            // swizzle
            Swizzle<T, 2, 0, 0> xx;
            Swizzle<T, 2, 0, 1> xy;
            Swizzle<T, 2, 0, 2> xz;
            Swizzle<T, 2, 0, 3> xw;
            Swizzle<T, 2, 1, 0> yx;
            Swizzle<T, 2, 1, 1> yy;
            Swizzle<T, 2, 1, 2> yz;
            Swizzle<T, 2, 1, 3> yw;
            Swizzle<T, 2, 2, 0> zx;
            Swizzle<T, 2, 2, 1> zy;
            Swizzle<T, 2, 2, 2> zz;
            Swizzle<T, 2, 2, 3> zw;
            Swizzle<T, 2, 3, 0> wx;
            Swizzle<T, 2, 3, 1> wy;
            Swizzle<T, 2, 3, 2> wz;
            Swizzle<T, 2, 3, 3> ww;
            Swizzle<T, 3, 0, 0, 0> xxx;
            Swizzle<T, 3, 0, 0, 1> xxy;
            Swizzle<T, 3, 0, 0, 2> xxz;
            Swizzle<T, 3, 0, 0, 3> xxw;
            Swizzle<T, 3, 0, 1, 0> xyx;
            Swizzle<T, 3, 0, 1, 1> xyy;
            Swizzle<T, 3, 0, 1, 2> xyz;
            Swizzle<T, 3, 0, 1, 3> xyw;
            Swizzle<T, 3, 0, 2, 0> xzx;
            Swizzle<T, 3, 0, 2, 1> xzy;
            Swizzle<T, 3, 0, 2, 2> xzz;
            Swizzle<T, 3, 0, 2, 3> xzw;
            Swizzle<T, 3, 1, 0, 0> yxx;
            Swizzle<T, 3, 1, 0, 1> yxy;
            Swizzle<T, 3, 1, 0, 2> yxz;
            Swizzle<T, 3, 1, 0, 3> yxw;
            Swizzle<T, 3, 1, 1, 0> yyx;
            Swizzle<T, 3, 1, 1, 1> yyy;
            Swizzle<T, 3, 1, 1, 2> yyz;
            Swizzle<T, 3, 1, 1, 3> yyw;
            Swizzle<T, 3, 1, 2, 0> yzx;
            Swizzle<T, 3, 1, 2, 1> yzy;
            Swizzle<T, 3, 1, 2, 2> yzz;
            Swizzle<T, 3, 1, 2, 3> yzw;
            Swizzle<T, 3, 2, 0, 0> zxx;
            Swizzle<T, 3, 2, 0, 1> zxy;
            Swizzle<T, 3, 2, 0, 2> zxz;
            Swizzle<T, 3, 2, 0, 3> zxw;
            Swizzle<T, 3, 2, 1, 0> zyx;
            Swizzle<T, 3, 2, 1, 1> zyy;
            Swizzle<T, 3, 2, 1, 2> zyz;
            Swizzle<T, 3, 2, 1, 3> zyw;
            Swizzle<T, 3, 2, 2, 0> zzx;
            Swizzle<T, 3, 2, 2, 1> zzy;
            Swizzle<T, 3, 2, 2, 2> zzz;
            Swizzle<T, 3, 2, 2, 3> zzw;
            Swizzle<T, 3, 3, 0, 1> wxy;
            Swizzle<T, 3, 3, 0, 2> wxz;
            Swizzle<T, 3, 3, 0, 3> wxw;
            Swizzle<T, 3, 3, 1, 0> wyx;
            Swizzle<T, 3, 3, 1, 1> wyy;
            Swizzle<T, 3, 3, 1, 2> wyz;
            Swizzle<T, 3, 3, 1, 3> wyw;
            Swizzle<T, 3, 3, 2, 0> wzx;
            Swizzle<T, 3, 3, 2, 1> wzy;
            Swizzle<T, 3, 3, 2, 2> wzz;
            Swizzle<T, 3, 3, 2, 3> wzw;
            Swizzle<T, 4, 0, 0, 0, 0> xxxx;
            Swizzle<T, 4, 0, 0, 0, 1> xxxy;
            Swizzle<T, 4, 0, 0, 0, 2> xxxz;
            Swizzle<T, 4, 0, 0, 0, 3> xxxw;
            Swizzle<T, 4, 0, 0, 1, 0> xxyx;
            Swizzle<T, 4, 0, 0, 1, 1> xxyy;
            Swizzle<T, 4, 0, 0, 1, 2> xxyz;
            Swizzle<T, 4, 0, 0, 1, 3> xxyw;
            Swizzle<T, 4, 0, 0, 2, 0> xxzx;
            Swizzle<T, 4, 0, 0, 2, 1> xxzy;
            Swizzle<T, 4, 0, 0, 2, 2> xxzz;
            Swizzle<T, 4, 0, 0, 2, 3> xxzw;
            Swizzle<T, 4, 0, 0, 3, 0> xxwx;
            Swizzle<T, 4, 0, 0, 3, 1> xxwy;
            Swizzle<T, 4, 0, 0, 3, 2> xxwz;
            Swizzle<T, 4, 0, 0, 3, 3> xxww;
            Swizzle<T, 4, 0, 1, 0, 0> xyxx;
            Swizzle<T, 4, 0, 1, 0, 1> xyxy;
            Swizzle<T, 4, 0, 1, 0, 2> xyxz;
            Swizzle<T, 4, 0, 1, 0, 3> xyxw;
            Swizzle<T, 4, 0, 1, 1, 0> xyyx;
            Swizzle<T, 4, 0, 1, 1, 1> xyyy;
            Swizzle<T, 4, 0, 1, 1, 2> xyyz;
            Swizzle<T, 4, 0, 1, 1, 3> xyyw;
            Swizzle<T, 4, 0, 1, 2, 0> xyzx;
            Swizzle<T, 4, 0, 1, 2, 1> xyzy;
            Swizzle<T, 4, 0, 1, 2, 2> xyzz;
            Swizzle<T, 4, 0, 1, 2, 3> xyzw;
            Swizzle<T, 4, 0, 1, 3, 0> xywx;
            Swizzle<T, 4, 0, 1, 3, 1> xywy;
            Swizzle<T, 4, 0, 1, 3, 2> xywz;
            Swizzle<T, 4, 0, 1, 3, 3> xyww;
            Swizzle<T, 4, 0, 2, 0, 0> xzxx;
            Swizzle<T, 4, 0, 2, 0, 1> xzxy;
            Swizzle<T, 4, 0, 2, 0, 2> xzxz;
            Swizzle<T, 4, 0, 2, 0, 3> xzxw;
            Swizzle<T, 4, 0, 2, 1, 0> xzyx;
            Swizzle<T, 4, 0, 2, 1, 1> xzyy;
            Swizzle<T, 4, 0, 2, 1, 2> xzyz;
            Swizzle<T, 4, 0, 2, 1, 3> xzyw;
            Swizzle<T, 4, 0, 2, 2, 0> xzzx;
            Swizzle<T, 4, 0, 2, 2, 1> xzzy;
            Swizzle<T, 4, 0, 2, 2, 2> xzzz;
            Swizzle<T, 4, 0, 2, 2, 3> xzzw;
            Swizzle<T, 4, 0, 2, 3, 0> xzwx;
            Swizzle<T, 4, 0, 2, 3, 1> xzwy;
            Swizzle<T, 4, 0, 2, 3, 2> xzwz;
            Swizzle<T, 4, 0, 2, 3, 3> xzww;
            Swizzle<T, 4, 0, 3, 0, 0> xwxx;
            Swizzle<T, 4, 0, 3, 0, 1> xwxy;
            Swizzle<T, 4, 0, 3, 0, 2> xwxz;
            Swizzle<T, 4, 0, 3, 0, 3> xwxw;
            Swizzle<T, 4, 0, 3, 1, 0> xwyx;
            Swizzle<T, 4, 0, 3, 1, 1> xwyy;
            Swizzle<T, 4, 0, 3, 1, 2> xwyz;
            Swizzle<T, 4, 0, 3, 1, 3> xwyw;
            Swizzle<T, 4, 0, 3, 2, 0> xwzx;
            Swizzle<T, 4, 0, 3, 2, 1> xwzy;
            Swizzle<T, 4, 0, 3, 2, 2> xwzz;
            Swizzle<T, 4, 0, 3, 2, 3> xwzw;
            Swizzle<T, 4, 0, 3, 3, 0> xwwx;
            Swizzle<T, 4, 0, 3, 3, 1> xwwy;
            Swizzle<T, 4, 0, 3, 3, 2> xwwz;
            Swizzle<T, 4, 0, 3, 3, 3> xwww;
            Swizzle<T, 4, 1, 0, 0, 0> yxxx;
            Swizzle<T, 4, 1, 0, 0, 1> yxxy;
            Swizzle<T, 4, 1, 0, 0, 2> yxxz;
            Swizzle<T, 4, 1, 0, 0, 3> yxxw;
            Swizzle<T, 4, 1, 0, 1, 0> yxyx;
            Swizzle<T, 4, 1, 0, 1, 1> yxyy;
            Swizzle<T, 4, 1, 0, 1, 2> yxyz;
            Swizzle<T, 4, 1, 0, 1, 3> yxyw;
            Swizzle<T, 4, 1, 0, 2, 0> yxzx;
            Swizzle<T, 4, 1, 0, 2, 1> yxzy;
            Swizzle<T, 4, 1, 0, 2, 2> yxzz;
            Swizzle<T, 4, 1, 0, 2, 3> yxzw;
            Swizzle<T, 4, 1, 0, 3, 0> yxwx;
            Swizzle<T, 4, 1, 0, 3, 1> yxwy;
            Swizzle<T, 4, 1, 0, 3, 2> yxwz;
            Swizzle<T, 4, 1, 0, 3, 3> yxww;
            Swizzle<T, 4, 1, 1, 0, 0> yyxx;
            Swizzle<T, 4, 1, 1, 0, 1> yyxy;
            Swizzle<T, 4, 1, 1, 0, 2> yyxz;
            Swizzle<T, 4, 1, 1, 0, 3> yyxw;
            Swizzle<T, 4, 1, 1, 1, 0> yyyx;
            Swizzle<T, 4, 1, 1, 1, 1> yyyy;
            Swizzle<T, 4, 1, 1, 1, 2> yyyz;
            Swizzle<T, 4, 1, 1, 1, 3> yyyw;
            Swizzle<T, 4, 1, 1, 2, 0> yyzx;
            Swizzle<T, 4, 1, 1, 2, 1> yyzy;
            Swizzle<T, 4, 1, 1, 2, 2> yyzz;
            Swizzle<T, 4, 1, 1, 2, 3> yyzw;
            Swizzle<T, 4, 1, 1, 3, 0> yywx;
            Swizzle<T, 4, 1, 1, 3, 1> yywy;
            Swizzle<T, 4, 1, 1, 3, 2> yywz;
            Swizzle<T, 4, 1, 1, 3, 3> yyww;
            Swizzle<T, 4, 1, 2, 0, 0> yzxx;
            Swizzle<T, 4, 1, 2, 0, 1> yzxy;
            Swizzle<T, 4, 1, 2, 0, 2> yzxz;
            Swizzle<T, 4, 1, 2, 0, 3> yzxw;
            Swizzle<T, 4, 1, 2, 1, 0> yzyx;
            Swizzle<T, 4, 1, 2, 1, 1> yzyy;
            Swizzle<T, 4, 1, 2, 1, 2> yzyz;
            Swizzle<T, 4, 1, 2, 1, 3> yzyw;
            Swizzle<T, 4, 1, 2, 2, 0> yzzx;
            Swizzle<T, 4, 1, 2, 2, 1> yzzy;
            Swizzle<T, 4, 1, 2, 2, 2> yzzz;
            Swizzle<T, 4, 1, 2, 2, 3> yzzw;
            Swizzle<T, 4, 1, 2, 3, 0> yzwx;
            Swizzle<T, 4, 1, 2, 3, 1> yzwy;
            Swizzle<T, 4, 1, 2, 3, 2> yzwz;
            Swizzle<T, 4, 1, 2, 3, 3> yzww;
            Swizzle<T, 4, 1, 3, 0, 0> ywxx;
            Swizzle<T, 4, 1, 3, 0, 1> ywxy;
            Swizzle<T, 4, 1, 3, 0, 2> ywxz;
            Swizzle<T, 4, 1, 3, 0, 3> ywxw;
            Swizzle<T, 4, 1, 3, 1, 0> ywyx;
            Swizzle<T, 4, 1, 3, 1, 1> ywyy;
            Swizzle<T, 4, 1, 3, 1, 2> ywyz;
            Swizzle<T, 4, 1, 3, 1, 3> ywyw;
            Swizzle<T, 4, 1, 3, 2, 0> ywzx;
            Swizzle<T, 4, 1, 3, 2, 1> ywzy;
            Swizzle<T, 4, 1, 3, 2, 2> ywzz;
            Swizzle<T, 4, 1, 3, 2, 3> ywzw;
            Swizzle<T, 4, 1, 3, 3, 0> ywwx;
            Swizzle<T, 4, 1, 3, 3, 1> ywwy;
            Swizzle<T, 4, 1, 3, 3, 2> ywwz;
            Swizzle<T, 4, 1, 3, 3, 3> ywww;
            Swizzle<T, 4, 2, 0, 0, 0> zxxx;
            Swizzle<T, 4, 2, 0, 0, 1> zxxy;
            Swizzle<T, 4, 2, 0, 0, 2> zxxz;
            Swizzle<T, 4, 2, 0, 0, 3> zxxw;
            Swizzle<T, 4, 2, 0, 1, 0> zxyx;
            Swizzle<T, 4, 2, 0, 1, 1> zxyy;
            Swizzle<T, 4, 2, 0, 1, 2> zxyz;
            Swizzle<T, 4, 2, 0, 1, 3> zxyw;
            Swizzle<T, 4, 2, 0, 2, 0> zxzx;
            Swizzle<T, 4, 2, 0, 2, 1> zxzy;
            Swizzle<T, 4, 2, 0, 2, 2> zxzz;
            Swizzle<T, 4, 2, 0, 2, 3> zxzw;
            Swizzle<T, 4, 2, 0, 3, 0> zxwx;
            Swizzle<T, 4, 2, 0, 3, 1> zxwy;
            Swizzle<T, 4, 2, 0, 3, 2> zxwz;
            Swizzle<T, 4, 2, 0, 3, 3> zxww;
            Swizzle<T, 4, 2, 1, 0, 0> zyxx;
            Swizzle<T, 4, 2, 1, 0, 1> zyxy;
            Swizzle<T, 4, 2, 1, 0, 2> zyxz;
            Swizzle<T, 4, 2, 1, 0, 3> zyxw;
            Swizzle<T, 4, 2, 1, 1, 0> zyyx;
            Swizzle<T, 4, 2, 1, 1, 1> zyyy;
            Swizzle<T, 4, 2, 1, 1, 2> zyyz;
            Swizzle<T, 4, 2, 1, 1, 3> zyyw;
            Swizzle<T, 4, 2, 1, 2, 0> zyzx;
            Swizzle<T, 4, 2, 1, 2, 1> zyzy;
            Swizzle<T, 4, 2, 1, 2, 2> zyzz;
            Swizzle<T, 4, 2, 1, 2, 3> zyzw;
            Swizzle<T, 4, 2, 1, 3, 0> zywx;
            Swizzle<T, 4, 2, 1, 3, 1> zywy;
            Swizzle<T, 4, 2, 1, 3, 2> zywz;
            Swizzle<T, 4, 2, 1, 3, 3> zyww;
            Swizzle<T, 4, 2, 2, 0, 0> zzxx;
            Swizzle<T, 4, 2, 2, 0, 1> zzxy;
            Swizzle<T, 4, 2, 2, 0, 2> zzxz;
            Swizzle<T, 4, 2, 2, 0, 3> zzxw;
            Swizzle<T, 4, 2, 2, 1, 0> zzyx;
            Swizzle<T, 4, 2, 2, 1, 1> zzyy;
            Swizzle<T, 4, 2, 2, 1, 2> zzyz;
            Swizzle<T, 4, 2, 2, 1, 3> zzyw;
            Swizzle<T, 4, 2, 2, 2, 0> zzzx;
            Swizzle<T, 4, 2, 2, 2, 1> zzzy;
            Swizzle<T, 4, 2, 2, 2, 2> zzzz;
            Swizzle<T, 4, 2, 2, 2, 3> zzzw;
            Swizzle<T, 4, 2, 2, 3, 0> zzwx;
            Swizzle<T, 4, 2, 2, 3, 1> zzwy;
            Swizzle<T, 4, 2, 2, 3, 2> zzwz;
            Swizzle<T, 4, 2, 2, 3, 3> zzww;
            Swizzle<T, 4, 2, 3, 0, 0> zwxx;
            Swizzle<T, 4, 2, 3, 0, 1> zwxy;
            Swizzle<T, 4, 2, 3, 0, 2> zwxz;
            Swizzle<T, 4, 2, 3, 0, 3> zwxw;
            Swizzle<T, 4, 2, 3, 1, 0> zwyx;
            Swizzle<T, 4, 2, 3, 1, 1> zwyy;
            Swizzle<T, 4, 2, 3, 1, 2> zwyz;
            Swizzle<T, 4, 2, 3, 1, 3> zwyw;
            Swizzle<T, 4, 2, 3, 2, 0> zwzx;
            Swizzle<T, 4, 2, 3, 2, 1> zwzy;
            Swizzle<T, 4, 2, 3, 2, 2> zwzz;
            Swizzle<T, 4, 2, 3, 2, 3> zwzw;
            Swizzle<T, 4, 2, 3, 3, 0> zwwx;
            Swizzle<T, 4, 2, 3, 3, 1> zwwy;
            Swizzle<T, 4, 2, 3, 3, 2> zwwz;
            Swizzle<T, 4, 2, 3, 3, 3> zwww;
            Swizzle<T, 4, 3, 0, 0, 0> wxxx;
            Swizzle<T, 4, 3, 0, 0, 1> wxxy;
            Swizzle<T, 4, 3, 0, 0, 2> wxxz;
            Swizzle<T, 4, 3, 0, 0, 3> wxxw;
            Swizzle<T, 4, 3, 0, 1, 0> wxyx;
            Swizzle<T, 4, 3, 0, 1, 1> wxyy;
            Swizzle<T, 4, 3, 0, 1, 2> wxyz;
            Swizzle<T, 4, 3, 0, 1, 3> wxyw;
            Swizzle<T, 4, 3, 0, 2, 0> wxzx;
            Swizzle<T, 4, 3, 0, 2, 1> wxzy;
            Swizzle<T, 4, 3, 0, 2, 2> wxzz;
            Swizzle<T, 4, 3, 0, 2, 3> wxzw;
            Swizzle<T, 4, 3, 0, 3, 0> wxwx;
            Swizzle<T, 4, 3, 0, 3, 1> wxwy;
            Swizzle<T, 4, 3, 0, 3, 2> wxwz;
            Swizzle<T, 4, 3, 0, 3, 3> wxww;
            Swizzle<T, 4, 3, 1, 0, 0> wyxx;
            Swizzle<T, 4, 3, 1, 0, 1> wyxy;
            Swizzle<T, 4, 3, 1, 0, 2> wyxz;
            Swizzle<T, 4, 3, 1, 0, 3> wyxw;
            Swizzle<T, 4, 3, 1, 1, 0> wyyx;
            Swizzle<T, 4, 3, 1, 1, 1> wyyy;
            Swizzle<T, 4, 3, 1, 1, 2> wyyz;
            Swizzle<T, 4, 3, 1, 1, 3> wyyw;
            Swizzle<T, 4, 3, 1, 2, 0> wyzx;
            Swizzle<T, 4, 3, 1, 2, 1> wyzy;
            Swizzle<T, 4, 3, 1, 2, 2> wyzz;
            Swizzle<T, 4, 3, 1, 2, 3> wyzw;
            Swizzle<T, 4, 3, 1, 3, 0> wywx;
            Swizzle<T, 4, 3, 1, 3, 1> wywy;
            Swizzle<T, 4, 3, 1, 3, 2> wywz;
            Swizzle<T, 4, 3, 1, 3, 3> wyww;
            Swizzle<T, 4, 3, 2, 0, 0> wzxx;
            Swizzle<T, 4, 3, 2, 0, 1> wzxy;
            Swizzle<T, 4, 3, 2, 0, 2> wzxz;
            Swizzle<T, 4, 3, 2, 0, 3> wzxw;
            Swizzle<T, 4, 3, 2, 1, 0> wzyx;
            Swizzle<T, 4, 3, 2, 1, 1> wzyy;
            Swizzle<T, 4, 3, 2, 1, 2> wzyz;
            Swizzle<T, 4, 3, 2, 1, 3> wzyw;
            Swizzle<T, 4, 3, 2, 2, 0> wzzx;
            Swizzle<T, 4, 3, 2, 2, 1> wzzy;
            Swizzle<T, 4, 3, 2, 2, 2> wzzz;
            Swizzle<T, 4, 3, 2, 2, 3> wzzw;
            Swizzle<T, 4, 3, 2, 3, 0> wzwx;
            Swizzle<T, 4, 3, 2, 3, 1> wzwy;
            Swizzle<T, 4, 3, 2, 3, 2> wzwz;
            Swizzle<T, 4, 3, 2, 3, 3> wzww;
            Swizzle<T, 4, 3, 3, 0, 0> wwxx;
            Swizzle<T, 4, 3, 3, 0, 1> wwxy;
            Swizzle<T, 4, 3, 3, 0, 2> wwxz;
            Swizzle<T, 4, 3, 3, 0, 3> wwxw;
            Swizzle<T, 4, 3, 3, 1, 0> wwyx;
            Swizzle<T, 4, 3, 3, 1, 1> wwyy;
            Swizzle<T, 4, 3, 3, 1, 2> wwyz;
            Swizzle<T, 4, 3, 3, 1, 3> wwyw;
            Swizzle<T, 4, 3, 3, 2, 0> wwzx;
            Swizzle<T, 4, 3, 3, 2, 1> wwzy;
            Swizzle<T, 4, 3, 3, 2, 2> wwzz;
            Swizzle<T, 4, 3, 3, 2, 3> wwzw;
            Swizzle<T, 4, 3, 3, 3, 0> wwwx;
            Swizzle<T, 4, 3, 3, 3, 1> wwwy;
            Swizzle<T, 4, 3, 3, 3, 2> wwwz;
            Swizzle<T, 4, 3, 3, 3, 3> wwww;
        };

        // IFixedVector constraints
    public:

        // construct from a single value
        constexpr explicit Vector4(const T value) noexcept : x(value), y(value), z(value), w(value) {}
        constexpr explicit Vector4(const T _x, const T _y, const T _z, const T _w) noexcept : x(_x), y(_y), z(_z), w(_w) {}

        // construct from a moveable array
        constexpr explicit Vector4(std::array<T, length>&& _data) noexcept : data(Utilities::exchange(_data, std::array<T, length>{})) {}

        // construct from a parameter pack
        template<typename...TS>
            requires(std::is_same_v<T, TS> && ...)
        constexpr explicit Vector4(TS&&... values) noexcept : data{ FWD(values)... } {}

        // constructo from a pointer
        constexpr explicit Vector4(const T* _data) {
            [[assume(_data != nullptr)]];
            AssumeAligned(T, _data);
            std::memcpy(data.data(), _data, length * sizeof(T));
        }

        // assign a moveable array
        constexpr Vector4& operator=(std::array<T, length>&& _data) noexcept {
            data = Utilities::exchange(_data, std::array<T, length>{});
            return *this;
        };

        // overload stream '<<' operator
        friend std::ostream& operator<<(std::ostream& xio_stream, const Vector4& vec) {
            return xio_stream << "{" << vec[0] << ", " << vec[1] << ", " << vec[2] << ", " << vec[3] << "}";
        }

        // overload operator '[]'
        constexpr T  operator[](const std::size_t i) const { assert(i < length); return data.at(i); }
        constexpr T& operator[](const std::size_t i) { assert(i < length); return data[i]; }

        // public API
    public:
        // constructo from scalar and vector3
        constexpr Vector4(const T x, const Vector3<T>& yzw) noexcept : data{ x,      yzw[0], yzw[1], yzw[2] } {}
        constexpr Vector4(const Vector3<T>& xyz, const T w = T{}) noexcept : data{ xyz[0], xyz[1], xyz[2], w } {}

        // constructo from two vector2
        constexpr Vector4(const Vector2<T>& xy, const Vector2<T>& zw) noexcept : data{ xy[0], xy[1], zw[0], zw[1] } {}
    };

    // Vector4 traits and concept
    template<typename>   struct is_vector4 : public std::false_type {};
    template<typename T> struct is_vector4<Vector4<T>> : public std::true_type {};
    template<typename T> constexpr bool is_vector4_v = is_vector4<T>::value;
    template<typename T> concept IVector4 = is_vector4_v<T>;

    /**
    *\brief 1xN numerical vector implementation of IFixedVector concept
    * @param{arithmetic, in} elements underlying type
    * @param{size_t,     in} amount of elements in vector
    * */
    template<typename T, std::size_t N>
        requires(std::is_arithmetic_v<T> && (N > 0))
    struct VectorN final {
        static constexpr std::integral_constant<std::size_t, N> length = {};
        using value_type = T;

        // accessors
        AlignedStorage(T) std::array<T, N> data{};

        // construct from a single value
        constexpr explicit VectorN(const T value) noexcept {
            Utilities::static_for<0, 1, length>([this, value](std::size_t i) {
                data[i] = value;
            });
        }

        // construct from a moveable array
        constexpr explicit VectorN(std::array<T, length>&& _data) noexcept : data(Utilities::exchange(_data, std::array<T, length>{})) {}

        // construct from a parameter pack
        template<typename...TS>
            requires(std::is_same_v<T, TS> && ...)
        constexpr explicit VectorN(TS&&... values) noexcept : data{ FWD(values)... } {}

        // constructo from a pointer
        constexpr explicit VectorN(const T* _data) {
            [[assume(_data != nullptr)]];
            AssumeAligned(T, _data);
            std::memcpy(data.data(), _data, length * sizeof(T));
        }

        // assign a moveable array
        constexpr VectorN& operator=(std::array<T, length>&& _data) noexcept {
            data = Utilities::exchange(_data, std::array<T, length>{});
            return *this;
        };

        // overload stream '<<' operator
        friend std::ostream& operator<<(std::ostream& xio_stream, const VectorN& vec) {
            xio_stream << "{";
            Utilities::static_for<0, 1, length - 1>([&xio_stream, &vec](std::size_t i) {
                xio_stream << vec[i] << ",";
             });
             xio_stream  << vec[length - 1] << "}";
             return xio_stream;
        }

        // overload operator '[]'
        constexpr T  operator[](const std::size_t i) const { assert(i < length); return data.at(i); }
        constexpr T& operator[](const std::size_t i) { assert(i < length); return data[i]; }
    };

    // VectorN traits and concept
    template<typename> struct is_vectorn : public std::false_type {};
    template<typename T, std::size_t N> struct is_vectorn<VectorN<T, N>> : public std::true_type {};
    template<typename T> constexpr bool is_vectorn_v = is_vectorn<T>::value;
    template<typename T> concept IVectorN = is_vectorn_v<T>;

    // trait to check if a type is GLSL vector
    template<typename>   struct is_vector : public std::false_type {};
    template<typename T> struct is_vector<Vector2<T>> : public std::true_type {};
    template<typename T> struct is_vector<Vector3<T>> : public std::true_type {};
    template<typename T> struct is_vector<Vector4<T>> : public std::true_type {};
    template<typename T, std::size_t N> struct is_vector<VectorN<T, N>> : public std::true_type {};
    template<typename T> constexpr bool is_vector_v = is_vector<T>::value;
    template<typename T> concept IGlslVector = is_vector_v<T>;

    //
    // IFixedCubicMatrix constructs
    //

    /**
    * \brief 2x2 numerical matrix (column major)
    * @param {arithmetic, in} elements underlying type
    **/
    template<typename T>
        requires(std::is_arithmetic_v<T>)
    struct Matrix2 final {
        static_assert(std::is_arithmetic_v<T>);
        static constexpr std::integral_constant<std::size_t, 4> length = {};
        static constexpr std::integral_constant<std::size_t, 2> columns = {};
        using value_type = T;
        using vector_type = Vector2<T>;

        // construct from a single value
        constexpr explicit Matrix2(const T value) noexcept : data{ value, value,
                                                                   value, value } {}
        constexpr explicit Matrix2(const T v0, const T v1, const T v2, const T v3) noexcept : data{ v0, v1,
                                                                                                    v2, v3} {}

        // construct from a moveable array
        constexpr explicit Matrix2(std::array<T, length>&& _data) noexcept : data(Utilities::exchange(_data, std::array<T, length>{})) {}

        // construct from a parameter pack (every two consecutive elements are a colume)
        template<typename...TS>
            requires(std::is_same_v<T, TS> && ...)
        constexpr explicit Matrix2(TS&&... values) noexcept : data{ FWD(values)... } {}

        // construct from a pointer
        constexpr explicit Matrix2(const T* _data) {
            [[assume(_data != nullptr)]];
            AssumeAligned(T, _data);
            std::memcpy(data.data(), _data, length * sizeof(T));
        }

        // assign a moveable array
        constexpr Matrix2& operator=(std::array<T, length>&& _data) noexcept {
            data = Utilities::exchange(_data, std::array<T, length>{});
            return *this;
        };

        // construct from two Vector2 (two columns)
        constexpr Matrix2(const vector_type& c0, const vector_type& c1) : c{ c0, c1 } {}

        // overload stream '<<' operator
        friend std::ostream& operator<<(std::ostream& xio_stream, const Matrix2& mat) {
            return xio_stream << '{' << mat(0,0) << ", " << mat(1, 0) << ",\n" <<
                                        mat(0,1) << ", " << mat(1, 1) << "}";
        }

        // overload operator '[]' to return column
        constexpr vector_type  operator[](const std::size_t i) const { assert(i < columns); return c[i]; }
        constexpr vector_type& operator[](const std::size_t i) { assert(i < columns); return c[i]; }

        // overload operator '()' to return value
        constexpr value_type  operator()(const std::size_t col, const std::size_t row) const { assert(col < columns); assert(row < columns); return data[col * columns + row]; }
        constexpr value_type& operator()(const std::size_t col, const std::size_t row)       { assert(col < columns); assert(row < columns); return data[col * columns + row]; }

        // return vectors by subscript
        constexpr vector_type x() const { return c[0]; }
        constexpr vector_type y() const { return c[1]; }

    private:
        union {
            AlignedStorage(T) std::array<T, length>            data;    // elements layed out column wise
            AlignedStorage(T) std::array<vector_type, columns> c;         // { column 0, column 1 }
        };
    };

    // Matrix2 traits and concept
    template<typename>   struct is_matrix2 : public std::false_type {};
    template<typename T> struct is_matrix2<Matrix2<T>> : public std::true_type {};
    template<typename T> constexpr bool is_matrix2_v = is_matrix2<T>::value;
    template<typename T> concept IMatrix2 = is_matrix2_v<T>;

    /**
    * \brief 3x3 numerical matrix (column major)
    * @param {arithmetic, in} elements underlying type
    **/
    template<typename T>
        requires(std::is_arithmetic_v<T>)
     struct Matrix3 final {
        static constexpr std::integral_constant<std::size_t, 9> length = {};
        static constexpr std::integral_constant<std::size_t, 3> columns = {};
        using value_type = T;
        using vector_type = Vector3<T>;

        // construct from a single value
        constexpr explicit Matrix3(const T value) noexcept : data{ value, value, value,
                                                                   value, value, value,
                                                                   value, value, value } {}
        constexpr explicit Matrix3(const T v0, const T v1, const T v2,
                                   const T v3, const T v4, const T v5,
                                   const T v6, const T v7, const T v8) noexcept : data{ v0, v1, v2,
                                                                                        v3, v4, v5,
                                                                                        v6, v7, v8 } {}

        // construct from a moveable array
        constexpr explicit Matrix3(std::array<T, length>&& _data) noexcept : data(Utilities::exchange(_data, std::array<T, length>{})) {}

        // construct from a pointer
        constexpr explicit Matrix3(const T* _data) {
            [[assume(_data != nullptr)]];
            AssumeAligned(T, _data);
            std::memcpy(data.data(), _data, length * sizeof(T));
        }

        // construct from a parameter pack (every two consecutive elements are a colume)
        template<typename...TS>
            requires(std::is_same_v<T, TS> && ...)
        constexpr explicit Matrix3(TS&&... values) noexcept : data{ FWD(values)... } {}

        // assign a moveable array
        constexpr Matrix3& operator=(std::array<T, length>&& _data) noexcept {
            data = Utilities::exchange(_data, std::array<T, length>{});
            return *this;
        };

        // construct from two Vector2 (two columns)
        constexpr Matrix3(const vector_type& c0, const vector_type& c1, const vector_type& c2) : c{ c0, c1, c2 } {}

        // overload stream '<<' operator
        friend std::ostream& operator<<(std::ostream& xio_stream, const Matrix3& mat) {
            return xio_stream << '{' << mat(0, 0) << ", " << mat(1, 0) << ", " << mat(2, 0) << ",\n" <<
                                        mat(0, 1) << ", " << mat(1, 1) << ", " << mat(2, 1) << ",\n" <<
                                        mat(0, 2) << ", " << mat(1, 2) << ", " << mat(2, 2) << "}";
        }

        // overload operator '[]' to return column
        constexpr vector_type  operator[](const std::size_t i) const { assert(i < columns); return c[i]; }
        constexpr vector_type& operator[](const std::size_t i) { assert(i < columns); return c[i]; }

        // overload operator '()' to return value
        constexpr value_type  operator()(const std::size_t col, const std::size_t row) const { assert(col < columns); assert(row < columns); return data[col * columns + row]; }
        constexpr value_type& operator()(const std::size_t col, const std::size_t row) { assert(col < columns); assert(row < columns); return data[col * columns + row]; }

        // return vectors by subscript
        constexpr vector_type x() const { return c[0]; }
        constexpr vector_type y() const { return c[1]; }
        constexpr vector_type z() const { return c[2]; }

    private:
        union {
            AlignedStorage(T) std::array<T, length>            data;    // elements layed out column wise
            AlignedStorage(T) std::array<vector_type, columns> c;         // { column 0, column 1, column 2 }
        };
    };

    // Matrix3 traits and concept
    template<typename>   struct is_matrix3 : public std::false_type {};
    template<typename T> struct is_matrix3<Matrix3<T>> : public std::true_type {};
    template<typename T> constexpr bool is_matrix3_v = is_matrix3<T>::value;
    template<typename T> concept IMatrix3 = is_matrix3_v<T>;

    /**
    * \brief 4x4 numerical matrix (column major)
    * @param {arithmetic, in} elements underlying type
    **/
    template<typename T>
        requires(std::is_arithmetic_v<T>)
    struct Matrix4 final {
        static constexpr std::integral_constant<std::size_t, 16> length = {};
        static constexpr std::integral_constant<std::size_t, 4> columns = {};
        using value_type = T;
        using vector_type = Vector4<T>;

        // construct from a single value
        constexpr explicit Matrix4(const T value) noexcept : data{ value, value, value, value,
                                                                   value, value, value, value,
                                                                   value, value, value, value,
                                                                   value, value, value, value } {}
        constexpr explicit Matrix4(const T v0,  const T v1,  const T v2,  const T v3,
                                   const T v4,  const T v5,  const T v6,  const T v7,
                                   const T v8,  const T v9,  const T v10, const T v11,
                                   const T v12, const T v13, const T v14, const T v15) noexcept : data{ v0,  v1,  v2,  v3,
                                                                                                        v4,  v5,  v6,  v7,
                                                                                                        v8,  v9,  v10, v11,
                                                                                                        v12, v13, v14, v15} {}

        // construct from a moveable array
        constexpr explicit Matrix4(std::array<T, length>&& _data) noexcept : data(Utilities::exchange(_data, std::array<T, length>{})) {}

        // construct from a parameter pack (every two consecutive elements are a colume)
        template<typename...TS>
            requires(std::is_same_v<T, TS> && ...)
        constexpr explicit Matrix4(TS&&... values) noexcept : data{ FWD(values)... } {}

        // construct from a pointer
        constexpr explicit Matrix4(const T* _data) {
            [[assume(_data != nullptr)]];
            AssumeAligned(T, _data);
            std::memcpy(data.data(), _data, length * sizeof(T));
        }

        // assign a moveable array
        constexpr Matrix4& operator=(std::array<T, length>&& _data) noexcept {
            data = Utilities::exchange(_data, std::array<T, length>{});
            return *this;
        };

        // construct from two Vector2 (two columns)
        constexpr Matrix4(const vector_type& c0, const vector_type& c1,
                          const vector_type& c2, const vector_type& c3) : c{ c0, c1, c2, c3 } {}

        // overload stream '<<' operator
        friend std::ostream& operator<<(std::ostream& xio_stream, const Matrix4& mat) {
            return xio_stream << '{' << mat(0,0) << ", " << mat(1, 0) << ", " << mat(2, 0) << ", " << mat(3, 0) << ",\n" <<
                                        mat(0,1) << ", " << mat(1, 1) << ", " << mat(2, 1) << ", " << mat(3, 1) << ",\n" <<
                                        mat(0,2) << ", " << mat(1, 2) << ", " << mat(2, 2) << ", " << mat(3, 2) << ",\n" <<
                                        mat(0,3) << ", " << mat(1, 3) << ", " << mat(2, 3) << ", " << mat(3, 3) << "}";
        }

        // overload operator '[]' to return column
        constexpr vector_type  operator[](const std::size_t i) const { assert(i < columns); return c[i]; }
        constexpr vector_type& operator[](const std::size_t i) { assert(i < columns); return c[i]; }

        // overload operator '()' to return value
        constexpr value_type  operator()(const std::size_t col, const std::size_t row) const { assert(col < columns); assert(row < columns); return data[col * columns + row]; }
        constexpr value_type& operator()(const std::size_t col, const std::size_t row) { assert(col < columns); assert(row < columns); return data[col * columns + row]; }

        // return vectors by subscript
        constexpr vector_type x() const { return c[0]; }
        constexpr vector_type y() const { return c[1]; }
        constexpr vector_type z() const { return c[2]; }
        constexpr vector_type w() const { return c[3]; }

    private:
        union {
            AlignedStorage(T) std::array<T, length>            data;    // elements layed out column wise
            AlignedStorage(T) std::array<vector_type, columns> c;       // { column 0, column 1, column 2, column 3 }
        };
    };

    // Matrix4 traits and concept
    template<typename>   struct is_matrix4 : public std::false_type {};
    template<typename T> struct is_matrix4<Matrix4<T>> : public std::true_type {};
    template<typename T> constexpr bool is_matrix4_v = is_matrix4<T>::value;
    template<typename T> concept IMatrix4 = is_matrix4_v<T>;

    /**
    * \brief cubic numerical matrix (column major)
    * @param {arithmetic, in} elements underlying type
    * @param {size_t,     in} amount of columns and rows
    **/
    template<typename T, std::size_t N>
        requires(std::is_arithmetic_v<T> && (N > 0))
    struct MatrixN final {
        static constexpr std::integral_constant<std::size_t, N * N> length = {};
        static constexpr std::integral_constant<std::size_t, N> columns = {};
        using value_type = T;
        using vector_type = VectorN<T, N>;

        // construct from a single value
        constexpr explicit MatrixN(const T value) noexcept {
            Utilities::static_for<0, 1, length>([this, value](std::size_t i) {
                data[i] = value;
            });
        }
        
        // construct from a moveable array
        constexpr explicit MatrixN(std::array<T, length>&& _data) noexcept : data(Utilities::exchange(_data, std::array<T, length>{})) {}

        // construct from a parameter pack (every N consecutive elements are a colume)
        template<typename...TS>
            requires((std::is_same_v<T, TS> && ...) && (sizeof...(TS) == length))
        constexpr explicit MatrixN(TS&&... values) noexcept : data{ FWD(values)... } {}

        // construct from a pointer
        constexpr explicit MatrixN(const T* _data) {
            [[assume(_data != nullptr)]];
            AssumeAligned(T, _data);
            std::memcpy(data.data(), _data, length * sizeof(T));
        }

        // assign a moveable array
        constexpr MatrixN& operator=(std::array<T, length>&& _data) noexcept {
            data = Utilities::exchange(_data, std::array<T, length>{});
            return *this;
        };

        // construct from N VectorN
        template<typename...TS>
            requires(std::is_same_v<vector_type, TS> && ...)
        constexpr explicit MatrixN(TS&&... vectors) noexcept {
            std::size_t i{};
            ([&] { c[i] = FWD(vectors); ++i; } (), ...);
        }

        // overload stream '<<' operator
        friend std::ostream& operator<<(std::ostream& xio_stream, const MatrixN& mat) {
            xio_stream << "{";
            Utilities::static_for<0, 1, columns - 1>([&xio_stream, &mat](std::size_t i) {
                xio_stream << mat[i] << ",\n";
             });
             xio_stream  << mat[columns - 1] << "}";
             return xio_stream;
        }

        // overload operator '[]' to return column
        constexpr vector_type  operator[](const std::size_t i) const { assert(i < columns); return c[i]; }
        constexpr vector_type& operator[](const std::size_t i) { assert(i < columns); return c[i]; }

        // overload operator '()' to return value
        constexpr value_type  operator()(const std::size_t col, const std::size_t row) const { assert(col < columns); assert(row < columns); return data[col * columns + row]; }
        constexpr value_type& operator()(const std::size_t col, const std::size_t row) { assert(col < columns); assert(row < columns); return data[col * columns + row]; }

    private:
        union {
            AlignedStorage(T) std::array<T, length>            data;    // elements layed out column wise
            AlignedStorage(T) std::array<vector_type, columns> c;       // { column 0, column 1, ..., column N-1 }
        };
    };

    // Matrix4 traits and concept
    template<typename>   struct is_matrixN : public std::false_type {};
    template<typename T, std::size_t N> struct is_matrixN<MatrixN<T, N>> : public std::true_type {};
    template<typename T> constexpr bool is_matrixn_v = is_matrixN<T>::value;
    template<typename T> concept IMatrixN = is_matrixn_v<T>;
    
    // trait to check if a type is a GLSL matrix
    template<typename>   struct is_matrix : public std::false_type {};
    template<typename T> struct is_matrix<Matrix2<T>> : public std::true_type {};
    template<typename T> struct is_matrix<Matrix3<T>> : public std::true_type {};
    template<typename T> struct is_matrix<Matrix4<T>> : public std::true_type {};
    template<typename T, std::size_t N> struct is_matrix<MatrixN<T, N>> : public std::true_type {};
    template<typename T> constexpr bool is_matrix_v = is_matrix<T>::value;
    template<typename T> concept IGlslMatrix = is_matrix_v<T>;
}

// glsl shorthands
using ivec2 = GLSL::Vector2<int>;
using ivec3 = GLSL::Vector3<int>;
using ivec4 = GLSL::Vector4<int>;
using vec2 = GLSL::Vector2<float>;
using vec3 = GLSL::Vector3<float>;
using vec4 = GLSL::Vector4<float>;
using dvec2 = GLSL::Vector2<double>;
using dvec3 = GLSL::Vector3<double>;
using dvec4 = GLSL::Vector4<double>;
using imat2 = GLSL::Matrix2<int>;
using imat3 = GLSL::Matrix3<int>;
using imat4 = GLSL::Matrix4<int>;
using mat2 = GLSL::Matrix2<float>;
using mat3 = GLSL::Matrix3<float>;
using mat4 = GLSL::Matrix4<float>;
using dmat2 = GLSL::Matrix2<double>;
using dmat3 = GLSL::Matrix3<double>;
using dmat4 = GLSL::Matrix4<double>;
