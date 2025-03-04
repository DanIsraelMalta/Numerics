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
#include <assert.h>
#include <bit> // std::bit_cast
#include <limits> // std::numeric_limits
#include <climits> // CHAR_BIT
#include <cmath> // std::signbit, std::isnan, std::isinf
#include "Utilities.h"
#include "Concepts.h"

/**
* numerical utilities
**/
namespace Numerics {

    /**
    * @param {floating point, out} value for floating point number comparisons
    **/
    template<typename T>
        requires (std::is_floating_point_v<T>)
    constexpr T equality_precision() noexcept {
        // for 32bit floating point use 5 digits
        if constexpr (sizeof(T) == sizeof(float)) {
            return 1.0e-5f;
        } // for 64bit floating point use 14 digits
        else if constexpr (sizeof(T) == sizeof(double)) {
            return 1.0e-14;
        } // 17 digits (except for MSVC, since they are same as doubles - https://msdn.microsoft.com/en-us/library/9cx8xs15.aspx)
        else {
#if !defined(_MSC_VER)
            return 1.0e-17;
#else
            return 1.0e-14;
#endif
        }
    }

    /**
    * \brief given amount of bits, return the number of size_t's needed to store them
    * @param {size_t, in}  number of bits
    * @param {size_t, out} number of size_t's needed to store given amount of bits
    **/
    constexpr std::size_t NumBitFields(const std::size_t num_bits) noexcept {
        constexpr std::size_t num_field_bits{ sizeof(std::size_t) * CHAR_BIT };
        return num_bits ? (1 + ((num_bits - 1) / num_field_bits)) : 0;
    }
    template<std::size_t num_bits>
    constexpr std::size_t NumBitFields() noexcept {
        constexpr std::size_t num_field_bits{ sizeof(std::size_t) * CHAR_BIT };
        return num_bits ? (1 + ((num_bits - 1) / num_field_bits)) : 0;
    }

    /**
    * \brief return true if all values are positive
    * @param {arithmetic..., in}  values
    * @param {boolean,       out} true if all values are positive
    **/
    template<typename T>
        requires(std::is_arithmetic_v<T>)
    constexpr bool allPositive(const T a) noexcept {
        return a > T{};
    }
    template<typename T, typename...Ts>
        requires(std::is_arithmetic_v<T> && (std::is_same_v<T, Ts> && ...))
    constexpr bool allPositive(const T a, const Ts... ts) noexcept {
        return allPositive(a) && allPositive(ts...);
    }

    /**
    * \brief return the minimal value among variadic amount of real numbers
    * @param {OrderedDescending..., in}  values
    * @param {OrderedDescending,    out} minimal value
    **/
    template<typename T>
    constexpr T min(const T a) noexcept {
        return a;
    }
    template<typename T, typename...Ts>
        requires(Concepts::OrderedDescending<T, T> && (std::is_same_v<T, Ts> && ...))
    constexpr T min(const T a, const Ts... ts) noexcept {
        return a < min(ts...) ? a : min(ts...);
    }

    /**
    * \brief return the maximal value among variadic amount of ordered ascending elements
    * @param {OrderedAscending..., in}  values
    * @param {OrderedAscending,    out} maximal value
    **/
    template<typename T>
    constexpr T max(const T a) noexcept {
        return a;
    }
    template<typename T, typename...Ts>
        requires(Concepts::OrderedAscending<T, T> && (std::is_same_v<T, Ts> && ...))
    constexpr T max(const T a, const Ts... ts) noexcept {
        return a > max(ts...) ? a : max(ts...);
    }

    /**
    * \brief return the dot product (sum of squares) of variadic amount of real numbers
    * @param {arithmetic..., in}  values
    * @param {arithmetic,    out} sum of squares of values
    **/
    template<typename T>
        requires(std::is_arithmetic_v<T>)
    constexpr T dot(const T a) noexcept {
        return a * a;
    }
    template<typename T>
        requires(std::is_arithmetic_v<T>)
    constexpr T dot(const T a, const T b) noexcept {
        return a * a + b * b;
    }
    template<typename T, typename... Ts>
        requires(std::is_arithmetic_v<T> && (std::same_as<T, Ts> && ...))
    constexpr T dot(const T a, const T b, const Ts... args) noexcept {
        return (dot(a, b) + dot(args...));
    }

    /**
    * \brief return the square root of sum of squares of variadic amount of real numbers, i.e. - their Euclidean norm (L2)
    * @param {arithmetic..., in}  values
    * @param {arithmetic,    out} square root of sum of squares of values (Euclidean norm; L2)
    **/
    template<typename T, typename... Ts>
        requires(std::is_arithmetic_v<T>)
    constexpr T norm(const T a) noexcept {
        const T _dot{ dot(a) };
        [[assume(_dot >= T{})]]
        return std::sqrt(_dot);
    }
    template<typename T, typename... Ts>
        requires(std::is_arithmetic_v<T> && (std::same_as<T, Ts> && ...))
    constexpr T norm(const T a, const T b, const Ts... ts) noexcept {
        return std::sqrt(dot(a, b, ts...));
    }

    /**
    * \brief Calculates the ULP ("unit in last place") distance between two floating point numbers (adhering to IEEE-754 format).
    *        The ULP distance of two floating point numbers is the count of valid floating point numbers representable between them.
    *        The following are changes between how this function counts the distance, and the interpretation of the standard as implemented:
    *        > `(x == y) => ulp_distance_between(x, y) == 0` (so `ulp_distance_between(-0, 0) == 0`)
    *        > `ulp_distance_between(maxFinite, INF) == 1`
    *        > `ulp_distance_between(x, -x) == 2 * ulp_distance_between(x, 0)`
    *
    *        notice that this function might recursively call itself twice in case input arguments differ by sign.
    *
    * @param {floating point, in}  floating point value #1
    * @param {floating point, in}  floating point value #2
    * @param {size_t,         out} ULP distance
    * @param {size_t,         out} ULP distance
    **/
    template<typename T>
        requires (std::is_floating_point_v<T>&& std::numeric_limits<T>::is_iec559)
    constexpr std::size_t ulp_distance_between(T lhs, T rhs) noexcept {
        using int_t = std::conditional_t<sizeof(T) == sizeof(float), std::uint32_t, std::uint64_t>;

        // handle NaN's & Infinities
        if (std::isnan(lhs) || std::isinf(lhs) || std::isnan(rhs) || std::isinf(rhs)) {
            return static_cast<std::size_t>(std::numeric_limits<T>::max());
        }

        // I want X == Y to imply 0 ULP distance even if X and Y aren't
        // bit-equal (-0 and 0), or X - Y != 0 (same sign infinities).
        if (lhs == rhs) { return 0; }

        // I want to ensure that +/- 0 is always represented as positive zero
        // (T{} == positive zero)
        if (lhs == T{}) { lhs = T{}; }
        if (rhs == T{}) { rhs = T{}; }

        // If arguments have different signs, I can handle them by summing their distance from positive zero
        if (std::signbit(lhs) != std::signbit(rhs)) {
            return (ulp_distance_between(std::abs(lhs), T{}) +
                    ulp_distance_between(std::abs(rhs), T{}));
        }

        // When both lhs and rhs are of the same sign, I can just
        // read the numbers bitwise as integers, and then subtract them
        int_t lc{ std::bit_cast<int_t>(lhs) };
        int_t rc{ std::bit_cast<int_t>(rhs) };

        // The ulp distance between two numbers is symmetric, so to avoid
        // dealing with overflows I want the bigger converted number on the lhs
        if (lc < rc) {
            Utilities::swap(lc, rc);
        }

        return static_cast<std::size_t>(lc - rc);
    }

    /**
    * \brief test two floating point numbers for comparison using absolute and relative errors.
    *        default parameter values are comparing within 0.1% relative error and 5 decimal places.
    * @param {floating point, in}  a (not INFINITY or NAN)
    * @param {floating point, in}  b (not INFINITY or NAN)
    * @param {floating point, in}  tolerance for absolute difference (default is 0.1%)
    * @param {floating point, in}  tolerance for relative difference (default is 5 decimal places)
    * @param {boolean,        out} true if (a == b), false otherwise
    **/
    template<typename T>
        requires (std::is_floating_point_v<T>)
    constexpr bool areEquals(const T lhs, const T rhs,
                             const T tol_for_absolute_diff = static_cast<T>(0.001),
                             const T tol_for_relative_diff = static_cast<T>(1e-5)) noexcept {
        [[assume(!std::isnan(lhs) && !std::isinf(lhs) && !std::isnan(rhs) && !std::isinf(rhs))]];

        // binary equal
        if (lhs == rhs) {
            return true;
        }
    
        // absolute error test
        // (equivalent check of std::fabs(lhs - rhs) >= tol, but without absolute value calculation)
        const T diff{ lhs - rhs };
        if ((diff <= -tol_for_absolute_diff) && (diff >= tol_for_absolute_diff)) {
            return false;
        }
    
        // if one side is zero, relative error transforms to be absolute error
        if (lhs == T{} || rhs == T{}) {
            return (diff <= tol_for_relative_diff) && (diff >= -tol_for_relative_diff);
        }

        // relative error
        // (equivalent check to (diff / (std::abs(lhs) + std::abs(rhs)) < tol_for_relative_diff), but without absolute value calculation and division)
        const T theoreticalMargin{ tol_for_relative_diff * (Numerics::max(lhs, -lhs) + Numerics::max(rhs, -rhs)) };
        return (diff <= theoreticalMargin) && (diff >= -theoreticalMargin);
    }

    /**
    * \brief returns the largest ULP magnitude within floating point range [a, b].
    *        this only makes sense for floating point types.
    * @param {floating point, in}  a (a < b)
    * @param {floating point, in}  b (a < b)
    * @param {floating point, out} ULP magnitude within floating point range [a, b]
    **/
    template<typename T>
        requires (std::is_floating_point_v<T>)
    constexpr T ulp_magnitude_within_range(const T a, const T b) noexcept {
        [[assume(a <= b)]];
        assert(a <= b);

        const T gamma_up{ std::nextafter(a, std::numeric_limits<T>::infinity()) - a };
        const T gamma_down{ b - std::nextafter(b, -std::numeric_limits<T>::infinity()) };

        return gamma_up < gamma_down ? gamma_down : gamma_up;
    }

    /**
    * \brief return the number of equi-distant floats in range [a, b]
    * @param {floating point,    in}  a (a < b)
    * @param {floating point,    in}  b (a < b)
    * @param {unsigned integral, out} equi-distant floats in range [a, b]
    **/
    template<typename T>
        requires (std::is_floating_point_v<T>)
    constexpr auto count_equidistant_floats(const T a, const T b) noexcept {
        [[assume(a <= b)]];
        using out_t = std::conditional_t<sizeof(T) == sizeof(float), std::uint32_t, std::uint64_t>;
        assert(a <= b);

        // since not every range can be split into equidistant floats exactly,
        // we actually calculate ceil(b / distance - a / distance), because in those cases we want to overcount.
        // notice the usage of modified Dekker's FastTwoSum algorithm to handle rounding. 
        const T distance{ Numerics::ulp_magnitude_within_range(a, b) };
        [[assume(distance > 0)]];
        const T ag{ a / distance };
        const T bg{ b / distance };
        const T s{ bg - ag };
        const T err{ (std::abs(a) <= std::abs(b)) ? -ag - (s - bg) : bg - (s + ag) };
        const T ceil_s{ std::ceil(s) };
        [[assume(ceil_s >= T{})]];
        const out_t ceil_s_i{ static_cast<out_t>(ceil_s) };

        return (ceil_s != s) ? ceil_s_i : ceil_s_i + static_cast<out_t>(err > T{});
    }

    /**
    * \brief check if integral type value (v) can be casted to floating point type (T).
    *        this is better implementation of std::in_range from utility header.
    * @param {integral, in}  integral value
    * @param {bool,     out} true if value can be casted to floating type 'T'
    **/
    template<class T, class V>
        requires(std::is_floating_point_v<T> && std::is_integral_v<V>)
    constexpr bool in_range(const V value) noexcept {
        constexpr std::size_t digits{ static_cast<std::size_t>(std::numeric_limits<T>::digits) };
        return static_cast<std::size_t>(std::ceil(std::log2l(value))) <= digits;
    }

    /**
    * \brief calculate a*b-c*d with maximal floating point error of 1.5*ULP.
    *        (if sign(ab) == sign(cd) - error will be +/-1.5*ULP ,) otherwise it will be +/-ULP
    *        see: Claude-Pierre Jeannerod, Nicolas Louvet, and Jean-Michel Muller,
    *             "Further Analysis of Kahan's Algorithm for the Accurate Computation
    *             of 2x2 Determinants". Mathematics of Computation, Vol. 82, No. 284,
    *             Oct. 2013, pp. 2245-2264
    * @param {floating point, in} a
    * @param {floating point, in} b
    * @param {floating point, in} c
    * @param {floating point, in} d
    * @param {floating point, in} a*b - c*d
    **/
    template<class T>
        requires(std::is_floating_point_v<T>)
    constexpr T diff_of_products(const T a, const T b, const T c, const T d) {
        const T w{ d * c };
        const T e{ std::fma(c, d, -w) };
        const T f{ std::fma(a, b, -w) };
        return (f - e);
    }

    /**
    * \brief implements std::copysign(1, value)
    * @param {arithmetic, in}  value
    * @param {arithmetic, out} std::copysign(1, value)
    **/
    template<typename T>
        requires(std::is_arithmetic_v<T>)
    constexpr T sign(T value) noexcept {
        if constexpr (std::is_integral_v<T> && !std::signed_integral<T>) {
            return static_cast<T>(1);
        } else if constexpr (std::signed_integral<T>) {
            return (value >= T{} ? static_cast<T>(1) : static_cast<T>(-1));
        } else {
            return std::copysign(static_cast<T>(1), value);
        }
    }

    /**
    * @param {unsigned, in}  integral value
    * @param {bool,     out} true if input argument is even
    **/
    template<typename T>
        requires(std::is_unsigned_v<T>)
    constexpr bool isEven(const T x) noexcept {
        return (x & 1);
    }
    template<std::size_t x>
    constexpr bool isEven() noexcept {
        return (x & 1);
    }

    /**
    * @param {unsigned, in}  integral value
    * @param {bool,     out} true if input argument is power of two
    **/
    template<typename T>
        requires(std::is_unsigned_v<T>)
    constexpr bool isPowerOfTwo(const T x) noexcept {
        return (x != 0) && !(x & (x - 1));
    }
    template<std::size_t x>
    constexpr bool isPowerOfTwo() noexcept {
        return (x != 0) && !(x & (x - 1));
    }

    /**
    * \brief align a given integral to next power-of-two value
    * @param {size_t, in}  number to align
    * @param {size_t, in}  next power-of-two to align to
    * @param {size_t, out} first input argument, aligned to upper multiplier of second argument
    **/
    constexpr std::size_t alignToNext(const std::size_t v, const std::size_t alignment) noexcept {
        [[assume(isPowerOfTwo(alignment))]];
        assert(isPowerOfTwo(alignment));
        return (v + alignment - 1) & ~(alignment - 1);
    }
    template<std::size_t alignment>
        requires(isPowerOfTwo<alignment>())
    constexpr auto alignToNext(const std::size_t v) noexcept {
        return (v + alignment - 1) & ~(alignment - 1);
    }
    template<std::size_t v, std::size_t alignment>
        requires(isPowerOfTwo<alignment>())
    constexpr auto alignToNext() noexcept {
        return (v + alignment - 1) & ~(alignment - 1);
    }

    /**
    * \brief align a given integral to previous power-of-two value
    * @param {size_t, in}  number to align
    * @param {size_t, in}  previous power-of-two to align to
    * @param {size_t, out} first input argument, aligned to lower multiplier of second argument
    **/
    constexpr std::size_t alignToPrev(const std::size_t v, const std::size_t alignment) noexcept {
        [[assume(isPowerOfTwo(alignment))]];
        assert(isPowerOfTwo(alignment));
        return (v & ~(alignment - 1));
    }
    template<std::size_t alignment>
        requires(isPowerOfTwo<alignment>())
    constexpr auto alignToPrev(const std::size_t v) noexcept {
        return (v & ~(alignment - 1));
    }
    template<std::size_t v, std::size_t alignment>
        requires(isPowerOfTwo<alignment>())
    constexpr auto alignToPrev() noexcept {
        return (v & ~(alignment - 1));
    }

    /**
    * \brief return the division of two integrals, rounded up
    * @param {arithmetic, in}  numerator
    * @param {arithmetic, in}  denominator
    * @param {arithmetic, out} rounded up (numerator / denominator)
    **/
    template<typename T>
        requires(std::is_integral_v<T>)
    constexpr T roundedUpDivision(const T num, const T den) {
        assert(den != T{});
        if constexpr (std::is_signed_v<T>) {
            const T offset{ num < T{} && den < T{} ? 1 : -1 };
            return (num + den + offset) / den;
        }
        else {
            return (num + den - 1) / den;
        }
    }
    template<auto num, auto den>
        requires(std::is_integral_v<decltype(num)> && std::is_integral_v<decltype(den)> && den != 0)
    constexpr auto roundedUpDivision() {
        using T = decltype(den);
        if constexpr (std::is_signed_v<T>) {
            constexpr T offset{ num < T{} && den < T{} ? 1 : -1 };
            return (num + den + offset) / den;
        }
        else {
            return (num + den - 1) / den;
        }
    }

    /**
    * \brief return the division of two integrals, rounded down
    * @param {integral, in}  numerator
    * @param {integral, in}  denominator
    * @param {integral, out} rounded low (numerator / denominator)
    **/
    template<typename T>
        requires(std::is_integral_v<T>)
    constexpr T roundedLowDivision(const T num, const T den) {
        T q{ num / den };
        if (const T r{ num % den }; (r != T{}) && ((r < T{}) != (den < T{}))) {
            --q;
        }
        return q;
    }
    template<auto num, auto den>
        requires(std::is_integral_v<decltype(num)> && std::is_integral_v<decltype(den)> && den != 0)
    constexpr auto roundedLowDivision() {
        using T = decltype(den);
        constexpr T q{ num / den };
        constexpr T r{ num % den };
        return ((r != T{}) && ((r < T{}) != (den < T{}))) ? q - 1 : q;
    }

    /**
    * \brief Default integer division is truncated, not rounded. this operation rounds the division instead of truncating it.
    *        notice that rounding ties (i.e., result % divisor == 0.5) are rounded up.
    * @param {integral, in}  numerator
    * @param {integral, in}  denominator
    * @param {integral, out} rounded (numerator / denominator)
    **/
    template<typename T>
        requires(std::is_integral_v<T>)
    constexpr T roundedivision(const T num, const T den) {
        if constexpr (std::is_signed_v<T>) {
            return (num + den / 2) / den;
        } else {
            return roundedLowDivision(num, den);
        }
    }
    template<auto num, auto den>
        requires(std::is_integral_v<decltype(num)> && std::is_integral_v<decltype(den)> && den != 0)
    constexpr auto roundedivision() {
        using T = decltype(den);
        if constexpr (std::is_signed_v<T>) {
            return (num + den / 2) / den;
        }
        else {
            return roundedLowDivision(num, den);
        }
    }

    /**
    * \brief limit a value into a given boundary in a circular manner
    * @param {floating point, in}  value
    * @param {floating point, in}  boundary lower limit
    * @param {floating point, in}  boundary upper limit
    * @param {floating point, out} limited value
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr T clampCircular(const T value, const T min, const T max) noexcept {
        return (min + std::remainder(value - min, max - min + static_cast<T>(1)));
    }

    /**
    * \brief return the difference between two angles, along a unit circle
    * @param {floating point, in}  start angle [rad]
    * @param {floating point, in}  end angle [rad]
    * @param {floating point, out} difference angles from start to end [rad]
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr T angleDifference(const T from, const T to) noexcept {
        constexpr T tau{ static_cast<T>(6.283185307179586476925286766559) };
        const T difference{ std::fmod(to - from, tau) };
        return std::fmod(static_cast<T>(2) * difference, tau) - difference;
    }

    /**
    * \brief clamp value to given region
    * @param {arithmetic, in}  value to clamp
    * @param {arithmetic, in}  range minimal value
    * @param {arithmetic, in}  range maximal value
    * @param {arithmetic, out} clamped value
    **/
    template<typename T>
        requires(std::is_arithmetic_v<T>)
    constexpr T clamp(const T x, const T minVal, const T maxVal) noexcept {
        assert(minVal < maxVal);
        return Numerics::min(Numerics::max(x, minVal), maxVal);
    }
    template<auto minVal, auto maxVal, class T = decltype(minVal)>
        requires(std::is_arithmetic_v<T> && std::is_same_v<T, decltype(maxVal)> && (minVal < maxVal))
    constexpr T clamp(const T x) noexcept {
        return Numerics::min(Numerics::max(x, minVal), maxVal);
    }

    /**
    * \brief stable numeric solution of a quadratic equation (a*x^2 + b*x + c = 0).
    *        solution will avoid numerical cancelation in the following cases:
    *        > b^2 >> || 4*a*c||
    *        > b^2 ~= 4*a*c
    * 
    * @param {floating_point,                   in}  a
    * @param {floating_point,                   in}  b
    * @param {floating_point,                   in}  c
    * @param {{floating_point, floating_point}, out} {smaller root, larger root}
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto SolveQuadratic(const T a, const T b, const T c) noexcept {
        using out_t = struct { T x1; T x2; };

        const T delta{ Numerics::diff_of_products(b, b, static_cast<T>(4.0) * a, c) };
        assert(delta >= T{});
        const T t0{ std::sqrt(delta) };
        const T t1{ b + std::copysign(t0, b) };

        return out_t{ (static_cast<T>(-2.0) * c) / t1,
                      t1 / (static_cast<T>(-2.0) * a) };
    }

    /**
    * \brief stable numeric solution of a cubic equation (x^3 + b*x^2 + c*x + d = 0)
    * 
    * @param {floating_point,            in}  b
    * @param {floating_point,            in}  c
    * @param {floating_point,            in}  d
    * @param {array<floating_point, 6>}, out} 1x6 array holding three paired solutions in the form (real solution #1, imag solution #1, ...)
    *                                         if 1 real root exists: {real root 1, 0, Re(root 2), Im(root 2), Re(root 3), Im(root 3)}
    *                                         if 3 real root exists: xo_roots[0] = real root 1, xo_roots[2] = real root 2, xo_roots[4] = real root 3
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr std::array<T, 6> SolveCubic(const T b, const T c, const T d) noexcept {
        constexpr T ov3{ static_cast<T>(1) / static_cast<T>(3) };
        constexpr T ov27{ static_cast<T>(1) / static_cast<T>(27) };
        const T sq3{ static_cast<T>(std::sqrt(3)) };
        const T ovsqrt27{ static_cast<T>(1) / std::sqrt(static_cast<T>(27)) };

        // housekeeping
        std::array<T, 6> roots{ { T{} } };

        // transform to: x^3 + p*x + q = 0
        const T bSqr{ b * b };
        const T p{ (static_cast<T>(3) * c - bSqr) * ov3 };
        const T q{ (static_cast<T>(9) * b * c - static_cast<T>(27) * d - static_cast<T>(2) * bSqr * b) * ov27 };

        // single real solution? ( x = w - (p / (3 * w)) -> (w^3)^2 - q*(w^3) - (p^3)/27 = 0)
        if (T h{ q * q / static_cast<T>(4) + p * p * p * ov27 }; h >= T{}) {
            [[assume(h >= T{})]];
            h = std::sqrt(h);

            const T qHalf{ q * static_cast<T>(0.5) };
            const T bThird{ b * ov3 };
            const T r{ qHalf + h };
            const T t{ qHalf - h };
            const T s{ std::cbrt(r) };
            const T u{ std::cbrt(t) };
            const T re{ -(s + u) / static_cast<T>(2) - bThird };
            const T im{  (s - u) * sq3 / static_cast<T>(2) };

            // real root
            roots[0] = (s + u) - bThird;

            // first complex root
            roots[2] = re;
            roots[3] = im;

            // second complex root
            roots[4] = re;
            roots[5] = -im;
        }  // three real solutions
        else {
            [[assume(-p >= T{})]];
            const T i{ p * std::sqrt(-p) * ovsqrt27 };     // p is negative (since h is positive)
            const T j{ std::cbrt(i) };
            const T k{ ov3 * std::acos((q / (static_cast<T>(2) * i))) };
            const T m{ std::cos(k) };
            const T n{ std::sin(k) * sq3 };
            const T s{ -b * ov3 };

            // roots
            roots[0] = static_cast<T>(2) * j * m + s;
            roots[2] = -j * (m + n) + s;
            roots[4] = -j * (m - n) + s;
        }

        return roots;
    }

    /**
    * \brief given a function (f(x)) and bounds [x0, x1], attempt to find the local minimizer 'x' of f(x).
    *        function uses golden section search combined with parabolic interpolation.
    * @param {callable,       in}  function to be minimized
    * @param {floating_point, in}  x lower bound
    * @param {floating_point, in}  x upper bound
    * @param {size_t,         in}  maximal amount of iterations (default is 500)
    * @param {floating_point, in}  tolerance on local minimizer convergence (default is 1e-4)
    * @param {struct,         out} output signature is {floating_point, floating_point, bool} where:
    *                              > first value is 'x' which minimizes f(x).
    *                              > value of f(x) at 'x'.
    *                              > flag which returns true if optimization converged ('x' was within 'tol_x')
    *                                and false if optimization terminated due to passing the maximal amount of iterations ('maxIter')
    **/
    template<typename FUNC, typename T>
        requires(std::is_floating_point_v<T>  &&
                 std::is_invocable_v<FUNC, T> &&
                 std::is_floating_point_v<typename std::invoke_result_t<FUNC, T>>)
    constexpr auto fminbnd(FUNC&& func, const T x0, const T x1,
                           const std::size_t maxIter = 500, const T tol_x = static_cast<T>(1e-4)) {
        using out_t = struct { T x; T function_value; bool converged; };
        assert(x0 < x1);
        assert(tol_x > T{});
        assert(maxIter > 0);

        // housekeeping
        const T c{ (static_cast<T>(3.0) - std::sqrt(static_cast<T>(5.0))) / static_cast<T>(2.0) };
        const T eps{ std::nextafter(static_cast<T>(1.0), static_cast<T>(2.0)) - static_cast<T>(1.0) };
        const T seps{ std::sqrt(eps) };
        std::size_t iter{};
        T a{ x0 };
        T b{ x1 };
        
        // start point
        T v{ a + c * (b - a) };
        T w{ v };
        T xf{ v };
        T x{ xf };
        T fx{ func(x) };
        T d{};
        T e{};

        // main loop
        T fv{ fx };
        T fw{ fx };
        T xm{ (a + b) / static_cast<T>(2.0) };
        T tol1{ std::fma(seps, std::abs(xf), tol_x / static_cast<T>(3.0)) };
        T tol2{ static_cast<T>(2.0) * tol1 };
        while (iter < maxIter &&
               std::abs(xf - xm) > (tol2 - (b - a) / static_cast<T>(2.0))) {
            bool gs{ true };

            // is parabolic fit possible?
            if (std::abs(e) > tol1) {
                gs = false;
                const T xfw{ xf - w };
                const T xfv{ xf - v };
                T r{ xfw * (fx - fv) };
                T q{ xfv * (fx - fw) };
                T p{ Numerics::diff_of_products(xfv, q, xfw, r) };
                q = static_cast<T>(2.0) * (q - r);
                if (q > T{}) {
                    p = -p;
                }
                q = std::abs(q);
                r = e;
                e = d;

                // is parabola acceptable>
                if ((std::abs(p) < std::abs(q * r / static_cast<T>(2.0))) &&
                    (p > q * (a - xf)) && (p < q * (b - xf))) {
                    assert(q != T{});
                    d = p / q;
                    x = xf + d;
                    
                    if (((x - a) < tol2) || ((b - x) < tol2)) {
                        T si{ Numerics::sign(xm - xf) };
                        if (Numerics::areEquals(xm - xf, T{})) {
                            si += static_cast<T>(1.0);
                        }
                        d = si * tol1;
                    }
                    
                }
                else {
                    gs = false;
                }
            }

            // is golden section step required?
            if (gs) {
                e = xf >= xm ? a - xf : b - xf;
                d = c * e;
            }

            // do no evaluate function to close to 'xf'
            T si{ Numerics::sign(d) };
            if (Numerics::areEquals(d, T{})) {
                si += static_cast<T>(1.0);
            }
            x = std::fma(si, Numerics::max(std::abs(d), tol1), xf);
            const T fu{ func(x) };

            // update parameters for next iteration
            if (fu <= fx) {
                if (x >= xf) {
                    a = xf;
                }
                else {
                    b = xf;
                }
                v = w;
                fv = fw;
                w = xf;
                fw = fx;
                xf = x;
                fx = fu;
            }
            else {
                if (x < xf) {
                    a = x;
                }
                else {
                    b = x;
                }

                if (fu <= fw || Numerics::areEquals(w, xf)) {
                    v = w;
                    fv = fw;
                    w = x;
                    fw = fu;
                }
                else if (fu <= fv || Numerics::areEquals(v, xf) || Numerics::areEquals(v, w)) {
                    v = x;
                    fv = fu;
                }
            }
            xm = (a + b) / static_cast<T>(2.0);
            tol1 = std::fma(seps, std::abs(xf), tol_x / static_cast<T>(3.0));
            tol2 = static_cast<T>(2.0) * tol1;

            ++iter;
        }

        // output
        return out_t{ x, fx, iter != maxIter };
    }
}
