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
        else if (sizeof(T) == sizeof(double)) {
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
    * \bried given amount of bits, return the number of size_t's needed to store them
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
    * @param {floating point, in}  a
    * @param {floating point, in}  b
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
        if (lhs == rhs) return true;
    
        // absolute error test
        // (equivalent check of std::fabs(lhs - rhs) <= tol, but without the subtraction to allow for INFINITY in comparison)
        if ((lhs + tol_for_absolute_diff >= rhs) &&
            (rhs + tol_for_absolute_diff >= lhs)) {
            return false;
        }
    
        // relative error
        /*
        old code which doesn't handle INFINITY. kept for documentation purposes.
        const T absLhs{ std::abs(lhs) };
        const T absRhs{ std::abs(rhs) };
        const T diff{ std::abs(lhs - rhs) };
        return (diff / (absLhs + absRhs) < tol_for_relative_diff);
        */
        const T theoreticalMargin{ tol_for_relative_diff * std::max(std::abs(lhs), std::abs(rhs)) };
        const T usedMargin{ std::isinf(theoreticalMargin) ? T{} : theoreticalMargin };
        return (lhs + usedMargin >= rhs) && (rhs + usedMargin >= lhs);
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
        // we actually compute ceil(b / distance - a / distance), because in those cases we want to overcount.
        // notice the usage of modified Dekker's FastTwoSum algorithm to handle rounding. 
        const T distance{ Numerics::ulp_magnitude_within_range(a, b) };
        [[assume(distance > 0)]];
        const T ag{ a / distance };
        const T bg{ b / distance };
        const T s{ bg - ag };
        const T err{ (std::abs(a) <= std::abs(b)) ? -ag - (s - bg) : bg - (s + ag) };
        const out_t ceil_s{ static_cast<out_t>(std::ceil(s)) };

        return (std::ceil(s) != s) ? ceil_s : ceil_s + static_cast<out_t>(err > T{});
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
    * \brief return the divison of two integrals, rounded up
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
    * \brief return the divison of two integrals, rounded down
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
        T q{ num / den };
        if (constexpr T r{ num % den }; (r != T{}) && ((r < T{}) != (den < T{}))) {
            --q;
        }
        return q;
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
    * \brief return the maximal value among variadic amount of ordererd ascending elements
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
    * \brief return the square root of sum of squares of variadic amount of real numbers, i.e. - their eulcidean norm (L2)
    * @param {arithmetic..., in}  values
    * @param {arithmetic,    out} square root of sum of squares of values (euclidean norm; L2)
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
    * \brief exact floating point numerical accumulation of a given collection using Neumaier variant of Kahan and Babuska summation.
    *        Its like Kahan summation but also covers the case where next value to be added is larger than absolute value of running sum.
    *        notice that on large vectors (10 million elements) tests show that this accumulation is 25% slower than std::accumulate and 220% slower than std::reduce.
    * @param {forward_iterator, in}  iterator to first element to sum
    * @param {forward_iterator, in}  iterator to last element to sum
    * @param {arithmetic,       in}  initial value
    * @param {arithmetic,       out} sum of elements
    **/
    template<std::forward_iterator InputIt, class T = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(std::is_arithmetic_v<T>)
    constexpr T accumulate(InputIt first, const InputIt last, T&& init = T{}) {
        // initialization
        T sum{ MOV(init) };
        T err{};

        // sum
        for (; first != last; ++first) {
            const T k{ *first };
            const T m{ sum + k };

            err += (std::abs(sum) >= std::abs(k)) ? (sum - m + k) : (k - m + sum);
            sum = m;
        }

        // output
        return (sum + err);
    }
    template<Concepts::Iterable COL>
    constexpr auto accumulate(const COL& collection) {
        return accumulate(collection.begin(), collection.end());
    }

    /**
    * \brief return the mean and standard deviation of a collection of arithmetic values.
    * @param {forward_iterator,         in}  iterator to first value
    * @param {forward_iterator,         in}  iterator to last value
    * @param {{arithmetic, arithmetic}, out} {mean, standard devialtion}
    **/
    template<std::forward_iterator InputIt, class T = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(std::is_arithmetic_v<T>)
    constexpr auto mean_and_std(InputIt first, const InputIt last) {
        using out_t = struct { T mean; T std; };
        T mean{};
        T std{};
        T count{};
        for (InputIt it{ first }; it != last; ++it) {
            ++count;
            const T delta{ *it - mean };
            mean += delta / count;
            const T delta2{ *it - mean };
            std += delta * delta2;
        }
        return out_t{ mean, std / count };
    }
    template<Concepts::Iterable COL>
    constexpr auto mean_and_std(const COL& collection) {
        return mean_and_std(collection.begin(), collection.end());
    }

    /**
    * \brief given collection of values, partition them into predetermined mount of bins and return the bin counts and bin edges.
    * @param {forward_iterator,                                     in}  iterator to first element to sum
    * @param {forward_iterator,                                     in}  iterator to last element to sum
    * @param {size_t,                                               in}  number of bins (a good initial choise can be ceil(sqrt(amount of values in input)) )
    * @param {{vector<size_t>, vector<arithmetic>, vector<size_t>}, out} {bin counts,
    *                                                                     bin edges - first element is the leading edge of the first bin. The last element is the trailing edge of the last bin,
    *                                                                     index array where 'i' cell holds the index of 'bin edges' in which *(first + i) value is catagroized }
    **/
    template<std::forward_iterator InputIt, class T = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(std::is_arithmetic_v<T>)
    constexpr auto histcounts(InputIt first, const InputIt last, const std::size_t nbins) {
        using out_t = struct { std::vector<std::size_t> N; std::vector<T> edges; std::vector<std::size_t> bin; };

        // housekeeping
        const std::size_t len{ static_cast<std::size_t>(std::distance(first, last)) };
        std::vector<T> edges(nbins + 1, T{});
        std::vector<std::size_t> N(nbins);
        std::vector<std::size_t> bins(len);

        // define bin edges
        T maxvalue{ *first };
        for (InputIt f{ first }; f != last; ++f) {
            maxvalue = Numerics::max(*f, maxvalue);
        }
        const T binWidth{ (maxvalue + static_cast<T>(1)) / static_cast<T>(nbins) };
        for (std::size_t i{1}; i < nbins + 1; ++i) {
            edges[i] = edges[i - 1] + binWidth;
        }

        // discretize values into bins
        std::size_t i{};
        for (InputIt f{ first }; f != last; ++f) {
            const std::size_t bin{ static_cast<std::size_t>(std::floor(*f /binWidth)) };
            ++N[bin];
            bins[i] = bin;
            ++i;
        }
        
        // output
        return out_t{ N, edges, bins };
    }
    template<Concepts::Iterable COL>
    constexpr auto histcounts(const COL& collection, const std::size_t nbins) {
        return histcounts(collection.begin(), collection.end(), nbins);
    }

    /**
    * \brief return the convolution between two collections.
    * @param {forward_iterator, in}  iterator to first collection first value
    * @param {forward_iterator, in}  iterator to first collection last value
    * @param {forward_iterator, in}  iterator to second collection first value
    * @param {forward_iterator, in}  iterator to second collection last value
    * @param {forward_iterator, out} iterator to begining of collection which will hold the convolution between first and second collections
    *                                output collection should be large enough to hold the amount of elements in first and second collections.
    **/
    template<std::forward_iterator It1, std::forward_iterator It2, std::forward_iterator Ot, class T = typename std::decay_t<decltype(*std::declval<It1>())>>
        requires(std::is_arithmetic_v<T> && std::is_same_v<T, typename std::decay_t<decltype(*std::declval<Ot>())>> &&
                 std::is_same_v<T, typename std::decay_t<decltype(*std::declval<It2>())>>)
    constexpr auto conv(const It1 u_first, const It1 u_last,
                        const It2 v_first, const It2 v_last, Ot out) {
        const std::size_t size_u{ static_cast<std::size_t>(std::distance(u_first, u_last)) };
        const std::size_t size_v{ static_cast<std::size_t>(std::distance(v_first, v_last)) };
        const std::size_t size{ size_u + size_v - 1};

        for (std::size_t i{}; i < size; ++i) {
            T sum{};
            std::size_t iter{ i };
            for (std::size_t j{}; j <= i; ++j) {
                if ((j < size_u) && (iter < size_v)) {
                    sum += *(u_first + j) * *(v_first + iter);
                }
                --iter;
            }
            *(out + i) = sum;
        }
    }

    /**
    * \brief apply "Transposed-Direct-Form-II structure of the IIR filter" on given signal/collection as time domain difference equation
    * @param {size_t} amount of coefficients in numerator (NB)
    * @param {size_t} amount of coefficients in denominator (NA; if NA > 1 - its an infinite impulse response filter, otherwise its a finite impulse response filter)
    * @param {forward_iterator,  in}  iterator to first element on which filter is applied
    * @param {forward_iterator,  in}  iterator to last element on which filter is applied
    * @param {forward_iterator,  out} iterator to begining of collection which will hold the filtered output
    * @param {array<arithmetic>, in}  Numerator coefficients of rational transfer function (has NB elements)
    * @param {array<arithmetic>, in}  Denominator coefficients of rational transfer function (has NA elements)
    * @param {array<arithmetic>, in}  Initial conditions for filter delays (default is zeros)
    **/
    template<std::size_t NB, std::size_t NA,
             std::forward_iterator It, std::forward_iterator Ot,
             class T = typename std::decay_t<decltype(*std::declval<It>())>>
        requires(std::is_arithmetic_v<T> && std::is_same_v<T, typename std::decay_t<decltype(*std::declval<Ot>())>>)
    constexpr void filter(const It x_first, const It x_last, Ot out,
                          const std::array<T, NB>& b, const std::array<T, NA>& a, const std::array<T, Numerics::max(NB, NA)>& z = {{T{}}}) {
        constexpr std::size_t coeff_size{ Numerics::max(NB, NA) };
        assert(!Numerics::areEquals(a[0], T{}));

        // local copy of delay vector
        std::array<T, coeff_size> Z(z);
        
        // normalized Numerator coefficients and equalize its size to Denominator
        std::array<T, coeff_size> bn{ {T{}} };
        const T a0{ a[0] };
        Utilities::static_for<0, 1, NB>([&bn, &b, a0](std::size_t i) {
            bn[i] = b[i] / a0;
        });

        // normalized Denominator coefficients and equalize its size to Numerator
        std::array<T, coeff_size> an{ {T{}} };
        Utilities::static_for<0, 1, NA>([&an, &a](std::size_t i) {
            an[i] = a[i] / a[0];
        });
        
        // perform filter operation
        const std::size_t size_x{ static_cast<std::size_t>(std::distance(x_first, x_last)) };
        for (std::size_t m{}; m < size_x; ++m) {
            const T x_m{ *(x_first + m) };
            const T out_m{ std::fma(bn[0], x_m, Z[0]) };
            *(out + m) = out_m;

            // infinite impulse response filter
            if constexpr (NA > 1) {
                Utilities::static_for<1, 1, coeff_size>([&bn, &an, &Z, x_m, out_m](std::size_t i) {
                    Z[i - 1] = std::fma(bn[i], x_m, Z[i]) - an[i] * out_m;
                });
            }
            //finite impulse response filter
            else {
                Utilities::static_for<1, 1, coeff_size>([&bn, &Z, x_m](std::size_t i) {
                    Z[i - 1] = std::fma(bn[i], x_m, Z[i]);
                });
            }
        }
    }

    /**
    * \brief Apply a kernal, in ordererd manner, on variadic amount of random access arithmetic ranges and return the result in a diffrent range.
    *        Although operation will be element wise, kernel should be given in scalar manner.
    * 
    * \example:
    *        std::array<int, 9> out;
    *        std::array<int, 6> a{{1, 1, 1, 0, 0, 0}};
    *        std::array<int, 6> b{{2, 2, 3, 3, 4, 5}};
    *        std::vector<int> c = {5, 5, 1, 1, 7, 7};
    *        const auto saxy = [](const int a, const int x, const int y) { return (a * x + y); };
    *        applyKernel<4>(saxy, out, a, b, c); // out = 7, 7, 4, 1, 7, 7, ?, ?, ?
    * 
    * @param {size_t}                       iteration block (register size)
    * @param {invocable,            in}     kernel {U <= @(T, T...)}
    * @param {RandomAccessSized,    in|out} out (range) = kernel(c, cc...)
    * @param {RandomAccessSized,    in}     c (range)
    * @param {RandomAccessSized..., in}     cc (ranges)
    **/
    template<std::size_t BLOCK, typename F, typename U, typename C, typename... Cc>
        requires(Numerics::isPowerOfTwo<BLOCK>() &&
                 Concepts::RandomAccessSized<U> &&
                 Concepts::RandomAccessSized<C> && std::is_arithmetic_v<typename C::value_type> &&
                 (Concepts::RandomAccessSized<Cc> && ...) && (std::is_same_v<typename C::value_type, typename Cc::value_type> && ...) &&
                 (std::is_invocable_v<F, typename C::value_type, typename Cc::value_type...>) &&
                 std::is_same_v<typename U::value_type, typename std::invoke_result_t<F, typename C::value_type, typename Cc::value_type...>>)
    constexpr void applyKernel(F&& kernel, U& out, const C& c, const Cc&... cc) {
        const std::size_t len{ Numerics::min(out.size(), c.size(), cc.size()...) };
        const std::size_t last{ Numerics::alignToPrev<BLOCK>(len) };

        std::size_t i{};
        for (; i < last; i+= BLOCK) {
            Utilities::static_for<0, 1, BLOCK>([i, _kernel = FWD(kernel), &out, &c, &cc...](std::size_t j) {
                out[i + j] = _kernel(c[i + j], cc[i + j]...);
            });
        }
        for (; i < len; ++i) {
            out[i] = kernel(c[i], cc[i]...);
        }
    }

    /**
    * \brief stable numeric solution of a quadratic equation (a*x^2 + b*x + c = 0)
    * 
    * @param {floating_point,                         in}  a
    * @param {floating_point,                         in}  b
    * @param {floating_point,                         in}  c
    * @param {{bool, floating_point, floating_point}, out} {true if a solution exists - false otherwise, smaller root, larger root}
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr auto SolveQuadratic(const T a, const T b, const T c) noexcept {
        using out_t = struct { bool found; T x1; T x2; };

        // trivial solution
        if (Numerics::areEquals(a, T{}) && Numerics::areEquals(b, T{})) [[unlikely]]  {
            return out_t{ true, T{}, T{} };
        }

        const T discriminant{ b * b - static_cast<T>(4) * a * c };
        if (discriminant < T{}) {
            return out_t{ false, T{}, T{} };
        }

        // solution
        [[assume(discriminant >= T{})]]
        const T t{ static_cast<T>(-0.5) * (b + Numerics::sign(b) * std::sqrt(discriminant)) };
        T x1{ t / a };
        T x2{ c / t };
        if (x1 > x2) {
            Utilities::swap(x1, x2);
        }

        return out_t{ true, x1, x2 };
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
}
