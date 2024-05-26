﻿#pragma once
#include <assert.h>
#include <bit> // std::bit_cast
#include <limits> // std::numeric_limits
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
        constexpr std::size_t digits{ std::numeric_limits<T>::digits };
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
        return std::sqrt(dot(a));
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
        requires(std::is_arithmetic_v<T> && std::is_same_v<T, decltype(maxVal) >&& (minVal < maxVal))
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
    * \brief given collection of numbers return their sum (numerically exact), mean and variance.
    *        using Welford’s method (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm)
    * @param {forward_iterator,                     in}  iterator to first element to sum
    * @param {forward_iterator,                     in}  iterator to last element to sum
    * @param {{arithmetic, arithmetic, arithmetic}, out} {sum of elements, mean of elements, sample variance}
    **/
    template<std::forward_iterator InputIt, class T = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(std::is_arithmetic_v<T>)
    constexpr auto statistics(InputIt first, const InputIt last) {
        using out_t = struct { T sum; T mean; T var; };

        T sum{};
        T err{};
        T mean{};
        T var{};
        std::size_t amount{};
        for (; first != last; ++first) {
            const T value{ *first };
            ++amount;
            
            // sum
            const T m{ sum + value };
            err += (std::abs(sum) >= std::abs(value)) ? (sum - m + value) : (value - m + sum);
            sum = m;

            // mean
            const T delta{ value - mean };
            mean += delta / static_cast<T>(amount);

            // variance
            var += delta * (value - mean);
        }

        return out_t{ sum + err, mean, var / static_cast<T>(amount - 1) };
    }
    template<Concepts::Iterable COL>
    constexpr auto statistics(const COL& collection) {
        return statistics(collection.begin(), collection.end());
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
    * \brief perform in place partition operations.
    *        if range is given by forward iterator - use std::partition which perform O(n*log(n)) swaps
    *        if range is given by bidirectional iterator - perform O(n/2) swaps using diffrent algorithm.
    * @param {forward_iterator | bidirectional_iterator, in}  iterator to range start
    * @param {forward_iterator | bidirectional_iterator, in}  iterator to range end
    * @param {invocable,                                 in}  unary predicate which returns ​true if the element should be ordered before other elements
    * @param {forward_iterator | bidirectional_iterator, out} iterator to first element of second group
    **/
    template<class It, class UnaryPred, class T = typename std::decay_t<decltype(*std::declval<It>())>>
        requires((std::forward_iterator<It> || std::bidirectional_iterator<It>) && std::is_invocable_v<UnaryPred, T>)
    It partition(It first, It last, UnaryPred&& p) {
        if constexpr (std::bidirectional_iterator<It>) {
            if (first == last) {
                return first;
            }
            --last;

            while (first != last) {
                while (first != last && p(*first)) {
                    ++first;
                }
                while (first != last && !p(*last)) {
                    --last;
                }

                Utilities::swap(*first, *last);
            }
            return first;
        } else if (std::forward_iterator<It>) {
            return std::partition(first, last, FWD(p));
        }
    }
}
