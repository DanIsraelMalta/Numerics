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
#include "Concepts.h"

/**
* Utilities to operate and handle variadic arguments
**/
namespace Variadic {

    /**
    * \brief in place iteration and mutation of variadic elements
    * @param {FUNC, in} unary operations
    * @param {...,  in} variadic arguments which can be mutated during operation
    **/
    template<class FUNC, class...Ts>
        requires(std::is_invocable_v<FUNC, Ts&> && ...)
    constexpr void forEach(FUNC&& f, Ts& ... ts) {
        ([&] { f(ts); } (), ...);
    }

    /**
    * \brief short circuited test to check if variadic amount of elements are lower than a given value
    * @param {OrderedDescending,    in}  value
    * @param {OrderedDescending..., in}  values
    * @param {OrderedDescending,    out} true if all values are lower than a given value, false otherwise
    **/
    template<typename T, typename...Ts>
        requires(Concepts::OrderedDescending<T, T> && (std::is_same_v<T, Ts> && ...))
    constexpr bool lowerThan(const T value, const Ts...ts) noexcept {
        return ((ts < value) && ...);
    };
    template<auto value, typename...Ts>
        requires(Concepts::OrderedDescending<decltype(value), decltype(value)> && (std::is_same_v<decltype(value), Ts> && ...))
    constexpr bool lowerThan(const Ts...ts) noexcept {
        return ((ts < value) && ...);
    };

    /**
    * \brief short circuited test to check if variadic amount of elements are greater than a given value
    * @param {OrderedAscending,    in}  value
    * @param {OrderedAscending..., in}  values
    * @param {OrderedAscending,    out} true if all values are greater than a given value, false otherwise
    **/
    template<typename T, typename...Ts>
        requires(Concepts::OrderedAscending<T, T> && (std::is_same_v<T, Ts> && ...))
    constexpr bool greaterThan(const T value, const Ts...ts) noexcept {
        return ((ts > value) && ...);
    };
    template<auto value, typename...Ts>
        requires(Concepts::OrderedAscending<decltype(value), decltype(value)> && (std::is_same_v<decltype(value), Ts> && ...))
    constexpr bool greaterThan(const Ts...ts) noexcept {
        return ((ts > value) && ...);
    };

    /**
    * \brief short circuited test to check if variadic amount of elements are within given range
    * @param {PartiallyOrdered,    in}  min value
    * @param {PartiallyOrdered,    in}  max value
    * @param {PartiallyOrdered..., in}  values
    * @param {PartiallyOrdered,    out} true if all values are within [min, max] region, false otherwise
    **/
    template<typename T, typename...Ts>
        requires(Concepts::PartiallyOrdered<T, T> && (std::is_same_v<T, Ts> && ...))
    constexpr bool within(const T min, const T max, const Ts ...ts) noexcept {
        assert(min < max);
        return (((ts >= min) && (ts <= max)) && ...);
    }
    template<auto min, auto max, typename...Ts>
        requires(Concepts::PartiallyOrdered<decltype(min), decltype(min)> &&
                 std::is_same_v<decltype(min), decltype(max)> && (std::is_same_v<decltype(min), Ts> && ...) && (min < max))
    constexpr bool within(const Ts ...ts) noexcept {
        [[assume(min < max)]];
        return (((ts >= min) && (ts <= max)) && ...);
    }
}
