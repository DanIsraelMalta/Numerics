//-------------------------------------------------------------------------------
//
// Copyright (c) 2016, Dan Israel Malta <malta.dan@gmail.com>
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
#include <type_traits>
#include <concepts>
#include <iterator>
#include <ranges>

/**
* usefull concepts and traits
**/
namespace Concepts {
    // numerical concepts
    template<class T> concept FloatingPoint = std::is_floating_point_v<T>;
    template<class T> concept SignedIntegral = std::is_signed_v<T>;
    template<class T> concept UnsignedIntegral = std::is_unsigned_v<T>;
    template<class T> concept Integral = std::is_unsigned_v<T> || std::is_signed_v<T>;
    template<class T> concept Arithmetic = std::is_arithmetic_v<T>;
    template<class T> concept Real = FloatingPoint<T> || Integral<T>;

    // concept of two elements which can be ordered in descending manner
    template<class T, class U>
    concept OrderedDescending = requires(const std::remove_reference_t<T>&t,
        const std::remove_reference_t<U>&u) {
            { t < u } -> std::convertible_to<bool>;
            { u < t } -> std::convertible_to<bool>;
    };

    // concept of two elements which can be ordered in ascending manner
    template<class T, class U>
    concept OrderedAscending = requires(const std::remove_reference_t<T>&t,
        const std::remove_reference_t<U>&u) {
            { t > u } -> std::convertible_to<bool>;
            { u > t } -> std::convertible_to<bool>;
    };

    // concept of two elements which can be ordered
    template<class T, class U>
    concept PartiallyOrdered = requires(const std::remove_reference_t<T>&t,
        const std::remove_reference_t<U>&u) {
        // lhs - rhs
            { t < u } -> std::convertible_to<bool>;
            { t <= u } -> std::convertible_to<bool>;
            { t > u } -> std::convertible_to<bool>;
            { t >= u } -> std::convertible_to<bool>;

            // rhs - lhs
            { u < t } -> std::convertible_to<bool>;
            { u <= t } -> std::convertible_to<bool>;
            { u > t } -> std::convertible_to<bool>;
            { u >= t } -> std::convertible_to<bool>;
    };

    // concept of a comparable and equatable objects
    template<class T, class U>
    concept Equateable = requires(const std::remove_reference_t<T>&t,
        const std::remove_reference_t<U>&u) {
        // lhs - rhs
            { t == u } -> std::convertible_to<bool>;
            { t != u } -> std::convertible_to<bool>;

            // rhs - lhs
            { u == t } -> std::convertible_to<bool>;
            { u != t } -> std::convertible_to<bool>;
    };

    // concept of two elements which are ordered
    template<class T, class U>
    concept TotallyOrdered = PartiallyOrdered<T, U>&& Equateable<T, U>;

    // concept of an incrementable object
    template<class I> concept Incrementable = requires(I i) {
        { ++i } -> std::same_as<I>;
        { i++ } -> std::same_as<I>;
    };

    // concept of a decrementable object
    template<class I> concept Decrementable = requires(I i) {
        { --i } -> std::same_as<I>;
        { i-- } -> std::same_as<I>;
    };

    // concept of a bidirectional object (both incrementable and decrementable)
    template<class I> concept Bidirectional = Incrementable<I> && Decrementable<I>;

    // concept of a collection iterable via 'begin' and 'end' iterators
    template<class T> concept Iterable = requires(T collection) {
        { collection.begin() } -> std::forward_iterator;
        { collection.end() } -> std::forward_iterator;
    };

    // randomly accessible buffer which knows its size (requires ranges.h)
    // its underlying object can be attained by: using T = typename std::ranges::range_value_t<BUFFER>;
    // its size type can be attained by: using I = typename std::ranges::range_size_t<BUFFER>;
    template<class BUFFER> concept RandomAccessSized = std::ranges::random_access_range<BUFFER> && std::ranges::sized_range<BUFFER>;
}
