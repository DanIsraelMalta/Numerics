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
#include <vector>

/**
* local STL replacements for <algorithm> library
**/
namespace Algoithms {

    /**
    * \brief local implementation of std::find_if_not
    **/
    template<class It, class UnaryPred, class T = typename std::decay_t<decltype(*std::declval<It>())>>
        requires((std::forward_iterator<It> || std::bidirectional_iterator<It>) && std::is_invocable_v<UnaryPred, T>)
    constexpr It find_if_not(It first, It last, UnaryPred&& q) noexcept {
        for (; first != last; ++first) {
            if (!q(*first)) {
                return first;
            }
        }

        return last;
    }

    /**
    * \brief local implementation of std::min_element
    **/
    template<class It, class Compare, class T = typename std::decay_t<decltype(*std::declval<It>())>>
        requires((std::forward_iterator<It> || std::bidirectional_iterator<It>) && std::is_invocable_v<Compare, T, T>)
    constexpr It min_element(It first, It last, Compare&& comp) noexcept {
        if (first == last) {
            return last;
        }

        It smallest{ first };

        while (++first != last) {
            if (comp(*first, *smallest)) {
                smallest = first;
            }
        }

        return smallest;
    }

    /**
    * \brief local implementation of std::ranges::nth_element for std::vector.
    **/
    template<class T, class Compare>
        requires(std::is_invocable_v<Compare, T, T>)
    constexpr void nth_element(std::vector<T>& vec, const std::size_t nth, Compare&& comp) {
        const std::size_t len{ vec.size() - 1 };
        if (vec.empty() || nth == 0 || nth > len) {
            return;
        }

        // lambda which implements partitioning with a pivot
        const auto partition_with_pivot = [&vec, COMP = FWD(comp)](const std::size_t left, const std::size_t right, std::size_t pivotIndex) -> std::size_t {
            const T pivot{ vec[pivotIndex] };
            Utilities::swap(vec[right], vec[pivotIndex]);
            pivotIndex = left;

            for (std::size_t i{ left }; i < right; ++i) {
                if (COMP(vec[i], pivot)) {
                    Utilities::swap(vec[i], vec[pivotIndex]);
                    ++pivotIndex;
                }
            }

            Utilities::swap(vec[pivotIndex], vec[right]);
            return pivotIndex;
        };

        // lambda to perform quick select algorithm in recursive manner
        const auto quickselect = [&partition_with_pivot, &vec, len](const std::size_t k, const std::size_t left, const std::size_t right, auto&& recursive_driver) {
            if (left == right) {
                return;
            }

            const std::size_t median{ left + (right - left) / 2 };
            const std::size_t medianIndex{ partition_with_pivot(left, right, median) };
            const std::size_t medianIndexNth{ medianIndex + k };

            if (medianIndexNth == len) {
                return;
            }
            else if (medianIndexNth < len) {
                recursive_driver(k, medianIndex + 1, right, recursive_driver);
            }
            else {
                recursive_driver(k, left, medianIndex - 1, recursive_driver);
            }
        };

        // perform nth_element using quick-select algorithm
        quickselect(nth, 0, len, quickselect);
    }

    /**
    * \brief local implementation of std::partition
    **/
    template<class It, class UnaryPred, class T = typename std::decay_t<decltype(*std::declval<It>())>>
        requires((std::forward_iterator<It> || std::bidirectional_iterator<It>) && std::is_invocable_v<UnaryPred, T>)
    It partition(It first, It last, UnaryPred&& p) noexcept {
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
        }
        else if (std::forward_iterator<It>) {
            first = Algoithms::find_if_not(first, last, FWD(p));
            if (first == last)
                return first;

            for (It i{ first + 1 }; i != last; ++i) {
                if (p(*i)) {
                    Utilities::swap(*i, *first);
                    ++first;
                }
            }

            return first;
        }
    }
}
