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
    * \brief local implementation of std::find_if
    **/
    template<class It, class UnaryPredicate, class T = typename std::decay_t<decltype(*std::declval<It>())>>
        requires((std::forward_iterator<It> || std::bidirectional_iterator<It>) && std::is_invocable_v<UnaryPredicate, T>)
    constexpr It find_if(It first, It last, UnaryPredicate&& p) {
        for (; first != last; ++first) {
            if (p(*first)) {
                return first;
            }
        }

        return last;
    }

    /**
    * \brief local implementation of std::count_if
    **/
    template<class It, class UnaryPredicate, class T = typename std::decay_t<decltype(*std::declval<It>())>>
        requires((std::forward_iterator<It> || std::bidirectional_iterator<It>) && std::is_invocable_v<UnaryPredicate, T>)
    constexpr std::size_t count_if(It first, It last, UnaryPredicate&& p) {
        typename std::size_t ret{};
        for (; first != last; ++first) {
            if (p(*first)) {
                ++ret;
            }
        }
        return ret;
    }

    /**
    * \brief local implementation of std::remove_if
    **/
    template<class It, class UnaryPredicate, class T = typename std::decay_t<decltype(*std::declval<It>())>>
        requires((std::forward_iterator<It> || std::bidirectional_iterator<It>) && std::is_invocable_v<UnaryPredicate, T>)
    constexpr It remove_if(It first, It last, UnaryPredicate&& p) {
        first = std::find_if(first, last, p);
        if (first != last) {
            for (It i{ first }; i != last; ++i) {
                if (!p(*i)) {
                    *first++ = std::move(*i);
                }
            }
        }

        return first;
    }

    /**
    * \brief local implementation of std::is_sorted
    **/
    template<class It, class Compare, class T = typename std::decay_t<decltype(*std::declval<It>())>>
        requires(std::forward_iterator<It> && std::is_invocable_v<Compare, T, T> &&
                 std::is_same_v<bool, typename std::invoke_result_t<Compare, T, T>>)
    constexpr bool is_sorted(const It first, const It last, Compare&& comp) {
        if (first == last) {
            return true;
        }

        for (It next{ first }; next != last; ++next) {
            if (comp(*next, *first)) {
                return false;
            }
        }

        return true;
    }

    /**
    * \brief local implementation of std::iota
    **/
    template<class It, class T = typename std::decay_t<decltype(*std::declval<It>())>>
        requires(std::forward_iterator<It>)
    constexpr void iota(It first, It last, T value) noexcept {
        for (; first != last; ++first, ++value) {
            *first = value;
        }
    }

    /**
    * \brief local implementation of std::reverse
    **/
    template<class It>
        requires(std::bidirectional_iterator<It>)
    constexpr void reverse(It first, It last) {
        using iter_cat = typename std::iterator_traits<It>::iterator_category;

        if constexpr (std::is_base_of_v<std::random_access_iterator_tag, iter_cat>) {
            if (first == last) {
                return;
            }

            for (--last; first < last; (void)++first, --last) {
                Utilities::swap(*first, *last);
            }
        }
        else {
            while (first != last && first != --last) {
                Utilities::swap(first++, last);
            }
        }
    }

    /**
    * \brief local implementation of std::rotate (without returned value)
    **/
    template<class It>
        requires(std::forward_iterator<It>)
    constexpr void rotate(It first, It middle, It last) {
        if constexpr (std::bidirectional_iterator<It>) {
            Algoithms::reverse(first, middle);
            Algoithms::reverse(middle, last);
            Algoithms::reverse(first, last);
        }
        else {
            if (first == middle) {
                return last;
            }

            if (middle == last) {
                return first;
            }

            It write{ first };
            It next_read{ first };
            for (It read{ middle }; read != last; ++write, ++read) {
                if (write == next_read) {
                    next_read = read;
                }
                Utilities::swap(*write, *read);
            }

            // rotate the remaining sequence into place
            Algoithms::rotate(write, next_read, last);
        }
    }

    /**
    * \brief local implementation of std::fill
    **/
    template<class It, class T = typename std::decay_t<decltype(*std::declval<It>())>>
        requires(std::forward_iterator<It>)
    constexpr void fill(It first, It last, const T& value) noexcept {
        for (; first != last; ++first) {
            *first = value;
        }
    }

    /**
    * \brief local implementation of std::lower_bound
    **/
    template<class It, class Compare, class T = typename std::decay_t<decltype(*std::declval<It>())>>
        requires((std::forward_iterator<It> || std::bidirectional_iterator<It>) && std::is_invocable_v<Compare, T, T>)
    constexpr It lower_bound(It first, It last, const T& value, Compare&& comp) {
        It it;
        std::size_t step{};
        std::size_t count{ static_cast<std::size_t>(std::distance(first, last)) };

        while (count > 0) {
            it = first;
            step = count / 2;
            it += step;

            if (comp(*it, value)) {
                first = ++it;
                count -= step + 1;
            }
            else {
                count = step;
            }
        }

        return first;
    }

    /**
    * \brief local implementation of std::unique
    **/
    template<class It, class BinaryPredicate, class T = typename std::decay_t<decltype(*std::declval<It>())>>
        requires((std::forward_iterator<It> || std::bidirectional_iterator<It>) && std::is_invocable_v<BinaryPredicate, T, T>)
    constexpr It unique(It first, It last, BinaryPredicate p) {
        if (first == last) {
            return last;
        }

        It result{ first };
        while (++first != last) {
            if (!p(*result, *first) && (++result != first)) {
                *result = MOV(*first);
            }
        }

        return ++result;
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
    * \brief local implementation of std::partial_sum
    **/
    template<class InputIt, class OutputIt, class BinaryOp, class T = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(std::forward_iterator<InputIt> && std::forward_iterator<OutputIt> && std::is_invocable_v<BinaryOp, T, T> &&
                 std::is_same_v<T, typename std::decay_t<decltype(*std::declval<OutputIt>())>>)
    constexpr OutputIt partial_sum(InputIt first, InputIt last, OutputIt d_first, BinaryOp&& op) {
        using input_t = typename std::iterator_traits<InputIt>::value_type;
        if (first == last) {
            return d_first;
        }

        input_t acc{ *first };
        *d_first = acc;

        while (++first != last) {
            acc = op(MOV(acc), *first);
            *++d_first = acc;
        }

        return ++d_first;
    }

    /**
    * \brief local specialized implementation of std::ranges::nth_element for std::vector.
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
        const auto quickselect = [&partition_with_pivot, len](const std::size_t k, const std::size_t left, const std::size_t right, auto&& recursive_driver) {
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
    * \brief local implementation of std::sort
    **/
    template<class It, class Compare, class T = typename std::decay_t<decltype(*std::declval<It>())>>
        requires(std::forward_iterator<It>&& std::is_invocable_v<Compare, T, T>)
    constexpr void sort(It first, It last, Compare&& comp) noexcept {
        // lambda to partition a collection using a pivot, i.e. - split collection into x<pivot and x>=pivot
        const auto partition_with_pivot = [C = FWD(comp)](T pivot, It f, It l) -> It {
            It res{ f };
            for (It it{ f }; it != l; ++it) {
                const bool r{ C(*it, pivot) };
                Utilities::swap(*res, *it);
                res += static_cast<std::size_t>(r);
            }
            return res;
        };

        // lambda to partition a collection using a pivot in reverse, i.e. - split collection into x<=pivot and x>pivot
        const auto partition_with_pivot_reverse = [C = FWD(comp)](T pivot, It f, It l) -> It {
            It res{ f };
            for (It it{ f }; it != l; ++it) {
                const bool r{ C(pivot, *it) };
                Utilities::swap(*res, *it);
                res += static_cast<std::size_t>(r);
            }
            return res;
        };

        // lambda to calculate the median of 5 elements in the array
        const auto median5 = [C = FWD(comp)](It f, It l) -> T {
            const std::size_t n{ static_cast<std::size_t>(l - f) };
            const std::size_t n4{ n / 4 };

            assert(n >= 5);

            T e0{ f[0] };
            T e1{ f[n4] };
            T e2{ f[n4 * 2] };
            T e3{ f[n4 * 3] };
            T e4{ f[n - 1] };

            if (C(e1, e0)) {
                Utilities::swap(e1, e0);
            }
            if (C(e4, e3)) {
                Utilities::swap(e4, e3);
            }
            if (C(e3, e0)) {
                Utilities::swap(e3, e0);
            }

            if (C(e1, e4)) {
                Utilities::swap(e1, e4);
            }
            if (C(e2, e1)) {
                Utilities::swap(e2, e1);
            }
            if (C(e3, e2)) {
                Utilities::swap(e2, e3);
            }

            if (C(e2, e1)) {
                Utilities::swap(e2, e1);
            }

            return e2;
        };

        // lambda to push root down through a heap
        const auto heap_sift = [C = FWD(comp)](It heap, std::size_t count, std::size_t root) {
            assert(count > 0);
            const std::size_t newLast{ (count - 1) / 2 };

            while (root < newLast) {
                assert(root * 2 + 2 < count);

                std::size_t next{ root };
                next = C(heap[next], heap[root * 2 + 1]) ? root * 2 + 1 : next;
                next = C(heap[next], heap[root * 2 + 2]) ? root * 2 + 2 : next;

                if (next == root) {
                    break;
                }
                Utilities::swap(heap[root], heap[next]);
                root = next;
            }

            if ((root == newLast) &&
                (root * 2 + 1 < count) &&
                C(heap[root], heap[root * 2 + 1])) {
                Utilities::swap(heap[root], heap[root * 2 + 1]);
            }
        };

        // lambda to sort a collection using heap sort
        const auto heap_sort = [&heap_sift](It f, It l) {
            if (f == l) {
                return;
            }

            It heap{ f };
            const std::size_t count{ static_cast<std::size_t>(l - f) };
            for (std::size_t i{ count / 2 }; i > 0; --i) {
                heap_sift(heap, count, i - 1);
            }

            for (std::size_t i{ count - 1 }; i > 0; --i) {
                Utilities::swap(heap[0], heap[i]);
                heap_sift(heap, i, 0);
            }
        };

        // lambda to perform fast sorting of small collections (smaller than 20)
        const auto small_sort = [C = FWD(comp)](It f, It l) {
            for (std::size_t i{ static_cast<std::size_t>(l - f) }; i > 1; i -= 2) {
                T x{ MOV(f[0]) };
                T y{ MOV(f[1]) };
                if (C(y, x)) {
                    Utilities::swap(y, x);
                }

                for (std::size_t j{ 2 }; j < i; j++) {
                    T z{ MOV(f[j]) };

                    if (C(x, z)) {
                        Utilities::swap(x, z);
                    }
                    if (C(y, z)) {
                        Utilities::swap(y, z);
                    }
                    if (C(y, x)) {
                        Utilities::swap(y, x);
                    }

                    f[j - 2] = MOV(z);
                }

                f[i - 2] = MOV(x);
                f[i - 1] = MOV(y);
            }
            };

        // lambda to sort a collection in recursive manner
        const auto sort = [&partition_with_pivot, &partition_with_pivot_reverse, &small_sort, &heap_sort, &median5]
                          (It f, It l, std::size_t limit, auto&& recursive_driver) {
            for (;;) {
                const std::size_t len{ static_cast<std::size_t>(l - f) };
                if (len < 16) {
                    small_sort(f, l);
                    return;
                }

                if (limit == 0) [[unlikely]] {
                    heap_sort(f, l);
                    return;
                }

                const T pivot{ median5(f, l) };
                const It mid{ partition_with_pivot(pivot, f, l) };

                // skewed partitions? calculate new midpoint by separating equal elements
                It midr{ mid };
                if (static_cast<std::size_t>(mid - f) <= len / 8) [[unlikely]] {
                    midr = partition_with_pivot_reverse(pivot, mid, l);
                }

                if (const std::size_t newLimit{ limit / 2 + limit / 4 }; mid - f <= l - midr) {
                    recursive_driver(f, mid, newLimit, recursive_driver);
                    f = midr;
                }
                else {
                    recursive_driver(midr, l, newLimit, recursive_driver);
                    l = mid;
                }
            }
            };

        // sort collections
        sort(first, last, static_cast<std::size_t>(last - first), sort);
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
