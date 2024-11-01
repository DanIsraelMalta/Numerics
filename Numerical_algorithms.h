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
#include "Algorithms.h"
#include "Numerics.h"

/**
* numerical algorithms on collections
**/
namespace NumericalAlgorithms {

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
    * @param {{arithmetic, arithmetic}, out} {mean, standard deviation}
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
    * @param {size_t,                                               in}  number of bins (a good initial choice can be ceil(sqrt(amount of values in input)) )
    * @param {{vector<size_t>, vector<arithmetic>, vector<size_t>}, out} {bin counts,
    *                                                                     bin edges - first element is the leading edge of the first bin. The last element is the trailing edge of the last bin,
    *                                                                     index array where 'i' cell holds the index of 'bin edges' in which *(first + i) value is categorized }
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
            const T floor_bin{ std::floor(*f / binWidth) };
            [[assume(floor_bin >= T{})]];
            const std::size_t bin{ static_cast<std::size_t>(floor_bin) };
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
    * @param {forward_iterator, out} iterator to beginning of collection which will hold the convolution between first and second collections
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
    * @param {forward_iterator,  out} iterator to beginning of collection which will hold the filtered output
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
        [[assume(a[0] != T{})]];

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
        Utilities::static_for<0, 1, NA>([&an, &a, a0](std::size_t i) {
            an[i] = a[i] / a0;
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
    * \brief circularly shift a collection
    * @param {forward_iterator,  in} iterator to first element of shifted collection
    * @param {forward_iterator,  in} iterator to last element of shifted collection
    * @param {integral,          in} Shift amount (positive value shifts to the right, negative values shift to the left
    **/
    template<class It, class T>
        requires(std::forward_iterator<It> && std::is_integral_v<T>)
    constexpr void circshift(It first, It last, const T shift) {
        const std::size_t dim{ static_cast<std::size_t>(std::distance(first, last)) };
        const std::size_t middle{ shift > 0 ? dim - shift : -shift };
        Algoithms::rotate(first, first + middle, last);
    }

    /**
    * \brief Apply a kernal, in ordered manner, on variadic amount of random access arithmetic ranges and return the result in a different range.
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
}
