//-------------------------------------------------------------------------------
//
// Copyright (c) 2025, Dan Israel Malta <malta.dan@gmail.com>
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
#include <algorithm>
#include <complex>

/**
* numerical algorithms on collections
**/
namespace NumericalAlgorithms {

    /**
    * \brief exact floating point numerical accumulation of a given collection using Neumaier variant of Kahan and Babuska summation.
    *        Its like Kahan summation but also covers the case where next value to be added is larger than absolute value of running sum.
    *        notice that on large vectors (10 million elements) tests show that this accumulation is 25% slower than std::accumulate
    *        and 220% slower than std::reduce.
    * 
    * @param {forward_iterator, in}  iterator to first element to sum
    * @param {forward_iterator, in}  iterator to last element to sum
    * @param {floating point,   in}  initial value
    * @param {floating point,   out} sum of elements
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
    * \brief return the floating point precise circular mean of a collection of angles in the range [0, 2*PI].
    * 
    * @param {forward_iterator, in}  iterator to first angle
    * @param {forward_iterator, in}  iterator to last angle
    * @param {floating,         out} circular mean [rad]
    **/
    template<std::forward_iterator InputIt, class T = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(std::is_arithmetic_v<T>)
    constexpr T circular_mean(InputIt first, const InputIt last) {
        // circular sine and cosine sums
        T sin_sum{};
        T cos_sum{};
        std::size_t i{};
        for (; first != last; ++first) {
            sin_sum += std::sin(*first);
            cos_sum += std::cos(*first);
            ++i;
        }
        sin_sum /= static_cast<T>(i);
        cos_sum /= static_cast<T>(i);

        // output
        return std::atan2(sin_sum, cos_sum);
    }
    template<Concepts::Iterable COL>
    constexpr auto circular_mean(const COL& collection) {
        return circular_mean(collection.begin(), collection.end());
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
    constexpr void conv(const It1 u_first, const It1 u_last,
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
    * \brief calculate the cumulative sum of a given collection
    * @param {forward_iterator,  in}  iterator to first element in collection
    * @param {forward_iterator,  in}  iterator to last element in collection
    * @param {forward_iterator,  out} iterator to beginning of collection which will hold the cumulative sum
    **/
    template<std::forward_iterator It, std::forward_iterator Ot, class T = typename std::decay_t<decltype(*std::declval<It>())>>
        requires(std::is_arithmetic_v<T> && std::is_same_v<T, typename std::decay_t<decltype(*std::declval<Ot>())>>)
    constexpr void cumsum(It x_first, It x_last, Ot y_first) {
        T sum{};
        for (; x_first != x_last; ++x_first) {
            sum += *x_first;
            *y_first++ = sum;
        }
    }

    /**
    * \brief generate a random sample from an arbitrary discrete/finite probability distribution
    * 
    * \example:
    *        std::vector<float> p{{0.2, 0.4, 0.4}}; // probability distribution
    *        std::vector<float> d{{1.0f, 2.0f, 4.0f}}; // probability domain
    *        std::vector<float> O(10, 0.0f);
    *        // on average, O should contain two 1's, four 2's and four 3's.
    *        generate_from_discrete_distribution(p.begin(), p.end(), d.begin(), d.end(), O.begin(), O.end());
    *
    * @param {forward_iterator,  in} iterator to first element of positive numbers whose values form the probability distribution (collection sum should be zero and sorted in ascending order)
    * @param {forward_iterator,  in} iterator to last element of positive numbers whose values form the probability distribution (collection sum should be zero and sorted in ascending order)
    * @param {forward_iterator,  in} iterator to first element of values defining probability distribution domain 
    * @param {forward_iterator,  in} iterator to last element of values defining probability distribution domain
    * @param {forward_iterator, out} iterator to first element in collection which will hold the generated numbers
    * @param {forward_iterator, out} iterator to last element in collection which will hold the generated numbers
    **/
    template<std::forward_iterator It1, std::forward_iterator It2, std::forward_iterator Ot,
             class T = typename std::decay_t<decltype(*std::declval<It1>())>,
             class U = typename std::decay_t<decltype(*std::declval<Ot>())>>
        requires(std::is_arithmetic_v<T> && std::is_arithmetic_v<U>)
    constexpr void generate_from_discrete_distribution(const It1 p_first, const It1 p_last,
                                                       const It2 d_first, const It2 d_last,
                                                       Ot x_first, const Ot x_last) {
        // check that probability distribution is normalized and sorted
        assert(Numerics::areEquals(NumericalAlgorithms::accumulate(p_first, p_last), static_cast<T>(1.0)));
        assert(Algoithms::is_sorted(p_first, p_last, [](const T lhs, const T rhs) { return lhs < rhs; }));

        // transform probability distribution to cumulative distribution
        const std::size_t p_length{ static_cast<std::size_t>(std::distance(p_first, p_last)) };
#ifdef DEBUG
        const std::size_t d_length{ static_cast<std::size_t>(std::distance(d_first, d_last)) };
        assert(p_length == d_length);
#endif
        std::vector<T> csum(p_length, T{});
        NumericalAlgorithms::cumsum(p_first, p_last, csum.begin());

        // generate samples
        const U d_first_value{ static_cast<U>(*d_first) };
        const U d_last_value{ static_cast<U>(*(d_last - 1)) };
        const T csum_front{ csum.front() };
        const T csum_back{ csum.back() };
        for (; x_first != x_last; ++x_first) {
            const T u{ static_cast<T>(rand()) / static_cast<T>(RAND_MAX) };
            if (u <= csum_front) {
                *x_first = d_first_value;
            }
            else if (u >= csum_back) {
                *x_first = d_last_value;
            }
            else {
                for (std::size_t i{}; i < p_length; ++i) {
                    if (u <= csum[i]) {
                        *x_first = static_cast<U>(*(d_first + i));
                        break;
                    }
                }
            }
        }
    }

    /**
    * \brief given a collection of objects and a metric, merge/weld/combine all objects whose metric is below given threshold.
    *        complexity is O(n*long(n)).
    *
    * @param {forward_iterator, in}  iterator to first object in object collection
    * @param {forward_iterator, in}  iterator to last object in object collection
    * @param {callable,         in}  metric which evaluates two objects and return evaluation in arithmetic type
    * @param {value_type,       in}  threshold (arithmetic) for objects to be welded / combined / merged
    * @param {vector<>,         out} collection of merged objects (not ordered same as input objects)
    **/
    template<std::forward_iterator InputIt, class METRIC,
             class OBJ = typename std::decay_t<decltype(*std::declval<InputIt>())>,
             class T = typename std::invoke_result_t<METRIC, OBJ, OBJ>>
        requires(std::is_invocable_v<METRIC, OBJ, OBJ> && std::is_arithmetic_v<T>)
    constexpr std::vector<OBJ> merge_close_objects(const InputIt first, const InputIt last, METRIC&& metric, const T tol) {
        // housekeeping
        std::vector<OBJ> objects(first, last);
        const std::size_t len{ objects.size() };

        // sort objects according to given metric
        std::sort(objects.begin(), objects.end(),
                  [m = FWD(metric)](const OBJ& a, const OBJ& b) { return m(a, b) < T{}; });

        // iterate over sorted objects and move "should be merged objects" to the end
        std::size_t length{};
        objects[length++] = objects[0];
        for (std::size_t i{}; i < len; ++i) {
            const OBJ obj{ objects[i] };
            const OBJ prev{ objects[length - 1] };
            if (metric(obj, prev) > tol) {
                objects[length++] = obj;
            }
        }
        objects.resize(length);

        // output
        return objects;
    }

    /**
    * \brief given a collection of arithmetic values, rearrange it and return its median value.
    *        complexity is O(n).
    *
    * @param {forward_iterator, in}  iterator to first value
    * @param {forward_iterator, in}  iterator to last value
    * @param {callable,         in}  comparison function object
    * @param {value_type,       out} median value
    **/
    template<std::forward_iterator InputIt, class Compare,
             class T = typename std::decay_t<decltype(*std::declval<InputIt>())>>
        requires(std::is_invocable_v<Compare, T, T>)
    constexpr void median(InputIt first, InputIt last, Compare&& comp) {
        const std::size_t N{ static_cast<std::size_t>(std::distance(first, last)) };
        const std::size_t N2{ N / 2 };

        if (N % 2 == 0) {
            const std::size_t N1{ (N - 1) / 2 };
            Algoithms::nth_element(first, first + N2, last, FWD(comp));
            Algoithms::nth_element(first, first + N1, last, FWD(comp));
            return (*(first + N1) + *(first + N2)) / static_cast<T>(2.0);
        }
        else {
            Algoithms::nth_element(first, first + N2, last, FWD(comp));
            return *(first + N2);
        }
    }

    /**
    * \brief perform Fast Fourier Transform on a collection of complex numbers.
    *        this function implements Cooley–Tukey FFT algorithm.
    *
    * @param {RandomAccessSized, in}  buffer of complex values
    * @param {vector<complex>,   out} (padded) Fourier transform of input vector
    **/
    template<class BUFFER>
        requires(std::ranges::random_access_range<BUFFER> && std::ranges::sized_range<BUFFER>)
    constexpr auto fft(const BUFFER& x) {
        using buffer_underlying_T = typename std::ranges::range_value_t<BUFFER>;
        static_assert(std::is_arithmetic_v<buffer_underlying_T>                 ||
                      std::is_same_v<buffer_underlying_T, std::complex<float>>  ||
                      std::is_same_v<buffer_underlying_T, std::complex<double>> ||
                      std::is_same_v<buffer_underlying_T, std::complex<long double>>);
        using complex_t = typename std::conditional<std::is_arithmetic_v<buffer_underlying_T>,
                                                    std::complex<buffer_underlying_T>,
                                                    buffer_underlying_T>::type;
        using T = typename complex_t::value_type;
        using vec_t = std::vector<complex_t>;

        if (x.empty()) {
            return vec_t{};
        }
        assert(x.size() < std::numeric_limits<std::int32_t>::max());

        // housekeeping
        const std::int32_t bits{ [n = static_cast<std::int32_t>(x.size())]() {
            for (std::int32_t i{}; i < 30; ++i) {
                if (const std::int32_t temp{ 1 << i };
                    temp >= n) {
                    return i;
                }
            }
            return 30;
        }() };
        const std::int32_t len{ 1 << bits };
        assert(len >= x.size());
        const vec_t points{ [len]() {
            constexpr T two_pi{ static_cast<T>(6.283185307179586476925286766559) };
            vec_t polar;
            polar.reserve(len);

            for (std::int32_t i{}; i < len; ++i) {
                const T phase{ -two_pi * static_cast<T>(i) / static_cast<T>(len) };
                polar.emplace_back(std::polar(static_cast<T>(1.0), phase));
            }

            return polar;
        }() };
        vec_t prev(len);
        vec_t temp(len);
        for (std::size_t i{}; i < x.size(); ++i) {
            prev[i] = x[i];
        }

        // lambda which performs in place bit reversal permutation of given buffer
        // this is done in O(n), thanks to: https://www.researchgate.net/publication/227682626_A_new_superfast_bit_reversal_algorithm
        const auto in_place_bit_reversal_permutation = [&prev, bits]() {
            const std::int32_t len{ static_cast<std::int32_t>(prev.size()) };

            if (len <= 2) {
                return;
            }

            if (len == 4) {
                Utilities::swap(prev[1], prev[3]);
                return;
            }

            std::vector<std::int32_t> bit_rerversal(len);
            bit_rerversal[0] = 0;
            bit_rerversal[1] = 1 << (bits - 1);
            bit_rerversal[2] = 1 << (bits - 2);
            bit_rerversal[3] = bit_rerversal[1] + bit_rerversal[2];
            for (std::int32_t k{ 3 }; k <= bits; ++k) {
                const std::int32_t nk{ (1 << k) - 1 };
                const std::int32_t n_km1{ (1 << (k - 1)) - 1 };
                bit_rerversal[nk] = bit_rerversal[n_km1] + (1 << (bits - k));
                for (std::int32_t i{ 1 }; i <= n_km1; ++i) {
                    bit_rerversal[nk - i] = bit_rerversal[nk] - bit_rerversal[i];
                }
            }

            for (std::int32_t i{}; i < len; ++i) {
                if (bit_rerversal[i] > i) {
                    Utilities::swap(prev[i], prev[bit_rerversal[i]]);
                }
            }
        };

        // lambda which performs one iteration of "butterfly forwarding".
        // 'phases' is vector of points on the unit circle
        // 'turn' is iteration indicator
        // 'bit_count' total number of iterations
        const auto butterfly_forwarding = [&prev, &temp, &points, bits](const std::int32_t turn) {
            if (turn == bits) {
                return;
            }

            const std::int32_t group_size{ 1 << (turn + 1) };
            const std::int32_t num_groups{ static_cast<std::int32_t>(prev.size()) / group_size };
            for (std::int32_t group_index{}; group_index < num_groups; ++group_index) {
                const std::int32_t base_index{ group_index * group_size };
                const std::int32_t half_size{ group_size / 2 };

                for (std::int32_t j{}; j < half_size; ++j) {
                    const std::int32_t x0_index{ base_index + j };
                    const std::int32_t x1_index{ base_index + half_size + j };
                    prev[x1_index] *= points[j * num_groups];
                    temp[x0_index] = prev[x0_index] + prev[x1_index];
                    temp[x1_index] = prev[x0_index] - prev[x1_index];
                }
            }
        };
        
        // Cooley–Tukey FFT algorithm
        in_place_bit_reversal_permutation();
        for (std::int32_t turn{}; turn < bits; ++turn) {
            butterfly_forwarding(turn);
            Utilities::swap(prev, temp);
        }

        // output
        return prev;
    }

    /**
    * \brief calculate the Inverse Fast Fourier Transform on a collection of complex numbers.
    *
    * @param {RandomAccessSized, in}  buffer of complex values (frequency domain)
    * @param {vector<complex>,   out} Inverse Fourier transform of input vector
    **/
    template<class BUFFER>
        requires(std::ranges::random_access_range<BUFFER>&& std::ranges::sized_range<BUFFER>)
    constexpr auto ifft(const BUFFER& x) {
        using buffer_underlying_T = typename std::ranges::range_value_t<BUFFER>;
        static_assert(std::is_arithmetic_v<buffer_underlying_T>                 ||
                      std::is_same_v<buffer_underlying_T, std::complex<float>>  ||
                      std::is_same_v<buffer_underlying_T, std::complex<double>> ||
                      std::is_same_v<buffer_underlying_T, std::complex<long double>>);
        using complex_t = typename std::conditional<std::is_arithmetic_v<buffer_underlying_T>,
                                                    std::complex<buffer_underlying_T>,
                                                    buffer_underlying_T>::type;
        using T = typename complex_t::value_type;
        using vec_t = std::vector<complex_t>;
        using iter_t = typename vec_t::iterator;

        // flip and normalized frequency spectrum
        const T len{ static_cast<T>(x.size()) };
        vec_t reverse_freq_spectrum(x);
        Algoithms::reverse(reverse_freq_spectrum.begin() + 1, reverse_freq_spectrum.end());
        for (std::size_t i{}; i < len; ++i) {
            reverse_freq_spectrum[i] /= len;
        }

        // output
        return NumericalAlgorithms::fft(reverse_freq_spectrum);
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
