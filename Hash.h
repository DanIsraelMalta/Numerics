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
#include <type_traits>
#include <limits> // std::numeric_limits
#include <cmath> // std::pow

/**
* Hashing and pairing functions
**/
namespace Hash {

    /**
    * \brief hash unsigned 2D/3D coordinate into 1D normalized floating point value.
    *        based upon:
    *        Mark Jarzynski and Marc Olano, Hash Functions for GPU Rendering,
    *        Journal of Computer Graphics Techniques (JCGT), vol. 9, no. 3, 21-38, 2020
    *        Available online http://jcgt.org/published/0009.
    * @param {unsigned,   in}  x coordinate
    * @param {unsigned,   in}  y coordinate
    * @param {unsigned,   in}  z coordinate (optional)
    * @param {floaintg,   out} hash value in region [0, 1]
    **/
    template<typename T>
        requires(std::is_unsigned_v<T>)
    constexpr auto hash_unsigned_coordinate_to_normalized_value(T x, T y, T z) {
        using out_t = std::conditional_t<sizeof(T) <= sizeof(float), float, double>;
        constexpr T bits{ std::numeric_limits<T>::digits + 1 };
        constexpr T shift{ bits / 2 };
        constexpr out_t divider{ std::pow(2, static_cast<out_t>(bits)) };

        // heptaplex collapse noise
#define HEPTAPLEX(X, Y, Z) ~(~(X) - (Y) - (Z)) * ~((X) - ~(Y) - (Z)) * ~((X) - (Y) - ~(Z))
        x = HEPTAPLEX(x, y, z);
        y = HEPTAPLEX(x, y, z);
        z = x ^ y ^ HEPTAPLEX(x, y, z);
#undef HEPTAPLEX

        // output
        return static_cast<out_t>(z ^ ~(~z >> shift)) / divider;
    }
    template<typename T>
        requires(std::is_unsigned_v<T>)
    constexpr auto hash_unsigned_coordinate_to_normalized_value(T x, T y) {
        return hash_unsigned_coordinate_to_normalized_value(x, y, x ^ y);
    }

    /**
    * \brief hash integral 2D/3D coordinate into 1D integral value, where grid size is determined at compile time.
    *        based upon: “VDB: High-Resolution Sparse Volumes with Dynamic Topology”, p. 27:9)
    *        notice that: X % Y == X & (1 << (log2(pow(2, Y))) - 1) == X & (1 << Y - 1)
    * @param {integral}      hash table size. 20 by deault.
    *                        N = 20 -> 2^20 = 1,048,576 ~= 100x100x100 grid size
    *                        N = 24 -> 2^24 = 16,777,216 = 256*256*256 grid size
    * @param {integral, in}  x coordinate
    * @param {integral, in}  y coordinate
    * @param {integral, in}  z coordinate (optional)
    * @param {integral, out} hash value
    **/
    template<std::size_t N = 20, typename T>
        requires(std::is_integral_v<T> &&
                 (N > 0 && N < static_cast<std::size_t>(std::numeric_limits<T>::digits) - 1))
    constexpr T hash_coordinate_to_integral(T x, T y, T z) {
        constexpr T A{ 73856093 };
        constexpr T B{ 19349663 };
        constexpr T C{ 83492791 };
        return ((1 << N) - 1) & (x * A ^ y * B ^ z * C);
    }
    template<std::size_t N = 20, typename T>
        requires(std::is_integral_v<T> &&
                 (N > 0 && N < static_cast<std::size_t>(std::numeric_limits<T>::digits) - 1))
    constexpr auto hash_coordinate_to_integral(T x, T y) {
        return hash_coordinate_to_integral<N>(x, y, x ^ y);
    }

    /**
    * \brief hash 2D/3D coordinate into 1D integral value, where grid size is determined at run time.
    *        based upon: “VDB: High-Resolution Sparse Volumes with Dynamic Topology”, p. 27:9)
    *        notice that: X % Y == X & (1 << (log2(pow(2, Y))) - 1) == X & (1 << Y - 1)
    * @param {integral, in}  hash table size. 20 by deault.
    *                        N = 20 -> 2^20 = 1,048,576 ~= 100x100x100 grid size
    *                        N = 24 -> 2^24 = 16,777,216 = 256*256*256 grid size
    * @param {integral, in}  x coordinate
    * @param {integral, in}  y coordinate
    * @param {integral, in}  z coordinate (optional)
    * @param {integral, out} hash value
    **/
    template<typename T>
        requires(std::is_integral_v<T>)
    constexpr T hash_coordinate_to_integral(std::size_t N, T x, T y, T z) {
        constexpr T A{ 73856093 };
        constexpr T B{ 19349663 };
        constexpr T C{ 83492791 };
        assert(N > 0 && N < static_cast<std::size_t>(std::numeric_limits<T>::digits) - 1);
        return ((1 << N) - 1) & (x * A ^ y * B ^ z * C);
    }
    template<typename T>
        requires(std::is_integral_v<T>)
    constexpr T hash_coordinate_to_integral(std::size_t N, T x, T y) {
        assert(N > 0 && N < static_cast<std::size_t>(std::numeric_limits<T>::digits) - 1);
        return hash_coordinate_to_integral(N, x, y, x ^ y);
    }

    /**
    * \brief hash 2D coordinate into 1D integral value, well - better terminology will
    *        be "sample white noise over 2D domain, i.e. - integer lattice".
    *        this "hash" uses concepts from "Equidistributed sequence", see:
             https://en.wikipedia.org/wiki/Equidistributed_sequence
    * @param {integral, in}  x coordinate (positive, at least 32bit)
    * @param {integral, in}  y coordinate (positive, at least 32bit)
    * @param {integral, out} hash value ("sampled white noise")
    **/
    template<typename T>
        requires(std::is_integral_v<T> && sizeof(T) >= sizeof(std::int32_t))
    constexpr T sample_white_noise_over_2D_domain(T x, T y) {
        constexpr T W0{ 0x3504f333 };   // = 3*2309*128413 
        constexpr T W1{ 0xf1bbcdcb };   // = 7*349*1660097 
        constexpr T M{ 741103597 };     // = 13*83*686843
        assert(x > 0);
        assert(y > 0);

        return M * (W0 * x) ^ (W1 * y);
    }

    /**
    * \brief 1D to 1D pseudo random number generator over unsigned integral numbers from PCG family
    *        see: https://www.pcg-random.org/
    * @param {uint32, in}  x coordinate
    * @param {uint32, out} pseudo random number generated from input values
    **/
    constexpr std::uint32_t pcg(std::uint32_t x) {
        const std::uint32_t state{ x * 747796405u + 2891336453u };
        const std::uint32_t word{ ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u };
        return (word >> 22u) ^ word;
    }

    /**
    * \brief generate a gaussian distributed random numbers with mean of 0 and variance of 1.
    *        see: "Natur als fraktale Grafik" by Reinhard Scholl.
    * @param {size_t} a variable that defines the precision of the distribution.
    *                 default is 15 which gives the smallest distance between two numbers (C3= 1 / (2^15 / 3) = 1/10922 = 0.000091)
    **/
    template<std::size_t Q = 15, typename T = float>
        requires(std::is_floating_point_v<T>)
    constexpr T normal_distribution() noexcept {
        constexpr T C1{ static_cast<T>((1 << Q) - 1) };
        constexpr T C2{ static_cast<T>((C1 / 3) + 1) };
        constexpr T C3{ static_cast<T>(1.0) / static_cast<T>(C1) };

#define RAND01(x) (static_cast<T>(x)) * (static_cast<T>(rand())) /  (static_cast<T>(RAND_MAX))
        return (static_cast<T>(2.0) * (RAND01(C2) + RAND01(C2) + RAND01(C2)) - static_cast<T>(3.0) * (C2 - static_cast<T>(1.0))) * C3;
#undef RAND01
    }

    /**
    * \brief using Szudzik pairing function, transform two unsigned values into one
    *        note: max input pair of Szudzik is the square root of the maximum integer value,
    *              so for 32bit unsigned value maximum input value without an overflow being 65,535.
    * @param {unsigned, in}  x coordinate
    * @param {unsigned, in}  y coordinate
    * @param {unsigned, out} pairing value
    **/
    template<typename T>
        requires(std::is_unsigned_v<T>)
    constexpr T SzudzikValueFromPair(const T x, const T y) {
        constexpr T bits{ (std::numeric_limits<T>::digits + 1) / 2 };
        constexpr T largest{ static_cast<T>(std::pow(2, bits)) };
        assert(x < largest);
        assert(y < largest);
        return (x < y) ? (x + y * y) : (x * x + x + y);
    }
    template<auto x, auto y, class T = decltype(x)>
        requires(std::is_unsigned_v<T> && std::is_same_v<T, decltype(y)>)
    constexpr auto SzudzikValueFromPair() {
        constexpr T bits{ (std::numeric_limits<T>::digits + 1) / 2 };
        constexpr T largest{ static_cast<T>(std::pow(2, bits)) };
        static_assert(x < largest);
        static_assert(y < largest);
        return (x < y) ? (x + y * y) : (x * x + x + y);
    }

    /**
    * \brief given Szudzik pairing value, return its two constructing coordinates
    * @param {unsigned,             in}  pairing value
    * @param {{unsigned, unsigned}, out} {.x = x coordinate, .y = y coordinate}
    **/
    template<typename T>
        requires(std::is_unsigned_v<T>)
    constexpr auto SzudzikPairFromValue(const T z) {
        constexpr T bits{ (std::numeric_limits<T>::digits + 1) / 2 };
        constexpr T largest{ static_cast<T>(std::pow(2, bits)) };
        assert(z < largest);
        using out_t = struct { T x; T y; };

        const T zsqrt{ static_cast<T>(std::floor(std::sqrt(z))) };
        const T z2{ zsqrt * zsqrt };
        const T temp{ z - z2 };

        return (temp < zsqrt) ? out_t{ temp, zsqrt } : out_t{ zsqrt, temp - zsqrt };
    }
    template<auto z, class T = decltype(z)>
        requires(std::is_unsigned_v<T>)
    constexpr auto SzudzikPairFromValue() {
        constexpr T bits{ (std::numeric_limits<T>::digits + 1) / 2 };
        constexpr T largest{ static_cast<T>(std::pow(2, bits)) };
        static_assert(z < largest);
        using out_t = struct { T x; T y; };

        const T zsqrt{ static_cast<T>(std::floor(std::sqrt(z))) };
        const T z2{ zsqrt * zsqrt };
        const T temp{ z - z2 };

        return (temp < zsqrt) ? out_t{ temp, zsqrt } : out_t{ zsqrt, temp - zsqrt };
    }
};
