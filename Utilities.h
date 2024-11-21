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
#include <memory> // std::assume_aligned
#include <new> // std::hardware_constructive_interference_size

// portable way to access the L1 data cache line size
#ifdef __cpp_lib_hardware_interference_size
using std::hardware_constructive_interference_size;
using std::hardware_destructive_interference_size;
#else
    // 64 bytes on x86-64 │ L1_CACHE_BYTES │ L1_CACHE_SHIFT │ __cacheline_aligned │ ...
[[maybe_unused]] constexpr std::size_t hardware_constructive_interference_size = 64;
[[maybe_unused]] constexpr std::size_t hardware_destructive_interference_size = 64;
#endif

// compiler friendly std::move replacement
#ifndef MOV
#define MOV(...) static_cast<std::remove_reference_t<decltype(__VA_ARGS__)>&&>(__VA_ARGS__)
#else
#undef MOV
#define MOV(...) static_cast<std::remove_reference_t<decltype(__VA_ARGS__)>&&>(__VA_ARGS__)
#endif

// compiler friendly std::forward replacement
#ifndef FWD
#define FWD(...) static_cast<decltype(__VA_ARGS__)&&>(__VA_ARGS__)
#else
#undef FWD
#define FWD(...) static_cast<decltype(__VA_ARGS__)&&>(__VA_ARGS__)
#endif

// define 'min' macro (to avoid include <algorithm> library)
#ifndef __NUMERICS_INTERNAL_MIN
#define __NUMERICS_INTERNAL_MIN(a,b) (a) < (b) ? (a) : (b)
#else
#undef __NUMERICS_INTERNAL_MIN
#define __NUMERICS_INTERNAL_MIN(a,b) (a) < (b) ? (a) : (b)
#endif

/**
* \brief align a given storage according to a given type
* @param {T, in} underlying element for structure to be aligned accordingly
*/
#ifndef AlignedStorage
#define AlignedStorage(T) alignas(__NUMERICS_INTERNAL_MIN(sizeof(T), hardware_constructive_interference_size))
#else
#undef AlignedStorage
#define AlignedStorage(T) alignas(__NUMERICS_INTERNAL_MIN(sizeof(T), hardware_constructive_interference_size))
#endif

/**
* \brief assume given points is aligned to given type size
* @param {T,   in} type assumed to be aligned to
* @param {PTR, in} pointer assumed to be aligned
**/
#ifndef AssumeAligned
#define AssumeAligned(T, PTR) [[assume(std::assume_aligned<__NUMERICS_INTERNAL_MIN(sizeof(T), hardware_constructive_interference_size)>(PTR))]]
#else
#undef AssumeAligned
#define AssumeAligned(T, PTR) [[assume(std::assume_aligned<__NUMERICS_INTERNAL_MIN(sizeof(T), hardware_constructive_interference_size)>(PTR))]]
#endif

/**
* \brief force inline
**/
#if defined(_MSC_VER)
#define ALWAYS_INLINE __forceinline
#else
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

/**
* generic utilities and local STL replacements
**/
namespace Utilities {
    /**
    * \brief compile time for loop (unrolls loop)
    * \usage example - iterate over numbers 0, 2, 4:
    *    static_for<0, 2, 6>([&](std::size_t i) {
    *        m_data[i] = static_cast<T>(MOV(v[i]));
    *    });
    **/
    template<std::size_t Start, std::size_t Inc, std::size_t End, class F>
        requires(std::is_invocable_v<F, decltype(Start)> || std::is_invocable_v<F>)
    constexpr void static_for(F&& f) noexcept {
        if constexpr (Start < End) {
            if constexpr (std::is_invocable_v<F, decltype(Start)>) {
                f(std::integral_constant<decltype(Start), Start>());
            }
            else {
                f();
            }
            static_for<Start + Inc, Inc, End>(FWD(f));
        }
    }

    /**
    * \brief swaps two moveable objects. local implementation of std::swap.
    **/
    template<typename T>
    constexpr void swap(T& t1, T& t2) noexcept {
        if constexpr (std::is_nothrow_move_assignable_v<T>) {
            T temp = MOV(t1);
            t1 = MOV(t2);
            t2 = MOV(temp);
        }
        else {
            T temp = FWD(t1);
            t1 = FWD(t2);
            t2 = FWD(temp);
        }
    }

    /**
    * \brief replaces the value of obj with new_value and returns the old value of obj.
    *        local implementation of std::exchange
    **/
    template<class T, class U = T>
    constexpr T exchange(T& obj, U&& new_value) noexcept(std::is_nothrow_assignable_v<T&, U>) {
        const T old_value{ MOV(obj) };
        obj = FWD(new_value);
        return old_value;
    }

    /**
    * \brief Obtains the actual address of the object or function.
    *        local implementation of std::addressof
    **/
    template<class T>
    T* addressof(T& arg) noexcept {
        if constexpr (std::is_object_v<T>) {
            return reinterpret_cast<T*>(&const_cast<char&>(reinterpret_cast<const volatile char&>(arg)));
        }
        else {
            return &arg;
        }
    }

    /**
    * \brief check if two iterators to same collection represent a continuous memory
    * @param {forward_iterator, in}  iterator to first element
    * @param {forward_iterator, in}  iterator to last element
    * @param {bool,             out} true if collection between 'first' and 'last' is continuous, false otherwise
    **/
    template<class It>
        requires(std::forward_iterator<It>)
    constexpr bool is_buffer_continuous(const It first, const It last) {
        return (&*last - &*first) == (last - first);
    }
};
