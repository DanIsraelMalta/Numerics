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
#include <cmath>

/**
* \brief namespace encapsulates operations on L1 angles (aka - "diamond angles"), i.e. - angles between points connected by straight lines instead of circular (L2) arcs.
*        diamond angles have the following charateristics:
*        > monotonic increasing and sign preservation like regular angles.
*        > full cycle = two pi radians = 4 diamond angles.
*        > conversion from diamond angles to degrees/radians is not straight forward other than in the following angles:
*          (angles are measured counter clockwise from positive X axis)
*          degrees: 0, 45,   90,   135,    180, 225,    270,    315,    360.
*          radians: 0, pi/4, pi/2, 3*pi/4, pi,  5*pi/4, 3*pi/2, 7*pi/4, 2*pi.
*          diamond: 0, 0.5,  1.0,  1.5,    2.0, 2.5,    3.0,    3.5,    4=0.
*        > diamond angles are usable in 3d dimension as well.
*          Given any two vectors A, B, setting x = dot(A,B) and y = cross(A,B)
*          then d = atan2(y,x) gives the angle between them.
**/
namespace DiamondAngle {
    /**
    * \brief return the diamond angle beween the positive x-axis and the ray from the origin to the point (x,y) in the Cartesian plane.
    *        when tested against regualr atan2<float>, it showed maximal error of 4.76837e-07 and hugh performance increase.
    *
    * @param {Arithmetic, in}  y
    * @param {Arithmetic, in}  x
    * @param {Arithmetic, out} diamond angle
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr T atan2(const T y, const T x) {
        if ((x == T{}) && (y == T{})) [[unlikely]] {
            return T{};
        }

        if (y >= T{}) {
            return x >= T{} ? (y / (x + y) ) : (static_cast<T>(1) + x / (x - y));
        } else {
            return x < T{} ? (static_cast<T>(2) + y / (x + y)) : (static_cast<T>(3) + x / (x - y));
        }
    }

    /**
    * \brief transform radian to diamond angle
    * @param {Arithmetic, in}  radian
    * @param {Arithmetic, out} diamond angle
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr T radianToDiamondAngle(const T rad) {
        return DiamondAngle::atan2(std::sin(rad), std::cos(rad));
    }

    /**
    * \brief transform diamond angle to radian
    * @param {Arithmetic, in}  diamond angle
    * @param {Arithmetic, out} radian
    **/
    template<typename T>
        requires(std::is_floating_point_v<T>)
    constexpr T diamondAngleToRadian(const T dia) {
        const T y = [dia]() {
            if (dia < static_cast<T>(3)) {
                return dia > static_cast<T>(1) ? static_cast<T>(2) - dia : dia;
            }
            else {
                return dia - static_cast<T>(4);
            }
        }();
        const T x{ dia < static_cast<T>(2) ? static_cast<T>(1) - dia : dia - static_cast<T>(3) };
        return std::atan2(y, x);
    }
};
