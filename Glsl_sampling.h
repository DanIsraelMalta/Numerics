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
#include "Hash.h" // for sample_circle
#include "Glsl.h"
#include "Algorithms.h"
#include <vector>

//
// collection of objects which allow for uniform sampling from various shapes
//
namespace Sample {

    /**
    * \brief uniformly sample inside a circle.
    *        see "a probabilistic approach to the geometry of the Lp^n ball" (https://arxiv.org/pdf/math/0503650)
    * @param {VEC,             in}  circle center
    * @param {VEC::value_type, in}  circle radius
    * @param {VEC,             out} point, uniformly sampled, inside a given circle
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(VEC::length() == 2)
    constexpr VEC sample_circle(const VEC& center, const T radius) {
        const VEC uv(Hash::normal_distribution(), Hash::normal_distribution());
        const T e{ static_cast<T>(-2.0) * std::log(static_cast<T>(rand()) / static_cast<T>(RAND_MAX)) };
        const T norm{ GLSL::dot(uv) + e };
        return !Numerics::areEquals(norm, T{}) ? (center + radius * uv / std::sqrt(norm)) : center;
    }

    /**
    * \brief uniformly sample inside an ellipse (non rotated).
    * @param {VEC,             in}  ellipse center
    * @param {VEC::value_type, in}  ellipse radius along X axis
    * @param {VEC::value_type, in}  ellipse radius along Y axis
    * @param {VEC,             out} point, uniformly sampled, inside a given ellipse
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(VEC::length() == 2)
    constexpr VEC sample_ellipse(const VEC& center, const T a, const T b) {
        constexpr T pi{ static_cast<T>(3.1415926535897932384626433832795) };

        assert(!Numerics::areEquals(a, T{}));
        assert(!Numerics::areEquals(b, T{}));

        // angle [-pi/2, 3*pi/2]
        const T u{ static_cast<T>(rand()) / static_cast<T>(RAND_MAX) / static_cast<T>(4.0) };
        const T ratio{ b / a };
        const T angle{ std::tan(static_cast<T>(2.0) * pi * u) };
        T theta{ std::atan(ratio * angle) };
        if (const T v{ static_cast<T>(rand()) / static_cast<T>(RAND_MAX) };
            v > static_cast<T>(0.25)) {
            if (v < static_cast<T>(0.5)) {
                theta = pi - theta;
            }
            else if (v < static_cast<T>(0.75)) {
                theta += pi;
            }
            else {
                theta = -theta;
            }
    }

        // radius
        const T sin_angle{ std::sin(theta) };
        const T cos_angle{ std::cos(theta) };
        const T max_radius{ a * b / std::sqrt(b * b * cos_angle * cos_angle + a * a * sin_angle * sin_angle) };
        const T radius{ max_radius * std::sqrt(static_cast<T>(rand()) / static_cast<T>(RAND_MAX)) };

        // output
        return !Numerics::areEquals(radius, T{}) ? (center + VEC(radius * cos_angle, radius * sin_angle)) : center;
    }

    /**
    * \brief uniformly sample n points inside a triangle
    *        see "shape distribution" (https://www.cs.princeton.edu/~funk/tog02.pdf)
    * @param {VEC, in}  triangle vertex #1
    * @param {VEC, in}  triangle vertex #2
    * @param {VEC, in}  triangle vertex #3
    * @param {VEC, out} point, uniformly sampled, inside a given triangle
    **/
    template<GLSL::IFixedVector VEC>
        requires(VEC::length() == 2)
    constexpr VEC sample_triangle(const VEC& p0, const VEC& p1, const VEC& p2) {
        using T = typename VEC::value_type;
        
        const T a{ Numerics::max(static_cast<T>(rand()) / static_cast<T>(RAND_MAX),
                                 static_cast<T>(rand()) / static_cast<T>(RAND_MAX)) };
        const T b{ static_cast<T>(rand()) / static_cast<T>(RAND_MAX) };
        const T a1{ static_cast<T>(1.0) - a };
        const T b1{ static_cast<T>(1.0) - b };
        return (a1 * p0 + a * (b1 * p1 + b * p2));
    }

    /**
    * \brief uniformly sample inside a rectangle
    * @param {VEC, in}  square min
    * @param {VEC, in}  square max
    * @param {VEC, out} point, uniformly sampled, inside a given rectangle
    **/
    template<GLSL::IFixedVector VEC>
        requires(VEC::length() == 2)
    constexpr VEC sample_rectangle(const VEC& min, const VEC& max) {
        using T = typename VEC::value_type;

        const T x{ std::abs(max.x - min.x) };
        const T y{ std::abs(max.y - min.y) };
        const VEC sample(x * static_cast<T>(rand()) / static_cast<T>(RAND_MAX),
                         y * static_cast<T>(rand()) / static_cast<T>(RAND_MAX));
        return (min + sample);
    }

    /**
    * \brief uniformly sample inside a parallelogram
    * @param {VEC, in}  parallelogram vertex #1
    * @param {VEC, in}  parallelogram vertex #2
    * @param {VEC, in}  parallelogram vertex #3 (not really needed...)
    * @param {VEC, in}  parallelogram vertex #4
    * @param {VEC, out} point, uniformly sampled, inside a given parallelogram
    **/
    template<GLSL::IFixedVector VEC>
        requires(VEC::length() == 2)
    constexpr VEC sample_parallelogram(const VEC& p0, const VEC& p1, [[maybe_unused]] const VEC& p2, const VEC& p3) {
        using T = typename VEC::value_type;

        const VEC e1{ p1 - p0 };
        const VEC e2{ p3 - p0 };
        const VEC u(static_cast<T>(rand()) / static_cast<T>(RAND_MAX),
                    static_cast<T>(rand()) / static_cast<T>(RAND_MAX));
        return (p0 + u.x * e1 + u.y * e2);
    }

    /**
    * \brief uniformly sample points inside a polygon.
    * @param {vector<IFixedVector>, in}  vector of vertices of polygon triangles.
    *                                    every three consecutive iterators define a triangle.
    *                                    see 'Algorithms2D::triangulate_polygon_delaunay' or "Algorithms2D::triangulate_polygon_earcut"
    * @param {value_type,           in}  twice polygon/triangulation area (can use 'Algorithms2D::Internals::get_area')
    * @param {size_t,               in}  amount of points to sample inside polygon
    * @param {vector<IFixedVector>, out} point, uniformly sampled, inside a given polygon
    **/
    template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
        requires(VEC::length() == 2)
    constexpr std::vector<VEC> sample_polygon(const std::vector<VEC>& triangulaion, const T total_area, const std::size_t count) {
        // housekeeping
        const T area_count{ static_cast<T>(count) / total_area };

        // sample each triangle according to its percentage of total polygon area
        std::vector<VEC> samples;
        samples.reserve(count);
        for (std::size_t i{}; i < triangulaion.size(); i += 3) {
            const VEC p0{ triangulaion[i]     };
            const VEC p1{ triangulaion[i + 1] };
            const VEC p2{ triangulaion[i + 2] };

            const VEC v1{ p0 - p2 };
            const VEC v2{ p1 - p2 };
            const T local_area{ std::abs(Numerics::diff_of_products(v1.x, v2.y, v1.y, v2.x)) };

            for (std::size_t j{}, len{ static_cast<std::size_t>(std::ceil(local_area * area_count)) }; j < len; ++j) {
                samples.emplace_back(Sample::sample_triangle(p0, p1, p2));
            }
        }

        // output
        if (samples.size() > count) {
            samples.resize(count);
        }
        return samples;
    }
}
