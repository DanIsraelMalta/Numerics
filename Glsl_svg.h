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
#include "Glsl.h"
#include <sstream>
#include <fstream>

/**
* \brief object which accepts GLSL vectors and generates scalable vector graphic schema
**/
template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
	requires(VEC::length() == 2)
class svg {
    public:

        /**
        * \brief constructor
        * @param {size_t, in} canvas width
        * @param {size_t, in} canvas height
        **/
        constexpr svg(const std::size_t width, const std::size_t height) {
            // SVG header
            this->xo_svg += "<?xml " + attribute("version", "1.0") + attribute("standalone", "no");
            this->xo_svg += "?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" ";
            this->xo_svg += "\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<svg ";
            this->xo_svg += attribute("width", std::to_string(width), "px");
            this->xo_svg += attribute("height", std::to_string(height), "px");
            this->xo_svg += attribute("xmlns", "http://www.w3.org/2000/svg");
            this->xo_svg += attribute("version", "1.1") + ">\n";

            // canvas frame
            this->xo_svg += elementStart("rect");
            this->xo_svg += attribute("x", std::to_string(0.0));
            this->xo_svg += attribute("y", std::to_string(0.0));
            this->xo_svg += attribute("width", std::to_string(width));
            this->xo_svg += attribute("height", std::to_string(height));
            this->xo_svg += attribute("style", "fill:rgb(255,255,255);stroke-width:1;stroke:black");
            this->xo_svg += "/>\n";
        }

        /**
        * \brief add line
        * @param {VEC,        in} line start
        * @param {VEC,        in} line end
        * @param {string,     in} fill color
        * @param {string,     in} border color
        * @param {value_type, in} border width
        **/
        constexpr void add_line(const VEC a, const VEC b, const std::string fill_color, const std::string border_color, const T width) {
            this->xo_svg += elementStart("line");
            this->xo_svg += attribute("x1", std::to_string(a.x));
            this->xo_svg += attribute("y1", std::to_string(a.y));
            this->xo_svg += attribute("x2", std::to_string(b.x));
            this->xo_svg += attribute("y2", std::to_string(b.y));
            this->xo_svg += style(fill_color, border_color, std::to_string(width));
            this->xo_svg += "/>\n";
        }

        /**
        * \brief add circle
        * @param {VEC,        in} circle center
        * @param {value_type, in} circle radius
        * @param {string,     in} fill color
        * @param {string,     in} border color
        * @param {value_type, in} border width
        **/
        constexpr void add_circle(const VEC center, const T radius, const std::string fill_color, const std::string border_color, const T width) {
            this->xo_svg += elementStart("circle");
            this->xo_svg += attribute("cx", std::to_string(center.x));
            this->xo_svg += attribute("cy", std::to_string(center.y));
            this->xo_svg += attribute("r", std::to_string(radius));
            this->xo_svg += style(fill_color, border_color, std::to_string(width));
            this->xo_svg += "/>\n";
        }

        /**
        * \brief add rectangle
        * @param {VEC,        in} rectangle min corner
        * @param {VEC,        in} rectangle extent (width, height)
        * @param {string,     in} fill color
        * @param {string,     in} border color
        * @param {value_type, in} border width
        **/
        constexpr void add_rectangle(const VEC min, const VEC extent, const std::string fill_color, const std::string border_color, const T width) {
            this->xo_svg += elementStart("rect");
            this->xo_svg += attribute("x", std::to_string(min.x));
            this->xo_svg += attribute("y", std::to_string(min.y));
            this->xo_svg += attribute("width", std::to_string(extent.x));
            this->xo_svg += attribute("height", std::to_string(extent.y));
            this->xo_svg += style(fill_color, border_color, std::to_string(width));
            this->xo_svg += "/>\n";
        }

        /**
        * \brief add polygon
        * @param {forward_iterator, in} polygon first point
        * @param {forward_iterator, in} polygon last point
        * @param {string,           in} fill color
        * @param {string,           in} border color
        * @param {value_type,       in} border width
        **/
        template<std::forward_iterator It, class V = typename std::decay_t<decltype(*std::declval<It>())>>
            requires(std::is_same_v<VEC, V> && V::length() == 2)
        constexpr void add_polygon(const It start, const It finish, const std::string fill_color, const std::string border_color, const T width) {
            this->xo_svg += elementStart("polygon ");
            this->xo_svg += attribute("points", vectorToString(start, finish));
            this->xo_svg += style(fill_color, border_color, std::to_string(width));
            this->xo_svg += "/>\n";
        }

        /**
        * \brief add poly line
        * @param {forward_iterator, in} poly line first point
        * @param {forward_iterator, in} poly line last point
        * @param {string,           in} fill color
        * @param {string,           in} border color
        * @param {value_type,       in} border width
        **/
        template<std::forward_iterator It, class V = typename std::decay_t<decltype(*std::declval<It>())>>
            requires(std::is_same_v<VEC, V> && V::length() == 2)
        constexpr void add_polyline(const It start, const It finish, const std::string fill_color, const std::string border_color, const T width) {
            this->xo_svg += elementStart("polyline ");
            this->xo_svg += attribute("points", vectorToString(start, finish));
            this->xo_svg += style(fill_color, border_color, std::to_string(width));
            this->xo_svg += "/>\n";
        }

        /**
        * @param {string, in} file name to export svg schema to
        **/
        constexpr void to_file(const std::string& file_name = "debug.svg") const {
            std::ofstream out;
            out.open(file_name);
            out << this->xo_svg << "</svg>\n";
            out.close();
        }

	private:

        // svg string
        std::string xo_svg;

        template<std::forward_iterator It, class V = typename std::decay_t<decltype(*std::declval<It>())>>
            requires(std::is_same_v<VEC, V> && V::length() == 2)
        constexpr std::string vectorToString(const It start, const It finish) {
            std::string out;
            for (It it{ start }; it != finish; ++it) {
                const VEC p{ *it };
                out += std::to_string(p.x) + "," + std::to_string(p.y) + " ";
            }
            return out;
        }

        // return svg attribute
        constexpr std::string attribute(const std::string attribute_name, const std::string value, const std::string extra = "") {
            return attribute_name + "=\"" + value + extra + "\" ";
        };

        // return svg style
        constexpr std::string style(const std::string fill_color, const std::string border_color, const std::string width) {
            return " style =\" fill: " + fill_color + " ;stroke-width: " + width + "; stroke: " + border_color + "\" ";
        }

        // return svg element start
        constexpr std::string elementStart(const std::string element_name) {
            return "\t<" + element_name + " ";
        };

        // return svg element end
        constexpr std::string elementEnd(const std::string element_name) {
            return "</" + element_name + ">\n";
        };
};
