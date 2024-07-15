#pragma once
#include "Glsl.h"
#include "Glsl_extra.h"
#include "Glsl_point_distance.h"
#include "DiamondAngle.h"
#include <limits>
#include <vector>
#include <algorithm>
#include <random>

//
// collection of algorithms for 2D cloud points and shapes
//
namespace Algorithms2D {

	//
	// utilities
	//
	namespace Internals {

		/**
		* \brief check if a point is counter clock wise relative to two other points
		* @param {IFixedVector, in}  point a
		* @param {IFixedVector, in}  point b
		* @param {IFixedVector, in}  point c
		* @param {value_type,   out} negative value means 'c' is counter clockwise to segmen a-b,
		*                            positive value means 'c' is clockwise to segmen a-b,
		*                            zero means 'c' is colinear with segmen a-b,
		**/
		template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
			requires(VEC::length() == 2)
		constexpr T are_points_ordered_counter_clock_wise(const VEC& a, const VEC& b, const VEC& c) noexcept {
			return (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y);
		}

		/**
		* \brief check if a point is "left" to another point
		* @param {IFixedVector, in}  point a
		* @param {IFixedVector, in}  point b
		* @param {bool,         out} true if point 'a' is left to point 'b'
		**/
		template<GLSL::IFixedVector VEC>
			requires(VEC::length() == 2)
		constexpr bool is_point_left_of(const VEC& a, const VEC& b) noexcept {
			return (a.x < b.x || (a.x == b.x && a.y < b.y));
		}

		/**
		* \brief return twice a triangle area
		* @param {IFixedVector, in}  point a
		* @param {IFixedVector, in}  point b
		* @param {IFixedVector, in}  point c
		* @param {floating,     out} twice triangle area
		**/
		template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
			requires(VEC::length() == 2)
		constexpr T triangle_twice_area(const VEC& a, const VEC& b, const VEC& c) noexcept {
			const VEC v1{ a - c };
			const VEC v2{ b - c };
			return std::abs(v1.x * v2.y - v1.y * v2.x);
		}

		/**
		* \brief return triangle area
		* @param {IFixedVector, in}  ray #1 origin
		* @param {IFixedVector, in}  ray #1 direction
		* @param {IFixedVector, in}  ray #2 origin
		* @param {IFixedVector, in}  ray #2 direction
		* @param {IFixedVector, out} ray #1 and ray #2 intersection point (vector filled with std::numeric_limits<T>::max() if intersection does not occure)
		**/
		template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
			requires(VEC::length() == 2)
		constexpr VEC get_rays_intersection_point(const VEC& ro0, const VEC& rd0, const VEC& ro1, const VEC& rd1) noexcept {
			assert(Extra::is_normalized(rd0));
			assert(Extra::is_normalized(rd1));
			if (GLSL::equal(ro0, ro1)) [[unlikely]] {
				return ro0;
			}

			const T det{ rd1.x * rd0.y - rd1.y * rd0.x };
			if (Numerics::areEquals(det, T{})) [[unlikely]] {
				return VEC(std::numeric_limits<T>::max());
			}
			[[assume(det > T{})]];

			const VEC d{ ro1 - ro0 };
			const T u{ (d.y * rd1.x - d.x * rd1.y) / det };
			if (const T v{ (d.y * rd0.x - d.x * rd0.y) / det }; u < T{} || v < T{}) [[unlikely]] {
				return VEC(std::numeric_limits<T>::max());
			}
			return ro0 + u * rd0;
		}

		/**
		* \brief project point on segment
		* @param {IFixedVector,               in}  segment point #1
		* @param {IFixedVector,               in}  segment point #2
		* @param {IFixedVector,               in}  point
		* @param {{IFixedVector, value_type}, out} {point projected on segment, interpolant along segment from point #1 to projected point}
		**/
		template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
			requires(VEC::length() == 2)
		constexpr auto project_point_on_segment(const VEC& a, const VEC& b, const VEC& p) noexcept {
			using out_t = struct { VEC point; T t; };
			const VEC ap{ p - a };
			const VEC ab{ b - a };

			const T ap_dot_ab{ GLSL::dot(ap, ab) };
			const T ab_dot{ GLSL::dot(ab) };
			assert(ab_dot > T{});

			const T t{ ap_dot_ab / ab_dot };
			return out_t{ a + t * ab, t };
		}

		/**
		* \brief get the circumcircle of two/three points
		* @param {IFixedVector,               in}  point #1
		* @param {IFixedVector,               in}  point #2
		* @param {IFixedVector,               in}  point #3 (optional)
		* @param {{IFixedVector, value_type}, out} {circumcircle center, circumcircle squared radius}
		*                                          center will be at center and squared radius will be negative in case of invalid calculation.
		**/
		template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
			requires(VEC::length() == 2)
		constexpr auto get_circumcircle(const VEC& a, const VEC& b) noexcept {
			using out_t = struct { VEC center; T radius_squared; };
			return out_t{ (a + b) / static_cast<T>(2), GLSL::dot(a - b) / static_cast<T>(4) };
		}
		template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
			requires(VEC::length() == 2)
		constexpr auto get_circumcircle(const VEC& a, const VEC& b, const VEC& c) noexcept {
			using out_t = decltype(Algorithms2D::Internals::get_circumcircle(a, b));

			const VEC ba{ b - a };
			const VEC ca{ c - a };
			const T B{ GLSL::dot(ba) };
			const T C{ GLSL::dot(ca) };
			const T D{ GLSL::cross(ba, ca) };
			if (Numerics::areEquals(D, T{})) {
				return out_t{ VEC(), static_cast<T>(-1) };
			}
			const VEC center{ a + VEC(ca.y * B - ba.y * C, ba.x * C - ca.x * B) / (static_cast<T>(2) * D) };
			return out_t{ center, GLSL::dot(center - a) };
		}
		template<GLSL::IFixedVector VEC, class T = typename VEC::value_type>
			requires(VEC::length() == 2)
		constexpr auto get_circumcircle(const VEC& a, const VEC& b, const VEC& c, const VEC& d) noexcept {
			using out_t = decltype(Algorithms2D::Internals::get_circumcircle(a, b));

			// does 'abc' circle include 'd'?
			out_t circle{ Internals::get_circumcircle(a, b, c) };
			bool inside{ GLSL::dot(d - circle.center) <= circle.radius_squared };
			if (inside) {
				return circle;
			}

			// does 'abd' circle include 'c'?
			circle = Internals::get_circumcircle(a, b, d);
			inside = GLSL::dot(c - circle.center) <= circle.radius_squared;
			if (inside) {
				return circle;
			}

			// does 'acd' circle include 'b'?
			circle = Internals::get_circumcircle(a, c, d);
			inside = GLSL::dot(b - circle.center) <= circle.radius_squared;
			if (inside) {
				return circle;
			}

			// does 'bcd' circle include 'a'?
			circle = Internals::get_circumcircle(b, c, d);
			inside = GLSL::dot(a - circle.center) <= circle.radius_squared;
			if (inside) {
				return circle;
			}

			// shouldn't reach this point
			assert(0 == 1);
		}

		/**
		* \brief given collection of points, return it sorted in clock wise manner
		* @param {vector<IFixedVector>, in}  cloud of points
		* @param {IFixedVector,         out} centroid
		**/
		template<GLSL::IFixedVector VEC>
			requires(VEC::length() == 2)
		constexpr VEC get_centroid(const std::vector<VEC>& points) {
			using T = typename VEC::value_type;

			VEC centroid;
			for (const VEC p : points) {
				centroid += p;
			}

			assert(points.size() > 0);
			return (centroid / static_cast<T>(points.size()));
		}

		/**
		* \brief given a point and collection of points whice define segments, return the indices of points which define the segment which is closest to the point
		* @param {vector<IFixedVector>, in}  cloud of points, each two consecutive points define a segment
		* @param {IFixedVector,         in}  point
		* @param {{value_type, size_t}, out} {squared distance, index of point #1 in collection which define closest segment}
		**/
		template<GLSL::IFixedVector VEC>
			requires(VEC::length() == 2)
		constexpr auto get_index_of_closest_segment(const std::vector<VEC>& segments, VEC point) noexcept {
			using T = typename VEC::value_type;
			using out_t = struct { T distance_squared; std::size_t index; };

			// housekeeping
			std::size_t index{};
			T distSquared{ std::numeric_limits<T>::max() };
			const auto update_point = [&point, &index, &distSquared](const VEC& a, const VEC& b, const std::size_t i) {
				const T dist2{ PointDistance::squared_udf_to_segment(point, a, b) };
				const bool closest{ dist2 > T{} && dist2 < distSquared };
				distSquared = closest ? dist2 : distSquared;
				index = closest ? i : index;
			};

			// iterate over all segments
			const std::size_t N{ segments.size() };
			for (std::size_t i{}; i < N - 1; ++i) {
				if (!Extra::are_vectors_identical(point, segments[i]) &&
					!Extra::are_vectors_identical(point, segments[i + 1])) {
					update_point(segments[i], segments[i + 1], i);
				}
			}

			// check "closed segment"
			if (!Extra::are_vectors_identical(point, segments[0]) &&
				!Extra::are_vectors_identical(point, segments[N - 1])) {
				update_point(segments[0], segments[N - 1], N - 1);
			}

			// output
			return out_t{ distSquared, index };
		}
	};

	/**
	* \brief calculate the convex hull of collection of 2D points (using Graham scan algorithm).
	*       
	* @param {forward_iterator,     in}  iterator to point cloud collection first point
	* @param {forward_iterator,     in}  iterator to point cloud collection last point
	* @param {vector<IFixedVector>, out} collection of points which define point cloud convex hull
	**/
	template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>>
		requires(GLSL::is_fixed_vector_v<VEC> && (VEC::length() == 2))
	constexpr std::vector<VEC> get_convex_hull(const InputIt first, const InputIt last) {
		using T = typename VEC::value_type;
		std::vector<VEC> points(first, last);

		// place left most point at start of point cloud
		std::swap(points[0], *std::ranges::min_element(points, [](const VEC& a, const VEC& b) noexcept -> bool {
			return (a.x < b.x || (a.x == b.x && a.y < b.y));
		}));

		// lexicographically sort all points using the smallest point as pivot
		const VEC v0( points[0] );
		std::sort(points.begin() + 1, points.end(), [v0](const VEC& b, const VEC& c) noexcept -> bool {
			return (b.y - v0.y) * (c.x - b.x) - (b.x - v0.x) * (c.y - b.y) < T{};
		});

		// build hull
		auto it = points.begin();
		std::vector<VEC> hull{ {*it++, *it++, *it++} };
		while (it != points.end()) {
			while (Internals::are_points_ordered_counter_clock_wise(*(hull.rbegin() + 1), *(hull.rbegin()), *it) >= T{}) {
				hull.pop_back();
			}
			hull.push_back(*it++);
		}

		return hull;
	}

	/**
	* \brief calculate the concave hull of collection of 2D points (using Graham scan algorithm).
	*
	* @param {forward_iterator,     in}  iterator to point cloud collection first point
	* @param {forward_iterator,     in}  iterator to point cloud collection last point
	* @param {value_type,           in}  concave threshold. point will be added to concave hull if the length of edge segment
	*                                    divided by the length of the segment divided at given point will be larger than this value.
	*                                    the smaller - the more points are added to concave.
	*                                    negative value will result in calculating the convex hull.
	* @param {vector<IFixedVector>, out} collection of points which define point cloud concave hull
	**/
	template<std::forward_iterator InputIt, class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>, class T = typename VEC::value_type>
		requires(GLSL::is_fixed_vector_v<VEC> && (VEC::length() == 2))
	constexpr std::vector<VEC> get_concave_hull(const InputIt first, const InputIt last, const T concave_threshold) {
		// get convex hull
		std::vector<VEC> hull{ Algorithms2D::get_convex_hull(first, last) };
		if (concave_threshold <= T{}) {
			return hull;
		}

		// "dig" into convex hull to create concave hull
		bool finished{ false };
		while (!finished) {
			// find closest point to hull
			VEC closest_point;
			std::size_t segment_index_start{};
			T distance_squared{ std::numeric_limits<T>::max() };
			for (auto it{ first }; it != last; ++it) {
				const VEC p{ *it };
				auto closest = Algorithms2D::Internals::get_index_of_closest_segment(hull, p);
				if (closest.distance_squared < distance_squared) {
					distance_squared = closest.distance_squared;
					segment_index_start = closest.index + 1;
					closest_point = p;
				}
			}

			// should point be part of concave hull?
			const std::size_t segment_index_end{ (segment_index_start + 1) % hull.size()};
			const T segment_length{ GLSL::distance(hull[segment_index_start], hull[segment_index_end]) };
			const T new_segment_length{ GLSL::distance(closest_point, hull[segment_index_end]) +
										GLSL::distance(hull[segment_index_start], closest_point) };
			assert(segment_length > T{});
			[[assume(segment_length > T{})]];
			if (segment_length / new_segment_length >= concave_threshold) {
				hull.insert(hull.begin() + segment_index_start, closest_point);
			} else {
				finished = true;
			}
		}

		// output
		return hull;
	}

	/**
	* \brief given convex hull, return its minimal area bounding rectangle
	* @param {vector<IFixedVector>, in}  convex hull
	* @param {{VEC, VEC, VEC, VEC}, out} vertices of minimal area bounding rectangle (ordererd counter clock wise)
	**/
	template<GLSL::IFixedVector VEC>
		requires(VEC::length() == 2)
	constexpr auto get_convex_hull_minimum_area_bounding_rectangle(const std::vector<VEC>& hull) {
		using T = typename VEC::value_type;
		using out_t = struct { VEC p0; VEC p1; VEC p2; VEC p3; };

		// iterate over all convex hull edges and find minimal area bounding rectangke
		T area{ std::numeric_limits<T>::max() };
		VEC p0;
		VEC p1;
		VEC maxNormal;
		for (std::size_t i{}; i < hull.size() - 2; ++i) {
			// segment
			VEC v0{ hull[i] };
			VEC v1{ hull[i + 1] };

			// segment distance to center
			const VEC center{ (v0 + v1) / static_cast<T>(2) };
			T v0_dist{ GLSL::dot(v0 - center) };
			T v1_dist{ v0_dist };
			T vNormal_dist{};

			// segment tangential and orthogonal directions
			const VEC dir{ GLSL::normalize(v1 - v0) };
			const VEC normal(-dir.y, dir.x);

			// point on orthogonal segment
			const VEC v2_0{ v0 + normal };
			const VEC v2_1{ v1 + normal };
			VEC vNormal(center);
			VEC ref;
			
			// project points on line connecting convex hull edge and find minimal/maximal points.
			// project points on orthogonal line to edge (should be going inside the convex hull) and find maximal point.
			for (const VEC point: hull) {
				// project points on segment tangential and orthogonal directions (orthogonal tawrds inside hull)
				const auto projOnDir{ Internals::project_point_on_segment(v0, v1, point) };
				const auto projOnNormal_0{ Internals::project_point_on_segment(v0, v2_0, point) };
				const auto projOnNormal_1{ Internals::project_point_on_segment(v1, v2_1, point) };
				
				// find furthest points along v0v1 segments which can constitute an extent to tight bounding box
				if (const T projOnDir_dist{ GLSL::dot(projOnDir.point - center) }; projOnDir.t < T{} && projOnDir_dist > v0_dist) {
					v0_dist = projOnDir_dist;
					v0 = projOnDir.point;
				} else if (projOnDir.t > T{} && projOnDir_dist > v1_dist) {
					v1_dist = projOnDir_dist;
					v1 = projOnDir.point;
				}
				
				// find furthest point from v0v1 segment along orthogonal direction to v0v1
				if (const T dist{ GLSL::dot(projOnNormal_0.point - v2_0) }; dist > vNormal_dist) {
					vNormal_dist = dist;
					vNormal = projOnNormal_0.point;
					ref = v0;
				}
				if (const T dist{ GLSL::dot(projOnNormal_1.point - v2_1) }; dist > vNormal_dist) {
					vNormal_dist = dist;
					vNormal = projOnNormal_1.point;
					ref = v1;
				}
			}

			// rectangle area
			const T rectangle_area{ GLSL::dot(v1 - v0) * GLSL::dot(vNormal - ref) };
			if (rectangle_area < area) {
				area = rectangle_area;
				p0 = v0;
				p1 = v1;
				maxNormal = vNormal;
			}
		}

		// calculate bounding rectangle vertices
		const VEC dir{ GLSL::normalize(p1 - p0) };
		const VEC normal{ -dir.y, dir.x };
		const VEC p2{ Internals::get_rays_intersection_point(p1, normal, maxNormal,  dir) };
		const VEC p3{ Internals::get_rays_intersection_point(p0, normal, maxNormal, -dir) };

		return out_t{ p0, p1, p2, p3 };
	}
	
	/**
	* \brief given convex hull of collection of points, return its diameter
	* @param {vector<IFixedVector>,           in}  points convex hull
	* @param {{value_type, array<size_t, 2>}, out} {squared diameter, <index of anti podal oint #1, index of anti podal point #2>}
	**/
	template<GLSL::IFixedVector VEC>
		requires(VEC::length() == 2)
	constexpr auto get_convex_diameter(const std::vector<VEC>& hull) {
		using T = typename VEC::value_type;
		using out_t = struct { T diamater_squared; std::array<std::size_t, 2> indices; };

		// housekeeping
		const std::size_t N{ hull.size() };
		out_t out{
			.diamater_squared = T{},
			.indices = std::array<std::size_t, 2>{{0, 0}}
		};
		const auto checkPoints = [N, &out, &hull](const std::size_t i, const std::size_t j) {
			const VEC a{ hull[i % N] };
			const VEC b{ hull[j % N] };
			const T furthest{ GLSL::dot(a - b) };
			const bool update{ furthest > out.diamater_squared };
			out.diamater_squared = update ? furthest : out.diamater_squared;
			out.indices = update ? std::array<std::size_t, 2>{{i, j}} : out.indices;
		};

		std::size_t k{ 1 };
		const VEC hull_0{ hull[0] };
		const VEC hull_n1{ hull[N - 1] };
		while (Internals::triangle_twice_area(hull_n1, hull_0, hull[(k + 1) % N]) >
			   Internals::triangle_twice_area(hull_n1, hull_0, hull[k])) {
			++k;
			checkPoints(N - 1, k);
		}

		for (std::size_t i{}, j{ k }; i <= k && j < N; ++i) {
			const VEC hull_i{ hull[i] };
			const VEC hull_i_1{ hull[(i + 1) % N] };

			while (j < N &&
				   Internals::triangle_twice_area(hull_i, hull_i_1, hull[(j + 1) % N]) >
				   Internals::triangle_twice_area(hull_i, hull_i_1, hull[j])) {
				checkPoints(i, (j + 1) % N);
				++j;
			}
		}

		return out;
	}

	/**
	* \brief given convex hull of collection of points, return the minimal bounding circle (circumcircle).
	*        based on https://www.cise.ufl.edu/~sitharam/COURSES/CG/kreveldnbhd.pdf.
	* @param {vector<IFixedVector>,       in}  points convex hull
	* @param {{IFixedVector, value_type}, out} {minimal bounding circle center, minimal bounding circle squared radius}
	**/
	template<GLSL::IFixedVector VEC>
		requires(VEC::length() == 2)
	constexpr auto get_minimal_bounding_circle(const std::vector<VEC>& hull) {
		using T = typename VEC::value_type;
		using out_t = decltype(Algorithms2D::Internals::get_circumcircle(hull[0], hull[0]));

		// is hulll composed of two/three/four points?
		const std::size_t N{ hull.size() };
		if (N == 2) {
			return Internals::get_circumcircle(hull[0], hull[1]);
		} else if (N == 3) {
			return Internals::get_circumcircle(hull[0], hull[1], hull[2]);
		} else if (N == 4) {
			return Internals::get_circumcircle(hull[0], hull[1], hull[2], hull[3]);
		}

		// lambda to check if a given point is inside given circle
		const auto is_point_in_circle = [](const out_t& circle, const VEC& point) -> bool {
			return GLSL::dot(point - circle.center) <= circle.radius_squared;
		};

		// find minimal bounding circle of set of points using two points
		const auto make_bounding_circle_two_points = [&hull, &is_point_in_circle](const std::size_t end, const VEC& p, const VEC& q) -> out_t {
			out_t circ{ Internals::get_circumcircle(p, q) };
			out_t left{ VEC(), static_cast<T>(-1) };
			out_t right{ VEC(), static_cast<T>(-1) };

			const VEC pq{ q - p };
			for (std::size_t i{}; i < end; ++i) {
				const VEC r{ hull[i] };
				if (is_point_in_circle(circ, r)) {
					continue;
				}

				const out_t c{ Internals::get_circumcircle(p, q, r) };
				if (c.radius_squared <= T{}) {
					continue;
				}

				const T cross{ GLSL::cross(pq, r - p) };
				const T pq_cross{ GLSL::cross(pq, c.center - p) };
				if (cross > T{} && (left.radius_squared < T{} || pq_cross > GLSL::cross(pq, left.center - p))) {
					left.center = c.center;
					left.radius_squared = c.radius_squared;
				}
				else if (cross < T{} && (right.radius_squared < T{} || pq_cross < GLSL::cross(pq, right.center - p))) {
					right.center = c.center;
					right.radius_squared = c.radius_squared;
				}
			}

			if (left.radius_squared <= T{} && right.radius_squared <= T{}) {
				return circ;
			}
			else if (left.radius_squared < T{}) {
				return right;
			}
			else if (right.radius_squared < T{}) {
				return left;
			}
			else {
				return (left.radius_squared <= right.radius_squared) ? left : right;
			}
		};

		// find minimal bounding circle of set of points using one point
		const auto make_bounding_circle_one_points = [&hull, &is_point_in_circle, &make_bounding_circle_two_points, N](const std::size_t end, const VEC& p) -> out_t {
			out_t circle{ p, static_cast<T>(-1) };

			for (std::size_t i{}; i < end; ++i) {
				const VEC q{ hull[i] };

				if (!is_point_in_circle(circle, q)) {
					if (circle.radius_squared <= T{}) {
						circle = Internals::get_circumcircle(p, q);
					} else {
						circle = make_bounding_circle_two_points(Numerics::min(i + 1, N), p, q);
					}
				}
			}

			return circle;
		};

		// iterate over remaining points and find minimal enclosling circle
		out_t circle{ Internals::get_circumcircle(hull[0], hull[1], hull[2], hull[3]) };
		for (std::size_t i{ 4 }; i < N; ++i) {
			const VEC p{ hull[i] };
			if (!is_point_in_circle(circle, p)) {
				circle = make_bounding_circle_one_points(Numerics::min(i + 1, N), p);
			}
		}

		return circle;
	}

	/**
	* \brief given polygon and a point - check if point is inside polygon
	*        (see: https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html)
	* @param {vector<IFixedVector>, in}  polygon
	* @param {IFixedVector,         in}  point
	* @param {bool,                 out} true if point is inside polygon, false otherwise
	**/
	template<GLSL::IFixedVector VEC>
		requires(VEC::length() == 2)
	constexpr bool is_point_inside_polygon(const std::vector<VEC>& poly, const VEC& point) {
		using T = typename VEC::value_type;

		bool inside{ false };
		for (std::size_t len{ poly.size() }, i{}, j{ len - 2 }; i < len - 1; j = i++) {
			const VEC pi{ poly[i] };
			const VEC pj{ poly[j] };
			const bool intersects{ pi.y > point.y != pj.y > point.y &&
				                   point.x < ((pj.x - pi.x) * (point.y - pi.y)) / (pj.y - pi.y) + pi.x };
			inside = intersects ? !inside : inside;
		}

		return inside;
	}

	/**
	* \brief given collection of points, return true if they are ordered in clock wise manner
	* @param {vector<IFixedVector>, in} cloud of points
	* @param {IFixedVector,         in} points geometric center
	* @param {bool,                 out} true if point are ordered in clock wise manner, false otherwise
	**/
	template<GLSL::IFixedVector VEC>
		requires(VEC::length() == 2)
	constexpr bool are_points_ordererd_clock_wise(const std::vector<VEC>& poly, const VEC& centroid) {
		using T = typename VEC::value_type;

		bool clockwise{ false };
		for (std::size_t len{ poly.size() }, i{}, j{ len - 2 }; i < len - 1; j = i++) {
			const VEC a{ poly[i] };
			const VEC b{ poly[j] };
			const T angle_a{ DiamondAngle::atan2(a.y - centroid.y, a.x - centroid.x) };
			const T angle_b{ DiamondAngle::atan2(b.y - centroid.y, b.x - centroid.x) };

			clockwise = angle_a > angle_b ? !clockwise : clockwise;
		}

		return clockwise;
	}

	/**
	* \brief given collection of points, return it sorted in clock wise manner
	* @param {vector<IFixedVector>, in} cloud of points
	* @param {IFixedVector,         in} points geometric center
	* @param {vector<IFixedVector>, out} points sorted in clock wise manner
	**/
	template<GLSL::IFixedVector VEC>
		requires(VEC::length() == 2)
	constexpr std::vector<VEC> sort_points_clock_wise(const std::vector<VEC>& cloud, const VEC& centroid) {
		using T = typename VEC::value_type;

		// housekeeping
		std::vector<VEC> points(cloud);

		// sort clock wise
		std::ranges::sort(points, [&centroid](const VEC& a, const VEC& b) noexcept -> bool {
			const T angla_a{ DiamondAngle::atan2(a.y - centroid.y, a.x - centroid.x) };
			const T angla_b{ DiamondAngle::atan2(b.y - centroid.y, b.x - centroid.x) };
			return angla_a > angla_b;
		});

		// output
		return points;
	}
}
