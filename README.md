# Welcome to Numerics
"Numerics" is a modern c++ (requires c++23) collection of numerical algorithms, structures and operations with particular focus on linear algebra and computational geometry.

Features include:
+ Generic, modern and extensible numerical toolkit which follows the syntax and functionality of the GLSL shading language.
+ Implementations of canonical linear algebra operations, ranging from decompositions to linear equation system set solvers.
+ Mandatory collection of coherent set of operations related to spatial transformations, sign distance fields, ray intersections and solution to general numerical/geometrical problems often encountered in the realms of 2D/3D geometry.
+ A suite of computational geometry tools ranging from acceleration structures used for fast nearest neighbours queries, clustering algorithms and 2D tailored operations.

**Sample #1 - glsl syntax (swizzling, column major, etc.):**
```cpp
// GLSL swizzle syntax
vec2 a(1.0f, -1.0f);
vec2 b(3.2f, 2.0f);
vec2 c = a.yx + b.xy; // c = {2.2f, 3.0f}
a.xy += b.xx; // a = {4.2f, 2.2f}

vec3 d(0.0f, 1.0f, 2.0f);
vec3 e(2.0f, 1.0f, 3.0f);
vec3 f = GLSL::mix<0.5f>(d, e); // f = {1.0f, 1.0f, 2.5f}

// build matrix from column vectors
// /4 7\
// \5 8/
mat2i b(ivec2(4, 5), ivec2(7, 8));

// multiply vector by matrix
//  (4 5) * /4 7\ = 4*4 + 5*5 = (41 68)
//          \5 8/   4*7 + 5*8
ivec2 xb = b.x * b;
assert(GLSL::equal(xb, ivec2(41, 68)));

// multiply matrix by vector
// /4 7\ * /4\ = 4*4 + 7*5 = (51, 60)
// \5 8/   \5/   5*4 + 8*5
ivec2 bx = b * b.x;
assert(GLSL::equal(bx, ivec2(51, 60)));

// basic operations
assert(GLSL::determinant(b) == -3);
assert(GLSL::equal(GLSL::transpose(b), mat2i(4, 7, 5, 8)));
```


**Sample #2 - spatial transformations:**
```cpp
// define axis and angle
vec3 axis = GLSL::normalize(vec3(1.0f, 2.0f, 3.0f));
float angle{ std::numbers::pi_v<float> / 4.0f };

// create rotation matrix and quaternion from axis and angle
const auto Rx = Transformation::rotation_matrix_from_axis_angle(vec3(1.0f, 0.0f, 0.0f), std::numbers::pi_v<float> / 2.0f);
vec4 quat = Transformation::create_quaternion_from_axis_angle(axis, angle);

// Euler angles from quaternion
vec3 euler = Transformation::create_euler_angles_from_quaternion(quat);

// create rotation matrix from quaternion and extract rotation axis
mat3 mat = Transformation::create_rotation_matrix_from_quaternion(quat);
auto axis_from_mat = Transformation::get_axis_angle_from_rotation_matrix(mat);

// decompose quaternion to twist and swing components
const auto decomp = Extra::decompose_quaternion_twist_swing(quat);
const vec4 qa = Extra::multiply_quaternions(decomp.Qz, decomp.Qr);
// quat = Extra::multiply_quaternions(decomp.Qz, decomp.Qr);
```


**Sample #3 - linear algebra (decompositions, linear equation system solvers, BLAS, etc.):**
```cpp
mat3 a(12.0, -51.0, 4.0,
       6.0,  167.0, -68.0,
       -4.0, 24.0,  -41.0);

// calculate eigenvalues
auto eigs = Decomposition::eigenvalues(a);
// eigs = {-34.196675001469160, 156.13668406196879, 16.059990939500377}

// perform QR decomposition
auto qr = Decomposition::QR(a);
// qr.Q = (0.228375, -0.9790593, 0.076125,
//         0.618929, 0.084383,   -0.780901,
//         0.751513, 0.225454,   0.619999);
// qr.R = (52.545219,   0.0,       -0.0,
//         -165.895209, 70.906839,  0.0,
//          -27.328842, 31.566433, -23.015097);

// solve set of linear equation using QR decomposition
vec3 b(70.0, 12.0, 50.0);
auto solution = Solvers::SolveQR(a, b);
// solution = {3.71118, 1.74416, -3.75020}

// create an 8x8 householder and companion matrices and calculate their generalized matrix-matrix multiplication (GEMM)
using mat8 = typename GLSL::MatrixN<double, 8>;
using vec8 = typename GLSL::VectorN<double, 8>;
vec8 axis8;
mat8 companion8;
Extra::make_random(axis8);
Extra::make_companion(companion8, axis8);
const mat8 reflect8{ Extra::Householder(GLSL::normalize(axis8)) };
assert(Decomposition::determinant_using_qr(reflect8) == 1); // calculate its determinant using QR decomposition
const GLSL::MatrixN<double, 8> general_matrix_multiplicatoin{ Extra::gemm(1.0f, companion8, reflect8, 2.0f, axis8) };
```


**Sample #4 - sign distance field calculation, ray-intersections and axis aligned bounding boxes:**
```cpp
// define polygon and calculate its SDF in various locations
std::array<vec2, 5> polygon{ {vec2(2.0f, 1.0f), vec2(1.0f, 2.0f), vec2(3.0f, 4.0f), vec2(5.0f, 5.0f), vec2(5.0f, 1.0f) }};
float distance = PointDistance::sdf_to_polygon<5>(polygon, vec2(2.0f, 0.0f)); // distance = 1
distance = PointDistance::sdf_to_polygon(polygon.begin(), polygon.end(), vec2(3.0f, 1.5f)); // distance = -0.5

// calculate ellipse SDF
distance = PointDistance::sdf_to_ellipse(vec2(5.0f, 0.0f), vec2(1.0f, 2.0f)); // distance = -4

// calculate triangle SDF
vec2 p0(1.0f);
vec2 p1(2.0f, 3.0f);
vec2 p2(0.0f, 3.0f);
distance = PointDistance::sdf_to_triangle(vec2(0.0f), p0, p1, p2); // distance = std::sqrt(2)

// ray-triangle intersections
vec3 p0(1.0f, 1.0f, 0.0f);
vec3 p1(2.0f, 3.0f, 0.0f);
vec3 p2(0.0f, 3.0f, 0.0f);
vec3 intersection = RayIntersections::triangle_intersect_cartesian(vec3(0.0f), GLSL::normalize(p0), p0, p1, p2); // intersection = vec3(-1.0f)
intersection = RayIntersections::triangle_intersect_barycentric(vec3(1.0f, 2.0f, 3.0f), vec3(0.0f, 0.0f, -1.0f), p0, p1, p2); // intersection = vec3(3.0f, 0.25f, 0.25f)))

// ray-ellipsoid intersections
intersections = RayIntersections::ellipsoid_intersection(vec3(0.0f, 10.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f), vec3(1.0f, 2.0f, 3.0f)); // intersection = vec2(8.0f, 12.0f)

// cloud points axis aligned bounding box
std::list<vec2> points{ {vec2(0.0f), vec2(1.0f), vec2(2.0f), vec2(-5.0f), vec2(10.0f), vec2(-17.0f, -3.0f)} };
auto aabb = AxisLignedBoundingBox::point_cloud_aabb(points.begin(), points.end());

// ellipse bounding box
vec3 center{10.0f, 10.0f, 0.0f};
vec3 axis1{2.0f, 0.0f, 0.0f};
vec3 axis2{0.0f, 2.0f, 0.0f};
auto aabb = AxisLignedBoundingBox::ellipse_aabb(center, axis1, axis2);
```


**Sample #5 - calculate polygon convex hull, bounding box, bounding circle, inscribed circle, triangulate it (earcut) and export as svg file:**
```cpp
// define polygon
std::vector<vec2> polygon{ {vec2(3.0f, 1.0f), vec2(5.0f, 1.0f), vec2(5.0f, 4.0f), vec2(4.0f, 6.0f), vec2(7.0f, 7.0f), vec2(10.0f, 7.0f), vec2(10.0f, 9.0f),
                            vec2(8.0f, 9.0f), vec2(6.0f, 10.0f), vec2(1.0f, 10.0f), vec2(1.0f, 8.0f), vec2(2.0f, 8.0f), vec2(2.0f, 6.0f), vec2(1.0f, 6.0f),
                            vec2(1.0f, 2.0f)} };

// calculate convex hull
auto convex = Algorithms2D::get_convex_hull(polygon.begin(), polygon.end());

// triangulate ("earcut" style) polygon
std::vector<std::vector<vec2>::iterator> earcut{ Algorithms2D::triangulate_polygon_earcut(polygon.begin(), polygon.end()) };

// calculate minimal area bounding rectangle
const auto obb = Algorithms2D::get_convex_hull_minimum_area_bounding_rectangle(convex);
std::vector<vec2> obbs{ {obb.p0, obb.p1, obb.p2, obb.p3} };

// close convex and obb for for drawing purposes
convex.emplace_back(convex.front());
obbs.emplace_back(obbs.front());

// scale polygons for drawing purposes
for (auto& p : polygon) {
   p = 50.0f * p + 50.0f;
}
for (auto& p : convex) {
   p = 50.0f * p + 50.0f;
}
for (auto& p : obbs) {
   p = 50.0f * p + 50.0f;
}

// calculate minimal bounding circle
const auto circle = Algorithms2D::get_minimal_bounding_circle(convex);

// calculate maximal inscribed circle
const std::vector<vec2> delaunay{ Algorithms2D::triangulate_points_delaunay(polygon.begin(), polygon.end()) };
const auto inscribed{ Algorithms2D::get_maximal_inscribed_circle(polygon.begin(), polygon.end(), delaunay) };

// export polygon and its bounding objects to SVG
svg<vec2> polygon_test_svg(650, 650);
polygon_test_svg.add_polygon(polygon.begin(), polygon.end(), "none", "black", 5.0f);
polygon_test_svg.add_polyline(convex.begin(), convex.end(), "none", "green", 1.0f);
polygon_test_svg.add_point_cloud(convex.begin(), convex.end(), 10.0f, "green", "green", 1.0f);
for (std::size_t i{}; i < earcut.size(); i += 3) {
    std::array<vec2, 3> tri{ { *(earcut[i]),
                               *(earcut[i + 1]),
                               *(earcut[i + 2]) } };
    polygon_test_svg.add_polygon(tri.begin(), tri.end(), "none", "black", 1.0f);
}
polygon_test_svg.add_polyline(obbs.begin(), obbs.end(), "none", "red", 2.0f);
polygon_test_svg.add_circle(circle.center, std::sqrt(circle.radius_squared), "none", "blue", 2.0f);
polygon_test_svg.add_circle(inscribed.center, inscribed.radius, "none", "chocolate", 2.0f);
polygon_test_svg.to_file("polygon_test_svg.svg");
```
![image](https://github.com/user-attachments/assets/73341e16-edf4-4de3-8dde-0f299bb74e03)


**Sample #6 - generate points, cluster/partition/segment them using density estimator (DBSCAN), triangulate them (delaunay) and export as svg file:**
```cpp
std::vector<vec2> points;
float sign{ 0.5f };

// cluster #1
const vec2 center(50.0f, 50.0f);
const float radius1{ 5.0f };
for (std::size_t i{}; i < 20; ++i) {
    float fi{ static_cast<float>(i) };
    points.emplace_back(vec2(center.x + radius1 * std::cos(fi) + sign * static_cast<float>(rand()) / RAND_MAX,
                             center.y + radius1 * std::sin(fi) + sign * static_cast<float>(rand()) / RAND_MAX));
    sign *= -1.0f;
}

// cluster #2
const float radius2{ 12.0f };
for (std::size_t i{}; i < 60; ++i) {
    float fi{ static_cast<float>(i) };
    points.emplace_back(vec2(center.x + radius2 * std::cos(fi) + sign * static_cast<float>(rand()) / RAND_MAX,
                             center.y + radius2 * std::sin(fi) + sign * static_cast<float>(rand()) / RAND_MAX));
    sign *= -1.0f;
}

// cluster #3
const float radius3{ 20.0f };
for (std::size_t i{}; i < 80; ++i) {
    float fi{ static_cast<float>(i) };
    points.emplace_back(vec2(center.x + radius3 * std::cos(fi) + sign * static_cast<float>(rand()) / RAND_MAX,
                             center.y + radius3 * std::sin(fi) + sign * static_cast<float>(rand()) / RAND_MAX));
    sign *= -1.0f;
}

// partition using kd-tree
SpacePartitioning::KDTree<vec2> kdtree;
const auto clusterIds = Clustering::get_density_based_clusters(points.cbegin(), points.cend(), kdtree, radius1, 3);
kdtree.clear();

// triangulate points
std::vector<vec2> delaunay{ Algorithms2D::triangulate_points_delaunay(points.begin(), points.end()) };

// draw
svg<vec2> dbscan_delaunay_test(800, 800);

for (const std::size_t i : clusterIds.clusters[0]) {
    dbscan_delaunay_test.add_circle(points[i] * 10.0f, 5.0f, "red", "black", 1.0f);
}
for (const std::size_t i : clusterIds.clusters[1]) {
    dbscan_delaunay_test.add_circle(points[i] * 10.0f, 5.0f, "blue", "black", 1.0f);
}
for (const std::size_t i : clusterIds.clusters[2]) {
    dbscan_delaunay_test.add_circle(points[i] * 10.0f, 5.0f, "green", "black", 1.0f);
}
for (const std::size_t i : clusterIds.noise) {
    dbscan_delaunay_test.add_circle(points[i] * 10.0f, 10.0f, "yellow", "black", 1.0f);
}
for (std::size_t i{}; i < delaunay.size(); i += 3) {
    std::array<vec2, 3> tri{ { (delaunay[i] * 10),
                               (delaunay[i + 1] * 10),
                               (delaunay[i + 2] * 10) } };
    dbscan_delaunay_test.add_polygon(tri.begin(), tri.end(), "none", "black", 1.0f);
}
dbscan_delaunay_test.to_file("dbscan_delaunay_test.svg");
```
![image](https://github.com/user-attachments/assets/483a4bc6-36fa-47f0-add8-8fa58ebf548f)


**Sample #7 - perform closest neighbour queries with different spatial structures (kd-tree, spatial grid) and in different shapes (circle, rectangle), calculate neighbours hull diameter, find closest pair among all points and export as svg file:**
```cpp
// place points on a plane
std::vector<vec2> points;
const std::size_t n{ 1000 };
points.reserve(n);
for (int i = 0; i < n; i++) {
    points.emplace_back(vec2(static_cast<float>(rand()) / RAND_MAX * 200.0f,
                             static_cast<float>(rand()) / RAND_MAX * 200.0f));
}

// find all points within given circle using kd-tree based nearest neighbors
SpacePartitioning::KDTree<vec2> kdtree;
kdtree.construct(points.begin(), points.end());
const vec2 circle_center(115.0f, 160.0f);
const float circle_radius{ 20.0f };
auto points_in_circle = kdtree.range_query(SpacePartitioning::RangeSearchType::Radius, circle_center, circle_radius);

// find extent of the diameter of the convex hull surrounding these points
std::vector<vec2> circle_points;
circle_points.reserve(points_in_circle.size());
for (std::size_t i{}; i < points_in_circle.size(); ++i) {
    circle_points.emplace_back(points[points_in_circle[i].second]);
}
auto circle_convex = Algorithms2D::get_convex_hull(circle_points.begin(), circle_points.end());
auto circle_diameter = Algorithms2D::get_convex_diameter(circle_convex);

// find all points within given rectangle using grid based nearest neighbors
SpacePartitioning::Grid<vec2> grid;
grid.construct(points.begin(), points.end());
const vec2 rectangle_center(50.0f, 60.0f);
const float rectangle_extent{ 25.0f };
auto points_in_rectangle = grid.range_query(SpacePartitioning::RangeSearchType::Manhattan, rectangle_center, rectangle_extent);

// find extent of the diameter of the convex hull surrounding these points
std::vector<vec2> rectangle_points;
rectangle_points.reserve(points_in_rectangle.size());
for (std::size_t i{}; i < points_in_rectangle.size(); ++i) {
    rectangle_points.emplace_back(points[points_in_rectangle[i].second]);
}
auto rectangle_convex = Algorithms2D::get_convex_hull(rectangle_points.begin(), rectangle_points.end());
auto rectangle_diameter = Algorithms2D::get_convex_diameter(rectangle_convex);

// find closest pair without using a spatial structure
auto closest_pair = Algorithms2D::get_closest_pair(points.begin(), points.end());

// export information to svg file
svg<vec2> point_query_svg(200, 200);

// points
point_query_svg.add_point_cloud(points.begin(), points.end(), 2.0f, "black", "none", 1.0f);

// circle range query
point_query_svg.add_point_cloud(circle_points.begin(), circle_points.end(), 2.0f, "red", "red", 1.0f);
point_query_svg.add_polygon(circle_convex.begin(), circle_convex.end(), "none", "red", 1.0f);
point_query_svg.add_line(circle_convex[circle_diameter.indices[0]], circle_convex[circle_diameter.indices[1]], "red", "red", 1.0f);

// rectangle range query
point_query_svg.add_point_cloud(rectangle_points.begin(), rectangle_points.end(), 2.0f, "blue", "blue", 1.0f);
point_query_svg.add_polygon(rectangle_convex.begin(), rectangle_convex.end(), "none", "blue", 1.0f);
point_query_svg.add_line(rectangle_convex[rectangle_diameter.indices[0]], rectangle_convex[rectangle_diameter.indices[1]], "red", "blue", 1.0f);

// closest pair
point_query_svg.add_line(*closest_pair.p0, *closest_pair.p1, "green", "green", 3.0f);
point_query_svg.add_circle((*closest_pair.p0 + *closest_pair.p1) / 2.0f, 5.0f, "none", "green", 1.0f);

// output
point_query_svg.to_file("point_query_svg.svg");
```
![image](https://github.com/user-attachments/assets/90581f6d-a791-45a8-9e41-81d40e014735)


## Files in repository:
+ Utilities.h - generic utilities and local STL replacements.
+ Algorithms.h - generic algorithms and local STL replacements.
+ Variadic.h - Utilities to operate and handle variadic arguments.
+ Concepts.h - useful concepts and traits.
+ Numerics.h - generic numerical utilities.
+ Numerical_algorithms.h - generic numerical algorithms on numerical collections.
+ DiamondAngle.h - operations on L1 angles.
+ Hash.h - useful hashing and pairing functions.
+ Glsl.h - A generic, modern and extensible numerical toolkit following the syntax and functionality of the GLSL shading language. compact implementation of [GLSL-CPP](https://github.com/DanIsraelMalta/GLSL-CPP).
+ Glsl_extra.h - assorted utilities using GLSL vectors and matrices and quaternions.
+ Glsl_triangle.h - triangle related functions using GLSL vectors and matrices.
+ Glsl_aabb.h - axis aligned bounding box related functions using GLSL vectors and matrices.
+ Glsls_transformation.h - spatial related operations using using GLSL vectors, matrices quaternions.
+ Glsl_solvers.h - decompositions and linear system solvers using GLSL vectors and matrices.
+ Glsl_axis_aligned_bounding_box.h - axis aligned bounding boxes functions of various geometric primitives using GLSL vectors and matrices.
+ Glsl_point_distance.h - Euclidean unsigned/signed distance functions of a point from a primitive using GLSL vectors and matrices.
+ Glsl_ray_intersections.h - ray-primitive intersection functions using GLSL vectors and matrices.
+ Glsl_algorithms_2D.h - collection of algorithms for 2D cloud points and shapes using GLSL vectors.
+ Glsl_space_partitioning.h - collection of Euclidean space partitioning data structure using GLSL vectors and matrices.
+ Glsl_clustering.h - collection of clustering algorithms using GLSL vectors and matrices.
+ Glsl_svg.h - generates scalable vector graphic schema from GLSL vectors.
+ Test.cpp - basic testing for the various files in this repository.
