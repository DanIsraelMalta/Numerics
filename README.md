# Welcome to Numerics
"Numerics" is a modern c++ (requires c++23) collection of numerical algorithms, structures and operations with particular focus on linear algebra and computational geometry.

Features include:
+ Generic, modern and extensible numerical toolkit which follows the syntax and functionality of the GLSL shading language.
+ Implementations of canonical linear algebra operations, ranging from decompositions to linear equation system set solvers.
+ A suite of computational geometry tools ranging from acceleration structures used for fast nearest neighbours queries and clustering algorithms to 2D tailored operations (oriented bounding box, enclosing circles, convex/concave hull...) etc.
+ Mandatory collection of coherent set of operations related to spatial transformations, sign distance fields, ray intersections and solution to general numerical/geometrical problems often encountered in the realms of 2D/3D geometry.

**Sample:**
```cpp
//
// GLSL swizzle syntax
//
vec2 a(1.0f, -1.0f);
vec2 b(3.2f, 2.0f);
vec2 c = a.yx + b.xy; // c = {2.2f, 3.0f}
a.xy += b.xx; // a = {4.2f, 2.2f}

vec3 d(0.0f, 1.0f, 2.0f);
vec3 e(2.0f, 1.0f, 3.0f);
vec3 f = GLSL::mix<0.5f>(d, e); // f = {1.0f, 1.0f, 2.5f}

//
// GLSL matrices
//

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

//
// basic spatial transformations
//

// define axis and angle
vec3 axis = GLSL::normalize(vec3(1.0f, 2.0f, 3.0f));
float angle{ std::numbers::pi_v<float> / 4.0f };

// create rotation matrix and quaternion from axis anf angle
const auto Rx = Transformation::rotation_matrix_from_axis_angle(vec3(1.0f, 0.0f, 0.0f), std::numbers::pi_v<float> / 2.0f);
vec4 quat = Transformation::create_quaternion_from_axis_angle(axis, angle);

// create rotation matrix from quaternion and extract rotation axis
mat3 mat = Transformation::create_rotation_matrix_from_quaternion(quat);
auto axis_from_mat = Transformation::get_axis_angle_from_rotation_matrix(mat);

//
// decompositions and linear equation system solvers
//

mat3 a(12.0, -51.0, 4.0,
       6.0,  167.0, -68.0,
       -4.0, 24.0,  -41.0);

// calculate eigenvalues
auto eigs = Decomposition::eigenvalues(a);
// eigs = {-34.196675001469160, 156.13668406196879, 16.059990939500377}

// perform QR decomposition
auto qr = Decomposition::QR_GivensRotation(a);
// qr.Q = (0.228375, -0.9790593, 0.076125,
//         0.618929, 0.084383,   -0.780901,
//         0.751513, 0.225454,   0.619999);
// qr.R = (52.545219,   0.0,       -0.0,
//         -165.895209, 70.906839,  0.0,
//          -27.328842, 31.566433, -23.015097);

// solve set of linear equation using LU decomposition
vec3 b(70.0, 12.0, 50.0);
auto solution = Solvers::SolveQR(a, b);
// solution = {3.71118, 1.74416, -3.75020}

//
// sign distance field calculation, ray-intersections and axis aligned bounding boxes
//

// define polygon and calculate its SDF in various locations
std::array<vec2, 5> polygon{ {vec2(2.0f, 1.0f), vec2(1.0f, 2.0f), vec2(3.0f, 4.0f), vec2(5.0f, 5.0f), vec2(5.0f, 1.0f) }};
float distance = PointDistance::sdf_to_polygon<5>(polygon, vec2(2.0f, 0.0f)); // distance = 1
distance = PointDistance::sdf_to_polygon<5>(polygon, vec2(3.0f, 1.5f)); // distance = -0.5

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

//
// computational geometry
//

// define polygon
std::vector<vec2> polygon{ {vec2(3.0f, 1.0f), vec2(5.0f, 1.0f), vec2(5.0f, 4.0f), vec2(4.0f, 6.0f), vec2(7.0f, 7.0f ), vec2(10.0f, 7.0f), vec2(10.0f, 9.0f),
                            vec2(8.0f, 9.0f), vec2(6.0f, 10.0f), vec2(1.0f, 10.0f), vec2(1.0f, 8.0f), vec2(2.0f, 8.0f), vec2(2.0f, 6.0f), vec2(1.0f, 6.0f),
                            vec2(1.0f, 2.0f)} };

// calculate bounding areas
const auto convex = Algorithms2D::get_convex_hull(polygon.begin(), polygon.end()); // get polygon convex hull
const auto concave = Algorithms2D::get_concave_hull(polygon.begin(), polygon.end(), 0.3f); // get polygon concave hull
const auto antipodal = Algorithms2D::get_convex_diameter(convex); // get polygon convex hull diameter
const auto obb = Algorithms2D::get_convex_hull_minimum_area_bounding_rectangle(convex); // get polygon minimal area boundingbox
const auto circle = Algorithms2D::get_minimal_bounding_circle(convex); // get polygon minimal bounding circle

// create random points on the plane
std::vector<vec2> points;
for (std::size_t i{}; i < 50; ++i) {
    points.emplace_back(vec2(static_cast<float>(rand() % 100 - 50),
                             static_cast<float>(rand() % 100 - 50)));
}

// sort points in clock wise manner and calculate its principle axis
const vec2 centroid = Algorithms2D::Internals::get_centroid(points.cbegin(), points.cend());
Algorithms2D::sort_points_clock_wise(points.begin(), points.end(), centroid);
const vec2 direction{ Algorithms2D::get_principle_axis(points.cbegin(), points.cend()) };

// create points in space
std::vector<vec2> points;
for (std::size_t i{}; i < 10000; ++i) {
    points.emplace_back(vec2(static_cast<float>(rand()) / RAND_MAX * 100.0f,
                             static_cast<float>(rand()) / RAND_MAX * 100.0f));
}

// calculate nearest neighbors
SpacePartitioning::KDTree<vec2> kdtree;
kdtree.construct(points.begin(), points.end());
auto pointsInCube = kdtree.range_query(SpacePartitioning::RangeSearchType::Manhattan, vec2(50.0f), 5.0f); // nearest neighbours in cube
auto pointsInSphere = kdtree.range_query(SpacePartitioning::RangeSearchType::Radius, vec2(50.0f), 5.0f); // nearest neighbours in sphere
const auto nearest10 = kdtree.nearest_neighbors_query(vec2(50.0f), 19); // get 19 nearest neighbors ot point at (50, 50)
kdtree.clear();

// cluster/partition points using DBSCAN with a kd-tree
const auto clusterIds = Clustering::get_density_based_clusters(points.cbegin(), points.cend(), kdtree, 1.0f, 10);
kdtree.clear();

// cluster/partition points using DBSCAN with a lattice-bin grid
SpacePartitioning::Grid<vec2> grid;
clusterIds0= Clustering::get_density_based_clusters(points.cbegin(), points.cend(), grid, 1.0f, 10);
grid.clear();

// k-mean might be faster...
clusterIds0 = Clustering::k_means(points.cbegin(), points.cend(), 3, 10, 0.01f);
```

It is also possible to export two dimensional calculation in scalable vector graphic format for debug purposes.
As an example, here is a graphic representation of the calculated top/bottom envelope of a sine signal and a different bounding shapes for a polygon:
```cpp
   // find a 2D signal envelope
   {
       // define 2D signal
       std::vector<vec2> points;
       for (std::size_t i{}; i < 200; ++i) {
           float fi{ static_cast<float>(i) };
           points.emplace_back(vec2(fi, 10.0f + 180.0f * std::abs(std::sin(fi))));
       }

       // calculate signal envelope
       const auto envelope = Algorithms2D::get_points_envelope(points.begin(), points.end());

       // export signal and envelope to SVG
       svg<vec2> envelope_test_svg(200, 200);
       for (const vec2 p : points) {
           envelope_test_svg.add_circle(p, 2.0f, "none", "black", 0.5f);
       }
       envelope_test_svg.add_polyline(envelope.top.begin(), envelope.top.end(), "none", "red", 1.0f);
       envelope_test_svg.add_polyline(envelope.bottom.begin(), envelope.bottom.end(), "none", "green", 1.0f);
       envelope_test_svg.to_file("envelope_test_svg.svg");
   }

   // find convex hull and bounding shapes
   {
       // define polygon
       std::vector<vec2> polygon{ {vec2(3.0f, 1.0f), vec2(5.0f, 1.0f), vec2(5.0f, 4.0f), vec2(4.0f, 6.0f), vec2(7.0f, 7.0f), vec2(10.0f, 7.0f), vec2(10.0f, 9.0f),
                                    vec2(8.0f, 9.0f), vec2(6.0f, 10.0f), vec2(1.0f, 10.0f), vec2(1.0f, 8.0f), vec2(2.0f, 8.0f), vec2(2.0f, 6.0f), vec2(1.0f, 6.0f),
                                    vec2(1.0f, 2.0f)} };

       // calculate convex hull
       auto convex = Algorithms2D::get_convex_hull(polygon.begin(), polygon.end());

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

       // export polygon and its bounding objects to SVG
       svg<vec2> polygon_test_svg(650, 650);
       polygon_test_svg.add_polygon(polygon.begin(), polygon.end(), "none", "black", 5.0f);
       polygon_test_svg.add_polyline(convex.begin(), convex.end(), "none", "green", 1.0f);
       for (const vec2 p : convex) {
           polygon_test_svg.add_circle(p, 10.0f, "green", "green", 1.0f);
       }
       polygon_test_svg.add_polyline(obbs.begin(), obbs.end(), "none", "red", 2.0f);
       polygon_test_svg.add_circle(circle.center, std::sqrt(circle.radius_squared), "none", "blue", 2.0f);
       polygon_test_svg.to_file("polygon_test_svg.svg");
   }
```
and here is the outcome (left image shows the signal as black circles, signal top envelope in green, signal bottom envelope in red; right image shows the polygon in black, convex hull in green, convex hull points in green circles, minimal bounding circle in blue and minimal area bounding rectangle in red):

![envelope_test_svg](https://github.com/user-attachments/assets/928004c9-6d36-4c52-b177-f0629d140632)
![image](https://github.com/user-attachments/assets/d3bed116-1f70-4dda-af5f-541473c032f3)

## Files in repository:
+ Utilities.h - generic utilities and local STL replacements.
+ Algorithms.h - generic algorithms and local STL replacements.
+ Variadic.h - Utilities to operate and handle variadic arguments.
+ Concepts.h - useful concepts and traits.
+ Numerics.h - generic numerical utilities.
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
