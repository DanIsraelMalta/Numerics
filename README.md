# Welcome to Numerics
"Numerics" is a modern c++ (requires c++23) collection of numerical algorithms, structures and operations with particular focus on linear algebra and computational geometry.

Features include:
+ Generic, modern and extensible numerical toolkit which follows the syntax and functionality of the GLSL shading language.
+ Implementations of canonical linear algebra operations including BLAS operations, decompositions, linear solvers and more.
+ Mandatory collection of coherent set of operations related to spatial transformations, sign distance fields, ray intersections and solution to general numerical/geometrical problems often encountered in the realms of computational geometry.
+ A suite of computational geometry tools ranging from acceleration structures used for fast nearest neighbours queries, clustering algorithms and 2D tailored operations for polygons and point clouds.

## Example 1 - messing around with polygons:

### define a polygon, triangulate it (delaunay) and use it to calculate the set of circles which cumulatively encircle all polygon vertices and segments:
```cpp
// define polygons
std::vector<vec2> polygon{ { vec2(18.0455f, -124.568f),  vec2(27.0455f, -112.568f),  vec2(26.0455f,  -91.5682f), vec2(11.0455f,   -74.5682f),
                             vec2(11.0455f, -61.5682f),  vec2(60.0455f, -55.5682f),  vec2(78.0455f,  -100.568f), vec2(78.0455f,   -119.568f),
                             vec2(102.045f, -120.568f),  vec2(102.045f, -101.568f),  vec2(83.0455f,  -36.5682f), vec2(60.0455f,   -26.5682f),
                             vec2(37.0455f, -27.5682f),  vec2(37.0455f,  42.4318f),  vec2(67.0455f,  101.432f),  vec2(76.0455f,   158.432f),
                             vec2(102.045f,  161.432f),  vec2(101.045f,  177.432f),  vec2(56.0455f,  177.432f),  vec2(56.0455f,   155.432f),
                             vec2(44.0455f,  110.432f),  vec2(-1.95454f, 66.4318f),  vec2(-43.9545f, 110.432f),  vec2(-55.9545f,  155.432f),
                             vec2(-55.9545f, 177.432f),  vec2(-100.955f, 177.432f),  vec2(-101.955f, 161.432f),  vec2(-75.9545f,  158.432f),
                             vec2(-66.9545f, 101.432f),  vec2(-36.9545f, 42.4318f),  vec2(-36.9545f, -27.5682f), vec2(-59.9545f,  -26.5682f),
                             vec2(-82.9545f, -36.5682f), vec2(-101.955f, -101.568f), vec2(-101.955f, -120.568f), vec2(-77.9545f,  -119.568f),
                             vec2(-77.9545f, -100.568f), vec2(-59.9545f, -55.5682f), vec2(-10.9545f, -61.5682f), vec2(-10.9545f,  -74.5682f),
                             vec2(-25.9545f, -91.5682f), vec2(-26.9545f, -112.568f), vec2(-17.9545f, -124.568f), vec2(0.0454559f, -128.568f) } };

// rotate polygons and place them on canvas side by side
constexpr float angle{ std::numbers::pi_v<float> / 8.0f };
const float sin_angle{ std::sin(angle) };
const float cos_angle{ std::cos(angle) };
const mat2 rot(cos_angle, -sin_angle, sin_angle, cos_angle);
for (vec2& p : polygon) {
    p = rot * p + vec2(150.0f, 200.0f);
}

// 'delaunay' triangulate 'polygon'
using circumcircle_t = decltype(Algorithms2D::Internals::get_circumcircle(vec2(), vec2()));
const auto aabb = AxisLignedBoundingBox::point_cloud_aabb(polygon.begin(), polygon.end());
const auto delaunay = Algorithms2D::triangulate_polygon_delaunay(polygon.begin(), polygon.end(), aabb);

// calculate triangles circumcircles
std::vector<circumcircle_t> circuumcircles;
circuumcircles.reserve(delaunay.size() / 3);
for (std::size_t i{}; i < delaunay.size(); i += 3) {
    circuumcircles.emplace_back(Algorithms2D::Internals::get_circumcircle(delaunay[i], delaunay[i + 1], delaunay[i + 2]));
}

// export as SVG for visualization
svg<vec2> canvas(400, 450);
canvas.add_polygon(polygon0.begin(), polygon0.end(), "none", "black", 3.0f);
for (std::size_t i{}; i < delaunay.size(); i += 3) {
    std::array<vec2, 3> tri{ { (delaunay[i]),
                               (delaunay[i + 1]),
                               (delaunay[i + 2]) } };
    canvas.add_polygon(tri.begin(), tri.end(), "none", "black", 3.0f);
}
for (circumcircle_t c : circuumcircles) {
    canvas.add_circle(c.center, std::sqrt(c.radius_squared), "none", "red", 1.0f);
}
canvas.to_file("canvas.svg");
```
![Image](https://github.com/user-attachments/assets/ddd44817-87da-4c12-8360-52b722aff354)


### calculate its medial axis joints and use it to find a set of locally largest inscribed circles:
```cpp
// find 'polygon' medial axis joints and their locally inscribed circles
const float step{ GLSL::distance(aabb.min, aabb.max) / 1000.0f };
const auto medial_axis = Algorithms2D::get_approximated_medial_axis(polygon.begin(), polygon.end(), step);

// export as SVG for visualization
svg<vec2> canvas(400, 450);
canvas.add_polygon(polygon0.begin(), polygon0.end(), "none", "black", 3.0f);
for (auto& mat : medial_axis) {
    canvas.add_circle(mat.point, std::sqrt(mat.squared_distance), "none", "blue", 1.0f);
    canvas.add_circle(mat.point, 5.0f, "red", "red", 1.0f);
}
canvas.to_file("canvas.svg");
```
![Image](https://github.com/user-attachments/assets/9cee1967-b4fd-44b8-b672-4494d104fbf5)


### uniformally sample the polygon and use the medial axis joints as initial seeds to partition the polygon, via k-means, to different components:
```cpp
 // sample polygon (1500 points)
 constexpr std::size_t sample_size{ 1500 };
 const float area{ Algorithms2D::Internals::get_area(polygon.begin(), polygon.end()) };
 auto polygon_samples = Sample::sample_polygon(delaunay, area, sample_size);

 // merge close medial axis joints
 for (std::size_t i{}; i < medial_axis.size(); ++i) {
     for (std::size_t j{}; j < medial_axis.size(); ++j) {
         if (GLSL::distance(medial_axis[i].point, medial_axis[j].point) < 10.0f) {
             Utilities::swap(medial_axis[j], medial_axis.back());
             medial_axis.pop_back();
         }
     }
 }

 // cluster sampled points, with medial axis joints as initial centers, using k-means
 std::vector<vec2> intial_centers;
 intial_centers.reserve(medial_axis.size());
 for (const auto& ma : medial_axis) {
     intial_centers.emplace_back(ma.point);
 }
 const auto clusterIds = Clustering::k_means(polygon_samples.cbegin(), polygon_samples.cend(), medial_axis.size(), 20, 0.01f, intial_centers);

 // extract clusters
 std::vector<std::vector<vec2>> clusters(medial_axis.size(), std::vector<vec2>{});
 for (std::size_t j{}; j < medial_axis.size(); ++j) {
     clusters[j].reserve(clusterIds[j].size());
     for (const std::size_t i : clusterIds[j]) {
         clusters[j].emplace_back(polygon_samples[i]);
     }
 }

 // calculate clusters convex hulls
std::vector<std::vector<vec2>> hulls;
 hulls.reserve(medial_axis.size());
 for (std::size_t j{}; j < medial_axis.size(); ++j) {
     hulls.emplace_back(Algorithms2D::get_convex_hull(clusters[j].begin(), clusters[j].end(), 17));
 }

// export as SVG for visualization
svg<vec2> canvas(400, 450);
std::vector<std::string> colors{ {"red", "green", "blue", "orange", "darkmagenta", "deeppink", "tan", "darkred",
                                  "darkolivegreen", "fuchsia", "plum", "tomato", "yellowgreen", "silver"}};
for (std::size_t j{}; j < clusters.size(); ++j) {
    canvas.add_point_cloud(clusters[j].begin(), clusters[j].end(), 1.0f, colors[j % colors.size()], colors[j % colors.size()], 1.0f);
}
 
// draw clusters concave hulls
for (const auto& h : hulls) {
    canvas.add_polyline(h.begin(), h.end(), "none", "black", 2.0f);
}
canvas.to_file("canvas.svg");
```
![Image](https://github.com/user-attachments/assets/5c7e2b17-e305-4e78-98d3-2585ef5e59cf)


### slice the polygon along its longitudianl axis, triangulate it ("earcut") and calculate its oriented bounding box and circumcircle:
```cpp
const auto centroid = Algorithms2D::Internals::get_centroid(polygon.begin(), polygon.end());
auto part = Algorithms2D::clip_polygon_by_infinte_line(polygon.begin(), polygon.end(), centroid, rot.x());
const auto convex = Algorithms2D::get_convex_hull(part.begin(), part.end());
const auto obb = Algorithms2D::get_bounding_rectangle(convex);
const auto circumcircle = Algorithms2D::get_minimal_bounding_circle(convex);
const auto earcut = Algorithms2D::triangulate_polygon_earcut(part.begin(), part.end());

export as SVG for visualization
svg<vec2> canvas(400, 450);
for (std::size_t i{}; i < earcut.size(); i += 3) {
    std::array<vec2, 3> tri{ { *(earcut[i]),
                               *(earcut[i + 1]),
                               *(earcut[i + 2]) } };
    canvas.add_polygon(tri.begin(), tri.end(), "none", "black", 2.0f);
}
std::vector<vec2> obbs{ {obb.p0, obb.p1, obb.p2, obb.p3, obb.p0} };
canvas.add_polyline(obbs.begin(), obbs.end(), "none", "red", 2.0f);
canvas.add_circle(circumcircle.center, std::sqrt(circumcircle.radius_squared), "none", "blue", 2.0f);
canvas.to_file("canvas.svg");
```
![Image](https://github.com/user-attachments/assets/bd580bf3-c794-4f7c-84e1-194ecf77fb7e)


## Example 2 - messing around with patterns and noise:

### generate two dimensional noisy patterns:
```cpp
// generate noisy patterns
std::vector<vec2> points;
float sign{ 3.0f };
vec2 center(150.0f, 150.0f);
const std::array<std::size_t, 3> count{ { 75, 125, 200} };
const std::array<float, 3> radius{ { 15.0f, 40.0f, 75.0f} };
for (std::size_t i{}; i < radius.size(); ++i) {
    const float r{ radius[i] };
    for (std::size_t j{}; j < count[i]; ++j) {
        float fi{ static_cast<float>(j) };
        points.emplace_back(vec2(center.x + r * std::cos(fi) + sign * static_cast<float>(rand()) / RAND_MAX,
                                 center.y + r * std::sin(fi) + sign * static_cast<float>(rand()) / RAND_MAX));
        sign *= -1.0f;
    }
}

center = vec2(250.0f, 250.0f);
for (std::size_t j{}; j < 50; ++j) {
    float fi{ static_cast<float>(j) };
    points.emplace_back(vec2(center.x + radius[0] * std::cos(fi) + sign * static_cast<float>(rand()) / RAND_MAX,
                             center.y + radius[0] * std::sin(fi) + sign * static_cast<float>(rand()) / RAND_MAX));
    sign *= -1.0f;
}

center = vec2(110.0f, 250.0f);
for (std::size_t j{}; j < 50; ++j) {
    float fi{ static_cast<float>(j) };
    points.emplace_back(vec2(center.x + radius[0] * std::cos(fi) + sign * static_cast<float>(rand()) / RAND_MAX,
                             center.y + radius[0] * std::sin(fi) + sign * static_cast<float>(rand()) / RAND_MAX));
    sign *= -1.0f;
}

sign = 5.0f;
constexpr float deg2rad{ 3.141592653589f / 180.0f };
float tanAngle{ std::tan(10.0f * deg2rad) };
for (std::size_t x{ 40 }; x < 250; x += 5) {
    const float xf{ static_cast<float>(x) };
    points.emplace_back(vec2(xf, tanAngle * xf + sign * static_cast<float>(rand()) / RAND_MAX));
    sign *= -1.0f;
}

tanAngle = std::tan(80.0f * deg2rad);
for (std::size_t x{}; x < 75; x += 1) {
    const float xf{ static_cast<float>(x) };
    points.emplace_back(vec2(xf, tanAngle * xf + sign * static_cast<float>(rand()) / RAND_MAX));
    sign *= -1.0f;
}

for (std::size_t i{}; i < 100; ++i) {
    points.emplace_back(vec2(static_cast<float>(rand()) / RAND_MAX * 300.0f,
                             static_cast<float>(rand()) / RAND_MAX * 300.0f));
}

// export as SVG for visualization
svg<vec2> cloud_points_svg(300, 280);
cloud_points_svg.add_point_cloud(points.cbegin(), points.cend(), 0.5f, "black", "black", 1.0f);
cloud_points_svg.to_file("cloud_points_svg.svg");
```
![Image](https://github.com/user-attachments/assets/0b4df27a-f394-484f-bccd-55487c274d8f)

## cluster/segment point cloud via density estimator (DBSCAN) and spatial query acceleration structure (kd-tree). mark different segments with different colors, use gray for noise:
```cpp
// partition space using kd-tree
SpacePartitioning::KDTree<vec2> kdtree;
kdtree.construct(points.begin(), points.end());

// use density estimator (DBSCAN) to segment/cluster the point cloud and identify "noise" 
const float density_radius{ 0.9f * radius[0] };
const std::size_t density_points{ 3 };
const auto segments = Clustering::get_density_based_clusters(points.cbegin(), points.cend(), kdtree, density_radius, density_points);
kdtree.clear();

// export as SVG for visualization
svg<vec2> cloud_points_svg(300, 280);
cloud_points_svg.add_point_cloud(points.cbegin(), points.cend(), 0.5f, "black", "black", 1.0f);

std::array<std::string, 7> colors{ {"red", "green", "blue", "orange", "magenta", "deeppink", "tan"}};
for (std::size_t i{}; i < segments.clusters.size(); ++i) {
    const std::string color{ colors[i % colors.size()] };
    for (const std::size_t j : segments.clusters[i]) {
        cloud_points_svg.add_circle(points[j], 3.0f, "none", color, 1.0f);
    }
}
for (const std::size_t i : segments.noise) {
    cloud_points_svg.add_circle(points[i], 3.0f, "none", "slategrey", 1.0f);
}

cloud_points_svg.to_file("cloud_points_svg.svg");
```
![Image](https://github.com/user-attachments/assets/0e44e285-d21e-441a-be36-3d1343be036a)


## Example 3 - messing around with samples and shape characteristics:

### uniformly sample different shapes:
```cpp
// how many points to sample
constexpr std::size_t count{ 1000 };
std::vector<vec2> points;
points.reserve(5 * count);

// create a circle and uniformly sample 1000 points within it
const vec2 center(130.0f, 130.0f);
const float radius{ 50.0f };
for (std::size_t i{}; i < count; ++i) {
    points.emplace_back(Sample::sample_circle(center, radius));
}

// creat a triangle and uniformly sample 1000 points within it
const vec2 v0(20.0f, 220.0f);
const vec2 v1(20.0f, 520.0f);
const vec2 v2(500.0f, 400.0f);
for (std::size_t i{}; i < count; ++i) {
    points.emplace_back(Sample::sample_triangle(v0, v1, v2));
}

// create a parallelogram and uniformly sample 2000 points within it
const vec2 p0(240.0f, 240.0f);
const vec2 p1(510.0f, 390.0f);
const vec2 p2(710.0f, 190.0);
const vec2 p3(310.0f, 90.0f);
for (std::size_t i{}; i < 2 * count; ++i) {
    points.emplace_back(Sample::sample_parallelogram(p0, p1, p2, p3));
}

// create an ellipse and uniformly sample 1000 points within it
const vec2 ellipse_center(160.0f, 220.0f);
const float xAxis{ 70.0f };
const float yAxis{ 30.0f };
std::vector<vec2> sampled_ellipse_points;
sampled_ellipse_points.reserve(count);
for (std::size_t i{}; i < count; ++i) {
    points.emplace_back(Sample::sample_ellipse(ellipse_center, xAxis, yAxis));
}

// export as SVG for visualization
svg<vec2> sample_test(800, 800);
sample_test.add_point_cloud(points.begin(), points.end(), 1.0f, "black", "none", 0.0f);
sample_test.to_file("sample_test.svg");
```
![Image](https://github.com/user-attachments/assets/1ab6885f-c1ee-43ba-ad43-7055d4cd6a70)


### cluster/segment point cloud via density estimator (DBSCAN) and spatial query acceleration structure (bin-lattice grid):
```cpp
// partition space using bin-lattice grid
SpacePartitioning::Grid<vec2> grid;
grid.construct(points.begin(), points.end());

// use density estimator (DBSCAN) to segment/cluster the point cloud
const float density_radius{ 0.3f * radius };
const std::size_t density_points{ 4 };
const auto segments = Clustering::get_density_based_clusters(points.begin(), points.end(), grid, density_radius, density_points);
grid.clear();

// export as SVG for visualization
svg<vec2> sample_test(800, 800);
std::array<std::string, 4> colors{ {"red", "green", "blue", "orange"} };
const std::size_t cluster_count{ segments.clusters.size() };
for (std::size_t i{}; i < cluster_count; ++i) {
    // get cluster points
    std::vector<vec2> cluster_points;
    cluster_points.reserve(segments.clusters[i].size());
    for (const std::size_t j : segments.clusters[i]) {
        cluster_points.emplace_back(points[j]);
    }

    // draw points
    sample_test.add_point_cloud(cluster_points.begin(), cluster_points.end(), 1.0f, colors[i % 4], "none", 0.0f);
}
sample_test.to_file("sample_test.svg");
```
![Image](https://github.com/user-attachments/assets/99cc53bf-eb86-4937-b12b-906551cff7b0)


### find shapes characteristics (concave hull, principle axis) and check if it matches the sampled shapes:
```cpp
// prepare drawing canvas
std::array<std::string, 4> colors{ {"red", "green", "blue", "orange"} };
svg<vec2> sample_test(800, 800);

// calculate clusters characteristics (concave hull, principle axis)
const std::size_t cluster_count{ segments.clusters.size() };
for (std::size_t i{}; i < cluster_count; ++i) {
    // get cluster points
    std::vector<vec2> cluster_points;
    cluster_points.reserve(segments.clusters[i].size());
    for (const std::size_t j : segments.clusters[i]) {
        cluster_points.emplace_back(points[j]);
    }

    // calculate points concave hull
    const std::size_t N{ cluster_points.size() / 20 };
    auto cluster_concave = Algorithms2D::get_concave_hull(cluster_points.begin(), cluster_points.end(), N);

    // calculate points principle axis
    const vec2 centroid{ Algorithms2D::Internals::get_centroid(cluster_points.begin(), cluster_points.end()) };
    const vec2 axis{ Algorithms2D::get_principle_axis(cluster_points.begin(), cluster_points.end(), centroid) };
    const std::array<vec2, 2> principle_axis_segments{ {centroid, centroid + 70.0f * axis} };

    // draw it all
    sample_test.add_point_cloud(cluster_points.begin(), cluster_points.end(), 1.0f, colors[i % 4], "none", 0.0f);
    cluster_concave.emplace_back(cluster_concave.front());
    sample_test.add_polygon(cluster_concave.begin(), cluster_concave.end(), "none", colors[i % 4], 2.0f);
    sample_test.add_circle(centroid, 4.0, "black", "black", 0.0);
    sample_test.add_polyline(principle_axis_segments.begin(), principle_axis_segments.end(), "black", "black", 2.0f);
}

sample_test.to_file("sample_test.svg");
```
![Image](https://github.com/user-attachments/assets/6a9d22ee-f257-4001-89e3-ea8b80a81090)
