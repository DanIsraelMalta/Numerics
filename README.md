# Welcome to Numerics
"Numerics" is a modern c++ (requires c++23) collection of numerical algorithms, structures and operations with particular focus on linear algebra and computational geometry.

Features include:
+ Generic, modern and extensible numerical toolkit which follows the syntax and functionality of the GLSL shading language.
+ Implementations of canonical linear algebra operations including BLAS operations, decompositions, linear solvers and more.
+ Mandatory collection of coherent set of operations related to spatial transformations, sign distance fields, ray intersections and solution to general numerical/geometrical problems often encountered in the realms of computational geometry.
+ A suite of computational geometry tools ranging from acceleration structures used for fast nearest neighbours queries, clustering algorithms and 2D tailored operations for polygons and point clouds.

## Examples:

**Example #1 - calculate its medial axis joints (and appropriate inscribed circles), triangulate it "delaunay" style (and draw triangles circumcircles), and then cut it in half, triangulate it "earcut" style and calculate its minimal bounding circle and oriented bounding box:**
```cpp
// define polygons
std::vector<vec2> polygon0{ { vec2(18.0455f, -124.568f),  vec2(27.0455f, -112.568f),  vec2(26.0455f,  -91.5682f), vec2(11.0455f,   -74.5682f),
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
std::vector<vec2> polygon1(polygon0);
std::vector<vec2> polygon2(polygon0);

// rotate polygons and place them on canvas side by side
constexpr float angle{ std::numbers::pi_v<float> / 8.0f };
const float sin_angle{ std::sin(angle) };
const float cos_angle{ std::cos(angle) };
const mat2 rot(cos_angle, -sin_angle, sin_angle, cos_angle);
for (vec2& p : polygon0) {
    p = rot * p + vec2(150.0f, 200.0f);
}
for (vec2& p : polygon1) {
    p = rot * p + vec2(360.0f, 200.0f);
}
for (vec2& p : polygon2) {
    p = rot * p + vec2(550.0f, 200.0f);
}

// find 'polygon0' medial axis joints and their locally inscribed circles
const auto aabb_polygon0 = AxisLignedBoundingBox::point_cloud_aabb(polygon0.begin(), polygon0.end());
const float step{ GLSL::distance(aabb_polygon0.min, aabb_polygon0.max) / 1000.0f };
const auto medial_axis = Algorithms2D::get_approximated_medial_axis(polygon0.begin(), polygon0.end(), step);

// 'delaunay' triangulate 'polygon1' and calculate its triangles circumcircles
using circumcircle_t = decltype(Algorithms2D::Internals::get_circumcircle(vec2(), vec2()));
const auto aabb1 = AxisLignedBoundingBox::point_cloud_aabb(polygon1.begin(), polygon1.end());
const auto delaunay = Algorithms2D::triangulate_polygon_delaunay(polygon1.begin(), polygon1.end(), aabb1);
std::vector<circumcircle_t> circuumcircles;
circuumcircles.reserve(delaunay.size() / 3);
for (std::size_t i{}; i < delaunay.size(); i += 3) {
    circuumcircles.emplace_back(Algorithms2D::Internals::get_circumcircle(delaunay[i], delaunay[i + 1], delaunay[i + 2]));
}

// slice 'polygon2' by a line to half, 'earcut' triangulate it and calculate its minimal bounding box and circle
const auto centroid = Algorithms2D::Internals::get_centroid(polygon2.begin(), polygon2.end());
auto part = Algorithms2D::clip_polygon_by_infinte_line(polygon2.begin(), polygon2.end(), centroid, rot.x());
for (vec2& p : part) {
    p = rot * p + vec2(0.0f, 230.f);
}
const auto convex = Algorithms2D::get_convex_hull(part.begin(), part.end());
const auto obb = Algorithms2D::get_bounding_rectangle(convex);
const auto circumcircle = Algorithms2D::get_minimal_bounding_circle(convex);
const auto earcut = Algorithms2D::triangulate_polygon_earcut(part.begin(), part.end());

//
// export calculations to SVG file
//

svg<vec2> canvas(800, 450);
canvas.add_polygon(polygon0.begin(), polygon0.end(), "none", "black", 3.0f);
for (auto& mat : medial_axis) {
    canvas.add_circle(mat.point, std::sqrt(mat.squared_distance), "none", "blue", 1.0f);
    canvas.add_circle(mat.point, 5.0f, "red", "red", 1.0f);
}

for (std::size_t i{}; i < delaunay.size(); i += 3) {
    std::array<vec2, 3> tri{ { (delaunay[i]),
                               (delaunay[i + 1]),
                               (delaunay[i + 2]) } };
    canvas.add_polygon(tri.begin(), tri.end(), "none", "black", 3.0f);
}
for (circumcircle_t c : circuumcircles) {
    canvas.add_circle(c.center, std::sqrt(c.radius_squared), "none", "red", 1.0f);
}

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
![Image](https://github.com/user-attachments/assets/b758e995-807b-4c26-88da-f4f7f4978f8a)


**How would it look if we sampled the polygon and used the medial axis joints as centroids for k-mean clustering?**


```cpp
// sample polygon (3000 points)
const float area{ Algorithms2D::Internals::get_area(polygon.begin(), polygon.end()) };
const auto polygon_samples = Sample::sample_polygon(delaunay, area, 3000);

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
    hulls.emplace_back(Algorithms2D::get_convex_hull(clusters[j].begin(), clusters[j].end()));
}

// draw clustered samples
svg<vec2> clustered_samples_svg(400, 450);
std::vector<std::string> colors{ {"red", "green", "blue", "orange", "darkmagenta", "deeppink", "tan", "darkred",
                                  "darkolivegreen", "fuchsia", "plum", "tomato", "yellowgreen", "silver"}};
for (std::size_t j{}; j < clusters.size(); ++j) {
    clustered_samples_svg.add_point_cloud(clusters[j].begin(), clusters[j].end(), 1.0f, colors[j % colors.size()], colors[j % colors.size()], 1.0f);
}

// draw clusters convex hulls
for (auto& h : hulls) {
    h.emplace_back(h.front());
}
for (std::size_t j{}; j < hulls.size(); ++j) {
    clustered_samples_svg.add_polygon(hulls[j].begin(), hulls[j].end(), "none", "black", 1.0f);
}

clustered_samples_svg.to_file("clustered_samples.svg");
```
![Image](https://github.com/user-attachments/assets/a3410607-e4c6-4d84-bcac-5521e5f5c03e)


**Example #2 - generate two dimensional noisy patterns and cluster/segment them using density estimator (each segemtn in different color, gray is noise):**
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

// partition the space using kd-tree an combine it with a density estimator (DBSCAN) to segment to shapes and noise
SpacePartitioning::KDTree<vec2> kdtree;
kdtree.construct(points.begin(), points.end());
const float density_radius{ 0.9f * radius[0] };
const std::size_t density_points{ 3 };
const auto segments = Clustering::get_density_based_clusters(points.cbegin(), points.cend(), kdtree, density_radius, density_points);
kdtree.clear();

//
// export calculations to SVG file
//

svg<vec2> cloud_points_svg(300, 280);

std::array<std::string, 7> colors{ {"red", "green", "blue", "orange", "magenta", "deeppink", "tan"}};
for (std::size_t i{}; i < segments.clusters.size(); ++i) {
    const std::string color{ colors[i % colors.size()] };
    for (const std::size_t j : segments.clusters[i]) {
        cloud_points_svg.add_circle(points[j], 3.0f, color, "black", 1.0f);
    }
}
for (const std::size_t i : segments.noise) {
    cloud_points_svg.add_circle(points[i], 3.0f, "slategrey", "black", 1.0f);
}

cloud_points_svg.to_file("cloud_points_svg.svg");
```
![Image](https://github.com/user-attachments/assets/8b742fd2-3224-4e7b-8e0c-b5c29701e8e9)
