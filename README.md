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

### lets find the lines and circles which best fit the various clusters by randomly sampling the plane:
```cpp
svg<vec2> cloud_points_svg(300, 280);

// iterate over clusters and attempt to find if its a line or a circle using RANSAC method
for (std::size_t i{}; i < segments.clusters.size(); ++i) {
    if (segments.clusters[i].size() < 4) {
        continue;
    }

    // get cluster
    std::vector<vec2> cluster;
    cluster.reserve(segments.clusters[i].size());
    for (const std::size_t j : segments.clusters[i]) {
        cluster.emplace_back(points[j]);
    }

    // detect line via RANSAC
    PatternDetection::RansacModels::Line<vec2> line_rnsc;
    auto line = PatternDetection::ransac_pattern_detection(cluster.begin(), cluster.end(), 100, line_rnsc, 2.0f);

    // detect circle via RANSAC
    const auto aabb = AxisLignedBoundingBox::point_cloud_aabb(cluster.begin(), cluster.end());
    const vec2 range{ aabb.max - aabb.min };
    const float max_radius{ GLSL::min(aabb.max - aabb.min) / 2.0f };
    using clamped_t = PatternDetection::clamped_value<float>;
    PatternDetection::RansacModels::Circle<vec2> circle_rnsc;
    circle_rnsc.set_model({ {
            clamped_t{.value = aabb.min.x, .min = aabb.min.x, .max = aabb.max.x }, // circle center x
            clamped_t{.value = aabb.min.y, .min = aabb.min.y, .max = aabb.max.y }, // circle center y
            clamped_t{.value = 1.0f,       .min = 1.0f,       .max = max_radius }  // circle radius
        } });
    auto circle = PatternDetection::ransac_pattern_detection(cluster.begin(), cluster.end(), 500, circle_rnsc, 4.0f);

    // if a model fits the data "good enough" - draw it
    std::cout << "circle: " << circle.score << ", " << line.score << '\n';
    constexpr std::size_t min_score_for_fit{ 10 };
    if (Numerics::max(circle.score, line.score) > min_score_for_fit) {
        // is it a circle?
        if (circle.score > line.score) {
            const vec2 _center(circle.model[0], circle.model[1]);
            const float _radius{ circle.model[2] };
            cloud_points_svg.add_circle(_center, _radius, "none", "black", 3.0f);
        } // if not a circle - isit is a line...
        else {
            const vec2 p0(line.model[0], line.model[1]);
            const vec2 p1(line.model[2], line.model[3]);
            cloud_points_svg.add_line(p0, p1, "none", "black", 3.0f);
        }
    }
}

cloud_points_svg.to_file("cloud_points_svg.svg");
```
![Image](https://github.com/user-attachments/assets/a51c1e70-3ff3-44a2-b79d-ffd2509c2557)

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

## Example 4 - playing with linear algebra, filters and noise:

### lets take 3k samples from an extremely noisy signal (more than 200% noise-to-signal ratio). signal is in black, signal with noise is in red:
```cpp
// define a signal with 200% noise-to-signal ration
const float step{ 0.01f };
const float max{ 12.0f * std::numbers::pi_v<float> };
const std::size_t len{ static_cast<std::size_t>(std::ceil(max / step)) };
std::vector<float> x, y, ys;
x.reserve(len);
y.reserve(len);
ys.reserve(len);
for (std::size_t i{}; i < len; ++i) {
    const float _x{ static_cast<float>(i) * step };
    const float _y{ 1.0f + std::sin(_x) * std::cos(_x) };
    x.emplace_back(_x);
    ys.emplace_back(_y);
    y.emplace_back(_y + 2.0f * Hash::normal_distribution());
}

// export as SVG for visualization
svg<vec2> data_svg(300, 50);
for (std::size_t i{}; i < len; ++i) {
    vec2 curr(x[i] * 10.0f, 20.0f + y[i] * 10.0f);
    data_svg.add_circle(curr, 1.0f, "red", "red", 0.0f);

    curr.y = 20.0f + ys[i] * 10.0f;
    data_svg.add_circle(curr, 1.0f, "black", "black", 0.0f);
}
data_svg.to_file("data.svg");
```
![Image](https://github.com/user-attachments/assets/3712efbc-52c2-452a-bfdf-47be7e2286ca)

### can a 40-tap zero-phase linear filter (with Hanning weights) be able to retrieve the signal? filter is in green:
```cpp
// Hanning window size
constexpr std::size_t W{ 40 };

// create "hanning" weights
std::array<float, W> weights;
float sum{};
for (std::size_t i{}; i < W; ++i) {
    const float value{ 2.0f * std::numbers::pi_v<float> * static_cast<float>(i) / static_cast<float>(W) };
    sum += 1.0f - value;
    weights[i] = value;
}
for (std::size_t i{}; i < W; ++i) {
    weights[i] /= sum;
}

// zero phase filtering
std::vector<float> smooth(len);
NumericalAlgorithms::filter<W, 1>(y.begin(), y.end(), smooth.begin(), weights, std::array<float, 1>{ 1.0f });
Algoithms::reverse(smooth.begin(), smooth.end());
NumericalAlgorithms::filter<W, 1>(smooth.begin(), smooth.end(), smooth.begin(), weights, std::array<float, 1>{ 1.0f });
Algoithms::reverse(smooth.begin(), smooth.end());

// export as SVG for visualization
svg<vec2> data_svg(300, 50);
for (std::size_t i{}; i < len; ++i) {
    vec2 curr(x[i] * 10.0f, 20.0f + y[i] * 10.0f);
    data_svg.add_circle(curr, 1.0f, "red", "red", 0.0f);

    curr.y = 20.0f + ys[i] * 10.0f;
    data_svg.add_circle(curr, 1.0f, "black", "black", 0.0f);

    const vec2 curr2(x[i] * 10.0f, 10.0f + smooth[i] * 10.0f);
    data_svg.add_circle(curr2, 1.0f, "green", "green", 0.0f);
}
data_svg.to_file("data.svg");
```
![Image](https://github.com/user-attachments/assets/a2cad841-a9f3-49ef-8ea5-ba492b6470f8)

### I bet first order linear smoothing (using QR decomposition) can both retrieve most of the information (i.e. - reduce noise to bareable levels) and decimate sample size from 30k to 40. extract signal is in blue:
```cpp
// define scatter reduction parameters
constexpr std::size_t N{ 40 }; // number of bins, i.e. - number of final data points
constexpr float beta{ 1.0f };  // smoothing parameters, the larger the smoother

// get observation x-axis min, max and range
const auto xmin_max_iter{ std::minmax_element(x.begin(), x.end()) };
const float xmin{ *xmin_max_iter.first };
const float xmax{ *xmin_max_iter.second };
const float dx{ (xmax - xmin) / static_cast<float>(N) };

// group the amount and sum of observations per bin
using vec_n = GLSL::VectorN<float, N>;
using mat_n = GLSL::MatrixN<float, N>;
vec_n c(0.0f), s(0.0f);
for (std::size_t i{}; i < len; ++i) {
    const std::size_t j{ Numerics::min(1 + static_cast<std::size_t>(std::floor((x[i] - xmin) / dx)), N - 1) };
    ++c[j];
    s[j] += y[i];
}

// use first order difference smoothing to calculate reduced Y coordinate values
mat_n p(0.0f);
for (std::size_t i{}; i < N - 1; ++i) {
    p(i,     i    ) = 2.0f * beta;
    p(i,     i + 1) = -beta;
    p(i + 1, i    ) = -beta;
}
p(0,     0    ) = beta;
p(N - 1, N - 1) = beta;

mat_n diag_c(0.0f);
for (std::size_t i{}; i < N - 1; ++i) {
    diag_c(i, i) = c[i];
}
mat_n A{ diag_c + p };
vec_n z{ Solvers::SolveQR(A, s) };

// calculate reduced X axis
vec_n u;
for (std::size_t i{}; i < N; ++i) {
    u[i] = xmin - dx / 2.0f + static_cast<float>(i) * dx;
}

// export as SVG for visualization
svg<vec2> data_svg(300, 50);
for (std::size_t i{}; i < len; ++i) {
    vec2 curr(x[i] * 10.0f, 20.0f + y[i] * 10.0f);
    data_svg.add_circle(curr, 1.0f, "red", "red", 0.0f);

    curr.y = 20.0f + ys[i] * 10.0f;
    data_svg.add_circle(curr, 1.0f, "black", "black", 0.0f);

    const vec2 curr2(x[i] * 10.0f, 10.0f + smooth[i] * 10.0f);
    data_svg.add_circle(curr2, 1.0f, "green", "green", 0.0f);
}
for (std::size_t i{ 1 }; i < N; ++i) {
    const vec2 prev(u[i - 1] * 10.0f, 20.0f + z[i - 1] * 10.0f);
    const vec2 curr(u[i] * 10.0f, 20.0f + z[i] * 10.0f);
    data_svg.add_circle(curr, 2.0f, "blue", "blue", 0.0f);
    data_svg.add_line(prev, curr, "blue", "blue", 1.0f);
}
data_svg.to_file("data.svg");
```
![Image](https://github.com/user-attachments/assets/d770aa71-cdcb-4e2d-bada-1982015391e7)
