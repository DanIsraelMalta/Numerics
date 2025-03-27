# Welcome to Numerics
"Numerics" is a modern c++ (requires c++23) collection of numerical algorithms, structures and operations with particular focus on linear algebra and computational geometry.

Features include:
+ Generic, modern and extensible numerical toolkit which follows the syntax and functionality of the GLSL shading language.
+ Implementations of canonical linear algebra operations including BLAS operations, decompositions, linear solvers and more.
+ Mandatory collection of coherent set of operations related to spatial transformations, sign distance fields, ray intersections and solution to general numerical/geometrical problems often encountered in the realms of computational geometry.
+ A suite of computational geometry tools ranging from acceleration structures used for fast nearest neighbours queries, clustering algorithms and 2D tailored operations for polygons and point clouds.

## Example 1 - partition polygon to concave components:

#### define a polygon and draw it:
```cpp
// define a shape
std::vector<vec2> shape{ { vec2(18.0455f, -124.568f),  vec2(27.0455f, -112.568f),  vec2(26.0455f,  -91.5682f), vec2(11.0455f,   -74.5682f),
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
for (vec2& p : shape) {
    p = rot * p + vec2(150.0f, 200.0f);
}

// draw shape
svg<vec2> pic1(400, 450);
pic1.add_point_cloud(shape.cbegin(), shape.cend(), 5.0f, "red", "red", 1.0f);
pic1.add_polygon(shape.begin(), shape.end(), "none", "black", 3.0f);
pic1.to_file("pic1.svg");
```
![Image](https://github.com/user-attachments/assets/8a547432-b0bb-49b0-a6c0-ad6d1b2ced57)


#### uniformaly sample it:
```cpp
// uniformly sample the shape with 3000 points
constexpr std::size_t sample_size{ 3000 };
const auto aabb = AxisLignedBoundingBox::point_cloud_aabb(shape.begin(), shape.end());
const float area{ Algorithms2D::Internals::get_area(shape.begin(), shape.end()) };
const auto delaunay = Algorithms2D::triangulate_polygon_delaunay(shape.begin(), shape.end(), aabb);
auto polygon_samples = Sample::sample_polygon(delaunay, area, sample_size);

// draw sampled shape
svg<vec2> pic2(400, 450);
pic2.add_point_cloud(polygon_samples.cbegin(), polygon_samples.cend(), 1.0f, "black", "black", 0.0f);
pic2.to_file("pic2.svg");
```
![Image](https://github.com/user-attachments/assets/f6acd045-6776-4a53-bf21-ae73808dde7b)


#### calculate polygon medial axis joints (and its locally largest inscribed circles):
```cpp
// find point cloud medial axis joints
const float step{ GLSL::distance(aabb.min, aabb.max) / 1000.0f };
auto medial_axis = Algorithms2D::get_approximated_medial_axis(shape.begin(), shape.end(), step);

// draw medial axis joints (and their locally largest inscribed circles)
svg<vec2> pic3(400, 450);
//pic3.add_polygon(shape.begin(), shape.end(), "none", "black", 3.0f);
pic3.add_point_cloud(polygon_samples.cbegin(), polygon_samples.cend(), 1.0f, "black", "black", 0.0f);
for (auto& mat : medial_axis) {
    pic3.add_circle(mat.point, std::sqrt(mat.squared_distance), "none", "blue", 1.0f);
    pic3.add_circle(mat.point, 5.0f, "red", "red", 1.0f);
}
pic3.to_file("pic3.svg");
```
![Image](https://github.com/user-attachments/assets/89261c62-cd4d-4bf4-897b-04aa88363778)


#### using medial axis joints as initial centers, cluster the sampled points via k-means:
```cpp
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

// draw clusters
svg<vec2> pic4(400, 450);
std::vector<std::string> colors{ {"red", "green", "blue", "orange", "darkmagenta", "deeppink", "tan", "darkred",
                                  "darkolivegreen", "fuchsia", "plum", "tomato", "yellowgreen", "silver"} };
for (std::size_t j{}; j < clusters.size(); ++j) {
    pic4.add_point_cloud(clusters[j].begin(), clusters[j].end(), 1.0f, colors[j % colors.size()], colors[j % colors.size()], 1.0f);
}
pic4.to_file("pic4.svg");
```
![Image](https://github.com/user-attachments/assets/253cfdb3-860f-49cb-92c7-8fc4bbf6cf6e)


#### calculate clusters concave hull:
```cpp
std::vector<std::vector<vec2>> hulls;
hulls.reserve(medial_axis.size());
const std::size_t concave_hull_max_count{ 2 * sample_size / 100 };
std::cout << " concave_hull_max_count = " << concave_hull_max_count << '\n';
for (std::size_t j{}; j < medial_axis.size(); ++j) {
    const std::size_t N{ Numerics::min(clusters[j].size(), concave_hull_max_count) };
    std::cout << " N = " << N << '\n';
    hulls.emplace_back(Algorithms2D::get_concave_hull(clusters[j].begin(), clusters[j].end(), N));
}

// draw clusters concave hulls
svg<vec2> pic5(400, 450);
for (std::size_t j{}; j < hulls.size(); ++j) {
    hulls[j].emplace_back(hulls[j].front());
    pic5.add_polyline(hulls[j].begin(), hulls[j].end(), "none", colors[j % colors.size()], 2.0f);
}
pic5.to_file("pic5.svg");
```
![Image](https://github.com/user-attachments/assets/122a2934-915d-4410-8e6b-e45c7a9b6cc6)


## Example 2 - extract patterns from noisy observations:

#### generate two dimensional noisy patterns:
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


#### cluster/segment point cloud via density estimator (DBSCAN) and spatial query acceleration structure (kd-tree). mark different segments with different colors, use gray for noise:
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


#### lets find the lines and circles which best fit the various clusters by randomly sampling the data:
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


## Example 3 - messing around with digital filters and model reductions:

#### lets take 3k samples from an extremely noisy non linear signal. signal is in black, signal with noise is in gray:
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
    const float _y{ 1.0f + std::sin(_x) + 2.0f * std::cos(_x / 2.0f) };
    x.emplace_back(_x);
    ys.emplace_back(_y);
    y.emplace_back(_y + 3.0f * Hash::normal_distribution());
}

// export as SVG for visualization
svg<vec2> data_svg(300, 150);
const float bias{ 50.0f };
const float scale{ 10.0f };
for (std::size_t i{}; i < len; ++i) {
    const vec2 curr_noise(x[i] * 10.0f, bias + y[i] * scale);
    data_svg.add_circle(curr_noise, 1.0f, "none", "gray", 1.0f);
}
for (std::size_t i{ 1 }; i < len; ++i) {
    const vec2 prev(x[i - 1] * 10.0f, bias + ys[i - 1] * scale);
    const vec2 curr(x[i] * 10.0f, bias + ys[i] * scale);
    data_svg.add_line(prev, curr, "black", "black", 1.0f);
}
data_svg.to_file("data.svg");
```
![Image](https://github.com/user-attachments/assets/07a82604-542e-45fa-ab72-b2a014187725)


#### can a two stage (forward and backward) Klaman filter extract the signal? Kalman output is in green:
```cpp
// Kalman parameters
constexpr float R{ 6.0f };        // measurement noise (sigma squared)
constexpr float Q{ R / 100.0f };  // process noise (sigma squared)

// housekeeping
std::vector<float> Ppred(len, 0.0f);
std::vector<float> Pcor(len, 0.0f);
std::vector<float> ypred(len, 0.0f);
std::vector<float> y_kalman(len, 0.0f);

// forward pass initial step
float K{ Ppred[0] / (Ppred[0] + R) };
ypred[0] = y[0];
y_kalman[0] = ypred[0] + K * (y[0] - ypred[0]);
Pcor[0] = (1.0f - K) * Ppred[0];

// forward pass iterations
for (std::size_t i{ 1 }; i < len; ++i) {
    Ppred[i] = Pcor[i - 1] + Q;
    ypred[i] = y_kalman[i - 1];
    K = Ppred[i] / (Ppred[i] + R);
    y_kalman[i] = ypred[i] + K * (y[i] - ypred[i]);
    Pcor[i] = (1 - K) * Ppred[i];
}

// backward pass
for (std::size_t i{ len - 2 }; i > 1; --i) {
    const float A{ Pcor[i] / Ppred[i + 1] };
    y_kalman[i] = y_kalman[i] + A * (y_kalman[i + 1] - ypred[i + 1]);
}

// export as SVG for visualization
svg<vec2> data_svg(300, 150);
const float bias{ 50.0f };
const float scale{ 10.0f };
for (std::size_t i{ 1 }; i < len; ++i) {
    const vec2 prev(x[i - 1] * 10.0f, bias + y_kalman[i - 1] * scale);
    const vec2 curr(x[i]     * 10.0f, bias + y_kalman[i]     * scale);
    data_svg.add_line(prev, curr, "green", "green", 1.0f);
}
data_svg.to_file("data.svg");
```
![Image](https://github.com/user-attachments/assets/1b389676-1b95-42a1-8e16-813a083ca48f)


#### can a simple 40-tap zero-phase (two stage) linear filter (with Hanning weights) be able to retrieve the signal better than Kalman? filter is in red:
```cpp
// Hanning window size
constexpr std::size_t W{ 40 };

// create "hanning" weights
std::array<float, W> weights;
float sum{};
for (std::size_t i{}; i < W; ++i) {
    const float value{ 2.0f * std::numbers::pi_v<float> * static_cast<float>(i) / static_cast<float>(W) };
    sum += value;
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
svg<vec2> data_svg(300, 150);
const float bias{ 50.0f };
const float scale{ 10.0f };
for (std::size_t i{ 1 }; i < len; ++i) {
    const vec2 prev(x[i - 1] * 10.0f, bias + smooth[i - 1] * scale);
    const vec2 curr(x[i]     * 10.0f, bias + smooth[i] * scale);
    data_svg.add_line(prev, curr, "red", "red", 1.0f);
}
data_svg.to_file("data.svg");
```
![Image](https://github.com/user-attachments/assets/81e7344a-ee12-4514-b376-4fdfd24c9c78)


#### both Kalman and Hannin filters did good job filtering the noise - but they were pretty resource intensive, required several iterations over the data and will require an extra step to reduce signal sample size. I bet we can simoultanously retrieve original signal and reduce sample size from 3k to 40, in a less resource intentsive manner, by using a first order grouped linear smoothing (using QR decomposition). result is in blue:
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
svg<vec2> data_svg(300, 150);
const float bias{ 50.0f };
const float scale{ 10.0f };
for (std::size_t i{ 1 }; i < N; ++i) {
    const vec2 prev(u[i - 1] * 10.0f, bias + z[i - 1] * scale);
    const vec2 curr(u[i] * 10.0f, bias + z[i] * scale);
    data_svg.add_circle(curr, 2.0f, "blue", "blue", 0.0f);
    data_svg.add_line(prev, curr, "blue", "blue", 1.0f);
}
data_svg.to_file("data.svg");
```
![Image](https://github.com/user-attachments/assets/c4008d67-f645-451e-86fe-1b982ae7ba4d)


#### can we simplify the first order grouped linear smoothing operation by approximating QR decomposition with a simple median calculation (which will reduce computational resources even more)? result in orange:
```cpp
const std::size_t WINDOW{ len / 40 };
std::vector<float> ismooth;
std::vector<float> xsmooth;
ismooth.reserve(len / WINDOW);
xsmooth.reserve(len / WINDOW);
for (std::size_t i{ WINDOW + 1 }; i < len; i += WINDOW) {
    std::vector<float> spn(y.begin() + i - WINDOW, y.begin() + i);
    const float e{ NumericalAlgorithms::median(spn.begin(), spn.end(),[](float a, float b) -> bool {return a < b; }) };

    std::int32_t p{};
    std::int32_t n{};
    float t{};
    for (std::size_t j{}; j < WINDOW; ++j) {
        if (const float sj{ spn[j] };
            sj > e) {
            ++p;
        }
        else if (sj < e) {
            ++n;
            t += sj;
        }
    }
    t -= e;
    t = std::abs(t);

    ismooth.emplace_back(e + static_cast<float>(p - n) * t / static_cast<float>(WINDOW * WINDOW));
    xsmooth.emplace_back(x[i - WINDOW / 2]);
}

// export as SVG for visualization
svg<vec2> data_svg(300, 150);
const float bias{ 50.0f };
const float scale{ 10.0f };
for (std::size_t i{ 1 }; i < ismooth.size(); ++i) {
    const vec2 prev(xsmooth[i - 1] * 10.0f, bias + ismooth[i - 1] * scale);
    const vec2 curr(xsmooth[i]     * 10.0f, bias + ismooth[i] * scale);
    data_svg.add_circle(curr, 2.0f, "orange", "orange", 0.0f);
    data_svg.add_line(prev, curr, "orange", "orange", 1.0f);
}
data_svg.to_file("data.svg");
```
![Image](https://github.com/user-attachments/assets/7daaa60a-7e9c-4004-a9f9-9ee1292f7174)
