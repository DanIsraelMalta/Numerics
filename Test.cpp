#include <assert.h>
#include <numbers>
#include <array>
#include <iostream>
#include <list>
#include <chrono>
#include <string>
#include <unordered_map>
#include <numeric>
#include "Algorithms.h"
#include "DiamondAngle.h"
#include "Hash.h"
#include "Variadic.h"
#include "Numerics.h"
#include "Numerical_algorithms.h"
#include "Glsl.h"
#include "Glsl_extra.h"
#include "Glsl_solvers.h"
#include "Glsl_triangle.h"
#include "Glsl_axis_aligned_bounding_box.h"
#include "Glsl_point_distance.h"
#include "Glsl_ray_intersections.h"
#include "Glsls_transformation.h"
#include "GLSL_algorithms_2D.h"
#include "Glsl_space_partitioning.h"
#include "GLSL_clustering.h"
#include "Glsl_sampling.h"
#include "Glsl_svg.h"
#include "Glsl_pattern_identification.h"

void test_diamond_angle() {
    // test atan2
    static_assert(static_cast<std::int32_t>(DiamondAngle::atan2(0.0f, 0.0f)) == 0);
    static_assert(static_cast<std::int32_t>(DiamondAngle::atan2(1.0f, 1.0f) * 10) == 5);
    static_assert(static_cast<std::int32_t>(DiamondAngle::atan2(1.0f, 0.0f) * 10) == 10);
    static_assert(static_cast<std::int32_t>(DiamondAngle::atan2(1.0f, -1.0f) * 10) == 15);
    static_assert(static_cast<std::int32_t>(DiamondAngle::atan2(0.0f, -1.0f) * 10) == 20);
    static_assert(static_cast<std::int32_t>(DiamondAngle::atan2(-1.0f, -1.0f) * 10) == 25);
    static_assert(static_cast<std::int32_t>(DiamondAngle::atan2(-1.0f, 0.0f) * 10) == 30);
    static_assert(static_cast<std::int32_t>(DiamondAngle::atan2(-1.0f, 1.0f) * 10) == 35);

    // test radianToDiamondAngle
    assert(static_cast<std::uint32_t>(DiamondAngle::radianToDiamondAngle(std::numbers::pi / 2.0) * 10.0) == 10);
    assert(static_cast<std::uint32_t>(DiamondAngle::radianToDiamondAngle(std::numbers::pi) * 10.0) == 19);

    // test diamondAngleToRadian
    assert(static_cast<std::int32_t>(DiamondAngle::diamondAngleToRadian(DiamondAngle::atan2(0.0f, -1.0f)) * 1000) == 3141); // pi
}

void test_hash() {
    // test SzudzikValueFromPair
    //static_assert(Hash::SzudzikValueFromPair<12u, 37u>() == 1381u);
    //static_assert(Hash::SzudzikValueFromPair(12u, 37u) == 1381u);

    // test SzudzikPairFromValue
    //assert(Hash::SzudzikPairFromValue(1381u).x == 12u);
    //assert(Hash::SzudzikPairFromValue(1381u).y == 37u);
    //assert(Hash::SzudzikPairFromValue<1381u>().x == 12u);
    //assert(Hash::SzudzikPairFromValue<1381u>().y == 37u);

    // test normal_distribution
    {
        std::vector<float> a(100000, 0.0f);
        float _max{ -2.0f };
        float _min{ 2.0f };
        for (std::size_t i{}; i < 100000; ++i) {
            a[i] = Hash::normal_distribution();
            if (a[i] > _max) {
                _max = a[i];
            }
            if (a[i] < _min) {
                _min = a[i];
            }
        }
        assert(1.0f - _max < 0.05f);
        assert(1.0f + _min < 0.05f);

        float mean{};
        float count{};
        for (const float _a : a) {
            ++count;
            const float delta{ _a - mean };
            mean += delta / count;
        }
        assert(std::abs(mean) < 0.002);
    }

    // test rand64
    {
        std::int32_t above_limit{};
        const std::int32_t lottery{ 100 };
        for (std::size_t i{}; i < 100; ++i) {
            const double r{ Hash::rand64() };
            if (std::abs(r) > static_cast<double>(std::numeric_limits<float>::max())) {
                ++above_limit;
            }
        }
        assert(above_limit >= lottery / 4);
    }
}

void test_variadic() {
    // test forEach
    int a{ 1 };
    int b{ 2 };
    int c{ 3 };
    Variadic::forEach([](int& x) { x*=2; }, a, b, c);
    assert(a == 2);
    assert(b == 4);
    assert(c == 6);

    // test lowerThan
    static_assert(Variadic::lowerThan<2>(0, 1, 1, 0, -5));
    static_assert(!Variadic::lowerThan<2>(0, 1, 1, 0, 5));
    static_assert(Variadic::lowerThan(2, 0, 1, 1, 0, -5));
    static_assert(!Variadic::lowerThan(2, 0, 1, 1, 0, 5));

    // test greaterThan
    static_assert(!Variadic::greaterThan<2>(0, 1, 1, 0, -5));
    static_assert(Variadic::greaterThan<2>(10, 11, 11, 10, 5));
    static_assert(!Variadic::greaterThan(2, 0, 1, 1, 0, -5));
    static_assert(Variadic::greaterThan(2, 10, 11, 11, 10, 5));

    // test within
    static_assert(Variadic::within<-7, 2>(0, 1, 1, 0, -5));
    static_assert(!Variadic::within<2, 4>(10, 11, 11, 10, 5));
    static_assert(Variadic::within(-7, 2, 0, 1, 1, 0, -5));
    static_assert(!Variadic::within(2, 4, 10, 11, 11, 10, 5));
}

void test_numerics() {
    // test NumBitFields
    assert(Numerics::NumBitFields(137) == 3);
    assert(Numerics::NumBitFields<137>() == 3);
    
    // test ulp_distance_between
    assert(Numerics::ulp_distance_between(0.0f, 0.0f) == 0);
    assert(Numerics::ulp_distance_between(0.0f, 1.0f) == 1065353216);

    // test areEquals
    static_assert(!Numerics::areEquals(1.0f, 1.001f));
    static_assert(Numerics::areEquals(1.0f, 1.00000001f));

    // test ulp_magnitude_within_range
    assert(static_cast<std::uint64_t>(Numerics::ulp_magnitude_within_range(1.0f, 100.0f) * 100000000000000u) == 762939456u);
    assert(static_cast<std::uint64_t>(Numerics::ulp_magnitude_within_range(1.0f, 1000.0f) * 10000000000000u) == 610351552u);

    // test count_equidistant_floats
    assert(Numerics::count_equidistant_floats(1.0f, 100.0f) == 12976128);
    assert(Numerics::count_equidistant_floats(1.0f, 1000.0f) == 16367616);

    // test in_range
    assert(Numerics::in_range<float>(1u));
    assert(!Numerics::in_range<float>(100000000u));

    // test sign
    assert(static_cast<std::int32_t>(Numerics::sign(3.0f)) == 1);
    assert(static_cast<std::int32_t>(Numerics::sign(-3.0f)) == -1);

    // test isEven
    static_assert(Numerics::isEven(3u));
    static_assert(!Numerics::isEven(4u));
    static_assert(Numerics::isEven<3>());
    static_assert(!Numerics::isEven<4>());

    // test isPowerOfTwo
    static_assert(Numerics::isPowerOfTwo(4u));
    static_assert(!Numerics::isPowerOfTwo(3u));
    static_assert(Numerics::isPowerOfTwo<4>());
    static_assert(!Numerics::isPowerOfTwo<3>());

    // test alignToNext
    static_assert(Numerics::alignToNext(2, 16) == 16);
    static_assert(Numerics::alignToNext<16>(2) == 16);
    static_assert(Numerics::alignToNext<2, 16>() == 16);

    // test alignToPrev
    static_assert(Numerics::alignToPrev(15, 4) == 12);
    static_assert(Numerics::alignToPrev<4>(15) == 12);
    static_assert(Numerics::alignToPrev<15, 4>() == 12);

    // test roundedUpDivision & roundedLowDivision
    static_assert(Numerics::roundedUpDivision(8, 3) == Numerics::roundedLowDivision(8, 3) + 1);
    static_assert(Numerics::roundedUpDivision<8, 3>() == Numerics::roundedLowDivision<8, 3>() + 1);

    // test clampCircular
    assert(static_cast<std::size_t>(Numerics::clampCircular(6.0, 2.0, 4.0)) == 3u);

    // test angleDifference
    assert(static_cast<std::int32_t>(Numerics::angleDifference(0.104720, 0.034907) - 0.069813) == 0);

    // test allPositive
    static_assert(Numerics::allPositive(6, 4, 5, 8));
    static_assert(!Numerics::allPositive(6, 4, 5, 8, 0, 6));

    // test min
    static_assert(Numerics::min(4, 8) == 4);
    static_assert(Numerics::min(6, 4, 5, 8) == 4);
    static_assert(Numerics::min(6, 4, 5, 8, 0, 6) == 0);
    static_assert(Numerics::min(6, 4, 5, 8, 0, -3, 6) == -3);

    // test max
    static_assert(Numerics::max(6, 8) == 8);
    static_assert(Numerics::max(6, 4, 5, 8) == 8);
    static_assert(Numerics::max(6, 4, 5, 8, 0, 16) == 16);
    static_assert(Numerics::max(6, 4, 5, -8, 0, -3, 6) == 6);

    // test dot
    static_assert(Numerics::dot(3, 4) == 25);

    // test norm
    assert(static_cast<std::int32_t>(Numerics::norm(3.0, 4.0)) == 5);
    assert(static_cast<std::int32_t>(Numerics::norm(3.0)) == 3);

    // test clamp
    static_assert(Numerics::clamp(3, 4, 5) == 4);
    static_assert(Numerics::clamp(-2, -8, -4) == -4);
    static_assert(Numerics::clamp<4, 5>(3) == 4);
    static_assert(Numerics::clamp<-8, -4>(-2) == -4);

    // test quadratic equation solver
    {
        auto sol = Numerics::SolveQuadratic(3.0f, -5.0f, 2.0f);
        assert(static_cast<std::uint32_t>(sol.x1 * 10000) == 6666);
        assert(static_cast<std::uint32_t>(sol.x2 * 10000) == 10000);

        sol = Numerics::SolveQuadratic(4.0f, -5.0f, -12.0f);
        assert(static_cast<std::int32_t>(sol.x1 * 10000) == -12163);
        assert(static_cast<std::int32_t>(sol.x2 * 10000) == 24663);
    }

    // test cubic equation solver (x^3 + b*x^2 + c*x + d = 0)
    {
        auto sol = Numerics::SolveCubic(2.0f, 5.0f, -8.0f);
        assert(static_cast<std::int32_t>(sol[0] * 10000) == 9999);
        assert(static_cast<std::int32_t>(sol[1] * 10000) == 0);
        assert(static_cast<std::int32_t>(sol[2] * 10000) == -15000);
        assert(static_cast<std::int32_t>(sol[3] * 10000) == 23979);
        assert(static_cast<std::int32_t>(sol[4] * 10000) == -15000);
        assert(static_cast<std::int32_t>(sol[5] * 10000) == -23979);
    }

    // test fminbnd
    {
        const auto func1 = [](const double x) -> double {
            double f{};
            for (std::int32_t i{ -10 }; i < 10; ++i) {
                const double k{ static_cast<double>(i) };
                f += (k + 1) * (k + 1) * std::cos(k * x) * std::exp(-(k * k) / 2.0);
            }
            return f;
        };
        const auto sol1 = Numerics::fminbnd(func1, 1.0, 3.0);
        assert(sol1.converged);
        assert(static_cast<std::int32_t>(sol1.x * 1000) == 2006);

        const auto func2 = [](const double x, const double a = 9.0 / 7.0) -> double {
            return std::sin(x - a);
        };
        const auto sol2 = Numerics::fminbnd(func2, 1.0, 2.0 * std::numbers::pi_v<double>);
        assert(sol2.converged);
        assert(static_cast<std::int32_t>(sol2.x * 1000) == 5998);
    }

    // test lps solver
    {
        constexpr std::int32_t M{ 4 };
        constexpr std::int32_t N{ 3 };
        const std::array<std::array<double, N>, M> A{ std::array<double, N>{  6.0, -1.0,  0.0 },
                                                      std::array<double, N>{ -1.0, -5.0,  0.0 },
                                                      std::array<double, N>{  1.0,  5.0,  1.0 },
                                                      std::array<double, N>{ -1.0, -5.0, -1.0 } };
        const std::array<double, M> b{ 10.0, -4.0, 5.0, -5.0 };
        const std::array<double, N> c{ 1.0, -1.0, 0.0 };
        auto maximizer = Numerics::linearProgramingSolve(A, b, c);
        assert(static_cast<std::int32_t>(std::floor(maximizer.value * 10000.0)) == 12903);
        assert(static_cast<std::int32_t>(std::floor(maximizer.x[0]  * 10000.0)) == 17419);
        assert(static_cast<std::int32_t>(std::floor(maximizer.x[1]  * 10000.0)) == 4516);
        assert(static_cast<std::int32_t>(std::floor(maximizer.x[2])) == 1);
    }

    // test approximations - sin / sincos
    {
        float angle{ -std::numbers::pi_v<float> };
        const float step{ 0.01f };
        while (angle < std::numbers::pi_v<float>) {
            const float sin_real{ std::sin(angle) };
            const float cos_real{ std::cos(angle) };

            const float sin_approx{ Numerics::Approximation::sin(angle) };
            const auto sincos = Numerics::Approximation::sincos(angle);

            assert(std::abs(sin_real - sin_approx) < 0.001f);
            assert(std::abs(sin_real - sincos.sin) < 0.001f);
            assert(std::abs(cos_real - sincos.cos) < 0.001f);

            angle += step;
        }
    }

    // test approximations - hypot
    {
        float x{ -1000.0f };
        float y{ -1000.0f };
        float step{ 5.0f };
        float max_diff{}, max_x{}, max_y{};
        while (x < 1000.0f) {
            while (y < 1000.0f) {
                float d = std::hypot(x, y);
                float da = Numerics::Approximation::hypot(x, y);
                if (std::abs(d - da) > max_diff) {
                    max_diff = std::abs(d - da);
                    max_x = x;
                    max_y = y;
                }
                y += step;
            }
            x += step;
        }
        assert((std::hypot(max_x, max_y) / Numerics::Approximation::hypot(max_x, max_y) - 1.0f) * 100.0f < 1.7);
    }
}

void test_numerical_algorithms() {
    // test accumulate
    const std::array<double, 4> tempArray{ 1.0, std::pow(10.0, 100), 0.01, -std::pow(10.0, 100) };
    assert(static_cast<std::int32_t>(NumericalAlgorithms::accumulate(tempArray) * 100) == 101);

    // test histcounts
    const std::array<double, 10> x{ {2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0} };
    const auto hists = NumericalAlgorithms::histcounts(x, 6);
    const std::array<std::size_t, 6> Nexpected{ {2,2,2,2,1,1} };
    const std::array<std::size_t, 10> binExpected{ {0,0,1,1,2,2,3,3,4,5} };
    for (std::size_t i{}; i < 6; ++i) {
        assert(hists.N[i] == Nexpected[i]);
    }
    for (std::size_t i{}; i < 10; ++i) {
        assert(hists.bin[i] == binExpected[i]);
    }

    // test mean and std
    const auto stats = NumericalAlgorithms::mean_and_std(x);
    assert(static_cast<std::size_t>(stats.mean * 10) == 129);
    assert(static_cast<std::size_t>(stats.std * 100) == 7329);

    // test circular_mean
    {
        constexpr float deg_to_rad{ std::numbers::pi_v<float> / 180.0f };
        std::array<float, 3> angles{ {350.0f * deg_to_rad, 5.0f * deg_to_rad, 20.0f * deg_to_rad } };
        const float mean{ NumericalAlgorithms::circular_mean(angles) };
        assert(Numerics::areEquals(mean, angles[1]));
    }

    // test convolution
    {
        const std::array<int, 3> u{ {1, 0, 1} };
        const std::array<int, 2> v{ {2, 7} };
        std::array<int, 4> w;
        NumericalAlgorithms::conv(u.begin(), u.end(), v.begin(), v.end(), w.begin());
        assert(w[0] == 2);
        assert(w[1] == 7);
        assert(w[2] == 2);
        assert(w[3] == 7);
    }
    {
        const std::array<int, 3> u{ {1, 1, 1} };
        const std::array<int, 7> v{ {1, 1, 0, 0, 0, 1, 1} };
        std::array<int, 9> w;
        NumericalAlgorithms::conv(u.begin(), u.end(), v.begin(), v.end(), w.begin());
        assert(w[0] == 1);
        assert(w[1] == 2);
        assert(w[2] == 2);
        assert(w[3] == 1);
        assert(w[4] == 0);
        assert(w[5] == 1);
        assert(w[6] == 2);
        assert(w[7] == 2);
        assert(w[8] == 1);
    }

    // test filter
    {
        // FIR filter (3 taps moving average)
        const std::array<double, 3> b{ {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0} };
        const std::array<double, 1> a{ {1.0} };
        const std::vector<double> xx{ {2.0, 1.0, 6.0, 2.0, 4.0, 3.0} };
        std::vector<double> y(6);
        NumericalAlgorithms::filter<3, 1>(xx.begin(), xx.end(), y.begin(), b, a);
        assert(static_cast<std::int32_t>(y[0] * 3.0) == 2);
        assert(static_cast<std::int32_t>(y[1] * 3.0) == 3);
        assert(static_cast<std::int32_t>(y[2] * 3.0) == 9);
        assert(static_cast<std::int32_t>(y[3] * 3.0) == 8); // 2.99999 ....
        assert(static_cast<std::int32_t>(y[4] * 3.0) == 12);
        assert(static_cast<std::int32_t>(y[5] * 3.0) == 9);;
    }

    // test partition
    {
        std::list<int> v = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        std::list<int> vExpected = { 0, 8, 2, 6, 4, 5, 3, 7, 1, 9 };
        auto it = Algoithms::partition(v.begin(), v.end(), [](int i) {return i % 2 == 0; });
        assert(v == vExpected);
    }

    // test circshift
    {
        std::array<std::size_t, 10> a{ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} };
        std::array<std::size_t, 10> ai{ {7, 8, 9, 0, 1, 2, 3, 4, 5, 6} };
        NumericalAlgorithms::circshift(a.begin(), a.end(), 3);
        for (std::size_t i{}; i < 10; ++i) {
            assert(a[i] == ai[i]);
        }
        NumericalAlgorithms::circshift(a.begin(), a.end(), -3);
        for (std::size_t i{}; i < 10; ++i) {
            assert(a[i] == i);
        }
    }

    // test generate_from_discrete_distribution
    {
        std::vector<float> p{ {0.2f, 0.4f, 0.4f} };
        std::vector<int> d{ {1, 2, 4} };
        std::size_t amount_of_ones{};
        std::size_t amount_of_twos{};
        std::size_t amount_of_fours{};
        std::vector<int> O(10, 0);
        for (std::size_t i{}; i < 1000; ++i) {
            NumericalAlgorithms::generate_from_discrete_distribution(p.begin(), p.end(), d.begin(), d.end(), O.begin(), O.end());
            amount_of_ones += Algoithms::count_if(O.begin(), O.end(), [](const int a) { return a == 1; });
            amount_of_twos += Algoithms::count_if(O.begin(), O.end(), [](const int a) { return a == 2; });
            amount_of_fours += Algoithms::count_if(O.begin(), O.end(), [](const int a) { return a == 4; });
        }
        assert(amount_of_ones + amount_of_twos + amount_of_fours == 10 * 1000);
        assert(std::abs(static_cast<double>(amount_of_ones) / 10000.0 - 0.2) <= 0.01);
        assert(std::abs(static_cast<double>(amount_of_twos) / 10000.0 - 0.4) <= 0.01);
        assert(std::abs(static_cast<double>(amount_of_fours) / 10000.0 - 0.4) <= 0.01);
    }

    // test FFT
    {
        std::vector<double> x{ 1.0,5.0,-9.0,8.0,-7.0,5.0,6.0,-2.0,6.0,5.0,4.0,-2.0,-3.0,5.0,8.0,-9.0,-7.0,1.0,2.0,4.0 };
        std::vector<std::complex<double>> vec(x.begin(), x.end());
        const auto freq = NumericalAlgorithms::fft(vec);
        const auto time = NumericalAlgorithms::ifft(freq);

        assert(x.size() <= time.size());
        for (std::size_t i{}; i < x.size(); ++i) {
            const std::int32_t t{ static_cast<std::int32_t>(std::round(time[i].real())) };
            const std::int32_t v{ static_cast<std::int32_t>(std::round(x[i])) };
            assert(t == v);
        }
    }
}

void test_glsl_basics() {
    static_assert(GLSL::IFixedVector<GLSL::Swizzle<float, 2, 0, 1>>);
    static_assert(GLSL::IFixedVector<GLSL::Swizzle<double, 3, 1, 0, 2>>);
    static_assert(sizeof(GLSL::Swizzle<double, 3, 1, 0, 2>) == 3 * sizeof(double));
    static_assert(sizeof(GLSL::Swizzle<float, 2, 0, 1>) == 2 * sizeof(float));

    static_assert(GLSL::IFixedVector<GLSL::Vector2<float>>);
    static_assert(GLSL::IFixedVector<GLSL::Vector2<double>>);
    static_assert(std::is_same_v<typename GLSL::Vector2<float>::value_type, float>);
    static_assert(GLSL::Vector2<float>::length() == 2);
    static_assert(sizeof(GLSL::Vector2<float>) == 2 * sizeof(float));

    static_assert(GLSL::IFixedVector<GLSL::Vector3<float>>);
    static_assert(GLSL::IFixedVector<GLSL::Vector3<double>>);
    static_assert(std::is_same_v<typename GLSL::Vector3<float>::value_type, float>);
    static_assert(GLSL::Vector3<float>::length() == 3);
    static_assert(sizeof(GLSL::Vector3<float>) == 3 * sizeof(float));

    static_assert(GLSL::IFixedVector<GLSL::Vector4<float>>);
    static_assert(GLSL::IFixedVector<GLSL::Vector4<double>>);
    static_assert(std::is_same_v<typename GLSL::Vector4<float>::value_type, float>);
    static_assert(GLSL::Vector4<float>::length() == 4);
    static_assert(sizeof(GLSL::Vector4<float>) == 4 * sizeof(float));

    static_assert(GLSL::IFixedVector<GLSL::VectorN<float, 16>>);
    static_assert(GLSL::IFixedVector<GLSL::VectorN<double, 9>>);
    static_assert(std::is_same_v<typename GLSL::VectorN<float, 16>::value_type, float>);
    static_assert(GLSL::VectorN<float, 9>::length() == 9);
    static_assert(sizeof(GLSL::VectorN<float, 9>) == 9 * sizeof(float));

    // check that Matrix2 is IFixedCubicMatrix
    static_assert(GLSL::IFixedCubicMatrix<GLSL::Matrix2<float>>);
    static_assert(GLSL::IFixedCubicMatrix<GLSL::Matrix2<double>>);
    static_assert(std::is_same_v<typename GLSL::Matrix2<float>::value_type, float>);
    static_assert(GLSL::Matrix2<float>::length() == 4);
    static_assert(GLSL::Matrix2<float>::columns() == 2);
    static_assert(sizeof(GLSL::Matrix2<float>) == 4 * sizeof(float));

    // check that Matrix3 is IFixedCubicMatrix
    static_assert(GLSL::IFixedCubicMatrix<GLSL::Matrix3<float>>);
    static_assert(GLSL::IFixedCubicMatrix<GLSL::Matrix3<double>>);
    static_assert(std::is_same_v<typename GLSL::Matrix3<float>::value_type, float>);
    static_assert(GLSL::Matrix3<float>::length() == 9);
    static_assert(GLSL::Matrix3<float>::columns() == 3);
    static_assert(sizeof(GLSL::Matrix3<float>) == 9 * sizeof(float));

    // check that Matrix4 is IFixedCubicMatrix
    static_assert(GLSL::IFixedCubicMatrix<GLSL::Matrix4<float>>);
    static_assert(GLSL::IFixedCubicMatrix<GLSL::Matrix4<double>>);
    static_assert(std::is_same_v<typename GLSL::Matrix4<float>::value_type, float>);
    static_assert(GLSL::Matrix4<float>::length() == 16);
    static_assert(GLSL::Matrix4<float>::columns() == 4);
    static_assert(sizeof(GLSL::Matrix4<float>) == 16 * sizeof(float));

    // check that MatrixN is IFixedCubicMatrix
    static_assert(GLSL::IFixedCubicMatrix<GLSL::MatrixN<float, 8>>);
    static_assert(GLSL::IFixedCubicMatrix<GLSL::MatrixN<double, 16>>);
    static_assert(std::is_same_v<typename GLSL::MatrixN<float, 5>::value_type, float>);
    static_assert(GLSL::MatrixN<float, 5>::length() == 25);
    static_assert(GLSL::MatrixN<float, 5>::columns() == 5);
    static_assert(sizeof(GLSL::MatrixN<float, 6>) == 36 * sizeof(float));

    {
        GLSL::Swizzle<int, 2, 1, 0> a(0, 1);
        assert(a[0] == 1);
        assert(a[1] == 0);

        GLSL::Swizzle<int, 2, 0, 1> b(0, 1);
        assert(b[0] == 0);
        assert(b[1] == 1);

        GLSL::Swizzle<int, 2, 0, 1> c = b;
        assert(c[0] == b[0]);
        assert(c[1] == b[1]);

        c += a;
        assert(c[0] == a[1] + a[1]);
        assert(c[1] == a[0] + a[0]);

        GLSL::Swizzle<int, 2, 0, 1> d(b);
        assert(d[0] == b[0]);
        assert(d[1] == b[1]);

        d += 3;
        assert(d[0] == b[0] + 3);
        assert(d[1] == b[1] + 3);

        GLSL::Swizzle<int, 3, 0, 2, 1> e(std::array<int, 3>{{0, 1, 2}});
        assert(e[0] == 0);
        assert(e[1] == 2);
        assert(e[2] == 1);
    }

    {
        ivec2 v(0, 0);
        assert(v.x == 0);
        assert(v.y == 0);

        v = std::array<int, 2>{{2, 3}};
        assert(v.x == 2);
        assert(v.y == 3);

        ivec2 u(3, 2);
        assert(u.x == 3);
        assert(u.y == 2);

        u.xy = v.yx + v.xy;
        assert(u.x == 5);
        assert(u.y == 5);

        ivec2 w(u);
        assert(w.x == u.x);
        assert(w.y == u.y);

        u -= v;
        assert(u.x == 3);
        assert(u.y == 2);

        u *= 2;
        assert(u.x == 6);
        assert(u.y == 4);

        u -= 2;
        assert(u.x == 4);
        assert(u.y == 2);

        u.xy += w.xx;
        assert(u.x == 9);
        assert(u.y == 7);

        u *= -1;
        assert(u.x == -9);
        assert(u.y == -7);

        u.xy = abs(u.yy);
        assert(u.x == 7);
        assert(u.y == 7);
    }

    {
        ivec3 f(2);
        assert(f.x == 2);
        assert(f.y == 2);
        f.x = 1;
        f.z = 3;

        ivec2 xi(3, 4);
        ivec3 x(xi, 5);
        assert(x.x == 3);
        assert(x.y == 4);
        assert(x.z == 5);

        x.zyx += f.xyz;
        assert(x.x == 5 + 1);
        assert(x.y == 4 + 2);
        assert(x.z == 3 + 3);

        f = clamp(f, 1, 2);
        assert(f.x == 1);
        assert(f.y == 2);
        assert(f.z == 2);

        ivec3 y(2, xi);
        assert(y.x == 2);
        assert(y.y == 3);
        assert(y.z == 4);

        assert(GLSL::equal(y, ivec3(2, 3, 4)));
    }

    {
        ivec2 a(0, 1);
        ivec2 b(2, 3);
        ivec3 c(1, 2, 3);

        ivec4 d(b, a);
        assert(d.x == 2);
        assert(d.y == 3);
        assert(d.z == 0);
        assert(d.w == 1);

        ivec4 e(c, 4);
        assert(e.x == 1);
        assert(e.y == 2);
        assert(e.z == 3);
        assert(e.w == 4);

        ivec4 f(0, c);
        assert(f.x == 0);
        assert(f.y == 1);
        assert(f.z == 2);
        assert(f.w == 3);

        assert(GLSL::equal(a, ivec2(0, 1)));
        assert(GLSL::greaterThan(a, ivec2(-1, -2)));
    }

    {
        vec3 a(0.0f, 1.0f, 2.0f);
        vec3 b(2.0f, 1.0f, 3.0f);

        vec3 c = GLSL::mix<0.5f>(a, b);
        assert(static_cast<std::uint32_t>(c.x) == 1u);
        assert(static_cast<std::uint32_t>(c.y) == 1u);
        assert(static_cast<std::uint32_t>(c.z * 10) == 25u);
        
        vec3 d = GLSL::mix(a, b, 0.25f);
        assert(static_cast<std::uint32_t>(d.x * 10) == 5u);
        assert(static_cast<std::uint32_t>(d.y) == 1u);
        assert(static_cast<std::uint32_t>(d.z * 100) == 225u);
    }

    {
        vec4 g(4.0f, 9.0f, 16.0f, 25.0f);
        vec4 gi = GLSL::inversesqrt(g);
        assert(static_cast<std::uint32_t>(gi.x * 10) == 5u);
        assert(static_cast<std::uint32_t>(gi.y * 1000) == 333u);
        assert(static_cast<std::uint32_t>(gi.z * 100) == 25u);
        assert(static_cast<std::uint32_t>(gi.w * 10) == 2u);
    }

    {
        using mat2i = GLSL::Matrix2<int>;
        // 0 2
        // 1 3
        mat2i a(0, 1, 2, 3);
        assert(a.x().x == 0);
        assert(a.x().y == 1);
        assert(a.y().x == 2);
        assert(a.y().y == 3);

        assert(GLSL::equal(GLSL::row(a, 1), ivec2(1, 3)));
        assert(GLSL::equal(a, mat2i(0, 1, 2, 3)));
        assert(GLSL::lessThan(a, mat2i(1, 2, 3, 4)));

        // 4 7
        // 5 8
        mat2i b(ivec2(4, 5), ivec2(7, 8));
        assert(b.x().x == 4);
        assert(b.x().y == 5);
        assert(b.y().x == 7);
        assert(b.y().y == 8);


        //  (4 5) * /4 7\ = 4*4 + 5*5 = (41 68)
        //          \5 8/   4*7 + 5*8
        ivec2 xb = b.x() * b;
        assert(GLSL::equal(xb, ivec2(41, 68)));

        // /4 7\ * /4\ = 4*4 + 7*5 = (51, 60)
        // \5 8/   \5/   5*4 + 8*5
        ivec2 bx = b * b.x();
        assert(GLSL::equal(bx, ivec2(51, 60)));

        assert(GLSL::determinant(b) == -3);
        assert(GLSL::equal(GLSL::transpose(b), mat2i(4, 7, 5, 8)));
        assert(GLSL::equal(GLSL::trace(b), ivec2(4, 8)));

        std::array<int, 4> arr = { 0, 1, 3, 2 };
        mat2i c(arr.data());
        assert(GLSL::equal(c, mat2i(0, 1, 3, 2)));

        mat2i d = c + a;
        assert(GLSL::equal(d, mat2i(0, 2, 5, 5)));
    }

    {
        using mat3i = GLSL::Matrix3<int>;

        // 0 3 6
        // 1 4 7
        // 2 5 8
        mat3i a(0, 1, 2, 3, 4, 5, 6, 7, 8);
        assert(a.x().x == 0);
        assert(a.x().y == 1);
        assert(a.x().z == 2);
        assert(a.y().x == 3);
        assert(a.y().y == 4);
        assert(a.y().z == 5);
        assert(a.z().x == 6);
        assert(a.z().y == 7);
        assert(a.z().z == 8);

        assert(GLSL::equal(GLSL::row(a, 0), ivec3(0, 3, 6)));
        assert(GLSL::equal(GLSL::row(a, 1), ivec3(1, 4, 7)));
        assert(GLSL::equal(GLSL::row(a, 2), ivec3(2, 5, 8)));

        assert(GLSL::equal(a, mat3i(0, 1, 2, 3, 4, 5, 6, 7, 8)));
        assert(GLSL::lessThan(a, mat3i(1, 2, 3, 4, 5, 6, 7, 8, 9)));

        // 4 7 1
        // 5 8 2
        // 6 9 3
        mat3i b(ivec3(4, 5, 6), ivec3(7, 8, 9), ivec3(1, 2, 3));
        assert(b.x().x == 4);
        assert(b.x().y == 5);
        assert(b.x().z == 6);
        assert(b.y().x == 7);
        assert(b.y().y == 8);
        assert(b.y().z == 9);
        assert(b.z().x == 1);
        assert(b.z().y == 2);
        assert(b.z().z == 3);

        mat3i d = a + b;
        assert(d.x().x == 4);
        assert(d.x().y == 6);
        assert(d.x().z == 8);
        assert(d.y().x == 10);
        assert(d.y().y == 12);
        assert(d.y().z == 14);
        assert(d.z().x == 7);
        assert(d.z().y == 9);
        assert(d.z().z == 11);

        //            (4 7 1)   
        // (1 2 3 ) * (5 8 2) = (32, 50, 14)
        //            (6 9 3)   
        ivec3 xb = ivec3(1, 2, 3) * b;
        assert(GLSL::equal(xb, ivec3(32, 50, 14)));

        // (4 7 1)   (1)
        // (5 8 2) * (2) = (21, 27, 33)
        // (6 9 3)   (3)
        ivec3 bx = b * ivec3(1, 2, 3);
        assert(GLSL::equal(bx, ivec3(21, 27, 33)));

        assert(GLSL::determinant(b) == 0);
        assert(GLSL::equal(GLSL::transpose(b), mat3i(4, 7, 1, 5, 8, 2, 6, 9, 3)));
        assert(GLSL::equal(GLSL::trace(b), ivec3(4, 8, 3)));

        std::array<int, 9> arr = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
        mat3i c(arr.data());
        assert(GLSL::equal(c, mat3i(0, 1, 2, 3, 4, 5, 6, 7, 8)));
    }

    {
        using mat4i = GLSL::Matrix4<int>;

        // 0 4 8 6
        // 1 5 9 5
        // 2 6 8 4
        // 3 7 7 3
        mat4i a(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3);
        assert(a.x().x == 0);
        assert(a.x().y == 1);
        assert(a.x().z == 2);
        assert(a.x().w == 3);
        assert(a.y().x == 4);
        assert(a.y().y == 5);
        assert(a.y().z == 6);
        assert(a.y().w == 7);
        assert(a.z().x == 8);
        assert(a.z().y == 9);
        assert(a.z().z == 8);
        assert(a.z().w == 7);

        assert(a.w().x == 6);
        assert(a.w().y == 5);
        assert(a.w().z == 4);
        assert(a.w().w == 3);

        assert(GLSL::equal(GLSL::row(a, 0), ivec4(0, 4, 8, 6)));
        assert(GLSL::equal(GLSL::row(a, 1), ivec4(1, 5, 9, 5)));
        assert(GLSL::equal(GLSL::row(a, 2), ivec4(2, 6, 8, 4)));
        assert(GLSL::equal(GLSL::row(a, 3), ivec4(3, 7, 7, 3)));

        assert(GLSL::equal(a, mat4i(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3)));
        assert(GLSL::lessThan(a, a + 1));

        // 4 8 6 2
        // 5 9 5 1
        // 6 8 4 2
        // 7 7 3 3

        mat4i b(ivec4(4, 5, 6, 7), ivec4(8, 9, 8, 7), ivec4(6, 5, 4, 3), ivec4(2, 1, 2, 3));
        assert(b.x().x == 4);
        assert(b.x().y == 5);
        assert(b.x().z == 6);
        assert(b.x().w == 7);
        assert(b.y().x == 8);
        assert(b.y().y == 9);
        assert(b.y().z == 8);
        assert(b.y().w == 7);
        assert(b.z().x == 6);
        assert(b.z().y == 5);
        assert(b.z().z == 4);
        assert(b.z().w == 3);
        assert(b.w().x == 2);
        assert(b.w().y == 1);
        assert(b.w().z == 2);
        assert(b.w().w == 3);

        mat4i d = a + b;
        assert(d.x().x == a.x().x + b.x().x);
        assert(d.x().y == a.x().y + b.x().y);
        assert(d.x().z == a.x().z + b.x().z);
        assert(d.x().z == a.x().z + b.x().z);
        assert(d.y().x == a.y().x + b.y().x);
        assert(d.y().y == a.y().y + b.y().y);
        assert(d.y().z == a.y().z + b.y().z);
        assert(d.y().z == a.y().z + b.y().z);
        assert(d.z().x == a.z().x + b.z().x);
        assert(d.z().y == a.z().y + b.z().y);
        assert(d.z().z == a.z().z + b.z().z);
        assert(d.z().z == a.z().z + b.z().z);
        assert(d.w().x == a.w().x + b.w().x);
        assert(d.w().y == a.w().y + b.w().y);
        assert(d.w().z == a.w().z + b.w().z);
        assert(d.w().z == a.w().z + b.w().z);

        // 1 2 3 4 *  4 8 6 2 = 60
        //            5 9 5 1   78
        //            6 8 4 2   40
        //            7 7 3 3   22
        ivec4 xb = ivec4(1, 2, 3, 4) * b;
        assert(GLSL::equal(xb, ivec4(60, 78, 40, 22)));

        // 4 8 6 2 * 1 = 46
        // 5 9 5 1   2   42
        // 6 8 4 2   3   42
        // 7 7 3 3   4   42
        ivec4 bx = b * ivec4(1, 2, 3, 4);
        assert(GLSL::equal(bx, ivec4(46, 42, 42, 42)));

        assert(GLSL::determinant(b) == 0);
        assert(GLSL::equal(GLSL::transpose(b), mat4i(4, 8, 6, 2, 5, 9, 5, 1, 6, 8, 4, 2, 7, 7, 3, 3)));
        assert(GLSL::equal(GLSL::trace(b), ivec4(4, 9, 4, 3)));

        std::array<int, 16> arr = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        mat4i c(arr.data());
        assert(GLSL::equal(c, mat4i(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)));

        mat4i e = GLSL::matrixCompMult(b, c);
        assert(e.x().x == b.x().x * c.x().x);
        assert(e.x().y == b.x().y * c.x().y);
        assert(e.x().z == b.x().z * c.x().z);
        assert(e.x().w == b.x().w * c.x().w);
        assert(e.y().x == b.y().x * c.y().x);
        assert(e.y().y == b.y().y * c.y().y);
        assert(e.y().z == b.y().z * c.y().z);
        assert(e.y().w == b.y().w * c.y().w);
        assert(e.z().x == b.z().x * c.z().x);
        assert(e.z().y == b.z().y * c.z().y);
        assert(e.z().z == b.z().z * c.z().z);
        assert(e.z().w == b.z().w * c.z().w);
        assert(e.w().x == b.w().x * c.w().x);
        assert(e.w().y == b.w().y * c.w().y);
        assert(e.w().z == b.w().z * c.w().z);
        assert(e.w().w == b.w().w * c.w().w);

        mat4 df(12.0f, 16.0f, 38.0f, 92.0f,
            13.0f, 15.0f, 75.0f, 32.0f,
            14.0f, 14.0f, -15.0f, 27.0f,
            15.0f, 13.0f, 5.0f, 5.0f);
        assert(static_cast<std::uint32_t>(GLSL::determinant(df)) == 108948u);

        mat4 dinv = GLSL::inverse(df);
        mat4 invExpected(0.242363f, -0.290946f, -0.609483f, 0.693781f,
            -0.292204f, 0.342824f, 0.726677f, -0.741583f,
            0.004369f, 0.005397f, -0.023901f, 0.014135f,
            0.028270f, -0.023901f, -0.037008f, 0.032639f);
        Utilities::static_for<0, 1, 4>([&dinv, &invExpected](std::size_t i) {
            Utilities::static_for<0, 1, 4>([&dinv, &invExpected, i](std::size_t j) {
                assert(std::abs(std::abs(dinv(i, j)) - std::abs(invExpected(i, j))) < 1e-6);
            });
        });
    }

    {
        ivec4 a(2, 1, 4, 3);
        assert(GLSL::min(a) == 1);
        assert(GLSL::max(a) == 4);
        assert(GLSL::equal(GLSL::clamp(a, 2, 3), ivec4(2, 2, 3, 3)));
        assert(GLSL::equal(GLSL::clamp<1, 2>(a), ivec4(2, 1, 2, 2)));

        assert(GLSL::equal(GLSL::sign(ivec4(-1, 2, 0, -4)), ivec4(-1, 1, 1, -1)));
        assert(GLSL::equal(GLSL::step(0, ivec4(-1, 2, 0, -4)), ivec4(0, 1, 1, 0)));
        assert(GLSL::equal(GLSL::step<2>(ivec4(1, 2, 0, 4)), ivec4(0, 1, 0, 1)));
        assert(GLSL::equal(GLSL::step(ivec4(1, 2, 2, 1), ivec4(0, 3, 0, 3)), ivec4(0, 1, 0, 1)));

        assert(GLSL::dot(ivec3(1, 2, 3)) == 14);
        assert(GLSL::dot(ivec3(1, 2, 3), ivec3(3, 2, 1)) == 10);

        assert(GLSL::prod(ivec3(1, 2, 3)) == 6);
        assert(GLSL::sum(ivec3(1, 2, 3)) == 6);

        assert(static_cast<std::uint32_t>(GLSL::distance(dvec2(0.0, 0.0), dvec2(0.0, 10.0))) == 10u);
    }

    {
        imat3 a(1, 2, 3, 4, 5, 6, 7, 8, 9);
        imat3 b(8, 9, 6, 7, 5, 4, 1, 2, 3);
        imat3 c = a * b;
        a *= b;

        imat3 expected(86, 109, 132, 55, 71, 87, 30, 36, 42);
        Utilities::static_for<0, 1, 3>([&a, &c, &expected](std::size_t i) {
            Utilities::static_for<0, 1, 3>([&a, &c, &expected, i](std::size_t j) {
                assert(c(i, j) == expected(i, j));
                assert(a(i, j) == expected(i, j));
            });
        });
    }

    {
        using ivec8 = GLSL::VectorN<std::int32_t, 8>;
        ivec8 a(0, 1, 2, 3, 4, 5, 6, 7);
        ivec8 b(7, 6, 5, 4, 3, 2, 1, 0);
        ivec8 c = a + b;
        ivec8 d(7);
        assert(GLSL::equal(c, d));
        assert(GLSL::sum(c) == 7 * 8);

        using vec8 = GLSL::VectorN<float, 8>;
        vec8 e = GLSL::mix<0.5f>(vec8(0.0f), vec8(4.0f));
        assert(GLSL::max(GLSL::abs(e - vec8(2.0f))) < 1e-6);
    }

    {
        using vec6 = GLSL::VectorN<std::int32_t, 6>;
        using mat6 = GLSL::MatrixN<std::int32_t, 6>;

        vec6 v(3);
        mat6 m0(5);
        mat6 m1(vec6(0), vec6(1), vec6(2), vec6(3), vec6(4), vec6(5));
        mat6 m2(vec6(5), vec6(4), vec6(3), vec6(2), vec6(1), vec6(0));
        mat6 m3 = m1 + m2;
        assert(GLSL::max(m3) == 5);
        assert(GLSL::min(m3) == 5);

        vec6 m4 = m1 * v;
        assert(GLSL::sum(m4) == 6 * 45);
    }
}

void test_glsl_extra() {
    {
        imat2 a = Extra::outer_product(ivec2(1, 2), ivec2(3, 4));
        assert(a(0, 0) == 1 * 3);
        assert(a(0, 1) == 1 * 4);
        assert(a(1, 0) == 2 * 3);
        assert(a(1, 1) == 2 * 4);

        using vec = GLSL::VectorN<int, 5>;
        using mat = appropriate_matrix_type<vec>::matrix_type;
        const vec v(1, 2, 3, 4, 5);
        const vec u(3, 4, 5, 6, 7);
        const mat b{ Extra::outer_product(v, u) };
        Utilities::static_for<0, 1, 5>([&b, &v, &u](std::size_t i) {
            Utilities::static_for<0, 1, 5>([&b, &v, &u, i](std::size_t j) {
                assert(b(i, j) == v[j] * u[i]);
            });
        });
    }

    {
        vec2 axis;
        Extra::make_random(axis);
        const mat2 reflect{ Extra::Householder(GLSL::normalize(axis)) };
        assert(std::abs(1.0 + GLSL::determinant(reflect)) < 1e-6);

        GLSL::VectorN<double, 8> axis8;
        Extra::make_random(axis8);
        const GLSL::MatrixN<double, 8> reflect8{ Extra::Householder(GLSL::normalize(axis8)) };
        assert(std::abs(1.0 + Decomposition::determinant_using_qr(reflect8)) < 1e-6);
    }

    {
        mat3 a({3.0f, 5.0f, -7.0f,
               -12.0f, 19.0f, 21.0f,
               2.0f, -8.0f, 1.0});
        mat3 b = Extra::orthonormalize(a);
        assert(Extra::is_orthonormal_matrix(b));
    }

    {
        vec3 x(GLSL::normalize(vec3(3.5f, -12.2f, 27.0f)));
        mat3 a = Extra::orthonomrmalBasis(x);
        assert(Extra::is_orthonormal_matrix(a));
    }

    {
        ivec3 a(0, 1, 2);
        ivec3 b(2, 3, 3);
        assert(Extra::manhattan_distance(a, b) == 5);
        assert(Extra::chebyshev_distance(a, b) == 2);
        assert(Extra::inverse_chebyshev_distance(a, b) == 1);
    }

    {
        ivec4 a(1, 2, 3, 4);
        ivec4 b(2, 3, 4, 5);
        assert(Extra::left_dot<3>(a, b) == 20);
        assert(Extra::left_dot<2>(b) == 13);
    }

    {
        ivec3 a(1, 2, 3);
        ivec3 b(3, 2, 3);
        assert(!Extra::are_vectors_identical(a, b, 1));
        assert(Extra::are_vectors_identical(a, b, 3));
    }

    {
        // check companion
        vec4 a(1.0f, -23.0f, 142.0f, -120.0f);
        mat4 b;
        Extra::make_companion(b, a);
        assert(std::abs(GLSL::sum(b[0]) - 1) < 1e-6);
        assert(std::abs(GLSL::sum(b[1]) - 1) < 1e-6);
        assert(std::abs(GLSL::sum(b[2]) - 1) < 1e-6);
        assert(std::abs(GLSL::max(b[3] + a)) < 1e-6);
    }

    {
        vec3 world(3.0f, 2.0f, 4.0f);
        for (std::size_t i{1}; i < GLSL::prod(world); ++i) {
            const vec3 pos{ Extra::index_to_vector(i, world) };
            const std::size_t index{ Extra::vector_to_index(pos, world) };
            assert(index == i);
        }
    }

    {
        const double alpha{ 10.0f * static_cast<double>(rand()) / RAND_MAX };
        const double beta{ 10.0f * static_cast<double>(rand()) / RAND_MAX };
        dmat4 A, B, C;
        dvec4 x, y;
        for (std::size_t i{}; i < 5; ++i) {
            Utilities::static_for<0, 1, 4>([&A, &B, &C, &x, &y](std::size_t i) {
                Utilities::static_for<0, 1, 4>([&A, &B, &C, i](std::size_t j) {
                    A(i, j) = static_cast<double>(10.0f * static_cast<double>(rand()) / RAND_MAX);
                    B(i, j) = static_cast<double>(10.0f * static_cast<double>(rand()) / RAND_MAX);
                    C(i, j) = static_cast<double>(10.0f * static_cast<double>(rand()) / RAND_MAX);
                });
                x[i] = static_cast<double>(10.0 * static_cast<double>(rand()) / RAND_MAX);
                y[i] = static_cast<double>(10.0 * static_cast<double>(rand()) / RAND_MAX);
            });
    
            dvec4 regular_s{ x * alpha + y };
            dvec4 specialized_s{ Extra::axpy(alpha, x, y) };
            assert(GLSL::max(GLSL::abs(regular_s - specialized_s)) < 1e-6);
    
            dvec4 regular_v{ A * x * alpha + y * beta };
            dvec4 specialized_v{ Extra::gemv(alpha, A, x, beta, y) };
            assert(GLSL::max(GLSL::abs(regular_v - specialized_v)) < 1e-6);
    
            dmat4 regular_m{ B * A * alpha + C * beta };
            dmat4 specialized_m{ Extra::gemm(alpha, B, A, beta, C) };
            Utilities::static_for<0, 1, 4>([&regular_m, &specialized_m](std::size_t j) {
                assert(GLSL::max(GLSL::abs(regular_m[j] - specialized_m[j])) < 1e-6);
            });
        }
    }

    {
        const ivec3 u(1, 0, 1);
        const ivec2 v(2, 7);
        auto w = Extra::conv(u, v);
        assert(w[0] == 2);
        assert(w[1] == 7);
        assert(w[2] == 2);
        assert(w[3] == 7);
    }
    {
        const ivec3 u(1, 1, 1);
        const GLSL::VectorN<int, 7> v(1, 1, 0, 0, 0, 1, 1);
        auto w = Extra::conv(u, v);
        assert(w[0] == 1);
        assert(w[1] == 2);
        assert(w[2] == 2);
        assert(w[3] == 1);
        assert(w[4] == 0);
        assert(w[5] == 1);
        assert(w[6] == 2);
        assert(w[7] == 2);
        assert(w[8] == 1);
    }

    {
        const vec3 axis = GLSL::normalize(vec3(1.0f, 2.0f, 3.0f));
        const float angle{ std::numbers::pi_v<float> / 3.0f };
        const vec4 quat{ Transformation::create_quaternion_from_axis_angle(axis, angle) };
        assert(Extra::is_normalized(quat));

        const auto decomp = Extra::decompose_quaternion_twist_swing(quat);
        const vec4 qa = Extra::multiply_quaternions(decomp.Qz, decomp.Qr);
        Utilities::static_for<0, 1, 4>([&quat, &qa](std::size_t i) {
            assert(std::abs(std::abs(quat[i]) - std::abs(qa[i])) <= 1e-6);
        });

        const vec4 half_angle_quat{ Extra::quaternion_sqrt(quat) };
        const float half_angle{ Extra::get_quaternion_angle(half_angle_quat) };
        assert(std::abs(std::fma(-2.0f, half_angle, angle)) <= 1e-6);
    }

    {
        const vec3 v0(0.0f, -2.0f, -3.0f);
        const vec3 v1(0.0f, 14.0f, 4.0f);
        const vec3 v2(0.0f, 7.0f, 7.0f);
        const vec3 v3(0.0f, -4.0f, 3.0f);

        const auto c0 = Extra::quad_barycentric_coordinate(v0, v1, v2, v3, vec3(0.0f, 1.0f, 1.0f));
        vec3 p_from_c{ c0[0] * v0 + c0[1] * v1 + c0[2] * v2 + c0[3] * v3 };
        assert(Extra::are_vectors_identical(vec3(0.0f, 1.0f, 1.0f), p_from_c));

        const auto c1 = Extra::quad_barycentric_coordinate(v0, v1, v2, v3, vec3(0.0f, 14.0f, 14.0f));
        assert(Numerics::areEquals(GLSL::sum(c1), -4.0f));

        const auto c2 = Extra::quad_barycentric_coordinate(v0, v1, v2, v3, v0);
        p_from_c = c2[0] * v0 + c2[1] * v1 + c2[2] * v2 + c2[3] * v3;
        assert(Extra::are_vectors_identical(v0, p_from_c));

        const auto c3 = Extra::quad_barycentric_coordinate(v0, v1, v2, v3, v1);
        p_from_c = c3[0] * v0 + c3[1] * v1 + c3[2] * v2 + c3[3] * v3;
        assert(Extra::are_vectors_identical(v1, p_from_c));

        const auto c4 = Extra::quad_barycentric_coordinate(v0, v1, v2, v3, v2);
        p_from_c = c4[0] * v0 + c4[1] * v1 + c4[2] * v2 + c4[3] * v3;
        assert(Extra::are_vectors_identical(v2, p_from_c));

        const auto c5 = Extra::quad_barycentric_coordinate(v0, v1, v2, v3, v3);
        p_from_c = c5[0] * v0 + c5[1] * v1 + c5[2] * v2 + c5[3] * v3;
        assert(Extra::are_vectors_identical(v3, p_from_c));
    }
}

void test_glsl_transformation() {
    {
        const auto Rx = Transformation::rotation_matrix_from_axis_angle(dvec3(1.0, 0.0, 0.0), std::numbers::pi_v<double> / 2);
        const auto RxExpected = dmat3(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0);
        Utilities::static_for<0, 1, 3>([&Rx, &RxExpected](std::size_t i) {
            Utilities::static_for<0, 1, 3>([&Rx, &RxExpected, i](std::size_t j) {
                assert(std::abs(std::abs(Rx(i, j)) - std::abs(RxExpected(i, j))) <= 1e-6);
            });
        });

        const auto Ry = Transformation::rotation_matrix_from_axis_angle(dvec3(0.0, 1.0, 0.0), std::numbers::pi_v<double> / 2);
        const auto RyExpected = dmat3(0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0);
        Utilities::static_for<0, 1, 3>([&Ry, &RyExpected](std::size_t i) {
            Utilities::static_for<0, 1, 3>([&Ry, &RyExpected, i](std::size_t j) {
                assert(std::abs(std::abs(Ry(i, j)) - std::abs(RyExpected(i, j))) <= 1e-6);
            });
        });
    }

    {
        dvec3 target(0.0, 0.0, 0.0);
        dvec3 eye(5.0, 4.0, 6.0);
        dvec3 world_up(0.0, 0.0, 1.0);

        // look-at matrix test #1
        dmat3 transformation_using_world_up = Transformation::create_look_at_matrix(eye, target, world_up); // look-at matrix using world up
        auto axis_angle_using_world_up = Transformation::get_axis_angle_from_rotation_matrix(transformation_using_world_up); // look-at matrix (using world up) axis and angle
        dmat3 dcm_using_world_up_axis_angle = Transformation::rotation_matrix_from_axis_angle(axis_angle_using_world_up.axis, std::acos(axis_angle_using_world_up.cosine)); // rotation matrix from axis and angle
        dmat3 dcm_using_forward = Transformation::create_look_at_matrix(GLSL::normalize(eye - target));

        Utilities::static_for<0, 1, 3>([&transformation_using_world_up, &dcm_using_world_up_axis_angle, &dcm_using_forward](std::size_t i) {
            Utilities::static_for<0, 1, 3>([&transformation_using_world_up, &dcm_using_world_up_axis_angle, &dcm_using_forward, i](std::size_t j) {
                assert(std::abs(std::abs(transformation_using_world_up(i, j)) - std::abs(dcm_using_world_up_axis_angle(j, i))) < 1e-6);
                assert(std::abs(std::abs(transformation_using_world_up(i, j)) - std::abs(dcm_using_forward(i, j))) < 1e-6);
            });
        });

        // look at matrix test #2
        dmat3 transformation_using_rotation = Transformation::create_look_at_matrix(eye, target, std::acos(axis_angle_using_world_up.cosine)); // look-at matrix using roll angle
        auto axis_angle_using_roll = Transformation::get_axis_angle_from_rotation_matrix(transformation_using_rotation); // look-at matrix (using roll angle) axis and angle
        dmat3 dcm_using_roll = Transformation::rotation_matrix_from_axis_angle(axis_angle_using_roll.axis, std::acos(axis_angle_using_roll.cosine)); // rotation matrix from axis and angle

        Utilities::static_for<0, 1, 3>([&transformation_using_rotation, &dcm_using_roll](std::size_t i) {
            Utilities::static_for<0, 1, 3>([&transformation_using_rotation, &dcm_using_roll, i](std::size_t j) {
                assert(std::abs(std::abs(transformation_using_rotation(i, j)) - std::abs(dcm_using_roll(j, i))) < 1e-6);
            });
        });
    }

    {
        vec3 xAxis(1.0f, 0.0f, 0.0f);
        vec3 zAxis(0.0f, 0.0f, 1.0f);
        float angle{ std::numbers::pi_v<float> / 2.0f };

        vec3 rotated = Transformation::rotate_point_around_axis(vec3(2.0f, 0.0f, 0.0f), zAxis, angle);
        assert(std::abs(rotated.x) < 1e-6);
        assert(std::abs(rotated.y - 2.0f) < 1e-6);
        assert(std::abs(rotated.z) < 1e-6);

        rotated = Transformation::rotate_point_around_axis(vec3(0.0f, 0.0f, -2.0f), xAxis, angle);
        assert(std::abs(rotated.x) < 1e-6);
        assert(std::abs(rotated.y - 2.0f) < 1e-6);
        assert(std::abs(rotated.z) < 1e-6);

        vec4 quat = Transformation::create_quaternion_from_axis_angle(zAxis, angle);
        vec3 rotated_using_quat = Transformation::rotate_point_using_quaternion(quat, vec3(2.0f, 0.0f, 0.0f));
        assert(std::abs(rotated_using_quat.x) < 1e-6);
        assert(std::abs(rotated_using_quat.y - 2.0f) < 1e-6);
        assert(std::abs(rotated_using_quat.z) < 1e-6);

        quat = Transformation::create_quaternion_from_axis_angle(xAxis, angle);
        rotated_using_quat = Transformation::rotate_point_using_quaternion(quat, vec3(0.0f, 0.0f, -2.0f));
        assert(std::abs(rotated_using_quat.x) < 1e-6);
        assert(std::abs(rotated_using_quat.y - 2.0f) < 1e-6);
        assert(std::abs(rotated_using_quat.z) < 1e-6);
    }

    {
        vec3 axis = GLSL::normalize(vec3(1.0f, 2.0f, 3.0f));
        float angle{ std::numbers::pi_v<float> / 4.0f };

        vec4 quat = Transformation::create_quaternion_from_axis_angle(axis, angle);
        mat3 mat = Transformation::create_rotation_matrix_from_quaternion(quat);
        auto axis_from_mat = Transformation::get_axis_angle_from_rotation_matrix(mat);

        assert(std::abs(angle - std::acos(axis_from_mat.cosine)) < 1e-6);
        assert(GLSL::max(GLSL::abs(GLSL::abs(axis) - GLSL::abs(axis_from_mat.axis))) < 1e-6);

        vec4 quat_from_mat = Transformation::create_quaternion_from_rotation_matrix(mat);
        assert(GLSL::max(GLSL::abs(GLSL::abs(quat_from_mat) - GLSL::abs(quat))) < 1e-6);
    }

    {
        constexpr float pi{ std::numbers::pi_v<float> };
        constexpr float step{ pi / 10.0f };
        float x{};
        float y{};
        float z{};
        while (x < pi) {
            while (y < pi) {
                while (z < pi) {
                    const vec4 quat{ Transformation::create_quaternion_from_euler_angles(vec3(x, y, z)) };
                    const vec3 euler{ Transformation::create_euler_angles_from_quaternion(quat) };

                    assert(std::abs(euler.x - x) <= 1e-6f);
                    assert(std::abs(euler.y - y) <= 1e-6f);
                    assert(std::abs(euler.z - z) <= 1e-6f);

                    z += step;
                }
                y += step;
            }
            x += step;
        }
    }
}

void test_glsl_solvers() {
    GLSL::Matrix3<double> a(12.0, -51.0, 4.0,
                            6.0, 167.0, -68.0,
                            -4.0, 24.0, -41.0);

    {
        auto eigs = Decomposition::eig(a);
        assert(static_cast<std::int32_t>(eigs[0] * 10000) == -341966);
        assert(static_cast<std::int32_t>(eigs[1] * 10000) == 1561366);
        assert(static_cast<std::int32_t>(eigs[2] * 10000) == 160599);


        mat2 b(51.0f, 13.0f, -24.0f, 7.0f);
        auto eigs2 = Decomposition::eig(b);
        assert(static_cast<std::int32_t>(eigs2[0] * 10000) == 421148);
        assert(static_cast<std::int32_t>(eigs2[1] * 10000) == 158851);
    }

    {
        dmat3 QExpected(0.228375, -0.9790593, 0.076125,
                        0.618929, 0.084383, -0.780901,
                        0.751513, 0.225454, 0.619999);
        dmat3 RExpected( 52.545219,  0.0,       -0.0,
                        -165.895209, 70.906839,  0.0,
                        -27.328842,  31.566433, -23.015097);

        auto rot = Decomposition::QR(a);
        Utilities::static_for<0, 1, 3>([&rot, &QExpected, &RExpected](std::size_t i) {
            assert(std::abs(GLSL::length(rot.Q[i]) - GLSL::length(QExpected[i])) < 1e-2);
            assert(std::abs(GLSL::length(rot.R[i]) - GLSL::length(RExpected[i])) < 1e-6);
        });

        auto gs = Decomposition::QR<Decomposition::QR_DEOMPOSITION_TYPE::SchwarzRutishauser>(a);
        Utilities::static_for<0, 1, 3>([&gs, &QExpected, &RExpected](std::size_t i) {
            assert(std::abs(GLSL::length(gs.Q[i]) - GLSL::length(QExpected[i])) < 1e-2);
            assert(std::abs(GLSL::length(gs.R[i]) - GLSL::length(RExpected[i])) < 1e-6);
        });
    }

    {
        const mat3 aa(13.0f, 21.0f, 92.0f,
                      21.0f, 5.0f, 57.0f,
                      92.0f, 57.0f, 72.0f);
        auto eigen33 = Decomposition::EigenSymmetric(aa);
        Utilities::static_for<0, 1, 3>([&aa, &eigen33](std::size_t i) {
            const vec3 lhs{ aa * eigen33.eigenvectors[i] };
            const vec3 rhs{ eigen33.eigenvalues[i] * eigen33.eigenvectors[i] };
            assert(std::abs(GLSL::length(lhs) - GLSL::length(rhs)) < 1e-2);
        });

        const mat4 bb(1.0f, 2.0f, 3.0f, 4.0f,
                      2.0f, 5.0f, 6.0f, 7.0f,
                      3.0f, 6.0f, 8.0f, 9.0f,
                      4.0f, 7.0f, 9.0f, 10.0f);
        auto eigen44 = Decomposition::EigenSymmetric(bb);
        Utilities::static_for<0, 1, 4>([&bb, &eigen44](std::size_t i) {
            const vec4 lhs{ bb * eigen44.eigenvectors[i] };
            const vec4 rhs{ eigen44.eigenvalues[i] * eigen44.eigenvectors[i] };
            assert(std::abs(GLSL::length(lhs) - GLSL::length(rhs)) < 1e-2);
        });
        const vec4 expected44(0.184913f, 0.558036f, 24.0625f, -0.805485f);
        assert(std::abs(GLSL::length(expected44) - GLSL::length(eigen44.eigenvalues)) < 1e-4);
    }

    {
        // check that balancing a matrix allows faster eigenvalue decomposition
        mat4 df(12.0f, 16.0f,  38.0f, 92.0f,
                13.0f, 15.0f,  75.0f, 32.0f,
                14.0f, 14.0f, -15.0f, 27.0f,
                15.0f, 13.0f,  5.0f,  5.0f);
        const auto shur_balanced_df = Decomposition::Schur(df, true, 3);
        auto _eig = Decomposition::eig(df, true, 3);
        assert(Extra::is_orthonormal_matrix(shur_balanced_df.eigenvectors));
        assert(static_cast<std::int32_t>(std::abs(shur_balanced_df.schur(0, 0))) == 76);
        assert(static_cast<std::int32_t>(std::abs(shur_balanced_df.schur(1, 1))) == 21);
        assert(static_cast<std::int32_t>(std::abs(shur_balanced_df.schur(2, 2))) == 38);
        assert(static_cast<std::int32_t>(std::abs(_eig[0])) == 76);
        assert(static_cast<std::int32_t>(std::abs(_eig[1])) == 21);
        assert(static_cast<std::int32_t>(std::abs(_eig[2])) == 38);
        assert(static_cast<std::int32_t>(std::abs(_eig[3])) == 1);
    }

    {
        dmat4 df(12.0, 16.0,  38.0, 92.0,
                 13.0, 15.0,  75.0, 32.0,
                 14.0, 14.0, -15.0, 27.0,
                 15.0, 13.0,  5.0,  5.0);
        dvec4 b(70.0, 12.0, 50.0, 9.0);
        auto solution = Solvers::SolveQR(df, b);

        assert(static_cast<std::int32_t>(std::abs(solution[0]) * 1e5) == 1393187);
        assert(static_cast<std::int32_t>(std::abs(solution[1]) * 1e5) == 1619759);
        assert(static_cast<std::int32_t>(std::abs(solution[2]) * 1e5) == 3547185);
        assert(static_cast<std::int32_t>(std::abs(solution[3]) * 1e5) == 4066615);
    }

    {
        GLSL::Matrix3<double> ap(4.0, 12.0, -16.0,
                                 12.0, 37.0, -43.0,
                                 -16.0, -43.0, 98.0);
        GLSL::Matrix3<double> cholExpectec(2.0, 0.0, 0.0,
                                           6.0, 1.0, 0.0,
                                           -8.0, 5.0, 3.0);
        const auto chol = Decomposition::Cholesky(ap);
        Utilities::static_for<0, 1, 3>([&chol, &cholExpectec](std::size_t i) {
            assert(std::abs(GLSL::length(chol[i]) - GLSL::length(cholExpectec[i])) < 1e-6);
        });

        dvec3 b(6.0, 3.0, 1.0);
        auto solution = Solvers::SolveCholesky(ap, b);
        const dvec3 x(257.611111, -70.555556, 11.111111);
        assert(std::abs(GLSL::length(solution) - GLSL::length(x)) < 1e-6);
    }

    {
        const auto det_via_qr = Decomposition::determinant_using_qr(a);
        const auto det = GLSL::determinant(a);
        assert(static_cast<std::int32_t>(det_via_qr) == static_cast<std::int32_t>(det));
        assert(static_cast<std::int32_t>(det_via_qr) == -85750);
    }

    {
        const auto a_inv_qr = Decomposition::invert_using_qr(a);
        const auto a_inv_reg = GLSL::inverse(a);
        Utilities::static_for<0, 1, 3>([&a_inv_qr, &a_inv_reg](std::size_t i) {
            assert(std::abs(GLSL::length(a_inv_qr[i]) - GLSL::length(a_inv_reg[i])) < 1e-6);
        });

        auto a4 = dmat4(12.0, 16.0,  38.0, 92.0,
                           13.0, 15.0,  75.0, 32.0,
                           14.0, 14.0, -15.0, 27.0,
                           15.0, 13.0,   5.0, 5.0);
        const auto a_inv_qr4 = Decomposition::invert_using_qr(a4);
        const auto a_inv_reg4 = GLSL::inverse(a4);
        Utilities::static_for<0, 1, 3>([&a_inv_qr4, &a_inv_reg4](std::size_t i) {
            assert(std::abs(GLSL::length(a_inv_qr4[i]) - GLSL::length(a_inv_reg4[i])) < 1e-6);
        });
    }

    {
        auto lambda = Decomposition::spectral_radius(a, 30);
        assert(static_cast<std::uint32_t>(lambda * 10000) == 1561366u);

        lambda = Decomposition::spectral_radius<30>(a);
        assert(static_cast<std::uint32_t>(lambda * 10000) == 1561366u);


        dmat3 a2(0.5, 0.5, 0.4, 0.2, 0.8, 0.5, 0.3, 0.9, 1.1);
        lambda = Decomposition::spectral_radius(a2);
        assert(static_cast<std::uint32_t>(lambda * 10000) == 18053u);
    }

    {
        auto eig = Decomposition::Schur(a, false, 20);
        auto _eig = Decomposition::eig(a, false, 20);
        assert(static_cast<std::int32_t>(eig.schur(0, 0) * 100) == 15613);
        assert(static_cast<std::int32_t>(eig.schur(1, 1) * 100) == -3419);
        assert(static_cast<std::int32_t>(eig.schur(2, 2) * 100) == 1605);
        assert(static_cast<std::int32_t>(_eig[0] * 100) == -3419);
        assert(static_cast<std::int32_t>(_eig[1] * 100) == 15613);
        assert(static_cast<std::int32_t>(_eig[2] * 100) == 1605);
        assert(Extra::is_orthonormal_matrix(eig.eigenvectors));

        eig = Decomposition::Schur(GLSL::Matrix3<double>(21.0, 6.0, 14.0,
                                                         -51.0, -51.0, 24.0,
                                                         4.0, 24.0, 321.0), false, 20);
        _eig = Decomposition::eig(GLSL::Matrix3<double>(21.0, 6.0, 14.0,
                                                        -51.0, -51.0, 24.0,
                                                         4.0, 24.0, 321.0), false, 20);
        assert(static_cast<std::int32_t>(eig.schur(0, 0) * 100) == 32257);
        assert(static_cast<std::int32_t>(eig.schur(1, 1) * 100) == -4881);
        assert(static_cast<std::int32_t>(eig.schur(2, 2) * 100) == 1723);
        assert(static_cast<std::int32_t>(_eig[0] * 100) == -4881);
        assert(static_cast<std::int32_t>(_eig[1] * 100) == 32257);
        assert(static_cast<std::int32_t>(_eig[2] * 100) == 1723);
        assert(Extra::is_orthonormal_matrix(eig.eigenvectors));

        auto eign = Decomposition::Schur<20>(a);
        auto _eign = Decomposition::eig<20>(a);
        assert(static_cast<std::int32_t>(eign.schur(0, 0) * 100) == 15613);
        assert(static_cast<std::int32_t>(eign.schur(1, 1) * 100) == -3419);
        assert(static_cast<std::int32_t>(eign.schur(2, 2) * 100) == 1605);
        assert(static_cast<std::int32_t>(_eign[0] * 100) == -3419);
        assert(static_cast<std::int32_t>(_eign[1] * 100) == 15613);
        assert(static_cast<std::int32_t>(_eign[2] * 100) == 1605);
        assert(Extra::is_orthonormal_matrix(eig.eigenvectors));

        auto eign4 = Decomposition::Schur(dmat4(12.0, 16.0,  38.0, 92.0,
                                                13.0, 15.0,  75.0, 32.0,
                                                14.0, 14.0, -15.0, 27.0,
                                                15.0, 13.0,   5.0, 5.0));
        auto _eign4 = Decomposition::eig<20>(dmat4(12.0, 16.0,  38.0, 92.0,
                                                13.0, 15.0,  75.0, 32.0,
                                                14.0, 14.0, -15.0, 27.0,
                                                15.0, 13.0,   5.0, 5.0));
        assert(Extra::is_orthonormal_matrix(eign4.eigenvectors));
        assert(static_cast<std::int32_t>(std::abs(eign4.schur(0, 0))) == 77);
        assert(static_cast<std::int32_t>(std::abs(eign4.schur(1, 1))) == 39);
        assert(static_cast<std::int32_t>(std::abs(eign4.schur(2, 2))) == 22);
        assert(static_cast<std::int32_t>(std::abs(eign4.schur(3, 3))) == 1);
        assert(static_cast<std::int32_t>(std::abs(_eign4[0])) == 77);
        assert(static_cast<std::int32_t>(std::abs(_eign4[1])) == 38);
        assert(static_cast<std::int32_t>(std::abs(_eign4[2])) == 23);
        assert(static_cast<std::int32_t>(std::abs(_eign4[3])) == 1);
    }

    {
        auto svd = Decomposition::SVD(a);
        assert(Extra::is_orthonormal_matrix(svd.U));
        assert(Extra::is_orthonormal_matrix(svd.V));
        assert(static_cast<std::int32_t>(svd.S[0] * 100) == 19056);
        assert(static_cast<std::int32_t>(svd.S[1] * 1000) == 32856);
        assert(static_cast<std::int32_t>(svd.S[2] * 1000) == -13694);
       
        auto svd4 = Decomposition::SVD<6>(dmat4(12.0, 16.0,  38.0, 92.0,
                                                13.0, 15.0,  75.0, 32.0,
                                                14.0, 14.0, -15.0, 27.0,
                                                15.0, 13.0,   5.0, 5.0));
        assert(static_cast<std::int32_t>(svd4.S[0] * 100) == 12440);
        assert(static_cast<std::int32_t>(svd4.S[1] * 1000) == 55681);
        assert(static_cast<std::int32_t>(svd4.S[2] * 1000) == 23746);
        assert(static_cast<std::int32_t>(svd4.S[3] * 1000) == 662);
    }

//    {
//        const auto roots1 = Decomposition::roots(dvec3(3.0, -2.0, -4.0));
//        std::cout << "roots = " << roots1 << "\n";
//    }
}

void test_glsl_triangle() {
    {
        assert(Triangle::is_valid(ivec2(-1), ivec2(1, -1), ivec2(0, 1)));
        assert(!Triangle::is_valid(ivec2(-1), ivec2(-1), ivec2(0, 1)));
    }

    {
        assert(Triangle::is_point_within_triangle(ivec2(0), ivec2(-1), ivec2(1, -1), ivec2(0, 1)));
        assert(!Triangle::is_point_within_triangle(ivec2(0, 2), ivec2(-1), ivec2(1, -1), ivec2(0, 1)));

        vec2 a(-1.0f);
        vec2 b(3.0f, 2.0f);
        vec2 c(0.0f, 4.0f);
        assert(Triangle::is_point_within_triangle(vec2(0.0f), a, b, c));
        assert(Triangle::is_point_within_triangle(vec2(-0.5f), a, b, c));
        assert(!Triangle::is_point_within_triangle(vec2(-0.5f, 2.0f), a, b, c));
        assert(!Triangle::is_point_within_triangle(vec2(3.0f, 1.0f), a, b, c));
        assert(Triangle::is_point_within_triangle(vec2(1.0f, 2.0f), a, b, c));
    }

    {
        vec2 a(-1.0f);
        vec2 b(1.0f, -1.0f);
        vec2 c(0.0f, 1.0f);

        vec3 bary = Triangle::get_point_in_barycentric(vec2(0.3f, -0.3f), a, b, c);
        vec2 euclid( bary.x * a + bary.y * b + bary.z * c );
        assert(std::abs(euclid.x - 0.3f) < 1e-6f);
        assert(std::abs(euclid.y - -0.3f) < 1e-6f);

        bary = Triangle::get_point_in_barycentric(vec2(-0.2f, -0.1f), a, b, c);
        euclid = bary.x * a + bary.y * b + bary.z * c;
        assert(std::abs(euclid.x - -0.2f) < 1e-6f);
        assert(std::abs(euclid.y - -0.1f) < 1e-6f);

        bary = Triangle::get_point_in_barycentric(vec2(-1.0f), a, b, c);
        assert(std::abs(bary.x - 1.0f) < 1e-6f);
        assert(std::abs(bary.y) < 1e-6f);
        assert(std::abs(bary.z) < 1e-6f);

        bary = Triangle::get_point_in_barycentric(vec2(1.0f, -1.0f), a, b, c);
        assert(std::abs(bary.x) < 1e-6f);
        assert(std::abs(bary.y - 1.0f) < 1e-6f);
        assert(std::abs(bary.z) < 1e-6f);

        bary = Triangle::get_point_in_barycentric(vec2(0.0f, 1.0f), a, b, c);
        assert(std::abs(bary.x) < 1e-6f);
        assert(std::abs(bary.y) < 1e-6f);
        assert(std::abs(bary.z - 1.0f) < 1e-6f);
    }

    {
        vec2 a(-1.0f);
        vec2 b(1.0f, -1.0f);
        vec2 c(0.0f, 1.0f);
        vec3 bary = Triangle::barycentric_from_cartesian(a, b, c);
        assert(std::abs(bary.x - 2.0f) < 1e-6f);
        assert(std::abs(bary.y - -0.5f) < 1e-6f);
        assert(std::abs(bary.z - -0.5f) < 1e-6f);
    }

    {
        vec2 a(-1.0f);
        vec2 b(1.0f, -1.0f);
        vec2 c(0.0f, 1.0f);
        auto closest = Triangle::closest_point(vec2(0.0f, 2.0f), a, b, c);
        assert(std::abs(closest.x) < 1e-6f);
        assert(std::abs(closest.y - 1.0f) < 1e-6f);

        closest = Triangle::closest_point(vec2(0.0f, -2.0f), a, b, c);
        assert(std::abs(closest.x) < 1e-6f);
        assert(std::abs(closest.y - -1.0f) < 1e-6f);

        closest = Triangle::closest_point(vec2(2.0f, -1.0f), a, b, c);
        assert(std::abs(closest.x - 1.0f) < 1e-6f);
        assert(std::abs(closest.y - -1.0f) < 1e-6f);
    }

    {
        // triangle #1
        vec3 a1(0.0f);
        vec3 b1(2.0f, 0.0f, 2.0f);
        vec3 c1(-2.0f,0.0f, 2.0f);

        // #triangle #2
        vec3 a2(-2.0f, -2.0f, 1.0f);
        vec3 b2(2.0f, -2.0f, 1.0f);
        vec3 c2(0.0f, 2.0f, 1.0f);
        const auto intersection_segment = Triangle::check_triangles_intersection(a1, b1, c1, a2, b2, c2);
        assert(std::abs(intersection_segment.p0.x - -1.0f) < 1e-6);
        assert(std::abs(intersection_segment.p0.y        ) < 1e-6);
        assert(std::abs(intersection_segment.p0.z -  1.0f) < 1e-6);
        assert(std::abs(intersection_segment.p1.x - 1.0f) < 1e-6);
        assert(std::abs(intersection_segment.p1.y       ) < 1e-6);
        assert(std::abs(intersection_segment.p1.z - 1.0f) < 1e-6);

        // #triangle #3
        vec3 a3(-2.0f, -2.0f, 0.001f);
        vec3 b3(2.0f, -2.0f, 0.001f);
        vec3 c3(0.0f, 2.0f, 0.001f);
        const auto intersection_point = Triangle::check_triangles_intersection(a1, b1, c1, a3, b3, c3);
        assert(std::abs(intersection_point.p0.x - -0.001f) < 1e-6);
        assert(std::abs(intersection_point.p0.y) < 1e-6);
        assert(std::abs(intersection_point.p0.z - 0.001f) < 1e-6);
        assert(std::abs(intersection_point.p1.x - 0.001f) < 1e-6);
        assert(std::abs(intersection_point.p1.y) < 1e-6);
        assert(std::abs(intersection_point.p1.z - 0.001f) < 1e-6);

        // #triangle #4
        vec3 a4(-2.0f, -2.0f, -0.001f);
        vec3 b4(2.0f, -2.0f, -0.001f);
        vec3 c4(0.0f, 2.0f, -0.001f);
        const auto no_intersection = Triangle::check_triangles_intersection(a1, b1, c1, a4, b4, c4);
        assert(no_intersection.p0.x > 0.9f * std::numeric_limits<float>::max());
        assert(no_intersection.p0.y > 0.9f * std::numeric_limits<float>::max());
        assert(no_intersection.p0.z > 0.9f * std::numeric_limits<float>::max());
        assert(no_intersection.p1.x > 0.9f * std::numeric_limits<float>::max());
        assert(no_intersection.p1.y > 0.9f * std::numeric_limits<float>::max());
        assert(no_intersection.p1.z > 0.9f * std::numeric_limits<float>::max());
    }
}

void test_glsl_axis_aligned_bounding_box() {
    {
        vec3 center(1.0f, 1.0f, 0.0f);
        vec3 normal(0.0f, 0.0f, 1.0f);
        float radius{ 1.0f };
        auto aabb = AxisLignedBoundingBox::disk_aabb(center, normal, radius);
        assert(GLSL::max(GLSL::abs(aabb.min - vec3(0.0f))) <= 1e-6);
        assert(GLSL::max(GLSL::abs(aabb.max - vec3(2.0f, 2.0f, 0.0f))) <= 1e-6);

        center = vec3(-1.0f, -1.0f, 0.0f);
        normal *= -1.0f;
        aabb = AxisLignedBoundingBox::disk_aabb(center, normal, radius);
        assert(GLSL::max(GLSL::abs(aabb.min - vec3(-2.0f, -2.0f, 0.0f))) <= 1e-6);
        assert(GLSL::max(GLSL::abs(aabb.max - vec3(0.0f))) <= 1e-6);
    }

    {
        vec3 p0{ 1.0f, 1.0f, 0.0f };
        vec3 p1{ 1.0f, 10.0f, 0.0f };
        float radius{ 1.0f };
        auto aabb = AxisLignedBoundingBox::cylinder_aabb(p0, p1, radius);
        assert(GLSL::max(GLSL::abs(aabb.min - vec3(0.0f, 1.0f, -1.0f))) <= 1e-6);
        assert(GLSL::max(GLSL::abs(aabb.max - vec3(2.0f, 10.0f, 1.0f))) <= 1e-6);
    }

    {
        vec3 p0{ 1.0f, 1.0f, 0.0f };
        vec3 p1{ 1.0f, 5.0f, 0.0f };
        float r0{ 2.0f };
        float r1{ 1.0f };
        auto aabb = AxisLignedBoundingBox::cone_aabb(p0, p1, r0, r1);
        assert(GLSL::max(GLSL::abs(aabb.min - vec3(-1.0f, 1.0f, -2.0f))) <= 1e-6);
        assert(GLSL::max(GLSL::abs(aabb.max - vec3(3.0f, 5.0f, 2.0f))) <= 1e-6);
    }

    {
        vec3 center{10.0f, 10.0f, 0.0f};
        vec3 axis1{2.0f, 0.0f, 0.0f};
        vec3 axis2{0.0f, 2.0f, 0.0f};
        auto aabb = AxisLignedBoundingBox::ellipse_aabb(center, axis1, axis2);
        assert(GLSL::max(GLSL::abs(aabb.min - vec3(8.0f, 8.0f, 0.0f))) <= 1e-6);
        assert(GLSL::max(GLSL::abs(aabb.max - vec3(12.0f, 12.0f, 0.0f))) <= 1e-6);

        axis1 = vec3{ 4.0f, 0.0f, 0.0f };
        axis2 = vec3{ 0.0f, 0.0f, 4.0f };
        aabb = AxisLignedBoundingBox::ellipse_aabb(center, axis1, axis2);
        assert(GLSL::max(GLSL::abs(aabb.min - vec3(6.0f, 10.0f, -4.0f))) <= 1e-6);
        assert(GLSL::max(GLSL::abs(aabb.max - vec3(14.0f, 10.0f, 4.0f))) <= 1e-6);
    }

    {
        std::list<vec2> points{ {vec2(0.0f), vec2(1.0f), vec2(2.0f), vec2(-5.0f), vec2(10.0f), vec2(-17.0f, -3.0f)} };
        auto aabb = AxisLignedBoundingBox::point_cloud_aabb(points.begin(), points.end());
        assert(GLSL::max(GLSL::abs(aabb.min - vec2(-17.0f, -5.0f))) <= 1e-6);
        assert(GLSL::max(GLSL::abs(aabb.max - vec2(10.0f, 10.0f))) <= 1e-6);
    }
}

void test_glsl_point_distance() {
    {
        vec3 p{ 5.0f, 5.0f, 5.0f };
        float distance = PointDistance::udf_to_segment(p, vec3(0.0f), vec3(10.0f));
        assert(std::abs(distance) < 1e-6);

        distance = PointDistance::udf_to_segment(p, vec3(0.0f, 0.0f, 5.0f), vec3(10.0f, 0.0f, 5.0f));
        assert(std::abs(distance - 5.0f) < 1e-6);

        distance = PointDistance::udf_to_segment(p, vec3(0.0f), vec3(5.0f));
        assert(std::abs(distance) < 1e-6);

        distance = PointDistance::udf_to_segment(p, vec3(6.0f, 6.0f, 6.0f), vec3(10.0f));
        assert(std::abs(distance - std::sqrt(3)) < 1e-6);
    }

    {
        vec2 p{ 5.0f, 5.0f };

        float distance = PointDistance::udf_to_segment(p, vec2(0.0f), vec2(10.0f));
        assert(std::abs(distance) < 1e-6);

        distance = PointDistance::udf_to_segment(p, vec2(6.0f, 6.0f), vec2(10.0f, 10.0f));
        assert(std::abs(distance - std::sqrt(2)) < 1e-6);

        distance = PointDistance::udf_to_segment(p, vec2(0.0f, 0.0f), vec2(0.0f, 10.0f));
        assert(std::abs(distance - 5.0f) < 1e-6);

        distance = PointDistance::udf_to_segment(p, vec2(10.0f, 0.0f), vec2(10.0f, 10.0f));
        assert(std::abs(distance - 5.0f) < 1e-6);
    }
    
    {
        float distanceSquared = PointDistance::squared_udf_to_segment(vec2(1.0f, 0.0f), vec2(0.0f), vec2(0.0f, 1.0f));
        assert(std::abs(distanceSquared - 1) < 1e-6);

        distanceSquared = PointDistance::squared_udf_to_segment(vec2(-1.0f, 0.0f), vec2(0.0f), vec2(0.0f, 1.0f));
        assert(std::abs(distanceSquared - 1) < 1e-6);

        distanceSquared = PointDistance::squared_udf_to_segment(vec2(-2.0f), vec2(0.0f), vec2(0.0f, 1.0f));
        assert(std::abs(distanceSquared - 8) < 1e-6);
    }

    {
        float distance = PointDistance::sdf_to_sphere(vec2(0.0f, 1.0f), vec2(0.0f, 0.0f), 1.0f);
        assert(std::abs(distance) < 1e-6);

        distance = PointDistance::sdf_to_sphere(vec2(0.0f, -0.5f), vec2(0.0f, 0.0f), 1.0f);
        assert(std::abs(distance - -0.5f) < 1e-6);

        distance = PointDistance::sdf_to_sphere(vec2(0.0f, 1.5f), vec2(0.0f, 0.0f), 1.0f);
        assert(std::abs(distance - 0.5f) < 1e-6);

        distance = PointDistance::sdf_to_sphere(vec2(1.21f, 1.0f), vec2(1.0f, 1.0f), 1.0f);
        assert(std::abs(distance - -0.79f) < 1e-6);

        distance = PointDistance::sdf_to_sphere(vec2(1.0f + std::sqrt(2.0f)/2.0f), vec2(1.0f, 1.0f), 1.0f);
        assert(std::abs(distance) < 1e-6);

        distance = PointDistance::sdf_to_sphere(vec2(std::sqrt(2.0f) / 2.0f), vec2(0.0f), 2.0f);
        assert(std::abs(distance - -1.0f) < 1e-6);
    }
    
    {
        float distance = PointDistance::sdf_to_box_at_center(vec2(0.0f), vec2(1.0f, 3.0f));
        assert(std::abs(distance - -1.0f) < 1e-6);

        distance = PointDistance::sdf_to_box_at_center(vec2(2.0f), vec2(1.0f, 3.0f));
        assert(std::abs(distance - 1.0f) < 1e-6);

        distance = PointDistance::sdf_to_box_at_center(vec2(0.0f, 4.0f), vec2(1.0f, 3.0f));
        assert(std::abs(distance - 1.0f) < 1e-6);
    }
    
    {
        float angle = std::numbers::pi_v<float> / 4.0f;
        float cos{ std::cos(angle) };
        float sin{ std::sin(angle) };
        mat2 rot(cos, sin, -sin, cos);
        float distance = PointDistance::sdf_to_oriented_box_at_center(vec2(0.0f), vec2(0.0f), vec2(1.0f), rot);
        assert(std::abs(distance - -1.0f) < 1e-6);

        distance = PointDistance::sdf_to_oriented_box_at_center(vec2(2.0f * sin, 2.0f * cos), vec2(0.0f), vec2(2.0f), rot);
        assert(std::abs(distance) < 1e-6);

        distance = PointDistance::sdf_to_oriented_box_at_center(vec2(-4.0f * sin, -4.0f * cos), vec2(0.0f), vec2(2.0f), rot);
        assert(std::abs(distance - 2.0f) < 1e-6);
    }

    {
        std::array<vec2, 5> polygon{ {vec2(2.0f, 1.0f), vec2(1.0f, 2.0f), vec2(3.0f, 4.0f), vec2(5.0f, 5.0f), vec2(5.0f, 1.0f) }};

        float distance = PointDistance::sdf_to_polygon(polygon.begin(), polygon.end(), vec2(2.0f, 0.0f));
        assert(!Algorithms2D::is_point_inside_polygon(polygon.begin(), polygon.end(), vec2(2.0f, 0.0f)));
        assert(std::abs(distance - 1.0f) < 1e-6);
        assert(std::abs(PointDistance::squared_udf_to_polygon(polygon.begin(), polygon.end(), vec2(2.0f, 0.0f)) - distance * distance) < 1e-6);

        distance = PointDistance::sdf_to_polygon(polygon.begin(), polygon.end(), vec2(3.0f, 1.5f));
        assert(Algorithms2D::is_point_inside_polygon(polygon.begin(), polygon.end(), vec2(3.0f, 1.5f)));
        assert(std::abs(distance - -0.5f) < 1e-6);
        assert(std::abs(PointDistance::squared_udf_to_polygon(polygon.begin(), polygon.end(), vec2(3.0f, 1.5f)) - distance * distance) < 1e-6);
    }

    {
        float distance = PointDistance::sdf_to_regular_poygon(vec2(0.0f), 1.0f, 2);
        assert(std::abs(distance) < 1e-6);

        distance = PointDistance::sdf_to_regular_poygon(vec2(0.0f), 1.0f, 3);
        assert(std::abs(distance - -0.5f) < 1e-6);

        distance = PointDistance::sdf_to_regular_poygon(vec2(0.0f), 1.0f, 4);
        assert(std::abs(distance - -std::sqrt(2.0f)/2.0f) < 1e-6);

        distance = PointDistance::sdf_to_regular_poygon(vec2(1.0f, 0.0f), 1.0f, 2);
        assert(std::abs(distance - 1.0f) < 1e-6);

        distance = PointDistance::sdf_to_regular_poygon(vec2(2.0f, 0.0f), 1.0f, 2);
        assert(std::abs(distance - 2.0f) < 1e-6);
    }
   
    {
        float distance = PointDistance::sdf_to_ellipse(vec2(0.5f, 0.0f), vec2(1.0f, 2.0f));
        assert(std::abs(distance - -0.5f) < 1e-6);

        distance = PointDistance::sdf_to_ellipse(vec2(5.0f, 0.0f), vec2(1.0f, 2.0f));
        assert(std::abs(distance - 4.0f) < 1e-6);
    }

    {
        std::vector<vec3> points{ {vec3(0.0f, 0.0f, 2.0f), vec3(1.0f, 0.0f, 2.0f), vec3(0.0f, 1.0f, 2.0f),
                                   vec3(3.0f, 0.0f, 2.0f), vec3(0.0f, 6.0f, 2.0f)} };
        auto plane = Extra::create_plane(points.begin(), points.begin() + 2);

        float distance = PointDistance::sdf_to_plane(vec3(0.0f), plane);
        assert(std::abs(distance - -2.0f) < 1e-6);

        distance = PointDistance::sdf_to_plane(vec3(0.0f, 0.0f, 1.0f), plane);
        assert(std::abs(distance - -1.0f) < 1e-6);

        distance = PointDistance::sdf_to_plane(vec3(0.0f, 0.0f, 3.0f), plane);
        assert(std::abs(distance - 1.0f) < 1e-6);

        //
        plane = Extra::create_plane(points.begin(), points.end());

        distance = PointDistance::sdf_to_plane(vec3(0.0f), plane);
        assert(std::abs(distance - -2.0f) < 1e-6);

        distance = PointDistance::sdf_to_plane(vec3(0.0f, 0.0f, 1.0f), plane);
        assert(std::abs(distance - -1.0f) < 1e-6);

        distance = PointDistance::sdf_to_plane(vec3(0.0f, 0.0f, 3.0f), plane);
        assert(std::abs(distance - 1.0f) < 1e-6);
    }

    {
        vec3 p0(1.0f);
        vec3 p1(1.0f, 10.0f, 1.0f);

        auto distance = PointDistance::sdf_to_capsule(vec3(1.0f), p0, p1, 1.0f);
        assert(std::abs(distance - -1.0f) < 1e-6);

        distance = PointDistance::sdf_to_capsule(vec3(1.0f, 5.0f, 2.0f), p0, p1, 1.0f);
        assert(std::abs(distance) < 1e-6);

        distance = PointDistance::sdf_to_capsule(vec3(1.0f, 6.0f, 3.0f), p0, p1, 1.0f);
        assert(std::abs(distance - 1.0f) < 1e-6);
    }

    {
        vec3 p0(1.0f);
        vec3 p1(1.0f, 10.0f, 1.0f);

        auto distance = PointDistance::sdf_to_capped_cylinder(vec3(1.0f), p0, p1, 1.0f);
        assert(std::abs(distance) < 1e-6);

        distance = PointDistance::sdf_to_capped_cylinder(vec3(1.0f, 5.0f, 2.0f), p0, p1, 1.0f);
        assert(std::abs(distance) < 1e-6);

        distance = PointDistance::sdf_to_capped_cylinder(vec3(1.0f, 6.0f, 3.0f), p0, p1, 1.0f);
        assert(std::abs(distance - 1.0f) < 1e-6);

        distance = PointDistance::sdf_to_capped_cylinder(vec3(1.0f, 11.0f, 1.0f), p0, p1, 1.0f);
        assert(std::abs(distance - 1.0f) < 1e-6);
    }

    {
        auto distance = PointDistance::sdf_to_bound_ellipsoied(vec3(0.0f), vec3(1.0f, 2.0f, 3.0f));
        assert(std::abs(distance - -1.0f) < 1e-6);

        distance = PointDistance::sdf_to_bound_ellipsoied(vec3(2.0f, 0.0f, 0.0f), vec3(1.0f, 2.0f, 3.0f));
        assert(std::abs(distance - 1.0f) < 1e-6);

        distance = PointDistance::sdf_to_bound_ellipsoied(vec3(0.0f, 4.0f, 0.0f), vec3(1.0f, 2.0f, 3.0f));
        assert(std::abs(distance - 2.0f) < 1e-6);
    }

    {
        vec2 p0_2(1.0f);
        vec2 p1_2(2.0f, 3.0f);
        vec2 p2_2(0.0f, 3.0f);
        vec3 p0_3(1.0f, 1.0f, 0.0f);
        vec3 p1_3(2.0f, 3.0f, 0.0f);
        vec3 p2_3(0.0f, 3.0f, 0.0f);

        float distance_2 = PointDistance::sdf_to_triangle(vec2(0.0f), p0_2, p1_2, p2_2);
        float distance_3 = PointDistance::udf_to_triangle(vec3(0.0f), p0_3, p1_3, p2_3);
        assert(std::abs(distance_3 - distance_2) < 1e-6);

        distance_2 = PointDistance::sdf_to_triangle(vec2(1.0f, -1.0f), p0_2, p1_2, p2_2);
        distance_3 = PointDistance::udf_to_triangle(vec3(1.0f, -1.0f, 0.0f), p0_3, p1_3, p2_3);
        assert(std::abs(distance_3 - distance_2) < 1e-6);

        distance_2 = PointDistance::sdf_to_triangle(vec2(1.0f, 2.5f), p0_2, p1_2, p2_2);
        distance_3 = PointDistance::udf_to_triangle(vec3(1.0f, 2.5f, 0.0f), p0_3, p1_3, p2_3);
        assert(distance_2 < 0.0f);
        assert(distance_3 < 1e-6);

        distance_2 = PointDistance::sdf_to_triangle(vec2(1.5f, 2.5f), p0_2, p1_2, p2_2);
        distance_3 = PointDistance::udf_to_triangle(vec3(1.5f, 2.5f, 0.0f), p0_3, p1_3, p2_3);
        assert(distance_2 < 0.0);
        assert(distance_3 < 1e-6);
    }

    {
        vec3 p0(1.0f, 1.0f, 0.0f);
        vec3 p1(4.0f, 1.0f, 0.0f);
        vec3 p2(4.0f, 4.0f, 0.0f);
        vec3 p3(1.0f, 4.0f, 0.0f);

        float distance = PointDistance::udf_to_quad(vec3(0.0f, 2.0f, 0.0f), p0, p1, p2, p3);
        assert(std::abs(distance - 1.0f) < 1e-6);

        distance = PointDistance::udf_to_quad(vec3(6.0f, 2.0f, 0.0f), p0, p1, p2, p3);
        assert(std::abs(distance - 2.0f) < 1e-6);

        distance = PointDistance::udf_to_quad(vec3(3.0f, 3.0f, 0.0f), p0, p1, p2, p3);
        assert(std::abs(distance) < 1e-6);
    }
}

void test_glsl_ray_intersection() {
    {
        vec3 p0(0.0f, 0.0f, 2.0f);
        vec3 p1(1.0f, 0.0f, 2.0f);
        vec3 p2(0.0f, 1.0f, 2.0f);
        auto plane = Extra::create_plane(p0, p1, p2);

        float distance = RayIntersections::plane_intersect(vec3(0.0f), vec3(1.0f, 0.0f, 0.0f), plane);
        assert(std::abs(distance - -1.0f) < 1e-6);

        distance = RayIntersections::plane_intersect(vec3(0.0f), vec3(0.0f, 0.0f, 1.0f), plane);
        assert(std::abs(distance - 2.0f) < 1e-6);

        distance = RayIntersections::plane_intersect(vec3(5.0f), vec3(0.0f, 0.0f, -1.0f), plane);
        assert(std::abs(distance - -1.0f) < 1e-6);
    }

    {
        dvec3 center(3.0);
        dvec3 normal(0.0, 0.0, 1.0);
        double radius{ 1.0 };

        double distance = RayIntersections::disk_intersect(dvec3(2.5, 2.5, 2.0), dvec3(0.0, 0.0, 1.0), center, normal, radius);
        assert(distance < 0.0);

        distance = RayIntersections::disk_intersect(dvec3(2.5, 2.5, 2.0), dvec3(0.0, 0.0, 1.0), center, -normal, radius);
        assert(std::abs(distance - 1.0f) < 1e-6);

        distance = RayIntersections::disk_intersect(dvec3(1.5, 1.5, 2.0), dvec3(0.0, 0.0, 1.0), center, -normal, radius);
        assert(distance < 0.0);

        distance = RayIntersections::disk_intersect(dvec3(0.0), GLSL::normalize(center), center, -normal, radius);
        assert(std::abs(distance - GLSL::distance(dvec3(0.0), center)) < 1e-6);

        distance = RayIntersections::disk_intersect(dvec3(2.5, 2.5, 5.0), dvec3(0.0, 0.0, -1.0), center, normal, radius);
        assert(std::abs(distance - 2.0f) < 1e-6);
    }

    {
        vec3 p0(1.0f, 1.0f, 0.0f);
        vec3 p1(2.0f, 3.0f, 0.0f);
        vec3 p2(0.0f, 3.0f, 0.0f);

        vec3 intersection = RayIntersections::triangle_intersect_cartesian(vec3(0.0f), GLSL::normalize(p0), p0, p1, p2);
        assert(GLSL::equal(intersection, vec3(-1.0f)));

        intersection = RayIntersections::triangle_intersect_barycentric(vec3(0.0f), GLSL::normalize(p0), p0, p1, p2);
        assert(GLSL::equal(intersection, vec3(-1.0f)));

        intersection = RayIntersections::triangle_intersect_cartesian(vec3(1.0f, 2.0f, 3.0f), vec3(0.0f, 0.0f, -1.0f), p0, p1, p2);
        assert(GLSL::equal(intersection, vec3(1.0f, 2.0f, 0.0f)));

        intersection = RayIntersections::triangle_intersect_barycentric(vec3(1.0f, 2.0f, 3.0f), vec3(0.0f, 0.0f, -1.0f), p0, p1, p2);
        assert(GLSL::equal(intersection, vec3(3.0f, 0.25f, 0.25f)));

        intersection = RayIntersections::triangle_intersect_cartesian(vec3(1.0f, 2.0f, -3.0f), vec3(0.0f, 0.0f, 1.0f), p0, p1, p2);
        assert(GLSL::equal(intersection, vec3(1.0f, 2.0f, 0.0f)));

        intersection = RayIntersections::triangle_intersect_barycentric(vec3(1.0f, 2.0f, -3.0f), vec3(0.0f, 0.0f, 1.0f), p0, p1, p2);
        assert(GLSL::equal(intersection, vec3(3.0f, 0.25f, 0.25f)));
    }

    {
        vec3 center(3.0f, 3.0f, 1.0f);
        vec3 u(1.0f, 0.0f, 0.0f);
        vec3 v(0.0f, 2.0f, 0.0f);

        vec3 intersection = RayIntersections::ellipse_intersect(vec3(3.0f), vec3(0.0f, 0.0f, -1.0f), center, u, v);
        assert(GLSL::equal(intersection, vec3(2.0f, 0.0f, 0.0f)));

        intersection = RayIntersections::ellipse_intersect(vec3(3.5f, 3.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f), center, u, v);
        assert(GLSL::equal(intersection, vec3(1.0f, 0.0f, 0.5f)));

        intersection = RayIntersections::ellipse_intersect(vec3(3.0f, 3.5f, 0.0f), vec3(0.0f, 0.0f, 1.0f), center, u, v);
        assert(GLSL::equal(intersection, vec3(1.0f, 1.0f, 0.0f)));

        intersection = RayIntersections::ellipse_intersect(vec3(3.5f, 3.5f, 0.0f), vec3(0.0f, 0.0f, 1.0f), center, u, v);
        assert(GLSL::equal(intersection, vec3(-1.0f)));
    }

    {
        vec3 center(3.0f);
        vec4 sphere(center, 2.0f);

        vec2 intersection = RayIntersections::sphere_intersect(vec3(3.0f, 3.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f), sphere);
        assert(GLSL::equal(intersection, vec2(1.0f, 5.0f)));

        intersection = RayIntersections::sphere_intersect(vec3(3.0f, 3.0f, 2.5f), vec3(0.0f, 0.0f, 1.0f), sphere);
        assert(intersection.x < 0.0f);
        assert(std::abs(intersection.y - 2.5f) < 1e-6);

        intersection = RayIntersections::sphere_intersect(vec3(3.0f, 2.5f, 3.0f), vec3(0.0f, 1.0f, 0.0f), sphere);
        assert(intersection.x < 0.0f);
        assert(std::abs(intersection.y - 2.5f) < 1e-6);
    }

    {
        vec2 intersections = RayIntersections::ellipsoid_intersection(vec3(10.0f, 0.0f, 0.0f), vec3(-1.0f, 0.0f, 0.0f), vec3(1.0f, 2.0f, 3.0f));
        assert(GLSL::max(GLSL::abs(intersections - vec2(9.0f, 11.0f))) < 1e-6);

        intersections = RayIntersections::ellipsoid_intersection(vec3(0.0f, 10.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f), vec3(1.0f, 2.0f, 3.0f));
        assert(GLSL::max(GLSL::abs(intersections - vec2(8.0f, 12.0f))) < 1e-6);

        intersections = RayIntersections::ellipsoid_intersection(vec3(0.0f, 0.0f, 10.0f), vec3(0.0f, 0.0f, -1.0f), vec3(1.0f, 2.0f, 3.0f));
        assert(GLSL::max(GLSL::abs(intersections - vec2(7.0f, 13.0f))) < 1e-6);
    }

    {
        const vec3 min(-2.0f);
        const vec3 max(3.5f);

        vec2 intersections = RayIntersections::aabb_intersection(vec3(-1.0f, -1.0f, -4.0f), vec3(0.0f, 0.0f, 1.0f), min, max);
        assert(static_cast<std::uint8_t>(intersections[0])      == 2);
        assert(static_cast<std::uint8_t>(intersections[1] * 10.0f) == 75);
    }
}

void test_GLSL_algorithms_2D() {
    {
        auto left = Algorithms2D::Internals::is_point_left_of(vec2(0.0f), vec2(0.0f));
        assert(!left);

        left = Algorithms2D::Internals::is_point_left_of(vec2(1.0f), vec2(0.0f));
        assert(!left);

        left = Algorithms2D::Internals::is_point_left_of(vec2(0.0f), vec2(1.0f));
        assert(left);
    }

    {
        auto area = Algorithms2D::Internals::triangle_twice_area(vec2(0.0f), vec2(0.0f, 1.0f), vec2(1.0f, 0.0f));
        assert(static_cast<int>(area * 10) == 10);
    }

    {
        auto point = Algorithms2D::Internals::get_rays_intersection_point(vec2(0.0f, 1.0f), vec2(1.0f, 0.0f), vec2(1.0f, 0.0f), vec2(0.0f, 1.0f));
        assert(static_cast<int>(point.x) == 1);
        assert(static_cast<int>(point.y) == 1);

        point = Algorithms2D::Internals::get_rays_intersection_point(vec2(0.0f, 0.0f), GLSL::normalize(vec2(1.0f)), vec2(1.0f, 0.0f), GLSL::normalize(vec2(-1.0f, 1.0f)));
        assert(static_cast<int>(point.x * 10) == 5);
        assert(static_cast<int>(point.y * 10) == 5);
    }

    {
        auto projected = Algorithms2D::Internals::project_point_on_segment(vec2(0.0f), vec2(0.0f, 1.0f), vec2(0.5f, 10.0f));
        assert(static_cast<int>(projected.point.x) == 0);
        assert(static_cast<int>(projected.point.y) == 10);
        assert(static_cast<int>(projected.t) == 10);

        projected = Algorithms2D::Internals::project_point_on_segment(vec2(1.0f, 0.0f), vec2(1.0f, 5.0f), vec2(2.0f, -10.0f));
        assert(static_cast<int>(projected.point.x) == 1);
        assert(static_cast<int>(projected.point.y) == -10);
        assert(static_cast<int>(projected.t) == -2);
    }

    {
        auto circumCircle = Algorithms2D::Internals::get_circumcircle(vec2(5.0f), vec2(10.0f));
        assert(static_cast<int>(circumCircle.center.x * 10) == 75);
        assert(static_cast<int>(circumCircle.center.y * 10) == 75);
        assert(static_cast<int>(circumCircle.radius_squared * 10) == 125);

        circumCircle = Algorithms2D::Internals::get_circumcircle(vec2(-5.0f), vec2(0.0f));
        assert(static_cast<int>(circumCircle.center.x * 10) == -25);
        assert(static_cast<int>(circumCircle.center.y * 10) == -25);
        assert(static_cast<int>(circumCircle.radius_squared * 10) == 125);

        auto circumCircle3 = Algorithms2D::Internals::get_circumcircle(vec2(-5.0f, 0.0f), vec2(5.0f, 0.0f), vec2(0.0f, 5.0f));
        assert(static_cast<int>(circumCircle3.center.x) == 0);
        assert(static_cast<int>(circumCircle3.center.y) == 0);
        assert(static_cast<int>(circumCircle3.radius_squared) == 25);

        circumCircle3 = Algorithms2D::Internals::get_circumcircle(vec2(-5.0f, 2.0f), vec2(5.0f, 0.0f), vec2(2.0f, 5.0f));
        assert(static_cast<int>(circumCircle3.center.x * 1e5) == -13636);
        assert(static_cast<int>(circumCircle3.center.y * 1e5) == 31818);
        assert(static_cast<int>(circumCircle3.radius_squared * 1e4) == 264834);
    }

    {
        std::vector<vec2> polygon{ {vec2(3.0f, 1.0f), vec2(5.0f, 1.0f), vec2(5.0f, 4.0f), vec2(4.0f, 6.0f), vec2(7.0f, 7.0f), vec2(10.0f, 7.0f), vec2(10.0f, 9.0f),
                                    vec2(8.0f, 9.0f), vec2(6.0f, 10.0f), vec2(1.0f, 10.0f), vec2(1.0f, 8.0f), vec2(2.0f, 8.0f), vec2(2.0f, 6.0f), vec2(1.0f, 6.0f),
                                    vec2(1.0f, 2.0f)} };
        auto area = Algorithms2D::Internals::get_polygon_area(polygon.begin(), polygon.end());
        assert(static_cast<std::int32_t>(area * 10.0f) == 445);

        for (auto it{ polygon.begin() }, nt{ polygon.begin() + 1 }; nt != polygon.end(); ++it, ++nt) {
            assert(Algorithms2D::is_line_connecting_polygon_vertices_inside_polygon(polygon.begin(), polygon.end(), area, it, nt));
        }

        assert(Algorithms2D::is_line_connecting_polygon_vertices_inside_polygon(polygon.begin(), polygon.end(), area, polygon.begin(), polygon.begin() + 2));
        assert(Algorithms2D::is_line_connecting_polygon_vertices_inside_polygon(polygon.begin(), polygon.end(), area, polygon.begin() + 3, polygon.begin() + 11));
        assert(Algorithms2D::is_line_connecting_polygon_vertices_inside_polygon(polygon.begin(), polygon.end(), area, polygon.begin() + 9, polygon.begin() + 11));
        assert(!Algorithms2D::is_line_connecting_polygon_vertices_inside_polygon(polygon.begin(), polygon.end(), area, polygon.begin() + 2, polygon.begin() + 8));
        assert(!Algorithms2D::is_line_connecting_polygon_vertices_inside_polygon(polygon.begin(), polygon.end(), area, polygon.begin() + 6, polygon.begin() + 8));
        assert(!Algorithms2D::is_line_connecting_polygon_vertices_inside_polygon(polygon.begin(), polygon.end(), area, polygon.begin(), polygon.begin() + 4));  // false
    }

    {
        std::vector<vec2> polygon{ {vec2(3.0f, 1.0f), vec2(5.0f, 1.0f), vec2(5.0f, 4.0f), vec2(4.0f, 6.0f), vec2(7.0f, 7.0f), vec2(10.0f, 7.0f), vec2(10.0f, 9.0f),
                                    vec2(8.0f, 9.0f), vec2(6.0f, 10.0f), vec2(1.0f, 10.0f), vec2(1.0f, 8.0f), vec2(2.0f, 8.0f), vec2(2.0f, 6.0f), vec2(1.0f, 6.0f),
                                    vec2(1.0f, 2.0f)} };

        // get reflex vertices
        const Algorithms2D::Winding winding{ Algorithms2D::get_polygon_winding(polygon.begin(), polygon.end()) };
        const auto reflex_vertices = Algorithms2D::get_reflex_vertices(polygon.begin(), polygon.end(), winding);
        const std::vector<vec2> reflexis{ { vec2(4.0f, 6.0f), vec2(7.0f, 7.0f), vec2(8.0f, 9.0f),
                                            vec2(2.0f, 8.0f), vec2(2.0f, 6.0f) } };
        for (std::size_t i{}; i < reflexis.size(); ++i) {
            assert(GLSL::max(GLSL::abs(reflexis[i] - *reflex_vertices[i])) < 1e-6f);
        }

        // get cusps in x
        const auto cusp_x_vertices = Algorithms2D::get_cusp_vertices(polygon.begin(), polygon.end());
        assert(cusp_x_vertices.size() == 1);
        assert(GLSL::max(GLSL::abs(vec2(4.0f, 6.0f) - *cusp_x_vertices[0])) < 1e-6f);

        // get cusps in y
        const auto cusp_y_vertices = Algorithms2D::get_cusp_vertices(polygon.begin(), polygon.end(), 1);
        assert(cusp_y_vertices.size() == 0);
    }

    {
        std::vector<vec2> polygon{ {vec2(3.0f, 1.0f), vec2(5.0f, 1.0f), vec2(5.0f, 4.0f), vec2(4.0f, 6.0f), vec2(7.0f, 7.0f), vec2(10.0f, 7.0f), vec2(10.0f, 9.0f),
                                    vec2(8.0f, 9.0f), vec2(6.0f, 10.0f), vec2(1.0f, 10.0f), vec2(1.0f, 8.0f), vec2(2.0f, 8.0f), vec2(2.0f, 6.0f), vec2(1.0f, 6.0f),
                                    vec2(1.0f, 2.0f)} };
        assert(!Algorithms2D::is_polygon_convex(polygon.begin(), polygon.end()));

        // convex hull test
        const auto convex = Algorithms2D::get_convex_hull(polygon.begin(), polygon.end());
        assert(Algorithms2D::is_polygon_convex(convex.begin(), convex.end()));
        const std::vector<vec2> expected_convex{ {vec2(1.0f, 2.0f), vec2(3.0f, 1.0f), vec2(5.0f, 1.0f), vec2(10.0f, 7.0f),
                                                  vec2(10.0f, 9.0f), vec2(6.0f, 10.0f), vec2(1.0f, 10.0f)} };
        //assert(expected_convex.size() == convex.size());
        for (std::size_t i{}; i < expected_convex.size(); ++i) {
            assert(GLSL::max(GLSL::abs(expected_convex[i] - convex[i])) < 1e-6);
        }

        // obb test
        const auto obb = Algorithms2D::get_bounding_rectangle(convex);
        assert(GLSL::max(GLSL::abs(vec2(1.0f, 1.0f) - obb.p0)) < 1e-6);
        assert(GLSL::max(GLSL::abs(vec2(10.0f, 1.0f) - obb.p1)) < 1e-6);
        assert(GLSL::max(GLSL::abs(vec2(10.0f, 10.0f) - obb.p2)) < 1e-6);
        assert(GLSL::max(GLSL::abs(vec2(1.0f, 10.0f) - obb.p3)) < 1e-6);

        // point inside polygon
        bool inside = Algorithms2D::is_point_inside_polygon(polygon.cbegin(), polygon.cend(), vec2(7.0f, 6.0f));
        assert(inside == false);
        inside = Algorithms2D::is_point_inside_polygon(polygon.cbegin(), polygon.cend(), vec2(7.0f, 8.0f));
        assert(inside);
        inside = Algorithms2D::is_point_inside_polygon(polygon.cbegin(), polygon.cend(), vec2(1.0f, 1.0f));
        assert(inside == false);
        inside = Algorithms2D::is_point_inside_polygon(polygon.cbegin(), polygon.cend(), vec2(1.5f, 7.0f));
        assert(inside == false);
        inside = Algorithms2D::is_point_inside_polygon(polygon.cbegin(), polygon.cend(), vec2(2.0f, 9.0f));
        assert(inside);

        // centroid
        std::vector<vec2> cents{ obb.p0, obb.p1, obb.p2, obb.p3 };
        const vec2 centroid = Algorithms2D::Internals::get_centroid(cents.cbegin(), cents.cend());
        assert(GLSL::max(GLSL::abs(vec2(5.5f, 5.5f) - centroid)) < 1e-6);

        // convex hull diameter
        const auto antipodal = Algorithms2D::get_convex_diameter(convex);
        assert(antipodal.indices[0] == 0);
        assert(antipodal.indices[1] == 4);

        // convex hull minimal bounding circle
        const auto circle = Algorithms2D::get_minimal_bounding_circle(convex);
        assert(static_cast<std::int32_t>(circle.center.x * 100) == 511);
        assert(static_cast<std::int32_t>(circle.center.y * 100) == 600);
        assert(static_cast<std::int32_t>(circle.radius_squared * 100) == 3290);
        for (const auto& p : convex) {
            assert(GLSL::dot(p - circle.center) <= circle.radius_squared);
        }

        // check orthogonality
        bool is_orthogonal{ Algorithms2D::is_polygon_orthogonal(polygon.begin(), polygon.end()) };
        assert(!is_orthogonal);

        // is monotone relative to Y axis?
        bool is_monotone = Algorithms2D::is_polygon_monotone_relative_to_line(polygon.begin(), polygon.end(), vec2(0.0f), vec2(0.0f, 1.0f));
        assert(is_monotone);

        // is monotone relative to X axis?
        is_monotone = Algorithms2D::is_polygon_monotone_relative_to_line(polygon.begin(), polygon.end(), vec2(0.0f), vec2(1.0f, 0.0f));
        assert(!is_monotone);

        // is convex hull monotone relative to Y axis?
        is_monotone = Algorithms2D::is_polygon_monotone_relative_to_line(convex.begin(), convex.end(), vec2(0.0f), vec2(0.0f, 1.0f));
        assert(is_monotone);

        // is convex hull monotone relative to X axis?
        is_monotone = Algorithms2D::is_polygon_monotone_relative_to_line(convex.begin(), convex.end(), vec2(0.0f), vec2(1.0f, 0.0f));
        assert(is_monotone);

        std::vector<vec2> mon{ {vec2(2.0, 0.0), vec2(5.0, 0.0), vec2(5.0, 10.0),
                                vec2(2.0, 10.0), vec2(1.0, 8.0), vec2(2.0, 6.0),
                                vec2(1.0, 3.0)} };

        // should not me monotone with regard to X axis
        is_monotone = Algorithms2D::is_polygon_monotone_relative_to_line(mon.begin(), mon.end(), vec2(0.0f), vec2(1.0f, 0.0f));
        assert(!is_monotone);

        // should be monotone with regard to Y axis
        is_monotone = Algorithms2D::is_polygon_monotone_relative_to_line(mon.begin(), mon.end(), vec2(0.0f), vec2(0.0f, 1.0f));
        assert(is_monotone);
    }

    {
        for (std::size_t angle{}; angle < 180; angle += 5) {
            std::vector<vec2> points;
            const float tanAngle{ std::tan(static_cast<float>(angle) * 3.141592653589f / 180.0f) };
            float sign{ 1.0f };
            for (std::size_t x{}; x < 100; ++x) {
                points.emplace_back(vec2(static_cast<float>(x), tanAngle * static_cast<float>(x) + sign * static_cast<float>(rand()) / RAND_MAX));
                sign *= -1.0f;
            }

            const vec2 centroid{ Algorithms2D::Internals::get_centroid(points.cbegin(), points.cend()) };
            const vec2 direction{ Algorithms2D::get_principle_axis(points.cbegin(), points.cend(), centroid) };
            const float error{ std::abs(angle - std::atan2(direction.y, direction.x) * 180.0f / 3.141592653589f) };
            assert(error <= 0.2f || std::abs(180.0f - error) <= 0.2f);
        }
    }

    {
        std::vector<vec2> polygon{ {vec2(0.0f,3.0f), vec2(1.0f,3.0f), vec2(1.0f,2.0f), vec2(2.0f,2.0f), vec2(2.0f,4.0f),
                                    vec2(3.0f,4.0f), vec2(3.0f,0.0f), vec2(4.0f,0.0f), vec2(4.0f,1.0f), vec2(5.0f,1.0f),
                                    vec2(5.0f,0.0f), vec2(6.0f,0.0f), vec2(6.0f,2.0f), vec2(7.0f,2.0f), vec2(7.0f,1.0f),
                                    vec2(8.0f,1.0f), vec2(8.0f,4.0f), vec2(9.0f,4.0f), vec2(9.0f,7.0f), vec2(8.0f,7.0f),
                                    vec2(8.0f,9.0f), vec2(7.0f,9.0f), vec2(7.0f,5.0f), vec2(6.0f,5.0f), vec2(6.0f,6.0f),
                                    vec2(5.0f,6.0f), vec2(5.0f,9.0f), vec2(4.0f,9.0f), vec2(4.0f,8.0f), vec2(6.0f,8.0f),
                                    vec2(6.0f,7.0f), vec2(2.0f,7.0f), vec2(2.0f,9.0f), vec2(1.0f,9.0f), vec2(1.0f,6.0f),
                                    vec2(0.0f,6.0f)} };

        // check orthogonality
        bool is_orthogonal{ Algorithms2D::is_polygon_orthogonal(polygon.begin(), polygon.end()) };
        assert(is_orthogonal);

        // is monotone relative to Y axis?
        bool is_monotone = Algorithms2D::is_polygon_monotone_relative_to_line(polygon.begin(), polygon.end(), vec2(0.0f), vec2(0.0f, 1.0f));
        assert(!is_monotone);

        // is monotone relative to X axis?
        is_monotone = Algorithms2D::is_polygon_monotone_relative_to_line(polygon.begin(), polygon.end(), vec2(0.0f), vec2(1.0f, 0.0f));
        assert(is_monotone);
    }

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
        envelope_test_svg.add_point_cloud(points.begin(), points.end(), 2.0f, "none", "black", 0.5f);
        envelope_test_svg.add_polyline(envelope.top.begin(), envelope.top.end(), "none", "red", 1.0f);
        envelope_test_svg.add_polyline(envelope.bottom.begin(), envelope.bottom.end(), "none", "green", 1.0f);
        envelope_test_svg.to_file("envelope_test_svg.svg");
    }

    {
        // define polygon
        std::vector<vec2> polygon{ {vec2(3.0f, 1.0f), vec2(5.0f, 1.0f), vec2(5.0f, 4.0f), vec2(4.0f, 6.0f), vec2(7.0f, 7.0f), vec2(10.0f, 7.0f), vec2(10.0f, 9.0f),
                                     vec2(8.0f, 9.0f), vec2(6.0f, 10.0f), vec2(1.0f, 10.0f), vec2(1.0f, 8.0f), vec2(2.0f, 8.0f), vec2(2.0f, 6.0f), vec2(1.0f, 6.0f),
                                     vec2(1.0f, 2.0f)} };

        // triangulate ("earcut" style) polygon
        std::vector<std::vector<vec2>::iterator> earcut{ Algorithms2D::triangulate_polygon_earcut(polygon.begin(), polygon.end()) };

        // calculate convex hull
        auto convex = Algorithms2D::get_convex_hull(polygon.begin(), polygon.end());

        // calculate minimal area bounding rectangle
        const auto obb = Algorithms2D::get_bounding_rectangle(convex);
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
        const auto aabb = AxisLignedBoundingBox::point_cloud_aabb(polygon.begin(), polygon.end());
        const std::vector<vec2> delaunay{ Algorithms2D::triangulate_polygon_delaunay(polygon.begin(), polygon.end(), aabb) };
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
    }

    {
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

        std::vector<vec2> points_in_circle_not_kdtree{ Algorithms2D::range_query(points.begin(), points.end(), circle_center, circle_radius) };
        assert(points_in_circle_not_kdtree.size() == points_in_circle.size());

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
    }

    {
        // uniformly place points on a grid
        std::vector<vec2> points;
        for (std::int32_t x{ -100 }; x < 100; x += 10) {
            for (std::int32_t y{ -100 }; y < 100; y += 10) {
                points.emplace_back(vec2(static_cast<float>(x), static_cast<float>(y)));
            }
        }

        // continuously place last point somewhere in the range of the grid
        for (std::int32_t x{ -95 }; x < 95; x += 60) {
            for (std::int32_t y{ -95 }; y < 95; y += 60) {
                const vec2 p(static_cast<float>(x), static_cast<float>(y));
                points.emplace_back(p);
                auto closest_pair = Algorithms2D::get_closest_pair(points.begin(), points.end());
                assert(Extra::are_vectors_identical(p, *closest_pair.p0) || Extra::are_vectors_identical(p, *closest_pair.p1));
                points.pop_back();
            }
        }
    }

    {
        std::vector<vec2> triangle{ {vec2(4.0f, 11.0f), vec2(4.0, 5.0f), vec2(9.0f, 9.0f) } };
        std::vector<vec2> quad{ {vec2(5.0f, 7.0f), vec2(7.0f, 3.0f), vec2(10.0f, 2.0f), vec2(12.0f, 7.0f) } };
        const vec2 quad_centroid{ Algorithms2D::Internals::get_centroid(quad.begin(), quad.end()) };
        vec2 triangle_centroid{ Algorithms2D::Internals::get_centroid(triangle.begin(), triangle.end()) };
        assert(Algorithms2D::do_convex_polygons_intersect(triangle, triangle_centroid, quad, quad_centroid));

        triangle[1] = vec2(5.1f, 6.9f);
        triangle_centroid = Algorithms2D::Internals::get_centroid(triangle.begin(), triangle.end());
        assert(Algorithms2D::do_convex_polygons_intersect(triangle, triangle_centroid, quad, quad_centroid));

        triangle[1] = vec2(5.1f, 7.1f);
        triangle_centroid = Algorithms2D::Internals::get_centroid(triangle.begin(), triangle.end());
        assert(!Algorithms2D::do_convex_polygons_intersect(triangle, triangle_centroid, quad, quad_centroid));
    }

    {
        // define polygon
        std::vector<vec2> polygon{ {vec2(3.0f, 1.0f), vec2(5.0f, 1.0f), vec2(5.0f, 4.0f), vec2(4.0f, 6.0f), vec2(7.0f, 7.0f), vec2(10.0f, 7.0f), vec2(10.0f, 9.0f),
                                     vec2(8.0f, 9.0f), vec2(6.0f, 10.0f), vec2(1.0f, 10.0f), vec2(1.0f, 8.0f), vec2(2.0f, 8.0f), vec2(2.0f, 6.0f), vec2(1.0f, 6.0f),
                                     vec2(1.0f, 2.0f)} };
        for (vec2& p : polygon) {
            p = 50.0f * p + 50.0f;
        }
        const vec2 centroid{ Algorithms2D::Internals::get_centroid(polygon.begin(), polygon.end()) };
        Algorithms2D::sort_points<Algorithms2D::Winding::CounterClockWise>(polygon.begin(), polygon.end(), centroid);

        std::vector<vec2> clipped1{ Algorithms2D::clip_polygon_by_infinte_line(polygon.begin(), polygon.end(), vec2(200.0f, 300.0f), vec2(1.0f, 1.0f)) };
        std::vector<vec2> clipped2{ Algorithms2D::clip_polygon_by_infinte_line(polygon.begin(), polygon.end(), vec2(200.0f, 300.0f), vec2(1.0f, -1.0f)) };

        // export information to svg file
        svg<vec2> polygon_clipped_svg(600, 600);
        polygon_clipped_svg.add_polygon(polygon.begin(), polygon.end(), "none", "black", 1.0f);

        polygon_clipped_svg.add_polygon(clipped1.begin(), clipped1.end(), "none", "red", 2.0f);
        polygon_clipped_svg.add_point_cloud(clipped1.begin(), clipped1.end(), 4.0f, "none", "red", 2.0f);

        polygon_clipped_svg.add_polygon(clipped2.begin(), clipped2.end(), "none", "green", 2.0f);
        polygon_clipped_svg.add_point_cloud(clipped2.begin(), clipped2.end(), 4.0f, "none", "green", 2.0f);

        polygon_clipped_svg.to_file("polygon_clipped_svg.svg");
    }

    {
        std::vector<vec2> polygon{ {vec2(90.0f, 60.0f), vec2(40.0f, 80.0f), vec2(0.0f,  50.0f),
                                    vec2(20.0f, 40.0f), vec2(60.0f, 40.0f), vec2(30.0f, 20.0f),
                                    vec2(10.0f, 10.0f), vec2(50.0f,  0.0f), vec2(90.0f, 30.0f) }};
        float min_area{ std::numeric_limits<float>::max() };
        for (std::size_t i{}; i < polygon.size(); ++i) {
            const std::size_t p{ (i - 1) % polygon.size() };
            const std::size_t n{ (i + 1) % polygon.size() };
            const float area{ Algorithms2D::Internals::triangle_twice_area(polygon[p], polygon[i], polygon[n]) / 2.0f };
            if (area < min_area) {
                min_area = area;
            }            
        }
        const auto simplified{ Algorithms2D::simplify_polygon(polygon.begin(), polygon.end(), 0.1f, min_area + 1.0f) };
        assert(simplified.size() == polygon.size() - 1);

        svg<vec2> polygon_simplification_svg(100, 100);
        polygon_simplification_svg.add_polygon(polygon.begin(),    polygon.end(), "none", "black", 2.0f);
        polygon_simplification_svg.add_polygon(simplified.begin(), simplified.end(), "none", "red", 2.0f);
        polygon_simplification_svg.to_file("polygon_simplification_svg.svg");
    }
}

void test_glsl_space_partitioning() {
    // 1000 points randomly created inside rectangle [0,0] to [100,100]
    std::vector<vec2> points;
    for (std::size_t i{}; i < 10000; ++i) {
        points.emplace_back(vec2(static_cast<float>(rand()) / RAND_MAX * 100.0f,
                                 static_cast<float>(rand()) / RAND_MAX * 100.0f));
    }

    // kd-tree
    {
        // construction
        SpacePartitioning::KDTree<vec2> kdtree;
        kdtree.construct(points.begin(), points.end());

        // search nearest neighbors in cube
        {
            const vec2 center(50.0f);
            const float extent{ 5.0f };

            auto pointsInCube = kdtree.range_query(SpacePartitioning::RangeSearchType::Manhattan, center, extent);

            for (std::size_t i{}; i < pointsInCube.size(); ++i) {
                assert(GLSL::max(GLSL::abs(points[pointsInCube[i].second] - center)) <= extent);
            }

            std::size_t amount_of_points_in_rectangle{};
            for (vec2 p : points) {
                amount_of_points_in_rectangle += GLSL::max(GLSL::abs(p - center)) <= extent;
            }
            assert(amount_of_points_in_rectangle - pointsInCube.size() <= 1);
        }

        // search nearest neighbors in radius
        {
            const vec2 center(50.0f);
            const float radius{ 5.0f };

            auto pointsInSphere = kdtree.range_query(SpacePartitioning::RangeSearchType::Radius, center, radius);
            auto max_point = std::ranges::max_element(pointsInSphere, [](const auto& a, const auto& b) {
                return (a.first < b.first);
            });
            assert((*max_point).first <= radius * radius);

            for (std::size_t i{}; i < pointsInSphere.size(); ++i) {
                assert(pointsInSphere[i].first <= radius * radius);
                assert(GLSL::dot(points[pointsInSphere[i].second] - center) <= radius * radius);
            }

            std::size_t amount_of_points_in_sphere{};
            for (vec2 p : points) {
                if (GLSL::dot(p - center) <= radius * radius) {
                    ++amount_of_points_in_sphere;
                }
            }
            assert(amount_of_points_in_sphere == pointsInSphere.size());
        }

        // search k nearest neighbors in cube
        {
            const vec2 center(50.0f);
            std::vector<float> closest;
            for (const vec2 p : points) {
                closest.emplace_back(GLSL::dot(center - p));
            }
            std::ranges::sort(closest, std::less<float>());

            const std::size_t len{ 19 };
            const auto nearest10 = kdtree.nearest_neighbors_query(center, len);

            for (std::size_t i{}; i < len; ++i) {
                assert(std::abs(closest[i] - nearest10[i].first) < std::numeric_limits<float>::min());
            }
        }

        // destruction
        kdtree.clear();
        assert(sizeof(kdtree) == 2 * sizeof(std::size_t));
    }

    // grid
    {
        // construction
        SpacePartitioning::Grid<vec2> grid;
        grid.construct(points.begin(), points.end());

        // search nearest neighbors in cube
        {
            const vec2 center(50.0f);
            const float extent{ 5.0f };

            auto pointsInCube = grid.range_query(SpacePartitioning::RangeSearchType::Manhattan, center, extent);

            for (std::size_t i{}; i < pointsInCube.size(); ++i) {
                assert(GLSL::max(GLSL::abs(points[pointsInCube[i].second] - center)) <= extent);
            }

            std::size_t amount_of_points_in_rectangle{};
            for (vec2 p : points) {
                if (GLSL::max(GLSL::abs(p - center)) <= extent) {
                    ++amount_of_points_in_rectangle;
                }
            }
            assert(amount_of_points_in_rectangle == pointsInCube.size());
        }

        // search nearest neighbors in radius
        {
            const vec2 center(50.0f);
            const float radius{ 5.0f };

            auto pointsInSphere = grid.range_query(SpacePartitioning::RangeSearchType::Radius, center, radius);
            auto max_point = std::ranges::max_element(pointsInSphere, [](const auto& a, const auto& b) {
                return (a.first < b.first);
            });
            assert((*max_point).first <= radius * radius);

            for (std::size_t i{}; i < pointsInSphere.size(); ++i) {
                assert(pointsInSphere[i].first <= radius * radius);
                assert(GLSL::dot(points[pointsInSphere[i].second] - center) <= radius * radius);
            }

            std::size_t amount_of_points_in_sphere{};
            for (vec2 p : points) {
                if (GLSL::dot(p - center) <= radius * radius) {
                    ++amount_of_points_in_sphere;
                }
            }
            assert(amount_of_points_in_sphere == pointsInSphere.size());
        }

        // search k nearest neighbors in cube
        {
            const vec2 center(50.0f);
            std::vector<float> closest;
            for (const vec2 p : points) {
                closest.emplace_back(GLSL::dot(center - p));
            }
            std::ranges::sort(closest, std::less<float>());

            const std::size_t len{ 19 };
            const auto nearest10 = grid.nearest_neighbors_query(center, len);

            for (std::size_t i{}; i < len; ++i) {
                assert(std::abs(closest[i] - nearest10[i].first) < std::numeric_limits<float>::min());
            }
        }

        // destruction
        grid.clear();
    }
}


void test_GLSL_clustering() {
   // dbscan
   {
       svg<vec2> dbscan_test_svg(600, 600);
       std::vector<vec2> points;
       float sign{ 0.5f };

       // cluster #1
       const vec2 center(20.0f, 12.0f);
       const float radius{ 3.0f };
       for (std::size_t i{}; i < 60; ++i) {
           float fi{ static_cast<float>(i) };
           points.emplace_back(vec2(center.x + radius * std::cos(fi) + sign * static_cast<float>(rand()) / RAND_MAX,
                                    center.y + radius * std::sin(fi) + sign * static_cast<float>(rand()) / RAND_MAX));
           sign *= -1.0f;
       }

       // cluster #2
       for (std::size_t i{}; i < 40; ++i) {
           float fi{ static_cast<float>(i) };
           float x{ fi + sign * static_cast<float>(rand()) / RAND_MAX };
           points.emplace_back(vec2(x, std::sqrt(x) + sign * static_cast<float>(rand()) / RAND_MAX));
           sign *= -1.0f;
       }

       // noise
       for (std::size_t i{}; i < 10; ++i) {
           points.emplace_back(vec2(50.0f * static_cast<float>(rand()) / RAND_MAX,
                                    50.0f * static_cast<float>(rand()) / RAND_MAX));
       }

       // partition #1 (using kd-tree)
       SpacePartitioning::KDTree<vec2> kdtree;
       const auto clusterIds0 = Clustering::get_density_based_clusters(points.cbegin(), points.cend(), kdtree, 1.0f, 10);
       assert(clusterIds0.clusters.empty());
       assert(clusterIds0.noise.size() == points.size());

       // partition #2 (using kd-tree)
       kdtree.clear();
       const auto clusterIds1 = Clustering::get_density_based_clusters(points.cbegin(), points.cend(), kdtree, radius, 4);
       assert(clusterIds1.clusters.size() == 2);
       assert(clusterIds1.clusters[0].size() >= 58 && clusterIds1.clusters[0].size() <= 62);
       assert(clusterIds1.clusters[1].size() >= 38 && clusterIds1.clusters[1].size() <= 42);
       assert(clusterIds1.noise.size() > 6);
       kdtree.clear();

       for (const std::size_t i : clusterIds1.clusters[0]) {
           dbscan_test_svg.add_circle(points[i] * 10.0f, 2.0f, "red", "red", 1.0f);
       }
       for (const std::size_t i : clusterIds1.clusters[1]) {
           dbscan_test_svg.add_circle(points[i] * 10.0f, 2.0f, "blue", "blue", 1.0f);
       }
       for (const std::size_t i : clusterIds1.noise) {
           dbscan_test_svg.add_circle(points[i] * 10.0f, 2.0f, "green", "green", 1.0f);
       }
       dbscan_test_svg.to_file("dbscan_test.svg");

       // partition #2 (using grid)
       SpacePartitioning::Grid<vec2> grid;
       const auto clusterIds2 = Clustering::get_density_based_clusters(points.cbegin(), points.cend(), grid, radius, 4);
       assert(clusterIds2.clusters.size() == 2);
       assert(clusterIds2.clusters[0].size() > 58 && clusterIds1.clusters[0].size() < 62);
       assert(clusterIds2.clusters[1].size() > 38 && clusterIds1.clusters[1].size() < 42);
       assert(clusterIds2.noise.size() > 6);
       grid.clear();
   }

   // k-means
   {
       svg<vec2> kmean_test_svg(250, 160);
       std::vector<vec2> points;
       float sign{ 0.5f };

       // cluster #1
       vec2 center(20.0f, 12.0f);
       float radius{ 3.0f };
       for (std::size_t i{}; i < 60; ++i) {
           float fi{ static_cast<float>(i) };
           points.emplace_back(vec2(center.x + radius * std::cos(fi) + sign * static_cast<float>(rand()) / RAND_MAX,
                                    center.y + radius * std::sin(fi) + sign * static_cast<float>(rand()) / RAND_MAX));
           sign *= -1.0f;
       }

       // cluster #2
       center = vec2(8.0f, 2.0f);
       radius = 3.0f;
       for (std::size_t i{}; i < 20; ++i) {
           float fi{ static_cast<float>(i) };
           points.emplace_back(vec2(center.x + radius * std::cos(fi) + sign * static_cast<float>(rand()) / RAND_MAX,
               center.y + radius * std::sin(fi) + sign * static_cast<float>(rand()) / RAND_MAX));
           sign *= -1.0f;
       }

       // cluster #3
       center = vec2(-2.0f, 2.0f);
       radius = 3.0f;
       for (std::size_t i{}; i < 80; ++i) {
           float fi{ static_cast<float>(i) };
           points.emplace_back(vec2(center.x + radius * std::cos(fi) + sign * static_cast<float>(rand()) / RAND_MAX,
               center.y + radius * std::sin(fi) + sign * static_cast<float>(rand()) / RAND_MAX));
           sign *= -1.0f;
       }

       // partition
       const auto clusterIds = Clustering::k_means(points.cbegin(), points.cend(), 3, 20, 0.01f);
       assert(clusterIds.size() == 3);
       std::array<std::size_t, 3> cluster_sizes{ {clusterIds[0].size(), clusterIds[1].size(), clusterIds[2].size()} };
       std::ranges::sort(cluster_sizes);
       assert(cluster_sizes[0] == 20);
       assert(cluster_sizes[1] == 60);
       assert(cluster_sizes[2] == 80);

       for (const std::size_t i : clusterIds[0]) {
           kmean_test_svg.add_circle(points[i] * 10.0f, 2.0f, "red", "red", 1.0f);
       }
       for (const std::size_t i : clusterIds[1]) {
           kmean_test_svg.add_circle(points[i] * 10.0f, 2.0f, "blue", "blue", 1.0f);
       }
       for (const std::size_t i : clusterIds[2]) {
           kmean_test_svg.add_circle(points[i] * 10.0f, 2.0f, "green", "green", 1.0f);
       }
       kmean_test_svg.to_file("kmean_test.svg");
   }

   // dbscan + triangulation
   {
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
       const auto aabb = AxisLignedBoundingBox::point_cloud_aabb(points.begin(), points.end());
       std::vector<vec2> delaunay{ Algorithms2D::triangulate_points_delaunay(points.begin(), points.end(), aabb) };

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
   }

   // mean-shift based clustering
   {
       svg<vec2> mean_shift_test_svg(250, 160);
       std::vector<vec2> points;
       float sign{ 0.5f };

       // cluster #1
       vec2 center(20.0f, 12.0f);
       float radius{ 3.0f };
       for (std::size_t i{}; i < 80; ++i) {
           float fi{ static_cast<float>(i) };
           points.emplace_back(vec2(center.x + radius * std::cos(fi) + sign * static_cast<float>(rand()) / RAND_MAX,
                                    center.y + radius * std::sin(fi) + sign * static_cast<float>(rand()) / RAND_MAX));
           sign *= -1.0f;
       }

       // cluster #2
       center = vec2(12.0f, 7.0f);
       radius = 3.0f;
       for (std::size_t i{}; i < 80; ++i) {
           float fi{ static_cast<float>(i) };
           points.emplace_back(vec2(center.x + radius * std::cos(fi) + sign * static_cast<float>(rand()) / RAND_MAX,
                                    center.y + radius * std::sin(fi) + sign * static_cast<float>(rand()) / RAND_MAX));
           sign *= -1.0f;
       }

       // cluster #3
       center = vec2(4.0f, 2.0f);
       radius = 2.0f;
       for (std::size_t i{}; i < 80; ++i) {
           float fi{ static_cast<float>(i) };
           points.emplace_back(vec2(center.x + radius * std::cos(fi) + sign * static_cast<float>(rand()) / RAND_MAX,
                                    center.y + radius * std::sin(fi) + sign * static_cast<float>(rand()) / RAND_MAX));
           sign *= -1.0f;
       }

       // cluster
       const auto clusterIds = Clustering::get_mean_shift_based_clusters(points.cbegin(), points.cend(), 2.0f);
       assert(clusterIds.centers.size() == 3);
       assert(clusterIds.clusters[0].size() == 80);
       assert(clusterIds.clusters[1].size() == 80);
       assert(clusterIds.clusters[2].size() == 80);

       for (const std::size_t i : clusterIds.clusters[0]) {
           mean_shift_test_svg.add_circle(points[i] * 10.0f, 2.0f, "none", "red", 1.0f);
       }
       mean_shift_test_svg.add_circle(clusterIds.centers[0] * 10.0f, 2.0f, "red", "red", 1.0f);

       for (const std::size_t i : clusterIds.clusters[1]) {
           mean_shift_test_svg.add_circle(points[i] * 10.0f, 2.0f, "none", "blue", 1.0f);
       }
       mean_shift_test_svg.add_circle(clusterIds.centers[1] * 10.0f, 2.0f, "blue", "blue", 1.0f);

       for (const std::size_t i : clusterIds.clusters[2]) {
           mean_shift_test_svg.add_circle(points[i] * 10.0f, 2.0f, "none", "green", 1.0f);
       }
       mean_shift_test_svg.add_circle(clusterIds.centers[2] * 10.0f, 2.0f, "green", "green", 1.0f);

       mean_shift_test_svg.to_file("mean_shift_test.svg");
   }
}

void test_samples() {
   {
       // how many points to sample
       constexpr std::size_t count{ 1000 };
       std::vector<vec2> points;
       points.reserve(5 * count);

       // create a circle and uniformly sample 1000 points within it
       const vec2 center(110.0f, 110.0f);
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
       const vec2 ellipse_center(160.0f, 210.0f);
       const float xAxis{ 70.0f };
       const float yAxis{ 30.0f };
       std::vector<vec2> sampled_ellipse_points;
       sampled_ellipse_points.reserve(count);
       for (std::size_t i{}; i < count; ++i) {
           points.emplace_back(Sample::sample_ellipse(ellipse_center, xAxis, yAxis));
       }
       
       // lets reduce amount of points by merging close vertices
       std::vector<vec2> points_reduced{ NumericalAlgorithms::merge_close_objects(points.begin(), points.end(),
                                         [](const vec2& a, const vec2& b)->float { return GLSL::distance(a, b); }, 10.0f) };

       // partition space using bin-lattice grid
       SpacePartitioning::Grid<vec2> grid;
       grid.construct(points_reduced.begin(), points_reduced.end());

       // use density estimator (DBSCAN) to segment/cluster the point cloud and identify "noise" 
       const float density_radius{ 0.3f * radius };
       const std::size_t density_points{ 4 };
       const auto segments = Clustering::get_density_based_clusters(points_reduced.begin(), points_reduced.end(), grid, density_radius, density_points);
       grid.clear();

       // prepare drawing canvas
       std::array<std::string, 4> colors{ {"red", "green", "blue", "orange"} };
       svg<vec2> sample_test(800, 800);

       // calculate clusters characteristics (concave hull, principle axis)
       const std::size_t cluster_count{ segments.clusters.size() };
       for (std::size_t i{}; i < cluster_count; ++i) {
           if (segments.clusters[i].size() < 4) {
               continue;
           }

           // get cluster points
           std::vector<vec2> cluster_points;
           cluster_points.reserve(segments.clusters[i].size());
           for (const std::size_t j : segments.clusters[i]) {
               cluster_points.emplace_back(points_reduced[j]);
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
   }
}

void test_for_show() {

   // show 1
   {
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

       // export calculations to SVG file 
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
   }

   // show 2
   {
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

       // export calculations to SVG file 
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

       // check if cluster is circle or line using RANSAC.
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
   }

   // show 3
   {
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

       // get polygon medial axis
       const auto aabb_polygon = AxisLignedBoundingBox::point_cloud_aabb(polygon.begin(), polygon.end());
       const float step{ GLSL::distance(aabb_polygon.min, aabb_polygon.max) / 1000.0f };
       auto medial_axis = Algorithms2D::get_approximated_medial_axis(polygon.begin(), polygon.end(), step);

       // sample polygon (1500 points)
       constexpr std::size_t sample_size{ 1500 };
       const float area{ Algorithms2D::Internals::get_area(polygon.begin(), polygon.end()) };
       const auto delaunay = Algorithms2D::triangulate_polygon_delaunay(polygon.begin(), polygon.end(), aabb_polygon);
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

       // calculate clusters concave hulls
       std::vector<std::vector<vec2>> hulls;
       hulls.reserve(medial_axis.size());
       for (std::size_t j{}; j < medial_axis.size(); ++j) {
           hulls.emplace_back(Algorithms2D::get_convex_hull(clusters[j].begin(), clusters[j].end()));
       }

       // draw clustered samples
       svg<vec2> clustered_samples_svg(400, 450);
       std::vector<std::string> colors{ {"red", "green", "blue", "orange", "darkmagenta", "deeppink", "tan", "darkred",
                                         "darkolivegreen", "fuchsia", "plum", "tomato", "yellowgreen", "silver"} };
       for (std::size_t j{}; j < clusters.size(); ++j) {
           clustered_samples_svg.add_point_cloud(clusters[j].begin(), clusters[j].end(), 1.0f, colors[j % colors.size()], colors[j % colors.size()], 1.0f);
       }

       // draw clusters concave hulls
       for (std::size_t j{}; j < clusters.size(); ++j) {
           hulls[j].emplace_back(hulls[j].front());
           clustered_samples_svg.add_polyline(hulls[j].begin(), hulls[j].end(), "none", colors[j % colors.size()], 2.0f);
       }

       clustered_samples_svg.to_file("clustered_samples.svg");
   }

   // show 4
   {
       // define a signal with high amount of noise
       const float step{ 0.01f };
       const float max{ 12.0f * std::numbers::pi_v<float> };
       const std::size_t len{ static_cast<std::size_t>(std::ceil(max / step)) };
       std::vector<vec2> observation;
       std::vector<float> x, y, ys;
       x.reserve(len);
       y.reserve(len);
       ys.reserve(len);
       observation.reserve(len);
       for (std::size_t i{}; i < len; ++i) {
           const float _x{ static_cast<float>(i) * step };
           const float _y{ 1.0f + std::sin(_x) + 2.0f * std::cos(_x / 2.0f) };
           x.emplace_back(_x);
           ys.emplace_back(_y);
           y.emplace_back(_y + 3.0f * Hash::normal_distribution());
           observation.emplace_back(vec2(x[i], y[i]));
       }

       //
       // reduce noise and sample size using first order difference
       // 

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
           p(i, i) = 2.0f * beta;
           p(i, i + 1) = -beta;
           p(i + 1, i) = -beta;
       }
       p(0, 0) = beta;
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

       //
       // filter data using "time domain Hannin filter"
       //

       // Hanning window size
       constexpr std::size_t W{ 40 };

       // create "hannin" weights
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

       //
       // filter data using a two stage (forward & backward) Kalman smoother
       // 

       // Kalman parameters
       constexpr float R{ 6.0f };      // measurement noise (sigma squared)
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

       //
       // impulse smoother
       // 

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

       // signal and observation
       for (std::size_t i{}; i < len; ++i) {
           const vec2 curr_noise(x[i] * 10.0f, bias + y[i] * scale);
           data_svg.add_circle(curr_noise, 1.0f, "none", "gray", 1.0f);
       }
       for (std::size_t i{ 1 }; i < len; ++i) {
           const vec2 prev(x[i - 1] * 10.0f, bias + ys[i - 1] * scale);
           const vec2 curr(x[i] * 10.0f, bias + ys[i] * scale);
           data_svg.add_line(prev, curr, "black", "black", 1.0f);
       }

       // Kalman filtered observation
       for (std::size_t i{ 1 }; i < len; ++i) {
           const vec2 prev(x[i - 1] * 10.0f, bias + y_kalman[i - 1] * scale);
           const vec2 curr(x[i] * 10.0f, bias + y_kalman[i] * scale);
           data_svg.add_line(prev, curr, "green", "green", 1.0f);
       }

       // Hannin filtered observation
       for (std::size_t i{ 1 }; i < len; ++i) {
           const vec2 prev(x[i - 1] * 10.0f, bias + smooth[i - 1] * scale);
           const vec2 curr(x[i] * 10.0f, bias + smooth[i] * scale);
           data_svg.add_line(prev, curr, "red", "red", 1.0f);
       }

       // scatter reduced observation
       for (std::size_t i{ 1 }; i < N; ++i) {
           const vec2 prev(u[i - 1] * 10.0f, bias + z[i - 1] * scale);
           const vec2 curr(u[i] * 10.0f, bias + z[i] * scale);
           data_svg.add_circle(curr, 2.0f, "blue", "blue", 0.0f);
           data_svg.add_line(prev, curr, "blue", "blue", 1.0f);
       }

       // impulse smoother observation
       for (std::size_t i{ 1 }; i < ismooth.size(); ++i) {
           const vec2 prev(xsmooth[i - 1] * 10.0f, bias + ismooth[i - 1] * scale);
           const vec2 curr(xsmooth[i] * 10.0f, bias + ismooth[i] * scale);
           data_svg.add_circle(curr, 2.0f, "orange", "orange", 0.0f);
           data_svg.add_line(prev, curr, "orange", "orange", 1.0f);
       }
       data_svg.to_file("data.svg");
   }
}

int main() {
    test_diamond_angle();
    test_hash();
    test_variadic();
    test_numerics();
    test_glsl_basics();
    test_glsl_transformation();
    test_glsl_solvers();
    test_glsl_triangle();
    test_glsl_axis_aligned_bounding_box();
    test_glsl_point_distance();
    test_glsl_ray_intersection();
    test_GLSL_algorithms_2D();
    test_glsl_space_partitioning();
    test_GLSL_clustering();
    test_glsl_extra();
    test_numerical_algorithms();
    test_for_show();
    test_samples();
	return 1;
}
