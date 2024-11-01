#include <assert.h>
#include <numbers>
#include <array>
#include <iostream>
#include <list>
#include <chrono>
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
#include "Glsl_svg.h"

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
    static_assert(Numerics::min(6, 4, 5, 8) == 4);
    static_assert(Numerics::min(6, 4, 5, 8, 0, 6) == 0);
    static_assert(Numerics::min(6, 4, 5, 8, 0, -3, 6) == -3);

    // test max
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
        const std::array<double, 6> xx{ {2.0, 1.0, 6.0, 2.0, 4.0, 3.0} };
        std::array<double, 6> y;
        NumericalAlgorithms::filter<3, 1>(xx.begin(), xx.end(), y.begin(), b, a);
        assert(static_cast<std::int32_t>(y[0] * 3.0) == 2);
        assert(static_cast<std::int32_t>(y[1] * 3.0) == 3);
        assert(static_cast<std::int32_t>(y[2] * 3.0) == 9);
        assert(static_cast<std::int32_t>(y[3] * 3.0) == 8); // 2.99999 ....
        assert(static_cast<std::int32_t>(y[4] * 3.0) == 12);
        assert(static_cast<std::int32_t>(y[5] * 3.0) == 9);;
    }

    // test partition
    std::list<int> v = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::list<int> vExpected = { 0, 8, 2, 6, 4, 5, 3, 7, 1, 9 };
    auto it = Algoithms::partition(v.begin(), v.end(), [](int i) {return i % 2 == 0; });
    assert(v == vExpected);
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

        Utilities::static_for<0, 1, 3>([&transformation_using_world_up, &dcm_using_world_up_axis_angle](std::size_t i) {
            Utilities::static_for<0, 1, 3>([&transformation_using_world_up, &dcm_using_world_up_axis_angle, i](std::size_t j) {
                assert(std::abs(std::abs(transformation_using_world_up(i, j)) - std::abs(dcm_using_world_up_axis_angle(j, i))) < 1e-6);
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
        auto eigen = Decomposition::EigenSymmetric3x3(aa);
        Utilities::static_for<0, 1, 3>([&aa, &eigen](std::size_t i) {
            const vec3 lhs{ aa * eigen.eigenvectors[i] };
            const vec3 rhs{ eigen.eigenvalues[i] * eigen.eigenvectors[i] };
            assert(std::abs(GLSL::length(lhs) - GLSL::length(rhs)) < 1e-2);
        });
    }

    {
        // chack that balancing a matrix allows faster eigenvalue decomposition
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

        float distance = PointDistance::sdf_to_polygon<5>(polygon, vec2(2.0f, 0.0f));
        assert(std::abs(distance - 1.0f) < 1e-6);

        distance = PointDistance::sdf_to_polygon<5>(polygon, vec2(3.0f, 1.5f));
        assert(std::abs(distance - -0.5f) < 1e-6);

        distance = PointDistance::sdf_to_polygon(std::vector<vec2>(polygon.begin(), polygon.end()), vec2(3.0f, 1.5f));
        assert(std::abs(distance - -0.5f) < 1e-6);
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
        auto plane = Extra::create_plane(vec3(0.0f, 0.0f, 2.0f), vec3(1.0f, 0.0f, 2.0f), vec3(0.0f, 1.0f, 2.0f));

        float distance = PointDistance::sdf_to_plane(vec3(0.0f), plane);
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
}

void test_GLSL_algorithms_2D() {
    {
        auto ccw = Algorithms2D::Internals::are_points_ordered_counter_clock_wise(vec2(0.0f), vec2(0.0f), vec2(0.0f));
        assert(static_cast<int>(ccw) == 0);

        ccw = Algorithms2D::Internals::are_points_ordered_counter_clock_wise(vec2(0.0f), vec2(0.0f, 1.0f), vec2(0.5f));
        assert(ccw > 0);

        ccw = Algorithms2D::Internals::are_points_ordered_counter_clock_wise(vec2(0.0f), vec2(0.0f, 1.0f), vec2(-0.5f));
        assert(ccw < 0);
    }

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

        assert(Algorithms2D::is_line_connecting_polygon_vertices_inside_polygon(polygon.begin(), polygon.end(), area, polygon.begin(),     polygon.begin() + 2));
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
        const auto reflex_vertices = Algorithms2D::get_reflex_vertices(polygon.begin(), polygon.end());
        const std::vector<vec2> reflexis{ { vec2(4.0f, 6.0f), vec2(7.0f, 7.0f), vec2(8.0f, 9.0f),
                                            vec2(2.0f, 8.0f), vec2(2.0f, 6.0f) }};
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
        std::vector<vec2> polygon{ {vec2(3.0f, 1.0f), vec2(5.0f, 1.0f), vec2(5.0f, 4.0f), vec2(4.0f, 6.0f), vec2(7.0f, 7.0f ), vec2(10.0f, 7.0f), vec2(10.0f, 9.0f),
                                    vec2(8.0f, 9.0f), vec2(6.0f, 10.0f), vec2(1.0f, 10.0f), vec2(1.0f, 8.0f), vec2(2.0f, 8.0f), vec2(2.0f, 6.0f), vec2(1.0f, 6.0f),
                                    vec2(1.0f, 2.0f)} };
        assert(!Algorithms2D::is_polygon_convex(polygon.begin(), polygon.end()));

        // convex hull test
        const auto convex = Algorithms2D::get_convex_hull(polygon.begin(), polygon.end());
        assert(Algorithms2D::is_polygon_convex(convex.begin(), convex.end()));
        const std::vector<vec2> expected_convex{ {vec2(1.0f, 2.0f), vec2(3.0f, 1.0f), vec2(5.0f, 1.0f), vec2(10.0f, 7.0f),
                                                  vec2(10.0f, 9.0f), vec2(6.0f, 10.0f), vec2(1.0f, 10.0f)} };
        assert(expected_convex.size() == convex.size());
        for (std::size_t i{}; i < expected_convex.size(); ++i) {
            assert(GLSL::max(GLSL::abs(expected_convex[i] - convex[i])) < 1e-6);
        }

        // obb test
        const auto obb = Algorithms2D::get_convex_hull_minimum_area_bounding_rectangle(convex);
        assert(GLSL::max(GLSL::abs(vec2(10.4705877f, 8.88235283f) - obb.p0)) < 1e-6);
        assert(GLSL::max(GLSL::abs(vec2(1.29411793f, 11.1764698f) - obb.p1)) < 1e-6);
        assert(GLSL::max(GLSL::abs(vec2(-0.999999285f, 1.99999905f) - obb.p2)) < 1e-6);
        assert(GLSL::max(GLSL::abs(vec2(8.1764698f, -0.294118881f) - obb.p3)) < 1e-6);

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
        assert(GLSL::max(GLSL::abs(vec2(4.73529434f, 5.44117594f) - centroid)) < 1e-6);

        // convex hull diamater
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

        // concave hull
        const auto concave0 = Algorithms2D::get_concave_hull(polygon.begin(), polygon.end());
        for (std::size_t i{}; i < expected_convex.size(); ++i) {
            assert(GLSL::max(GLSL::abs(expected_convex[i] - concave0[i])) < 1e-6);
        }

        const auto concave2 = Algorithms2D::get_concave_hull(polygon.begin(), polygon.end(), 0.5f);
        assert(!Algorithms2D::is_polygon_convex(concave2.begin(), concave2.end()));
        const std::vector<vec2> expected_concave2{ {vec2(1.0f, 2.0f), vec2(3.0f, 1.0f), vec2(5.0f, 1.0f), vec2(10.0f, 7.0f),
                                                   vec2(10.0f, 9.0f), vec2(6.0f, 10.0f), vec2(1.0f, 10.0f), vec2(1.0f, 6.0f) } };
        for (std::size_t i{}; i < expected_concave2.size(); ++i) {
            assert(GLSL::max(GLSL::abs(expected_concave2[i] - concave2[i])) < 1e-6);
        }

        const auto concave3 = Algorithms2D::get_concave_hull(polygon.begin(), polygon.end(), 1.2f);
        assert(!Algorithms2D::is_polygon_convex(concave2.begin(), concave2.end()));
        const std::vector<vec2> expected_concave3{ {vec2(1.0f, 2.0f), vec2(3.0f, 1.0f), vec2(5.0f, 1.0f), vec2(5.0f, 4.0f),
                                                   vec2(7.0f, 7.0f), vec2(10.0f, 7.0f), vec2(10.0f, 9.0f), vec2(8.0f, 9.0f),
                                                   vec2(6.0f, 10.0f), vec2(1.0f, 10.0f), vec2(1.0f, 6.0f)} };
        for (std::size_t i{}; i < expected_concave3.size(); ++i) {
            assert(GLSL::max(GLSL::abs(expected_concave3[i] - concave3[i])) < 1e-6);
        }

        // chcek orthogonality
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
                                vec2(1.0, 3.0)}};

        // should not me monotone with regard to X axis
        is_monotone = Algorithms2D::is_polygon_monotone_relative_to_line(mon.begin(), mon.end(), vec2(0.0f), vec2(1.0f, 0.0f));
        assert(!is_monotone);

        // should be monotone with regard to Y axis
        is_monotone = Algorithms2D::is_polygon_monotone_relative_to_line(mon.begin(), mon.end(), vec2(0.0f), vec2(0.0f, 1.0f));
        assert(is_monotone);
    }

   {
       std::vector<vec2> points;
       for (std::size_t i{}; i < 50; ++i) {
           points.emplace_back(vec2(static_cast<float>(rand() % 100 - 50),
                                    static_cast<float>(rand() % 100 - 50)));
       }

       const vec2 centroid = Algorithms2D::Internals::get_centroid(points.cbegin(), points.cend());
       Algorithms2D::sort_points_clock_wise(points.begin(), points.end(), centroid);
       assert(Algorithms2D::are_points_ordererd_clock_wise(points.cbegin(), points.cend(), centroid));
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

           const vec2 direction{ Algorithms2D::get_principle_axis(points.cbegin(), points.cend()) };
           const float error{ std::abs(angle - std::atan2(direction.y, direction.x) * 180.0f / 3.141592653589f) };
           assert(error <= 0.2f || std::abs(180.0f-error) <= 0.2f);
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
       // chcek orthogonality
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
       for (const vec2 p : points) {
           envelope_test_svg.add_circle(p, 2.0f, "none", "black", 0.5f);
       }
       envelope_test_svg.add_polyline(envelope.top.begin(), envelope.top.end(), "none", "red", 1.0f);
       envelope_test_svg.add_polyline(envelope.bottom.begin(), envelope.bottom.end(), "none", "green", 1.0f);
       envelope_test_svg.to_file("envelope_test_svg.svg");
   }

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

        // search nearest neighbours in cube
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

        // search nearest neighbours in radius
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

        // search k nearest neighbours in cube
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

        // search nearest neighbours in cube
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

        // search nearest neighbours in radius
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

        // search k nearest neighbours in cube
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
        assert(sizeof(grid) == 3 * (sizeof(std::array<std::size_t, 2>) + sizeof(vec2)) + sizeof(vec2*) + sizeof(std::vector<vec2>));
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

       // clster #2
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
       assert(clusterIds1.clusters[0].size() > 58 && clusterIds1.clusters[0].size() < 62);
       assert(clusterIds1.clusters[1].size() > 38 && clusterIds1.clusters[1].size() < 42);
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
}


int main() {
    test_diamond_angle();
    test_hash();
    test_variadic();
    test_numerics();
    test_numerical_algorithms();
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
	return 1;
}
