//-------------------------------------------------------------------------------
//
// Copyright (c) 2025, Dan Israel Malta <malta.dan@gmail.com>
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
#include <iterator>
#include <array>
#include <vector>
#include <thread>
#include <algorithm>

//
// collection of pattern identification algorithms
//
namespace PatternDetection {

	/**
	* \brief a 'type' which holds a value, a minimal limit of the value and a maximal value of the limit
	**/
	template<typename T>
		requires(std::is_arithmetic_v<T>)
	struct clamped_value {
		T value{}; // "current" value
		T min{};   // minimal possible value
		T max{};   // maximal possible value
	};
	template<typename>   struct is_clamped_value                   : public std::false_type {};
	template<typename T> struct is_clamped_value<clamped_value<T>> : public std::true_type {};
	template<typename T> constexpr bool is_clamped_value_v = is_clamped_value<T>::value;
	template<typename T> concept IClampledValue = is_clamped_value_v<T>;

	/**
	* \brief requirements of data used in/by RANSAC model
	**/
	template<class RANSAC>
	concept IRansacData = std::is_integral_v<decltype(RANSAC::MODEL_SIZE())>  && // amount of model parameters
		                  std::is_integral_v<decltype(RANSAC::POINT_COUNT())> && // amount of points needed for model calculation
		                  std::is_integral_v<decltype(RANSAC::EVAL_COUNT())>  && // model outputs used for score evaluation
					      GLSL::IFixedVector<typename RANSAC::vector_type>    && // vector type
                          std::is_arithmetic_v<typename RANSAC::value_type>   && // value type
		                  IClampledValue<typename RANSAC::model_type>         && // type of model parameter
		                  // holding model parameters
		                  std::is_same_v<typename RANSAC::model_t, std::array<typename RANSAC::model_type, RANSAC::MODEL_SIZE()>> &&
		                  // points used as input to calculate parameters used for score evaluation
					      std::is_same_v<typename RANSAC::points_t, std::array<typename RANSAC::vector_type, RANSAC::POINT_COUNT()>> &&
		                  // evaluation arguments
		                  std::is_same_v<typename RANSAC::eval_t, std::array<typename RANSAC::value_type, RANSAC::EVAL_COUNT()>> &&
		                  requires(RANSAC ransac, RANSAC::model_t model) {
		/**
		* \brief get model parameters
		* @param {model_t, out} model parameters
		**/
		{ ransac.get_model() } -> std::same_as<typename RANSAC::model_t>;

		/**
		* \brief set model parameters
		* @param {model_t, in} model parameters
		**/
		{ ransac.set_model(model) } -> std::same_as<void>;

		/**
		* \brief is model valid?
		* @param {bool, put} true if model is valid, false otherwise
		**/
		{ ransac.is_valid() } -> std::same_as<bool>;
	};

	/**
	* \brief concept of RANSAC model
	**/
	template<class RANSAC>
	concept IRansac = IRansacData<RANSAC> &&
		              requires(RANSAC ransac, RANSAC::points_t points, RANSAC::vector_type vec) {
		/**
		* \brief apply model, given by by its params, on a collection of points and calculate model evaluations
		* @param {points_t, in} points for model to operate upon
		**/
		{ ransac.evaluate_score_parameters(points) } -> std::same_as<void>;

		/**
		* \brief given a point, use 'evaluations' to calculate this point RANSAC score,
		*        i.e. - how much point is "close"/"far away" from model
		* @param {vector_type, in}  points for model to operate upon
		* @param {value_type,  out} point score
		**/
		{ ransac.calculate_distance(vec) } -> std::same_as<typename RANSAC::value_type>;
	};

	//
	// collection of useful RANSAC models
	//
	namespace RansacModels {

		/**
		* \brief basic data structure holding RASAC model data
		* @param {size_t} amount of parameters in model
		* @param {size_t} amount of points required to evaluate model parameter
		* @param {size_t} amount of parameters created during model evaluations, used to calculate model "score" on points
		* @param {IFixedVector} type of points on which model operates
		**/
		template<std::size_t N0, std::size_t N1, std::size_t N2, class VEC>
			requires(GLSL::IFixedVector<VEC> && N0 > 0)
		struct RansacDataModel {
			// compile time parameters
			static constexpr std::integral_constant<std::size_t, N0> MODEL_SIZE  = {}; // amount of model parameters
			static constexpr std::integral_constant<std::size_t, N1> POINT_COUNT = {}; // amount of points needed for model calculation
			static constexpr std::integral_constant<std::size_t, N2> EVAL_COUNT  = {}; // model outputs used for score evaluation

			// types
			using vector_type = VEC;
			using value_type  = typename vector_type::value_type;
			using model_type  = clamped_value<value_type>;
			using model_t     = std::array<model_type,  MODEL_SIZE>;
			using points_t    = std::array<vector_type, POINT_COUNT>;
			using eval_t      = std::array<value_type,  EVAL_COUNT>;

			// properties
			model_t parameters{ {model_type{}} }; // RANSAC model parameters
			eval_t evaluations{ {value_type{}} }; // RANSAC model output arguments used for evaluation
			bool valid{ true };					  // RANSAC model validty

			void set_model([[maybe_unused]] model_t model) noexcept {
				this->parameters = model;
			}

			model_t get_model() const {
				return this->parameters;
			}

			bool is_valid() const {
				return this->valid;
			}
		};
		static_assert(IRansacData<RansacDataModel<3, 0, 0, vec2>>);
		static_assert(IRansacData<RansacDataModel<4, 2, 5, vec2>>);

		/**
		* \brief RANSAC model to detect a line segment
		*        RANSAC model holds line segment points in the following manner: {p0.x, p0.y, p1.x, p1.y}
		**/
		template<GLSL::IFixedVector VEC>
		struct Line final : RansacDataModel<4, 2, 5, VEC> {
			using base = RansacDataModel<4, 2, 5, VEC>;
			using vec_t = typename base::vector_type;
			using T = typename base::value_type;

			/**
			* \brief override 'set_model'. not needed here.
			**/
			[[maybe_unused]] void set_model([[maybe_unused]] base::model_t model) noexcept {
				Utilities::unreachable();
			}

			/**
			* \brief implement 'evaluate_score_parameters'. also sets model parameters.
			**/
			void evaluate_score_parameters(const base::points_t& points) noexcept {
				// extract points and set model
				const vec_t p0(points[0]);
				const vec_t p1(points[1]);
				this->parameters[0].value = p0.x;
				this->parameters[1].value = p0.y;
				this->parameters[2].value = p1.x;
				this->parameters[3].value = p1.y;

				// calculate evaluations
				const vec_t dp{ p1 - p0 };
				const T det{ GLSL::length(dp) };
				if (Numerics::areEquals(det, T{})) {
					this->valid = false;
				}

				// calculate 'evaluations'
				this->evaluations = { {p0.x, p0.y, dp.x, dp.y, det} };
			}

			/**
			* \brief implement 'calculate_distance'. calculate unsigned distance from line.
			**/
			base::value_type calculate_distance(const base::vector_type& point) const noexcept {
				const vec_t dp(this->evaluations[0], this->evaluations[1]);
				const vec_t p0(this->evaluations[2], this->evaluations[3]);
				return std::abs(GLSL::cross(dp, point - p0) / this->evaluations[4]);
			}
		};
		static_assert(IRansac<Line<vec2>>);

		/**
		* \brief RANSAC model to detect a circle
		*        RANSAC model is: {center.x, center.y, radius}
		**/
		template<GLSL::IFixedVector VEC>
		struct Circle final : RansacDataModel<3, 0, 0, VEC> {
			using base = RansacDataModel<3, 0, 0, VEC>;
			using value_type  = typename base::value_type;
			using vector_type = typename base::vector_type;
			using points_t    = typename base::points_t;
			using model_t     = typename base::model_t;

			/**
			* \brief override 'set_model'.
			**/
			void set_model(model_t model) noexcept {
				Utilities::static_for<0, 1, base::MODEL_SIZE()>([this, &model](std::size_t i) {
					this->parameters[i].min = model[i].min;
					this->parameters[i].max = model[i].max;
					this->parameters[i].value = model[i].min + (model[i].max - model[i].min) * static_cast<value_type>(rand()) / RAND_MAX;
				});
			}

			/**
			* \brief implement 'evaluate_score_parameters'. not required here.
			**/
			[[maybe_unused]] void evaluate_score_parameters([[maybe_unused]] const points_t& points) const noexcept {
				Utilities::unreachable();
			}

			/**
			* \brief implement 'calculate_distance'. calculate unsigned distance from circle.
			**/
			value_type calculate_distance(const vector_type& point) const noexcept {
				const vector_type center(this->parameters[0].value, this->parameters[1].value);
				const value_type radius{ this->parameters[2].value };
				return std::abs(GLSL::distance(point, center) - radius);
			}
		};
		static_assert(IRansac<Circle<vec2>>);
	};

	/**
	* \brief given a collection of points, identify a given pattern using RANSAC method.
	* @param {forward_iterator,            in}  iterator to point cloud first point
	* @param {forward_iterator,            in}  iterator to point cloud last point
	* @param {size_t,                      in}  amount of iterations for RANSAC procedure to run
	* @param {IRansac,                     in}  initial RANSAC model
	* @param {value_type,                  in}  threshold used for a point to be considered part of a model
	* @param {{array<value_type>, size_t}, out} {parameters of RANSAC model with highest score, model score}
	**/
	template<std::forward_iterator InputIt, IRansac MODEL,
		     class VEC = typename std::decay_t<decltype(*std::declval<InputIt>())>,
		     class T = typename VEC::value_type>
		requires(GLSL::is_fixed_vector_v<VEC> && std::same_as<VEC, typename MODEL::vector_type>)
	constexpr auto ransac_pattern_detection(const InputIt first, const InputIt last,
		                                    const std::size_t count, const MODEL& model, const T threshold) {
		using points_t    = typename MODEL::points_t;
		using model_out_t = std::array<T, MODEL::MODEL_SIZE()>;
		using kernel_t    = struct { typename MODEL::model_t parameters; std::size_t score; };
		using out_t       = struct { model_out_t model; std::size_t score; };

		// housekeeping
		const T len{ static_cast<T>(std::distance(first, last) - 1) };

		// lambda which creates RANSAC model
		const auto create_model = [first, len, model]() -> MODEL {
			MODEL rand_model{ model };

			// fill model with random values
			rand_model.set_model(model.parameters);

			// if required by model - calculate model score evaluation parameters
			if constexpr (MODEL::POINT_COUNT() > 0) {
				// get random points to calculate model score evaluation parameters
				points_t points;
				std::array<std::size_t, MODEL::POINT_COUNT()> indices{ { 0 }};
				Utilities::static_for<0, 1, MODEL::POINT_COUNT()>([first, len, &points, &indices](std::size_t i) {
					std::size_t index{};
					do {
						index = static_cast<std::size_t>(std::floor(static_cast<T>(rand()) / RAND_MAX * len));
					} while (std::find(indices.begin(), indices.end(), index) != indices.end());
					indices[i] = index;
					points[i] = *(first + index);
				});

				// calculate model score evaluation parameters
				rand_model.evaluate_score_parameters(points);
			}

			// return model
			return rand_model;
		};

		// lambda which calculates RANSAC model score
		const auto calculate_score = [first, last, threshold](const MODEL& model) -> std::size_t {
			std::size_t score{};
			if (!model.is_valid()) {
				return score;
			}

			for (InputIt it{ first }; it != last; ++it) {
				if (model.calculate_distance(*it) < threshold) {
					++score;
				}
			}

			return score;
		};

		// lambda to perform RANSAC 'count' times and return best model
		const auto ransac_kernel = [&create_model, &calculate_score, count]() -> kernel_t {
			kernel_t best_outcome;
			best_outcome.score = 0;

			for (std::size_t i{}; i < count; ++i) {
				const MODEL model{ create_model() };
				if (const std::size_t score{ calculate_score(model) };
					score > best_outcome.score) {
					best_outcome.score = score;
					best_outcome.parameters = model.parameters;
				}
			}
			return best_outcome;
		};

		// RANSAC
		const kernel_t best_model{ ransac_kernel() };

		// output
		model_out_t model_out{ {T{}} };
		Utilities::static_for<0, 1, MODEL::MODEL_SIZE()>([&model_out, &best_model](std::size_t i) {
			model_out[i] = best_model.parameters[i].value;
		});
		return out_t{ model_out, best_model.score };
	}

}
