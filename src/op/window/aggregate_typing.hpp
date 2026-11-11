/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "expression/aggregate_id.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <vector>

namespace sirius::op {

/// 10^@p digits as a signed 128 bit integer.
constexpr __int128_t int128_pow10(int digits)
{
  __int128_t value = 1;
  for (int i = 0; i < digits; ++i) {
    value *= 10;
  }
  return value;
}

/**
 * @brief One aggregate of a window: the function and the column it reads.
 */
struct window_aggregate {
  /// One of `sum` (`sum_no_overflow` is treated as `sum`), `avg`, `count`, `count_star`, `min` and
  /// `max`. Any other id is rejected.
  sirius::aggregate_id function;
  /// Index of the argument column in the input table. `count_star` counts rows including NULLs, so
  /// any column serves as its argument.
  cudf::size_type argument_column;
};

/**
 * @brief The ORDER BY of a window, in cuDF terms.
 *
 * `directions` and `nulls` have one entry for each of `columns`.
 */
struct window_order {
  /// Indices of the order key columns in the input table.
  std::vector<cudf::size_type> columns;
  /// Direction of each order key.
  std::vector<cudf::order> directions;
  /// Where NULLs sort for each order key, in the cuDF sense: applied in the ascending frame and
  /// reversed for a descending direction. `to_cudf_null_order` converts SQL `NULLS FIRST` and
  /// `NULLS LAST`.
  std::vector<cudf::null_order> nulls;
};

/**
 * @brief The type a window `sum` over @p input accumulates in and returns.
 *
 * Signed integers give DECIMAL128 with scale 0 (the `HUGEINT` carrier), decimals give DECIMAL128 at
 * the input scale and floating point gives FLOAT64.
 *
 * @throws std::invalid_argument for any other type.
 */
cudf::data_type window_sum_type(cudf::data_type input);

/**
 * @brief The result type of the window aggregate @p function over an argument of type @p input.
 *
 * `sum` is window_sum_type(), `avg` is FLOAT64, `count` and `count_star` are INT64, and `min` and
 * `max` keep the argument type.
 *
 * @throws std::invalid_argument for an unsupported function, or a `sum` over an unsupported type.
 */
cudf::data_type window_result_type(sirius::aggregate_id function, cudf::data_type input);

/**
 * @brief @p function as the window evaluators compute it: `sum_no_overflow` becomes `sum`.
 *
 * @throws std::invalid_argument for any id other than `sum`, `sum_no_overflow`, `avg`, `count`,
 * `count_star`, `min` and `max`.
 */
sirius::aggregate_id normalize_window_aggregate(sirius::aggregate_id function);

/**
 * @brief Check that every valid value of the DECIMAL128 column @p sums is inside DECIMAL(38).
 *
 * Synchronizes @p stream.
 *
 * @throws std::overflow_error if a value is at least 10^38 in magnitude.
 */
void check_decimal38(cudf::column_view sums,
                     ::cuda::stream_ref stream,
                     rmm::device_async_resource_ref mr);

}  // namespace sirius::op
