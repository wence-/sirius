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

#include "op/window/whole_partition.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <stdexcept>

namespace sirius::op {
namespace {

cudf::data_type accumulator_type(cudf::data_type input)
{
  using cudf::type_id;
  switch (input.id()) {
    case type_id::INT8:
    case type_id::INT16:
    case type_id::INT32:
    case type_id::INT64: return cudf::data_type{type_id::DECIMAL128, 0};
    case type_id::DECIMAL32:
    case type_id::DECIMAL64:
    case type_id::DECIMAL128: return cudf::data_type{type_id::DECIMAL128, input.scale()};
    case type_id::FLOAT32:
    case type_id::FLOAT64: return cudf::data_type{type_id::FLOAT64};
    default: throw std::invalid_argument("Unsupported whole-partition window argument type");
  }
}

void validate_key(cudf::data_type type)
{
  using cudf::type_id;
  switch (type.id()) {
    case type_id::BOOL8:
    case type_id::INT8:
    case type_id::INT16:
    case type_id::INT32:
    case type_id::INT64:
    case type_id::TIMESTAMP_DAYS:
    case type_id::TIMESTAMP_SECONDS:
    case type_id::TIMESTAMP_MILLISECONDS:
    case type_id::TIMESTAMP_MICROSECONDS:
    case type_id::TIMESTAMP_NANOSECONDS:
    case type_id::STRING: return;
    default: throw std::invalid_argument("Unsupported whole-partition window key type");
  }
}

// SUM_OVERFLOW preserves the accumulator type (including decimal scale). Its validity
// belongs to the parent struct, not to the sum child: all-null groups must remain null.
std::unique_ptr<cudf::column> checked_sum(std::unique_ptr<cudf::column> result,
                                          ::cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  if (result->type().id() != cudf::type_id::STRUCT) { return result; }
  auto child = result->child(1).view();
  cudf::column_view flags(
    child.type(), child.size(), child.head(), result->view().null_mask(), result->null_count());
  auto overflow = cudf::reduce(flags,
                               *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
                               cudf::data_type{cudf::type_id::BOOL8},
                               stream,
                               mr);
  if (overflow->is_valid(stream) &&
      static_cast<cudf::numeric_scalar<bool> const&>(*overflow).value(stream)) {
    throw std::overflow_error("Whole-partition window sum exceeds its 128-bit accumulator");
  }
  auto const nulls = result->null_count();
  auto contents    = result->release();
  auto sum         = std::move(contents.children[0]);
  if (nulls) { sum->set_null_mask(std::move(*contents.null_mask), nulls); }
  return sum;
}

struct argument_state {
  cudf::size_type column;
  bool needs_avg;
  std::unique_ptr<cudf::column> widened;
  std::unique_ptr<cudf::column> sum;
  std::unique_ptr<cudf::column> avg;
};

}  // namespace

std::unique_ptr<cudf::table> evaluate_whole_partition_window(
  cudf::table_view input,
  std::vector<cudf::size_type> const& partition_columns,
  std::vector<whole_partition_aggregate> const& aggregates,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (aggregates.empty()) { throw std::invalid_argument("Window requires at least one aggregate"); }
  auto check_index = [&](cudf::size_type index) {
    if (index < 0 || index >= input.num_columns()) {
      throw std::invalid_argument("Whole-partition window column is outside the input schema");
    }
  };
  for (auto index : partition_columns) {
    check_index(index);
    validate_key(input.column(index).type());
  }
  std::vector<argument_state> arguments;
  std::vector<std::size_t> argument_indices;
  for (auto const& agg : aggregates) {
    check_index(agg.argument_column);
    (void)accumulator_type(input.column(agg.argument_column).type());
    if (agg.function != whole_partition_function::sum &&
        agg.function != whole_partition_function::avg) {
      throw std::invalid_argument("Unsupported whole-partition window function");
    }
    auto found = std::find_if(arguments.begin(), arguments.end(), [&](auto const& arg) {
      return arg.column == agg.argument_column;
    });
    if (found == arguments.end()) {
      argument_indices.push_back(arguments.size());
      arguments.push_back(
        {agg.argument_column, agg.function == whole_partition_function::avg, {}, {}, {}});
    } else {
      argument_indices.push_back(static_cast<std::size_t>(found - arguments.begin()));
      found->needs_avg |= agg.function == whole_partition_function::avg;
    }
  }
  if (input.num_rows() == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty;
    for (auto const& agg : aggregates) {
      auto type = agg.function == whole_partition_function::avg
                    ? cudf::data_type{cudf::type_id::FLOAT64}
                    : accumulator_type(input.column(agg.argument_column).type());
      empty.push_back(cudf::make_empty_column(type));
    }
    return std::make_unique<cudf::table>(std::move(empty));
  }

  std::vector<cudf::groupby::aggregation_request> requests;
  for (auto& arg : arguments) {
    auto values = input.column(arg.column);
    auto type   = accumulator_type(values.type());
    if (values.type() != type) {
      arg.widened = cudf::cast(values, type, stream, mr);
      values      = arg.widened->view();
    }
    cudf::groupby::aggregation_request request;
    request.values = values;
    if (type.id() == cudf::type_id::DECIMAL128) {
      request.aggregations.push_back(
        cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>());
    } else {
      request.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    }
    if (arg.needs_avg) {
      request.aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
    }
    requests.push_back(std::move(request));
  }

  std::unique_ptr<cudf::table> grouped_keys;
  std::vector<cudf::groupby::aggregation_result> results;
  if (!partition_columns.empty()) {
    cudf::groupby::groupby groupby(input.select(partition_columns), cudf::null_policy::INCLUDE);
    auto grouped = groupby.aggregate(requests, stream, mr);
    grouped_keys = std::move(grouped.first);
    results      = std::move(grouped.second);
  } else {
    // No key table, hash table, or row-sized constant column is needed for a global window.
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto const& request = requests[i];
      bool const exact    = request.values.type().id() == cudf::type_id::DECIMAL128;
      std::unique_ptr<cudf::reduce_aggregation> agg =
        exact ? cudf::make_sum_overflow_aggregation<cudf::reduce_aggregation>()
              : cudf::make_sum_aggregation<cudf::reduce_aggregation>();
      auto sum =
        cudf::reduce(request.values,
                     *agg,
                     exact ? cudf::data_type{cudf::type_id::STRUCT} : request.values.type(),
                     stream,
                     mr);
      cudf::groupby::aggregation_result result;
      result.results.push_back(cudf::make_column_from_scalar(*sum, 1, stream, mr));
      if (arguments[i].needs_avg) {
        auto count = cudf::reduce(request.values,
                                  *cudf::make_count_aggregation<cudf::reduce_aggregation>(),
                                  cudf::data_type{cudf::type_id::INT64},
                                  stream,
                                  mr);
        result.results.push_back(cudf::make_column_from_scalar(*count, 1, stream, mr));
      }
      results.push_back(std::move(result));
    }
  }

  auto const f64 = cudf::data_type{cudf::type_id::FLOAT64};
  for (std::size_t i = 0; i < arguments.size(); ++i) {
    auto& arg = arguments[i];
    arg.widened.reset();
    arg.sum = checked_sum(std::move(results[i].results[0]), stream, mr);
    // DuckDB DECIMAL(38, scale) has a stricter bound than the signed 128-bit carrier.
    // AVG's internal sum need not fit DECIMAL(38), but a requested decimal SUM must.
    auto const decimal_sum =
      cudf::is_fixed_point(input.column(arg.column).type()) &&
      std::any_of(aggregates.begin(), aggregates.end(), [&](auto const& aggregate) {
        return aggregate.argument_column == arg.column &&
               aggregate.function == whole_partition_function::sum;
      });
    if (decimal_sum) {
      __int128_t limit = 1;
      for (int digit = 0; digit < 38; ++digit) {
        limit *= 10;
      }
      auto bounds = cudf::minmax(arg.sum->view(), stream, mr);
      for (auto const* bound : {bounds.first.get(), bounds.second.get()}) {
        if (bound->is_valid(stream)) {
          auto const value =
            static_cast<cudf::fixed_point_scalar<numeric::decimal128> const&>(*bound).value(stream);
          if (value <= -limit || value >= limit) {
            throw std::overflow_error("Whole-partition window sum exceeds DECIMAL(38)");
          }
        }
      }
    }
    if (arg.needs_avg) {
      auto sum   = cudf::cast(arg.sum->view(), f64, stream, mr);
      auto count = cudf::cast(results[i].results[1]->view(), f64, stream, mr);
      arg.avg    = cudf::binary_operation(
        sum->view(), count->view(), cudf::binary_operator::DIV, f64, stream, mr);
    }
  }
  std::vector<cudf::column_view> output;
  for (std::size_t i = 0; i < aggregates.size(); ++i) {
    auto const& arg = arguments[argument_indices[i]];
    output.push_back(aggregates[i].function == whole_partition_function::sum ? arg.sum->view()
                                                                             : arg.avg->view());
  }
  if (partition_columns.empty()) {
    std::vector<std::unique_ptr<cudf::column>> expanded;
    for (auto col : output) {
      auto scalar = cudf::get_element(col, 0, stream, mr);
      expanded.push_back(cudf::make_column_from_scalar(*scalar, input.num_rows(), stream, mr));
    }
    return std::make_unique<cudf::table>(std::move(expanded));
  }

  // Groupby emits unique keys including null keys. Probe those exact keys with nulls equal.
  // distinct_hash_join guarantees one build index per input row, in input order. Thus no
  // join multiplicity, sorting, payload gather, or inverse permutation is required.
  cudf::distinct_hash_join lookup(grouped_keys->view(), cudf::null_equality::EQUAL, 0.5, stream);
  auto indices = lookup.left_join(input.select(partition_columns), stream, mr);
  cudf::column_view map(
    cudf::data_type{cudf::type_id::INT32}, input.num_rows(), indices->data(), nullptr, 0);
  return cudf::gather(
    cudf::table_view(output), map, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
}

}  // namespace sirius::op
