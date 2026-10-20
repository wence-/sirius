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
#include "operator/operator_test_utils.hpp"
#include "utils/parquet_fixture_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <duckdb.hpp>

#include <limits>

namespace {
using sirius::op::evaluate_whole_partition_window;
using sirius::op::whole_partition_aggregate;
using function = sirius::op::whole_partition_function;

void set_validity(cudf::column& column,
                  std::vector<bool> const& valid,
                  ::cuda::stream_ref stream,
                  rmm::device_async_resource_ref mr)
{
  if (valid.empty()) { return; }
  REQUIRE(valid.size() == static_cast<std::size_t>(column.size()));
  auto mask = cudf::create_null_mask(column.size(), cudf::mask_state::ALL_VALID, stream, mr);
  cudf::size_type nulls = 0;
  for (cudf::size_type i = 0; i < column.size(); ++i) {
    if (!valid[i]) {
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), i, i + 1, false, stream);
      ++nulls;
    }
  }
  column.set_null_mask(std::move(mask), nulls);
}

template <typename T>
std::unique_ptr<cudf::column> column(std::vector<T> const& values,
                                     cudf::data_type type,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr,
                                     std::vector<bool> const& valid = {})
{
  auto result =
    (type.id() == cudf::type_id::DECIMAL32 || type.id() == cudf::type_id::DECIMAL64 ||
     type.id() == cudf::type_id::DECIMAL128)
      ? cudf::make_fixed_point_column(
          type, values.size(), cudf::mask_state::UNALLOCATED, stream, mr)
      : cudf::make_numeric_column(type, values.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  if (!values.empty()) {
    REQUIRE(cudaMemcpyAsync(result->mutable_view().template data<T>(),
                            values.data(),
                            values.size() * sizeof(T),
                            cudaMemcpyHostToDevice,
                            stream.get()) == cudaSuccess);
  }
  set_validity(*result, valid, stream, mr);
  // Input vectors can be temporaries; finish their host-to-device transfers before returning.
  stream.sync();
  return result;
}

template <typename T>
std::vector<T> host(cudf::column_view input, ::cuda::stream_ref stream)
{
  std::vector<T> values(input.size());
  if (!values.empty()) {
    REQUIRE(cudaMemcpyAsync(values.data(),
                            input.data<T>(),
                            values.size() * sizeof(T),
                            cudaMemcpyDeviceToHost,
                            stream.get()) == cudaSuccess);
  }
  stream.sync();
  return values;
}

std::vector<bool> validity(cudf::column_view input, ::cuda::stream_ref stream)
{
  std::vector<bool> result(input.size(), true);
  if (!input.nullable()) { return result; }
  std::vector<cudf::bitmask_type> mask((input.size() + input.offset() + 31) / 32);
  REQUIRE(cudaMemcpyAsync(mask.data(),
                          input.null_mask(),
                          mask.size() * sizeof(cudf::bitmask_type),
                          cudaMemcpyDeviceToHost,
                          stream.get()) == cudaSuccess);
  stream.sync();
  for (cudf::size_type i = 0; i < input.size(); ++i) {
    auto bit  = input.offset() + i;
    result[i] = (mask[bit / 32] & (1u << (bit % 32))) != 0;
  }
  return result;
}

__int128_t wide(duckdb::Value const& value)
{
  auto v = value.GetValue<duckdb::hugeint_t>();
  return static_cast<__int128_t>(v.upper) * (static_cast<__int128_t>(1) << 64) + v.lower;
}
}  // namespace

TEST_CASE("whole-partition evaluator matches DuckDB while preserving unsorted input rows",
          "[window_evaluator][gpu]")
{
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto setup = con.Query(
    "CREATE TABLE w(id INTEGER, k INTEGER, q VARCHAR, v BIGINT, f FLOAT, payload INTEGER);"
    "INSERT INTO w VALUES (0,2,'b',10,1,99),(1,1,'a',20,2,99),(2,2,'b',NULL,3,99),"
    "(3,NULL,'a',5,4,99),(4,1,'a',40,NULL,99),(5,NULL,'a',7,6,99),"
    "(6,2,'c',NULL,NULL,99),(7,2,'c',NULL,NULL,99),(8,1,NULL,3,9,99),(9,1,NULL,4,10,99)");
  REQUIRE_FALSE(setup->HasError());
  rmm::cuda_stream owning_stream;
  auto stream = owning_stream.view();
  auto mr     = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    column<int32_t>({2, 1, 2, 0, 1, 0, 2, 2, 1, 1},
                    cudf::data_type{cudf::type_id::INT32},
                    stream,
                    mr,
                    {true, true, true, false, true, false, true, true, true, true}));
  auto strings = sirius::test::operator_utils::make_string_column(
    {"b", "a", "b", "a", "a", "a", "c", "c", "", ""}, stream, mr);
  set_validity(
    *strings, {true, true, true, true, true, true, true, true, false, false}, stream, mr);
  columns.push_back(std::move(strings));
  columns.push_back(
    column<int64_t>({10, 20, 0, 5, 40, 7, 0, 0, 3, 4},
                    cudf::data_type{cudf::type_id::INT64},
                    stream,
                    mr,
                    {true, true, false, true, true, true, false, false, true, true}));
  columns.push_back(column<float>({1, 2, 3, 4, 0, 6, 0, 0, 9, 10},
                                  cudf::data_type{cudf::type_id::FLOAT32},
                                  stream,
                                  mr,
                                  {true, true, true, true, false, true, false, false, true, true}));
  columns.push_back(column<int32_t>(
    {99, 99, 99, 99, 99, 99, 99, 99, 99, 99}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  cudf::table input(std::move(columns));
  auto const keyed  = GENERATE(false, true);
  auto const sliced = GENERATE(false, true);
  auto view         = sliced ? cudf::slice(input.view(), {2, 9}, stream).front() : input.view();
  auto result       = evaluate_whole_partition_window(
    view,
    keyed ? std::vector<cudf::size_type>{1, 0} : std::vector<cudf::size_type>{},
    {{function::sum, 2}, {function::avg, 2}, {function::sum, 3}, {function::avg, 2}},
    stream,
    mr);
  auto window   = keyed ? "PARTITION BY k,q" : "";
  auto expected = con.Query(std::string("SELECT sum(v) OVER (") + window + "), avg(v) OVER (" +
                            window + "),sum(f) OVER (" + window + "),avg(v) OVER (" + window +
                            ") FROM w " + (sliced ? "WHERE id>=2 AND id<9 " : "") + "ORDER BY id");
  REQUIRE_FALSE(expected->HasError());
  REQUIRE(result->num_rows() == expected->RowCount());
  REQUIRE(result->num_columns() == 4);
  CHECK(result->get_column(0).type() == cudf::data_type{cudf::type_id::DECIMAL128, 0});
  auto expected_rows = expected->Fetch();
  REQUIRE(expected_rows);
  REQUIRE(expected_rows->size() == expected->RowCount());
  auto sums = host<__int128_t>(result->view().column(0), stream);
  for (cudf::size_type c = 0; c < result->num_columns(); ++c) {
    auto valid   = validity(result->view().column(c), stream);
    auto doubles = c ? host<double>(result->view().column(c), stream) : std::vector<double>{};
    for (cudf::size_type row = 0; row < result->num_rows(); ++row) {
      auto value = expected_rows->GetValue(c, row);
      REQUIRE(valid[row] == !value.IsNull());
      if (!valid[row]) { continue; }
      if (c == 0) {
        CHECK(sums[row] == wide(value));
      } else {
        CHECK(doubles[row] == Catch::Approx(value.GetValue<double>()));
      }
    }
  }
}

TEST_CASE("whole-partition evaluator widens before summation and handles cancellation",
          "[window_evaluator][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream    = owning_stream.view();
  auto mr        = cudf::get_current_device_resource_ref();
  auto const max = std::numeric_limits<int64_t>::max();
  auto const min = std::numeric_limits<int64_t>::min();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    column<int32_t>({0, 0, 1, 1, 2, 2, 2}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  columns.push_back(column<int64_t>(
    {max, max, min, min, max, -max, 1}, cudf::data_type{cudf::type_id::INT64}, stream, mr));
  auto input  = cudf::table(std::move(columns));
  auto result = evaluate_whole_partition_window(
    input.view(), {0}, {{function::sum, 1}, {function::avg, 1}}, stream, mr);
  auto sums = host<__int128_t>(result->view().column(0), stream);
  auto avgs = host<double>(result->view().column(1), stream);
  CHECK(sums == std::vector<__int128_t>{2 * static_cast<__int128_t>(max),
                                        2 * static_cast<__int128_t>(max),
                                        2 * static_cast<__int128_t>(min),
                                        2 * static_cast<__int128_t>(min),
                                        1,
                                        1,
                                        1});
  CHECK(avgs[0] == static_cast<double>(max));
  CHECK(avgs[2] == static_cast<double>(min));
  CHECK(avgs[4] == Catch::Approx(1.0 / 3));

  auto floats =
    column<float>({16777216, 1, -16777216}, cudf::data_type{cudf::type_id::FLOAT32}, stream, mr);
  auto floating = evaluate_whole_partition_window(
    cudf::table_view{{floats->view()}}, {}, {{function::sum, 0}, {function::avg, 0}}, stream, mr);
  CHECK(host<double>(floating->view().column(0), stream) == std::vector<double>{1, 1, 1});
  CHECK(host<double>(floating->view().column(1), stream)[0] == Catch::Approx(1.0 / 3));
}

TEMPLATE_TEST_CASE("whole-partition evaluator preserves decimal scale and null results",
                   "[window_evaluator][gpu]",
                   int32_t,
                   int64_t,
                   __int128_t)
{
  rmm::cuda_stream owning_stream;
  auto stream      = owning_stream.view();
  auto mr          = cudf::get_current_device_resource_ref();
  auto id          = sizeof(TestType) == 4   ? cudf::type_id::DECIMAL32
                     : sizeof(TestType) == 8 ? cudf::type_id::DECIMAL64
                                             : cudf::type_id::DECIMAL128;
  auto const keyed = GENERATE(false, true);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    column<int32_t>({0, 0, 1, 1, 2, 2}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  columns.push_back(column<TestType>({100, 200, 0, 0, 450, 50},
                                     cudf::data_type{id, -2},
                                     stream,
                                     mr,
                                     {true, true, false, false, true, true}));
  auto input  = cudf::table(std::move(columns));
  auto result = evaluate_whole_partition_window(
    input.view(),
    keyed ? std::vector<cudf::size_type>{0} : std::vector<cudf::size_type>{},
    {{function::sum, 1}, {function::avg, 1}},
    stream,
    mr);
  CHECK(result->get_column(0).type() == cudf::data_type{cudf::type_id::DECIMAL128, -2});
  auto valid = validity(result->view().column(0), stream);
  CHECK(valid == (keyed ? std::vector<bool>{true, true, false, false, true, true}
                        : std::vector<bool>(6, true)));
  CHECK(validity(result->view().column(1), stream) == valid);
  auto sums = host<__int128_t>(result->view().column(0), stream);
  auto avgs = host<double>(result->view().column(1), stream);
  CHECK(sums[0] == (keyed ? 300 : 800));
  CHECK(sums[4] == (keyed ? 500 : 800));
  CHECK(avgs[0] == Catch::Approx(keyed ? 1.5 : 2.0));
  CHECK(avgs[4] == Catch::Approx(keyed ? 2.5 : 2.0));
}

TEST_CASE("whole-partition evaluator returns typed empties and nulls for all-null input",
          "[window_evaluator][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream      = owning_stream.view();
  auto mr          = cudf::get_current_device_resource_ref();
  auto const keyed = GENERATE(false, true);
  auto const empty = GENERATE(false, true);
  auto input       = column<int32_t>(empty ? std::vector<int32_t>{} : std::vector<int32_t>{0, 0, 0},
                               cudf::data_type{cudf::type_id::INT32},
                               stream,
                               mr,
                               empty ? std::vector<bool>{} : std::vector<bool>(3, false));
  auto result = evaluate_whole_partition_window(
    cudf::table_view{{input->view()}},
    keyed ? std::vector<cudf::size_type>{0} : std::vector<cudf::size_type>{},
    {{function::sum, 0}, {function::avg, 0}},
    stream,
    mr);
  REQUIRE(result->num_rows() == (empty ? 0 : 3));
  REQUIRE(result->num_columns() == 2);
  CHECK(result->get_column(0).type() == cudf::data_type{cudf::type_id::DECIMAL128, 0});
  CHECK(result->get_column(1).type() == cudf::data_type{cudf::type_id::FLOAT64});
  CHECK(result->get_column(0).null_count() == result->num_rows());
  CHECK(result->get_column(1).null_count() == result->num_rows());
}

TEST_CASE("whole-partition evaluator reports decimal accumulator overflow",
          "[window_evaluator][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream      = owning_stream.view();
  auto mr          = cudf::get_current_device_resource_ref();
  __int128_t limit = 1;
  for (int i = 0; i < 38; ++i) {
    limit *= 10;
  }
  --limit;
  auto const carrier_overflow = GENERATE(false, true);
  // Two half-limits cross DECIMAL(38)'s precision bound without overflowing int128.
  auto const negative  = GENERATE(false, true);
  auto const magnitude = carrier_overflow ? limit : limit / 2 + 1;
  auto const value     = negative ? -magnitude : magnitude;
  auto input =
    column<__int128_t>({value, value}, cudf::data_type{cudf::type_id::DECIMAL128, 0}, stream, mr);
  auto const keyed = GENERATE(false, true);
  auto keys        = column<int32_t>({0, 0}, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  CHECK_THROWS_AS(evaluate_whole_partition_window(
                    cudf::table_view{{keys->view(), input->view()}},
                    keyed ? std::vector<cudf::size_type>{0} : std::vector<cudf::size_type>{},
                    {{function::sum, 1}},
                    stream,
                    mr),
                  std::overflow_error);
}
