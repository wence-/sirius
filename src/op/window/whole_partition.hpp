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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <vector>

namespace sirius::op {

enum class whole_partition_function { sum, avg };

struct whole_partition_aggregate {
  whole_partition_function function;
  cudf::size_type argument_column;
};

/// An evaluator for one complete hash bucket, which may contain many SQL partitions.
/// Input is already prepared: keys and arguments refer to columns in this table.
/// Their logical types (including decimal scales) must be restored before evaluation;
/// this interface does not interpret compressed physical carriers.
/// Returns only the requested result columns, in request order and original row order.
/// Input columns are neither copied nor reordered. The caller retains/appends payload,
/// owns complete-bucket assembly and admission, and removes hidden preparation columns.
///
/// Signed integer SUM uses DECIMAL128(scale=0) as its exact HUGEINT carrier; decimal
/// SUM uses DECIMAL128 at the input scale. Floating SUM and all AVG results are DOUBLE.
/// Integer/decimal arguments are widened before summation; overflow throws.
/// An empty input returns typed empty result columns, including for an unkeyed window.
///
/// No repository, physical operator, or DuckDB planner state is used here. All work is
/// submitted to stream. distinct_hash_join uses cuDF's current resource for its build;
/// callers must install their admitted resource there as well as passing it as mr.
std::unique_ptr<cudf::table> evaluate_whole_partition_window(
  cudf::table_view input,
  std::vector<cudf::size_type> const& partition_columns,
  std::vector<whole_partition_aggregate> const& aggregates,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace sirius::op
