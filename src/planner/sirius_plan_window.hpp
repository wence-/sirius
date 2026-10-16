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

#include "duckdb/common/types.hpp"
#include "duckdb/parser/expression/window_expression.hpp"
#include "expression/ast/node.hpp"

#include <memory>
#include <vector>

namespace duckdb {
class LogicalWindow;
}

namespace sirius::planner {

enum class whole_partition_function { sum, avg };

struct whole_partition_expression {
  whole_partition_function function;
  duckdb::idx_t argument_slot;
  duckdb::LogicalType argument_type;
  // Keep DuckDB's exact result type, including HUGEINT and DECIMAL(38, scale).
  duckdb::LogicalType result_type;
  duckdb::WindowBoundary start;
  duckdb::WindowBoundary end;
  duckdb::idx_t output_slot;
};

// Planning IR only: not an executable operator. All slots index the prepared input
// (original columns followed by hidden expressions), except output_slot, which
// indexes the visible output (original columns followed by window results).
struct whole_partition_window_plan {
  duckdb::idx_t input_width;
  duckdb::vector<duckdb::LogicalType> prepared_types;
  duckdb::vector<duckdb::LogicalType> output_types;
  std::vector<duckdb::idx_t> partition_slots;
  std::vector<std::unique_ptr<ast::node>> hidden_expressions;
  std::vector<whole_partition_expression> expressions;
};

// Requires resolved column references. Does not consume or modify the logical
// plan. Throws NotImplementedException for unsupported combinations, even on
// empty input. Hidden columns must survive through reduction/expansion and be
// removed when the visible output is published.
whole_partition_window_plan plan_whole_partition_window(duckdb::LogicalWindow const& window);

}  // namespace sirius::planner
