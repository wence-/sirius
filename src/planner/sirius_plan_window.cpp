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

#include "planner/sirius_plan_window.hpp"

#include "cudf/cudf_utils.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"
#include "duckdb/planner/operator/logical_window.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/reference.hpp"
#include "expression/ast/utils.hpp"
#include "helper/type_conversions.hpp"
#include "op/partition/partition_sizing.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_partition.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"

#include <limits>
#include <stdexcept>

namespace sirius::planner {
namespace {

[[noreturn]] void decline(char const* reason)
{
  throw duckdb::NotImplementedException("Unsupported whole-partition window: %s", reason);
}

bool numeric_argument(duckdb::LogicalType const& type)
{
  using duckdb::LogicalTypeId;
  switch (type.id()) {
    case LogicalTypeId::TINYINT:
    case LogicalTypeId::SMALLINT:
    case LogicalTypeId::INTEGER:
    case LogicalTypeId::BIGINT:
    case LogicalTypeId::FLOAT:
    case LogicalTypeId::DOUBLE:
    case LogicalTypeId::DECIMAL: return true;
    default: return false;
  }
}

bool partition_key(duckdb::LogicalType const& type)
{
  using duckdb::LogicalTypeId;
  switch (type.id()) {
    case LogicalTypeId::BOOLEAN:
    case LogicalTypeId::TINYINT:
    case LogicalTypeId::SMALLINT:
    case LogicalTypeId::INTEGER:
    case LogicalTypeId::BIGINT:
    case LogicalTypeId::DATE:
    case LogicalTypeId::TIMESTAMP:
    case LogicalTypeId::TIMESTAMP_MS:
    case LogicalTypeId::TIMESTAMP_NS:
    case LogicalTypeId::TIMESTAMP_SEC: return true;
    case LogicalTypeId::VARCHAR: return duckdb::StringType::GetCollation(type).empty();
    default: return false;
  }
}

whole_partition_function validate(duckdb::BoundWindowExpression const& expression)
{
  using duckdb::WindowBoundary;
  if (expression.GetExpressionType() != duckdb::ExpressionType::WINDOW_AGGREGATE ||
      !expression.AggregateFunction()) {
    decline("only SUM and AVG aggregates are planned");
  }
  if (expression.Distinct()) { decline("DISTINCT"); }
  if (expression.Filter()) { decline("FILTER"); }
  if (expression.WindowExclude() != duckdb::WindowExcludeMode::NO_OTHER) { decline("EXCLUDE"); }
  if (!expression.ArgOrders().empty()) { decline("argument ORDER BY"); }
  if (!expression.OrderBy().empty()) { decline("window ORDER BY"); }
  if (expression.IgnoreNulls()) { decline("IGNORE NULLS"); }
  if (expression.StartExpr() || expression.EndExpr()) {
    decline("frame offsets or navigation arguments");
  }
  // Without ORDER BY all rows are peers. DuckDB retains CURRENT_ROW_RANGE for
  // the default frame; CURRENT_ROW_ROWS still ends at the physical current row.
  if (expression.WindowStart() != WindowBoundary::UNBOUNDED_PRECEDING ||
      (expression.WindowEnd() != WindowBoundary::UNBOUNDED_FOLLOWING &&
       expression.WindowEnd() != WindowBoundary::CURRENT_ROW_RANGE)) {
    decline("frame does not cover the whole partition");
  }
  if (expression.GetChildren().size() != 1 ||
      !numeric_argument(expression.GetChildren()[0]->GetReturnType())) {
    decline("aggregate argument type or arity");
  }
  auto const& argument = expression.GetChildren()[0]->GetReturnType();
  auto const& name     = expression.AggregateFunction()->GetName().GetIdentifierName();
  if (name == "avg" && expression.GetReturnType() == duckdb::LogicalType::DOUBLE) {
    return whole_partition_function::avg;
  }
  if (name == "sum") {
    duckdb::LogicalType expected = duckdb::LogicalType::HUGEINT;
    if (argument.id() == duckdb::LogicalTypeId::DECIMAL) {
      expected = duckdb::LogicalType::DECIMAL(38, duckdb::DecimalType::GetScale(argument));
    } else if (argument.id() == duckdb::LogicalTypeId::FLOAT ||
               argument.id() == duckdb::LogicalTypeId::DOUBLE) {
      expected = duckdb::LogicalType::DOUBLE;
    }
    if (expression.GetReturnType() == expected) { return whole_partition_function::sum; }
  }
  decline("aggregate function or result type");
}

}  // namespace

whole_partition_window_plan plan_whole_partition_window(duckdb::LogicalWindow const& window)
{
  if (window.children.size() != 1 || window.expressions.empty()) {
    decline("expected one child and at least one expression");
  }
  whole_partition_window_plan plan;
  plan.input_width    = window.children[0]->types.size();
  plan.prepared_types = window.children[0]->types;
  plan.output_types   = window.types;
  if (window.types.size() != plan.input_width + window.expressions.size()) {
    decline("output width");
  }
  for (duckdb::idx_t i = 0; i < plan.input_width; ++i) {
    if (window.types[i] != plan.prepared_types[i]) { decline("input column mapping"); }
  }

  // Compare bound expressions before translation. Reuse references directly and
  // evaluate each deterministic hidden expression once for the entire stage.
  std::vector<duckdb::Expression const*> hidden;
  auto prepare = [&](duckdb::Expression const& expression) -> duckdb::idx_t {
    if (expression.IsVolatile()) { decline("volatile key or argument"); }
    if (expression.GetExpressionType() == duckdb::ExpressionType::BOUND_REF) {
      auto const& ref = expression.Cast<duckdb::BoundReferenceExpression>();
      if (ref.Index() >= plan.input_width ||
          expression.GetReturnType() != plan.prepared_types[ref.Index()]) {
        decline("input reference");
      }
      return ref.Index();
    }
    for (std::size_t i = 0; i < hidden.size(); ++i) {
      if (expression.Equals(*hidden[i])) { return plan.input_width + i; }
    }
    auto translated = ast::from_duckdb(expression);
    if (!translated) { decline("key or argument expression cannot be translated"); }
    auto const slot = plan.prepared_types.size();
    hidden.push_back(&expression);
    plan.hidden_expressions.push_back(std::move(translated));
    plan.prepared_types.push_back(expression.GetReturnType());
    return slot;
  };

  duckdb::BoundWindowExpression const* organization = nullptr;
  for (duckdb::idx_t i = 0; i < window.expressions.size(); ++i) {
    auto const& bound = *window.expressions[i];
    if (bound.GetExpressionClass() != duckdb::ExpressionClass::BOUND_WINDOW) {
      decline("expected a bound window expression");
    }
    auto const& expression = bound.Cast<duckdb::BoundWindowExpression>();
    auto const function    = validate(expression);
    if (organization && (!organization->PartitionsAreEquivalent(expression) ||
                         !expression.PartitionsAreEquivalent(*organization))) {
      decline("incompatible partition specifications");
    }
    for (auto const& key : expression.Partitions()) {
      if (!partition_key(key->GetReturnType())) { decline("partition key type or collation"); }
      if (key->IsVolatile()) { decline("volatile key or argument"); }
    }
    if (!organization) {
      organization = &expression;
      for (auto const& key : expression.Partitions()) {
        plan.partition_slots.push_back(prepare(*key));
      }
    }
    auto const output_slot = plan.input_width + i;
    if (window.types[output_slot] != expression.GetReturnType()) {
      decline("result column mapping");
    }
    plan.expressions.push_back({function,
                                prepare(*expression.GetChildren()[0]),
                                expression.GetChildren()[0]->GetReturnType(),
                                expression.GetReturnType(),
                                expression.WindowStart(),
                                expression.WindowEnd(),
                                output_slot});
  }
  return plan;
}

duckdb::unique_ptr<op::sirius_physical_operator> build_whole_partition_window_input(
  duckdb::unique_ptr<op::sirius_physical_operator> child,
  whole_partition_window_plan const& plan,
  std::size_t estimated_cardinality,
  uint64_t hash_partition_bytes)
{
  if (hash_partition_bytes == 0) {
    throw std::invalid_argument("Partition byte target must be positive");
  }
  if (!child || child->types.size() != plan.input_width ||
      plan.prepared_types.size() != plan.input_width + plan.hidden_expressions.size()) {
    throw duckdb::InternalException("Window preparation schema does not match its child");
  }
  auto prepared = sirius::from_duckdb_vec(plan.prepared_types);
  duckdb::vector<std::unique_ptr<ast::node>> select;
  for (duckdb::idx_t i = 0; i < plan.input_width; ++i) {
    if (child->types[i] != prepared[i]) {
      throw duckdb::InternalException("Window child type mismatch");
    }
    select.push_back(
      std::make_unique<ast::node>(ast::reference{static_cast<uint32_t>(i), prepared[i]}));
  }
  for (auto const& expression : plan.hidden_expressions) {
    select.push_back(ast::clone(*expression));
  }
  child = push_projection(std::move(child), prepared, std::move(select), estimated_cardinality);
  std::vector<int> keys;
  for (auto slot : plan.partition_slots) {
    if (slot > static_cast<duckdb::idx_t>(std::numeric_limits<int>::max())) {
      throw duckdb::NotImplementedException("Window partition slot exceeds the native index range");
    }
    keys.push_back(static_cast<int>(slot));
  }
  auto assembly = duckdb::make_uniq<op::sirius_physical_concat>(
    prepared, estimated_cardinality, /*concat_all=*/true);
  op::partition_spec spec;
  spec.mode = keys.empty() ? op::partition_mode::single : op::partition_mode::hash;
  for (auto key : keys) {
    spec.key_cast_types.push_back(sirius::get_cudf_type(prepared[key]));
  }
  spec.key_columns = std::move(keys);
  auto exchange    = duckdb::make_uniq<op::sirius_physical_partition>(
    prepared, estimated_cardinality, std::move(spec), op::MemoryBarrierType::FULL);
  if (plan.partition_slots.empty()) {
    exchange->set_fixed_layout({op::partition_mode::single, op::partition_placement::unpinned(1)});
  } else {
    op::partition_exchange::bind(
      {exchange.get()},
      0,
      false,
      [target = hash_partition_bytes, receiver = assembly.get()](auto const& stats, bool) {
        auto count = op::natural_num_partitions(stats[0].bytes, target, receiver->num_gpus());
        return std::vector<op::partition_layout>{
          {op::partition_mode::hash, op::partition_placement::unpinned(count)}};
      });
  }
  exchange->children.push_back(std::move(child));
  assembly->children.push_back(std::move(exchange));
  return assembly;
}

duckdb::unique_ptr<op::sirius_physical_operator> sirius_physical_plan_generator::create_plan(
  duckdb::LogicalWindow& window)
{
  auto bound_plan = plan_whole_partition_window(window);
  // The input builder now implements complete-bucket assembly, but its schema
  // contains prepared input columns, not SQL results. Keep plan-time fallback
  // until reduction/expansion upholds the bound output/type contracts.
  throw duckdb::NotImplementedException(
    "Whole-partition window execution is not implemented (%llu expressions planned)",
    static_cast<unsigned long long>(bound_plan.expressions.size()));
}

}  // namespace sirius::planner
