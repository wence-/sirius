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

#include "catch.hpp"
#include "duckdb.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"
#include "duckdb/planner/operator/logical_window.hpp"
#include "duckdb/planner/planner.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_window.hpp"
#include "utils/parquet_fixture_utils.hpp"

using sirius::planner::plan_whole_partition_window;
using sirius::planner::whole_partition_function;

namespace {

struct window_fixture {
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db{nullptr};
  duckdb::Connection con{db};

  window_fixture()
  {
    auto result = con.Query(
      "CREATE TABLE w(p INTEGER, q INTEGER, v INTEGER, f DOUBLE, "
      "d DECIMAL(18,2), wide DECIMAL(38,2), s VARCHAR)");
    REQUIRE_FALSE(result->HasError());
    con.BeginTransaction();
  }
  ~window_fixture() { con.Rollback(); }

  duckdb::unique_ptr<duckdb::LogicalOperator> bind(std::string const& sql, bool optimize = false)
  {
    duckdb::Parser parser(*con.context);
    parser.ParseQuery(sql);
    duckdb::Planner planner(*con.context);
    planner.CreatePlan(std::move(parser.statements[0]));
    auto plan = std::move(planner.plan);
    if (optimize) {
      duckdb::Optimizer optimizer(*planner.binder, *con.context);
      plan = optimizer.Optimize(std::move(plan));
    }
    plan->ResolveOperatorTypes();
    duckdb::ColumnBindingResolver resolver;
    resolver.VisitOperator(*plan);
    return plan;
  }
};

duckdb::LogicalWindow& find_window(duckdb::LogicalOperator& plan)
{
  if (plan.type == duckdb::LogicalOperatorType::LOGICAL_WINDOW) {
    return plan.Cast<duckdb::LogicalWindow>();
  }
  REQUIRE(plan.children.size() == 1);
  return find_window(*plan.children[0]);
}

}  // namespace

TEST_CASE("whole-partition window planning preserves slots and bound types",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind(
    "SELECT v, sum(v) OVER (PARTITION BY p), "
    "avg(v) OVER (PARTITION BY p) FROM w");
  auto& window = find_window(*logical);
  auto plan    = plan_whole_partition_window(window);
  REQUIRE(plan.input_width == window.children[0]->types.size());
  REQUIRE(plan.output_types == window.types);
  REQUIRE(plan.prepared_types == window.children[0]->types);
  REQUIRE(plan.hidden_expressions.empty());
  REQUIRE(plan.partition_slots.size() == 1);
  REQUIRE(plan.expressions.size() == 2);
  CHECK(plan.expressions[0].function == whole_partition_function::sum);
  CHECK(plan.expressions[1].function == whole_partition_function::avg);
  CHECK(plan.expressions[0].argument_slot == plan.expressions[1].argument_slot);
  CHECK(plan.expressions[0].argument_type == duckdb::LogicalType::INTEGER);
  CHECK(plan.expressions[0].result_type == duckdb::LogicalType::HUGEINT);
  CHECK(plan.expressions[1].result_type == duckdb::LogicalType::DOUBLE);
  CHECK(plan.expressions[0].output_slot == plan.input_width);
  CHECK(plan.expressions[1].output_slot == plan.input_width + 1);
  CHECK(plan.expressions[0].start == duckdb::WindowBoundary::UNBOUNDED_PRECEDING);
  CHECK(plan.expressions[0].end == duckdb::WindowBoundary::CURRENT_ROW_RANGE);
}

TEST_CASE("whole-partition window planning shares hidden expressions", "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind(
    "SELECT sum(v+1) OVER (PARTITION BY p+1, q), "
    "avg(v+1) OVER (PARTITION BY q, p+1) FROM w");
  auto plan = plan_whole_partition_window(find_window(*logical));
  REQUIRE(plan.hidden_expressions.size() == 2);
  REQUIRE(plan.partition_slots.size() == 2);
  CHECK(plan.partition_slots[0] == plan.input_width);
  CHECK(plan.expressions[0].argument_slot == plan.input_width + 1);
  CHECK(plan.expressions[1].argument_slot == plan.input_width + 1);
  CHECK(plan.prepared_types.size() == plan.input_width + 2);
  CHECK(plan.output_types.size() == plan.input_width + 2);

  // An expression used as both a partition key and an argument gets one column.
  logical = fixture.bind("SELECT sum(p+1) OVER (PARTITION BY p+1) FROM w");
  plan    = plan_whole_partition_window(find_window(*logical));
  REQUIRE(plan.hidden_expressions.size() == 1);
  CHECK(plan.expressions[0].argument_slot == plan.partition_slots[0]);
}

TEST_CASE("whole-partition window planning admits only whole unordered frames",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto const clause =
    GENERATE("",
             "PARTITION BY p",
             "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
             "PARTITION BY p RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING");
  auto logical = fixture.bind(std::string("SELECT sum(v) OVER (") + clause + ") FROM w");
  auto plan    = plan_whole_partition_window(find_window(*logical));
  REQUIRE(plan.expressions.size() == 1);
}

TEST_CASE("whole-partition window planning rejects unsupported bound semantics",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto const expression = GENERATE(
    "sum(v) OVER (ORDER BY p)",
    "sum(v) OVER (ORDER BY p ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)",
    "sum(v) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)",
    "sum(v) OVER (ROWS BETWEEN 1 PRECEDING AND UNBOUNDED FOLLOWING)",
    "sum(v) OVER (ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)",
    "sum(v) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE CURRENT ROW)",
    "sum(v) OVER (RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE GROUP)",
    "sum(v) OVER (RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE TIES)",
    "sum(DISTINCT v) OVER ()",
    "sum(v) FILTER (WHERE v > 0) OVER ()",
    "sum(v ORDER BY p) OVER ()",
    "count(*) OVER ()",
    "min(v) OVER ()",
    "max(v) OVER ()",
    "row_number() OVER ()",
    "rank() OVER (ORDER BY p)",
    "lag(v) OVER ()",
    "sum(v::HUGEINT) OVER ()",
    "sum(v::UBIGINT) OVER ()",
    "sum(v) OVER (PARTITION BY [p])",
    "sum(v) OVER (PARTITION BY f)",
    "sum(v) OVER (PARTITION BY s COLLATE nocase)",
    "sum(random()) OVER ()",
    "sum(atan2(f, 1)) OVER ()",
    "sum(v) OVER (), avg(v) OVER (PARTITION BY p)",
    "sum(v) OVER (PARTITION BY p, p), avg(v) OVER (PARTITION BY p, q)");
  INFO(expression);
  // The table is empty. Capability rejection must still occur before execution.
  auto logical = fixture.bind(std::string("SELECT ") + expression + " FROM w");
  REQUIRE_THROWS_AS(plan_whole_partition_window(find_window(*logical)),
                    duckdb::NotImplementedException);
}

TEST_CASE("whole-partition window planning retains decimal widening", "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind("SELECT sum(d) OVER (), avg(d) OVER (), sum(f) OVER () FROM w");
  auto plan    = plan_whole_partition_window(find_window(*logical));
  REQUIRE(plan.expressions.size() == 3);
  CHECK(plan.expressions[0].argument_type == duckdb::LogicalType::DECIMAL(18, 2));
  CHECK(plan.expressions[0].result_type == duckdb::LogicalType::DECIMAL(38, 2));
  CHECK(plan.expressions[1].result_type == duckdb::LogicalType::DOUBLE);
  CHECK(plan.expressions[2].result_type == duckdb::LogicalType::DOUBLE);
}

TEST_CASE("whole-partition window planner seam keeps runtime fallback closed",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto const supported = GENERATE(false, true);
  auto logical         = fixture.bind(supported ? "SELECT sum(v) OVER () FROM w"
                                        : "SELECT sum(v) OVER (ORDER BY p) FROM w");
  sirius::planner::sirius_physical_plan_generator generator(*fixture.con.context);
  REQUIRE_THROWS_WITH(generator.create_plan(std::move(logical)),
                      Catch::Matchers::ContainsSubstring(supported ? "execution is not implemented"
                                                                   : "window ORDER BY"));
}

TEST_CASE("whole-partition window planning rejects invalid mappings and null treatment",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical     = fixture.bind("SELECT sum(v) OVER (PARTITION BY p) FROM w");
  auto& window     = find_window(*logical);
  auto& expression = window.expressions[0]->Cast<duckdb::BoundWindowExpression>();
  SECTION("unsupported null treatment") { expression.IgnoreNullsMutable() = true; }
  SECTION("incorrect visible result type") { window.types.back() = duckdb::LogicalType::DOUBLE; }
  SECTION("incorrect input mapping") { window.types[0] = duckdb::LogicalType::VARCHAR; }
  SECTION("argument outside input schema")
  {
    expression.GetChildrenMutable()[0]->Cast<duckdb::BoundReferenceExpression>().IndexMutable() =
      window.children[0]->types.size();
  }
  REQUIRE_THROWS_AS(plan_whole_partition_window(window), duckdb::NotImplementedException);
}

TEST_CASE("whole-partition window planning checks optimized TPC-DS shapes",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto const aggregate = GENERATE("sum", "avg");
  // Q12/Q20/Q98 sum grouped decimal sales; Q53/Q63/Q89 average them.
  // Use the real optimizer: an input SUM widens DECIMAL(18,2) to DECIMAL(38,2).
  auto logical = fixture.bind(
    std::string("SELECT ") + aggregate + "(sum(d)) OVER (PARTITION BY p) FROM w GROUP BY p, q",
    true);
  auto plan = plan_whole_partition_window(find_window(*logical));
  REQUIRE(plan.expressions.size() == 1);
  CHECK(plan.expressions[0].argument_type == duckdb::LogicalType::DECIMAL(38, 2));
}
