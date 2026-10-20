# Plan 1: whole-partition aggregate windows

## Scope and execution contract

Deliver a narrow end-to-end GPU implementation of whole-partition SUM and AVG,
with COUNT/MIN/MAX implemented as follow-ups on the same reduction/expansion path. Start with partition-only specifications,
including no partition keys. An explicit whole frame with ORDER BY can be admitted
later when bound-semantic validation proves that order does not affect the admitted
aggregate and modifiers. An ordered default frame must not be treated as whole.

```text
input + evaluated keys/arguments
  -> hash exchange on partition keys (or gather with no keys)
  -> FULL barrier
  -> claim and concatenate one entire bucket
  -> group by actual keys and reduce
  -> expand aggregate results to original rows
  -> append result columns and publish
```

One bucket may contain multiple SQL partitions. No SQL partition crosses buckets.
This is not a global merge-sort dependency and needs no new cuCascade order API.
If reusing existing merge output in a test or future fast path, independently
establish complete SQL partition coverage; a sorted range alone is insufficient.

## Barrier communication and bucket claims

Use existing FULL input ports and the existing consumer-to-producer readiness
protocol. The relevant implementation seams are:

| Seam | Responsibility |
| --- | --- |
| `sirius_physical_operator::get_next_task_hint()` | Gate FULL ports on producer pipeline completion; return READY or WAITING with the producer to pursue |
| `sirius_physical_operator::is_source_pipeline_finished()` | Check upstream pipelines associated with the operator's ports |
| `task_creator::get_operator_for_next_task()` | Follow WAITING hints upstream to find work that can make progress |
| `sirius_pipeline::update_pipeline_status()` | Establish completion using source/input exhaustion and task accounting, including intentional LIMIT termination |
| `sirius_pipeline::notify_downstream_pipelines()` | Schedule consumers to recheck readiness after completion |
| CONCAT source `get_next_task_input_data()` | Claim one whole completed bucket under coordinated task creation |

Source locations: `src/op/sirius_physical_operator.cpp`,
`src/creator/task_creator.cpp`, `src/pipeline/sirius_pipeline.cpp`, and
`src/pipeline/gpu_pipeline_task.cpp`. Recheck these against the implementation
revision rather than duplicating their lifecycle logic inside the window operator.

```text
consumer get_next_task_hint()
  producer unfinished -> WAITING_FOR_INPUT_DATA, naming producer
                          task creator follows the dependency upstream
  producer finished   -> READY if a nonempty unclaimed bucket remains
                          otherwise no input task available

producer pipeline completion
  -> schedules downstream consumers
  -> consumer rechecks hint
  -> coordinated input acquisition claims an entire bucket
```

Reuse the base hint implementation where its port/data checks suffice. Any
window-specific override must preserve FULL gating and correctly handle empty
buckets. A hint does not reserve data: concurrent callers must not split a bucket
or claim it twice. Coordinate whole-bucket extraction under the consumer's drain
lock and existing task-creation synchronization; individual repository pop locks
alone do not make a multi-pop bucket claim atomic. Recheck acquisition conditions
within that coordination and tolerate another creator having already claimed work.

The barrier covers the entire producing pipeline. Do not attempt per-bucket
readiness while that pipeline is still publishing. Once it is finished, every
bucket is complete, including empty ones. Claiming the final bucket does not mean
the window pipeline has completed: its created tasks must still execute, publish,
and retire under ordinary accounting.

On the successful GPU path, `publish_output()` calls the sink before task
destruction increments completed-task accounting. Thus consumer gating must wait
through sink deposition, not just exhaustion of producer input or completion of
GPU computation. Task destruction also participates in failure cleanup, so the
finished flag is not independently a success certificate. Preserve the existing
query lifecycle's fatal-error and cancellation gates; do not add a second success
protocol or treat failed partial publication as a valid complete input.

The repository provides payload storage only. No completion flag, completed-reader
type, per-bucket ledger, or new cuCascade API is needed for this barrier.

## Slices

### 1. Bound planning and capability gate

Implementation status: the planning slice lives in
`src/planner/sirius_plan_window.{hpp,cpp}`. The `LogicalWindow` dispatcher calls
the builder, which validates and prepares a planning IR. Slice 4 now lowers that IR to
an executable WINDOW adapter. Unsupported bound semantics retain plan-time CPU fallback.

The initial **planning** matrix is:

| Item | Accepted |
| --- | --- |
| Functions | SUM, AVG, COUNT, MIN and MAX with one argument; COUNT(*) with no argument |
| Argument types | Signed TINYINT/SMALLINT/INTEGER/BIGINT, FLOAT/DOUBLE, DECIMAL with precision 5 through 38 |
| SUM result | Exact bound HUGEINT for signed integers, DOUBLE for floating arguments, DECIMAL(38, input scale) for decimals |
| AVG result | Bound DOUBLE; decimal argument scale is retained |
| Partition keys | BOOLEAN, signed integers through BIGINT, DATE, TIMESTAMP/TIMESTAMP_S/TIMESTAMP_MS/TIMESTAMP_NS, uncollated VARCHAR |
| Frames | Unordered UNBOUNDED PRECEDING to UNBOUNDED FOLLOWING, or to CURRENT ROW RANGE (all rows are peers without ordering) |
| Multiple expressions | Equivalent partition keys, including different key order; result order follows the logical window expression order |

Ordered frames (even explicitly whole), current-row ROWS frames, DISTINCT,
FILTER, EXCLUDE, argument ordering, IGNORE NULLS, functions outside this matrix,
unsupported types, volatile expressions, and untranslatable key/argument
expressions are rejected during planning. COUNT/MIN/MAX share the existing complete-bucket evaluator.
DuckDB may bind a SQL argument through a cast; the matrix applies to the bound
argument, after that cast.

Original input slots remain intact. Bare references reuse their input slots;
equal deterministic hidden expressions share one appended preparation slot,
including an expression used as both a key and an argument. One shared key-slot
list describes the entire stage. Hidden slots remain in `prepared_types` through
their final use; visible `output_types` contain only original input columns plus
window results. Each result records its argument slot/type, exact bound result
type, original bound frame endpoints, and visible output slot. The IR owns Sirius
AST expressions and does not consume the DuckDB logical plan.

In particular, no HUGEINT downcast or decimal result narrowing happens here.
Execution must honor these contracts before activation, or narrow the capability
matrix with corresponding tests. DECIMAL(38, scale) arguments matter for the
TPC-DS shapes `sum(sum(sales)) OVER (...)` and `avg(sum(sales)) OVER (...)`:
the inner grouped SUM already widens a DECIMAL sales column to precision 38.
`test/cpp/planner/test_window_plan.cpp` checks these shapes through DuckDB's
optimizer; it does not establish full TPC-DS query support.

Add a `LogicalWindow` plan-builder seam at the current rejection in
`src/planner/sirius_physical_plan_generator.cpp`. Inspect the current
`duckdb/.../bound_window_expression.hpp` and DuckDB physical window planner.
Record argument/partition expression slots, bound result types, frame endpoints,
and output-column mapping. Prepare hidden expressions once per applicable stage
and retain their columns through their final use.

Start with a small explicit function/type/frame matrix. Reject DISTINCT, FILTER,
EXCLUDE, argument ORDER BY, and other unsupported combinations explicitly. Do not
silently ignore modifiers. Multiple expressions sharing the same organization
should share assembly. Until Plan 2 supports incompatible specifications, reject
them as a whole during planning. Preserve the established fallback/error route.

Acceptance: planner tests distinguish whole frames from current-row frames,
unsupported cases cannot reach execution, and result slots/types are correct.

### 2. Complete-bucket input organization

Implementation: `build_whole_partition_window_input()` constructs a prepared
projection, a shared PARTITION router with explicit keys, and the generic
`sirius_physical_concat` with `concat_all = true`. This input-stage builder
returns the prepared schema, without SQL window results. SQL activation uses the same
preparation and exchange builders, inserting PARTITION/CONCAT after compressed-schema
propagation alongside the existing pipeline wrappers.

The planner supplies normalized key types and binds a byte-target sizing policy through the
shared `partition_exchange`. Without keys it installs a fixed single-destination layout.
The router's declared ingress is FULL so sizing observes the completed input, subject to the
existing producer protocol. CONCAT consumes the resulting layout and does not select keys or
partition counts. Placement stays unpinned, using ordinary task locality and cross-device
preparation. No window-specific constructor, sizing branch, or consumer pointer is needed
inside PARTITION.

CONCAT requires producer completion before claiming an entire nonempty partition
under its drain lock, skipping empty gaps. Its ordinary publication preserves
partition indices and row order within the repository's batch sequence. Generic
CONCAT preparation checks decoded input capacity and native row-count limits;
concatenation uses its existing memory estimates and native error propagation.
No-key windows must fit as one complete input. Slice 3 must extend accounting for
aggregate outputs, expansion maps, and scratch before activation.

The planner suite checks the PARTITION -> CONCAT tree through pipeline conversion
and repository materialization, including the FULL exchange ingress and the
producing pipeline reference. CONCAT retains its ordinary PARTIAL port; its
`concat_all` hint and claim paths enforce producer-completion gating. Generic CONCAT tests cover complete claims, empty gaps, concurrent
claims, output partition indices, capacity/row-count rejection, and cross-device
preparation. Window input-stage tests exercise the actual builder for hash sizing and no-key gather.
These checks do not establish end-to-end SQL window execution. Validation results
from the superseded window-owned assembly operator do not validate this wiring.

Account for retained input plus assembly, outputs, and scratch in admission.
Define the behavior of impossible working sets and native row-count overflow;
avoid endless retries. Spilling input is useful but does not make an oversized
concatenation fit. Reservations and estimates are not hard skew guarantees.

Acceptance: fragmented input and delayed producers still yield one complete claim
per bucket; null/composite keys and forced collisions retain complete groups;
empty input, cancellation, and preparation failures follow existing lifecycle.

Barrier-specific acceptance:

* Hold the last producer before or during sink deposition. Even with batches
  already resident, the consumer remains waiting and claims nothing.
* Release the producer and verify normal pipeline completion schedules the
  consumer; progress must not depend on an unrelated later task or manual wakeup.
* Exercise concurrent task creation and empty gaps between buckets. Each nonempty
  bucket is claimed whole exactly once; later buckets remain reachable.
* Complete an empty producer and verify downstream termination with zero window
  rows. Neither an empty repository nor a `nullopt` hint prematurely closes an
  upstream pipeline that can still publish.
* Delay a claimed window task after all input buckets have been drained. The
  window pipeline must remain unfinished until task publication and retirement.
* Inject producer failure, including partial sink publication, and cancellation.
  The query must follow its existing terminal-error path rather than report a
  successful result from incomplete buckets. If current lifecycle behavior does
  not satisfy this, stop activation and address that concrete issue separately.

### 3. Reduction and expansion

Implementation: `src/op/window/whole_partition.{hpp,cpp}` provides a table evaluator,
independent of repositories, physical operators, and the DuckDB planning IR. It accepts a
read-only, type-normalized prepared bucket and returns only the requested result columns,
in request order and aligned with the original input rows. The caller retains payload,
appends these results, and drops hidden preparation columns.

The baseline batches SUM and non-null COUNT requests by argument column in one groupby,
including null partition keys. A null-safe `distinct_hash_join` against the unique group
keys supplies one aggregate-row index per input row; one gather expands all results.
No explicit key sort, payload reordering, or inverse permutation is required. Without
keys, scalar reductions and broadcast avoid both groupby and join.

Arguments are widened before summation: signed integers use DECIMAL128(scale 0) as an
exact HUGEINT carrier, decimal arguments use DECIMAL128 at their logical scale, and
floating arguments use DOUBLE. Integer/decimal summation uses checked native aggregation;
overflow throws rather than publishing wrapped values. AVG divides the widened sum by
the non-null count and returns DOUBLE; all-null groups remain null. The interface documents
the stream and allocator contract, including the distinct hash table's current-resource
requirement.

SQL activation is provided by slice 4. The adapter restores compressed argument types
before calling the evaluator and retains its explicit HUGEINT carrier. This evaluator
slice itself does not own a physical operator or modify partition/CONCAT ownership.

Before SQL activation, `sirius_library` and `sirius_unittest` built and linked. The GPU
selection `[window_evaluator],[window_plan],[window_input_plan]` recorded **678 assertions
in 17 test cases**, including DuckDB comparisons for nullable composite keys, global
windows and sliced inputs, wide integer sums, decimal scale/nulls/overflow, and
existing input-stage checks. This establishes evaluator correctness for these
cases, not SQL GPU window activation or a performance result.

Use cuDF/Sirius aggregation primitives where their semantics match the admitted
matrix. Key grouping must include nulls and distinguish collisions by actual key
equality. Expand each group result to every original row through a row-to-group
mapping, or a proven null-safe unique-group lookup. A lookup must not duplicate or
drop rows. Never assume groupby result order equals input order.

With no keys, reduce the entire batch and broadcast to its existing rows; empty
input still produces zero rows. AVG must use correct sum/count or a verified
equivalent implementation, with correct validity and output type. Do not reuse
an aggregation wrapper that discards payload or changes row association.

Keep table input/output and SQL partition metadata separate from repository/task
ownership. The physical adapter appends the evaluator's result columns; all retained
input columns must stay aligned.

Acceptance: exact row multiplicity, duplicate payloads, null keys, all-null values,
empty input, admitted numeric boundaries, and several expressions on one group.

### 4. Activate and validate the SQL path

Implementation: `src/op/sirius_physical_window.{hpp,cpp}` is a pipelineable adapter after
the generic complete-bucket CONCAT. It restores evaluated key/argument carriers, calls
the table evaluator, and publishes a view of original payload plus owned result columns.
The output owner retains the input read lock and result table; hidden preparation columns
are omitted. Its source acquisition drains assembled batches across every repository
partition under the operator lock, skipping gaps and rechecking the FULL barrier. The base
operator's partition-0-only pop cannot consume CONCAT's indexed multi-bucket publication.
Empty upstream input terminates through the ordinary source protocol.
`WINDOW` task deployment is observable through existing task telemetry and execute has
its own NVTX range. Runtime task reporters now use the context's existing public event
publisher, so subscribers can observe the real task deployments rather than a separate
private event source. No window-specific repository, completion, or cancellation protocol
is introduced.

The cold-start estimate includes restored/widened values, per-group state, copied keys,
hash lookup scratch, the row-to-group map, and expanded results. A known lower bound
(input plus result data) rejects impossible working sets before reduction. Conservative
estimates are admission hints rather than hard-fit proofs; remaining native OOM follows
the existing bounded retry policy. A whole hash bucket must fit on one GPU, and an unkeyed
window requires the entire input to fit. CONCAT retains its decoded-input and native
row-count checks.

DECIMAL arguments of precision 1–4 fall back: the generic Sirius mapping has no cuDF
carrier for DuckDB's DECIMAL16 storage. The planning gate rejects them explicitly.
DECIMAL128 inputs are exercised through Parquet and preceding GPU grouped SUMs; the existing
DuckDB-native scan still declines stored DECIMAL128 columns before window execution.

Integer SUM results remain DECIMAL128(scale=0) through payload forwarding and the result
collector's conversion to DuckDB HUGEINT. Until generic expression consumers support an
exact HUGEINT carrier, computed expressions/predicates over these integer SUM results
fall back during planning. Bare projection, ordering and LIMIT remain admitted. Decimal
SUM and DOUBLE AVG results can feed supported downstream expressions, including TPC-DS
revenue ratios. With upstream ABS support, the deviation predicates in Q53/Q63/Q89
also execute on GPU. Multiple logical window stages remain a
plan-time fallback until Plan 2; the activation does not implicitly enable chained specifications.

Differential SQL tests live in `test/cpp/integration/test_gpu_execution_window.cpp`.
They check production task-deployment events as well as transparent execution/fallback
counters; EXPLAIN alone does not establish activation. Original TPC-DS Q12/Q20/Q53/Q63/
Q89/Q98 are probed both with Sirius's default optimizer mask and with `window_self_join`
additionally disabled. The test preserves Sirius's required exclusions when changing that
setting, so elimination of WINDOW is distinguishable from actual GPU window execution.

SQL CLI validation on 2026-10-20: two RTX A6000 GPUs were available outside the sandbox;
the SQL configuration uses one GPU, and the local build targets SM86. Original query text
from `duckdb/extension/tpcds/dsdgen/queries/` runs against the committed SF 0.01 database.
Both optimizer modes produced the following results:

| Query | Rows | Result | Window evidence |
| --- | --- | --- | --- |
| Q12 | 14 | GPU execution; matches DuckDB | 3 WINDOW task deployments |
| Q20 | 24 | GPU execution; matches DuckDB | 4 WINDOW task deployments |
| Q98 | 24 | GPU execution; matches DuckDB | 4 WINDOW task deployments |
| Q53 | 3 | GPU execution; matches DuckDB | 1 WINDOW task deployments |
| Q63 | 10 | GPU execution; matches DuckDB | 1 WINDOW task deployments |
| Q89 | 21 | GPU execution; matches DuckDB | 2 WINDOW task deployments |

These are correctness runs on a small dataset, not performance measurements or full TPC-DS
coverage. The activation regression suite separately requires production WINDOW task-deployment
events on successful nonempty GPU paths.

Final GPU validation after rebasing: the focused evaluator/planner/adapter/SQL, CONCAT,
partition exchange, late-materialization, lifecycle, and publisher selection passes
**119,815 assertions in 169 test cases**. The original TPC-DS probe passes
**1,822 assertions in one generated test case**, covering all 12 query/mode combinations.
All six queries deploy GPU WINDOW tasks in both optimizer modes, with zero CPU fallback. Formatting, lint, and test-registration
hooks pass. Decimal precision-overflow tests retain the existing read-only CPU replay
policy and verify that both GPU evaluation and DuckDB report overflow.


Wire the operator through pipeline construction, memory estimation, telemetry,
and ordinary publication. Add differential SQL tests and confirm actual GPU
window execution. Test several input batches and buckets, shuffled arrival,
cross-device preparation and spill/restore where available. Record environment
blockers separately from passing tests.

Probe the optimized TPC-DS whole-frame examples to determine which still contain
windows. Do not require the full benchmark suite to enable a correctly bounded
isolated feature, or claim full-query support from operator tests.

### COUNT/MIN/MAX follow-up

COUNT(expr), COUNT(*), MIN and MAX use the same whole-partition capability gate,
complete-bucket assembly, and null-safe row expansion as SUM/AVG. Grouped requests
are submitted together; unkeyed requests use scalar reduction and broadcast.
COUNT is widened to BIGINT, including zero for all-null arguments. MIN/MAX retain
the argument type and return null for all-null groups. Floating extrema explicitly
match DuckDB: NaNs sort above all other values, MIN returns a non-NaN when one exists,
and MAX returns NaN if any non-null NaN is present. Neither evaluates an
unrequested SUM, so counting or finding extrema cannot raise a SUM overflow.

COUNT(*) prepares a shared hidden non-null constant, retaining input cardinality
when every original payload column has been pruned. Variable-width extrema use
string offsets as the adapter's output-size lower bound; character storage remains
subject to normal allocation and bounded OOM handling. DISTINCT, FILTER, ordered
frames and multiple window stages retain their existing planning restrictions.

Validation: the rebuilt library and test binary pass **243,794 assertions in 31 test
cases** for `[window_evaluator],[window_plan],[window_adapter],[window_input_plan],
[window_sql],[window_tpcds]`. Coverage includes null/NaN groups, empty input,
COUNT-only plans with pruned payload, strings/dates/booleans/decimals, fragmented
multi-bucket input, and all six original TPC-DS probes in both optimizer modes.
Formatting, lint and test-registration checks pass.

## Completion and exclusions

Complete when admitted whole-partition SUM/AVG/COUNT/MIN/MAX SQL windows execute correctly on GPU with
complete-bucket assembly and verified row association. Report the supported type
matrix and whole-bucket-fit restriction. Ranking, cumulative frames, permutations,
cross-batch summaries, and new repository ordering semantics are not prerequisites.
