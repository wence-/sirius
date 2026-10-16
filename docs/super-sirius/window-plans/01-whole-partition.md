# Plan 1: whole-partition aggregate windows

## Scope and execution contract

Deliver a narrow end-to-end GPU implementation of whole-partition SUM and AVG,
then COUNT/MIN/MAX as small follow-ups. Start with partition-only specifications,
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
| Window source `get_next_task_input_data()` | Claim one whole completed bucket under coordinated task creation |

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
the builder, which validates and prepares a planning IR, then deliberately throws
`NotImplementedException` until slices 2–4 provide execution. Ordinary SQL still
uses the existing CPU fallback; no GPU window execution is enabled by this slice.

The initial **planning** matrix is:

| Item | Accepted |
| --- | --- |
| Functions | SUM and AVG with exactly one argument |
| Argument types | Signed TINYINT/SMALLINT/INTEGER/BIGINT, FLOAT/DOUBLE, DECIMAL through precision 38 |
| SUM result | Exact bound HUGEINT for signed integers, DOUBLE for floating arguments, DECIMAL(38, input scale) for decimals |
| AVG result | Bound DOUBLE; decimal argument scale is retained |
| Partition keys | BOOLEAN, signed integers through BIGINT, DATE, TIMESTAMP/TIMESTAMP_S/TIMESTAMP_MS/TIMESTAMP_NS, uncollated VARCHAR |
| Frames | Unordered UNBOUNDED PRECEDING to UNBOUNDED FOLLOWING, or to CURRENT ROW RANGE (all rows are peers without ordering) |
| Multiple expressions | Equivalent partition keys, including different key order; result order follows the logical window expression order |

Ordered frames (even explicitly whole), current-row ROWS frames, DISTINCT,
FILTER, EXCLUDE, argument ordering, IGNORE NULLS, non-SUM/AVG functions,
unsupported types, volatile expressions, and untranslatable key/argument
expressions are rejected during planning. COUNT/MIN/MAX remain follow-ups.
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

Reuse indexed hash routing and repository registration. Require a FULL barrier
and atomically coordinated whole-bucket task claims; skip empty buckets without
losing later work. A drained repository is not itself producer completion.
Implement the barrier communication and bucket-claim contract above using the
existing hint traversal and downstream scheduling, without polling or blocking
a GPU worker. Ensure wiring retains the correct upstream pipeline reference.
For no partition keys, route all input to one work unit. Use ordinary placement
and cross-device preparation rather than new affinity machinery.

Inspect reuse boundaries carefully: current `sirius_physical_concat` is configured
through a downstream join and can emit partial buckets. Do not construct a dummy
join or reuse partial concatenation for windows. Reuse the concatenation primitive
and partition-consumer mechanics, with a narrow window-owned complete-bucket path
or a small justified extraction of shared code. Avoid broad join-policy refactors.

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
ownership. Evaluation appends columns; all retained input columns stay aligned.

Acceptance: exact row multiplicity, duplicate payloads, null keys, all-null values,
empty input, admitted numeric boundaries, and several expressions on one group.

### 4. Activate and validate the SQL path

Wire the operator through pipeline construction, memory estimation, telemetry,
and ordinary publication. Add differential SQL tests and confirm actual GPU
window execution. Test several input batches and buckets, shuffled arrival,
cross-device preparation and spill/restore where available. Record environment
blockers separately from passing tests.

Probe the optimized TPC-DS whole-frame examples to determine which still contain
windows. Do not require the full benchmark suite to enable a correctly bounded
isolated feature, or claim full-query support from operator tests.

## Completion and exclusions

Complete when admitted whole-partition SQL windows execute correctly on GPU with
complete-bucket assembly and verified row association. Report the supported type
matrix and whole-bucket-fit restriction. Ranking, cumulative frames, permutations,
cross-batch summaries, and new repository ordering semantics are not prerequisites.
