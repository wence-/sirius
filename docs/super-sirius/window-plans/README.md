# SQL window implementation plans

Status: Plan 1 slice 1 has a bound-planning implementation with execution still
gated on CPU fallback. Later slices remain proposed. No GPU window validation
or performance result is implied. These plans supersede the implementation route
in the older `wence/fea/window-functions` branch. Do not import its ordering
directory, placement, bounded-merge, or continuation machinery as prerequisites.

## Decisions

Start with complete hash buckets materialized as single tables. With `PARTITION
BY`, hash on the SQL partition keys only. Equal keys must reach the same bucket,
including null keys under SQL partition semantics. Hash collisions mean a bucket
can contain several SQL partitions; actual key equality still determines groups.
Without `PARTITION BY`, gather the entire input into one work unit.

Whole-partition order-insensitive aggregates use grouped reduction and expansion
back to input rows. They need no sort. Order-sensitive windows sort each complete
bucket by partition keys followed by window order keys, then evaluate within SQL
partition boundaries. Hash bucket indices do not define output order.

The initial capacity restriction is **one complete hash bucket fits**, or the
whole input fits without partition keys. It is stronger than saying each SQL
partition fits. Fit includes retained input, concatenation, sort temporaries,
window output, and scratch, and respects native table-size limits.

## No dependency on new cuCascade ordering APIs

The first milestones use ordinary indexed repositories, existing FULL barriers,
and normal task ownership. No batch positions, ordered partition declarations,
ordered traversal, or new cuCascade release is needed for their correctness.
They should work against the existing dependency as well as after the parallel
repository adaptation lands, using whichever ordinary retrieval API is current.

The implicit physical contracts must still be documented and tested:

1. The exchange routes every row of a SQL partition to one bucket.
2. On successful execution, producer completion follows sink deposition; errors
   and cancellation remain governed by the query lifecycle.
3. A coordinated claim obtains the entire completed bucket exactly once.
4. Concatenation keeps columns aligned, although batch arrival order is arbitrary.
5. All rows of a SQL partition are available to one evaluator invocation.

Barrier communication uses the existing consumer hint protocol. The consumer's
`get_next_task_hint()` checks its producer pipeline's `is_pipeline_finished()`.
An unfinished FULL dependency returns `WAITING_FOR_INPUT_DATA` naming the producer;
the task creator follows that dependency upstream. Pipeline completion schedules
downstream consumers to recheck readiness. No GPU worker waits at the barrier.

This is a pipeline-wide completion barrier, not per-bucket completion. After it
opens, the window source coordinates whole-bucket claims. A readiness hint does
not reserve input, and a `nullopt` hint is not a completion certificate. Repository
emptiness and the pipeline finished flag alone do not establish successful query
execution. Keep failure/cancellation handling in the existing lifecycle.

Completeness belongs to the producer pipeline, storage to the repository, and
readiness/claims to the consumer. No new repository completion state, producer
counter, typed completed-reader handle, or barrier protocol is required. See
[Plan 1's barrier contract](01-whole-partition.md#barrier-communication-and-bucket-claims)
for implementation seams and acceptance tests.

Inside a table, row offsets provide sufficient order for local evaluation. No
ordering of tasks or output batches is required. A final SQL `ORDER BY` remains
the responsibility of the existing sort path; window order is not a substitute.
Do not claim this work repairs unrelated final-sort ordering limitations.

## Row association across specifications

Every stage carries the entire evolving row, including earlier window outputs:

```text
(a, b, c)
  -> sort all columns by b
  -> append cumulative_sum_b
  -> sort all columns, including cumulative_sum_b, by c
  -> append cumulative_sum_c
  -> project requested columns
```

Repartitioning for a different `PARTITION BY` also moves every retained column.
Stages preserve row count and association, not incoming physical row order.
No inverse permutation, synthetic row identity, or independent result join is
needed. Repeated movement of wide rows is an accepted baseline cost.

## Sequence

| Plan | Deliverable | Dependency |
| --- | --- | --- |
| [1 Whole-partition aggregates](01-whole-partition.md) | SQL planning, complete-bucket assembly, SUM/AVG, grouped expansion | Existing exchange and completion infrastructure |
| [2 Ordered windows and chained specifications](02-ordered-windows.md) | Local sorting, ranking, cumulative ROWS, different specifications | Plan 1 |
| [3 Local frame extensions](03-local-frames.md) | Bounded ROWS, navigation, selected RANGE semantics | Plan 2 |
| [4 Scaling and movement optimizations](04-scaling.md) | Smaller complete work units, then cross-batch algorithms | Validated earlier contracts |

Each plan is split into reviewable implementation slices. Initial slices may
restrict the type matrix or reject multiple incompatible specifications, but must
reject unsupported plans before execution rather than partially implementing them.
Full chained-specification support is an explicit Plan 2 deliverable.

## Shared implementation and validation rules

Use a dedicated window planner/operator, not projection-side interpretation of
window expressions. Separate input organization, partition/peer/frame discovery,
and function evaluation. Add only metadata required by the active function family.
Do not require one generic frame representation for all evaluators.

Inspect bound DuckDB expressions, including endpoints, FILTER, DISTINCT, EXCLUDE,
null treatment, argument ordering, offsets, result types, and expression keys.
A similarly named cuDF function is not sufficient evidence of SQL compatibility.
Unsupported combinations follow established planning fallback. Do not assume an
oversized runtime task can transparently restart on CPU after execution begins.

Empty input yields no window rows, unlike an ungrouped aggregate's singleton
result. Null partition keys group together; all-null SUM/AVG results remain null;
COUNT semantics distinguish rows from non-null values. Preserve duplicate rows.
Define supported widening, decimal, overflow, and floating comparison behavior.

Compare full row tuples against DuckDB, as multisets unless final ordering is
specified. Use unique order keys for exact ROWS comparisons when ties would
permit different results. Test peer-sensitive functions separately with ties.
Assert that supported SQL actually creates and executes GPU window operators;
transparent fallback must not masquerade as coverage.

Read current Super Sirius architecture and module context before implementing.
Use `pixi run` for build/test commands. Recheck code seams after the concurrent
repository migration. Preserve unrelated work and keep each PR within the
repository's branching and reviewability requirements.

## TPC-DS targets

The reviewed checked-in DuckDB SQL includes whole-partition SUM in Q12/Q20/Q98,
AVG in Q47/Q53/Q57/Q63/Q89, ranking in several queries including Q47/Q57, and
cumulative ROWS SUM/MAX in Q51. Q47/Q57 also need ranking, so whole-frame AVG
alone does not complete them. Capture actual optimized plans before attributing
coverage: an optimizer rewrite can remove windows, and unrelated operators can
still block a full query. Benchmark coverage is separate from isolated semantics.
