# Plan 4: scale beyond complete-bucket evaluation

Status: follow-up design gates, not implementation-ready algorithms. Preserve the
validated SQL planner and row-association contracts from Plans 1–3. Choose each
extension from measured limitations; do not make this plan a prerequisite for
shipping those milestones.

## A. Reduce work-unit size while keeping SQL partitions complete

Measure bucket sizes, largest SQL partition, assembly/sort working sets, skew,
spill traffic, and task placement. More hash buckets can separate distinct groups
but cannot split an oversized SQL partition and do not eliminate collisions.

Investigate recursive hash redistribution, explicit complete-group extraction,
or partition-aware sorted ranges. Any range approach must avoid splitting a SQL
partition. Group extraction itself must not require concatenating the oversized
parent bucket. Detect non-progressing subdivision and retain an explicit capacity
limit rather than retrying the same impossible task.

Acceptance: input with many individually small SQL partitions and an oversized
original bucket executes without materializing that entire bucket. Demonstrate
bounded assembly and exactly-once rows. Do not claim that one oversized SQL
partition is handled by this extension.

## B. Evaluate one SQL partition across batches

Choose algorithms by function family rather than introducing a universal halo
mechanism:

* Whole-partition aggregates: reduce partial states, combine by actual SQL key,
  then expand to retained original rows. AVG carries sum and non-null count.
  This path may remain unordered and need no positioned repository contract.
* Cumulative windows: ordered fragments plus local prefixes and combined incoming
  state, resetting at SQL partition boundaries.
* Ranking: ordered fragments plus partition counts and peer continuation. A row
  count offset alone is insufficient for a peer spanning several fragments.
* Bounded frames/navigation: choose neighbor access, summaries, or tiled evaluation
  according to the function and frame; account for borrowed input and ownership.

This is where the adapted cuCascade position APIs may become a prerequisite for
order-sensitive families. Positions establish sequence, not successful completion,
SQL partition identity, peer equality, or frame readiness. Reuse existing barriers
first and introduce more incremental progress contracts only when justified.
The baseline communication is consumer hints checking producer-pipeline completion
with downstream rescheduling, as specified in Plan 1. Per-bucket or per-fragment
readiness before pipeline completion would be a new contract and needs separate
design; do not infer it from a position being present or a repository being empty.

Keep boundary discovery and aggregate evaluation separate. For every algorithm,
define retained payload lifetime, admission, spill/restore, cancellation, retry,
cross-device preparation, and exactly-once output before activation. No new sidecar
payload owner or ordered directory is presumed necessary.

Acceptance: force one SQL partition and one peer group across several batches,
delay earlier fragments, and compare actual GPU SQL output. Measure summary/halo
bytes and working sets separately from input/output size. Local fit is not evidence
of cross-batch support.

## C. Reduce repeated movement across specifications

After measuring chained full-row sorts, consider sorting keys/arguments with a
row index and scattering results into a common layout. A table-local row index
suffices only while the common table and its ownership remain intact. Repartitioned
or independently evaluated work needs identity valid across those movements.

Specify identity, retained-source lifetime, permutation memory, and result assembly
before replacing chained execution. Preserve full-tuple equivalence and duplicate
row multiplicity. Independent evaluation/joins and parallel specification execution
are optional optimizations, not the default architecture.

This optimization is independent of adding new SQL functions and can be scheduled
earlier if profiling shows wide-row movement dominates. It must not weaken the
baseline invariant that every selected column refers to the same original row.
