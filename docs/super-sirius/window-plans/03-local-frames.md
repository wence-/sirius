# Plan 3: broader frames within complete partitions

Depends on Plan 2. Expand semantics without changing complete-bucket ownership.
There are no halos or inter-batch continuation states in this plan.

## Slices

1. Add constant bounded ROWS frame discovery. Compute partition-clipped intervals
   independently of aggregation. Specify inclusive SQL endpoints and the internal
   interval convention, empty frames, mixed preceding/following bounds, and wide
   arithmetic before native-index conversion. Admit functions/types individually.
2. Evaluate SUM/COUNT/MIN/MAX/AVG for those bounds with verified primitives or a
   simple correct multipass implementation. Keep boundary discovery independently
   testable. SQL empty-frame validity and COUNT zero behavior must be explicit;
   do not infer them solely from a rolling API's minimum-period setting.
3. Add constant-offset LEAD/LAG with typed defaults and partition-edge behavior.
   Begin with the supported ordinary null policy; reject dynamic offsets and
   IGNORE NULLS until separately implemented and tested.
4. Add peer-aware cumulative RANGE, including the common unbounded-preceding to
   current-row shape. All peers share the appropriate endpoint. Then separately
   assess bounded value RANGE: comparator direction, nulls, arithmetic overflow,
   dates/intervals, and bound-key restrictions require their own capability gate.

EXCLUDE, DISTINCT, argument ordering, additional navigation/distribution functions,
and dynamic frame expressions are separate features, not incidental side effects
of having a general-looking evaluator API. Order these slices by actual workload
need; they are not all required to call Plan 2 complete.

## Acceptance

Compare each admitted shape with DuckDB using singleton and empty input, multiple
SQL partitions, clipped and empty frames, all-null values, and both ordering
directions. Test peers explicitly for RANGE and use deterministic tie ordering
where ROWS comparisons need it. Preserve row association when these windows are
chained with earlier whole-partition or ranking stages.

Complete each capability only after SQL tests establish GPU execution and the
type/null/error matrix. Do not postpone local correctness until a cross-batch
algorithm exists. Conversely, do not claim partitions larger than one admitted
work unit are supported by this phase.
