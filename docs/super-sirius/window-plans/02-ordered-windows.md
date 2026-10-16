# Plan 2: local ordering and multiple specifications

Depends on Plan 1. Retain complete-bucket assembly and ordinary unordered
repositories. Ordered rows live inside one evaluator table; positioned batches
are not required for this milestone either.

Inherit [Plan 1's barrier and claim contract](01-whole-partition.md#barrier-communication-and-bucket-claims).
Each new exchange between incompatible specifications uses producer-pipeline
completion and consumer hints before whole-bucket acquisition. Fused local sort
and evaluation on an already complete table need no additional barrier.

## Slices

### 1. Local sort and boundary discovery

Within each completed bucket, sort the entire evolving table by SQL partition
keys followed by the window's order keys. With no partition keys, sort the entire
gathered input. Preserve direction, null ordering, and admitted comparison/type
semantics. Reject unsupported collations or types rather than substituting a
different comparator.

Discover partition boundaries using partition keys only. Discover peer boundaries
using the function's order keys within each SQL partition. Keep these interfaces
separate from evaluation. No rank or cumulative state crosses evaluator tasks.

Reuse local sort primitives; do not route through a range-splitting global sort
that may split SQL partitions. Retain hidden keys until boundary/evaluation users
are finished rather than applying final-sort projections too early.

Acceptance: nullable composite partitions, multiple groups per bucket, descending
and mixed order directions, peers, and keys supplied by expressions.

### 2. Ranking and cumulative ROWS

Add ROW_NUMBER, RANK, and DENSE_RANK in focused slices. RANK is one plus the
partition-relative row offset of the peer group's first row; DENSE_RANK counts
peer groups. Preserve DuckDB result types. Admit frame/modifier combinations only
after checking the bound function's semantics; ranking is not a frame aggregate.

Add SUM/MAX for `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, followed by
COUNT/MIN/AVG where justified. Reset scans at SQL partition boundaries. Maintain
validity for all-null prefixes and correct widening/overflow behavior. An ordered
default RANGE frame is not interchangeable with cumulative ROWS when peers exist.

Acceptance: ranks with ties, all-null order keys, negative MAX inputs, null prefixes,
partition resets, and exact ROWS comparisons using unique order keys. Confirm
Q51-shaped isolated windows on GPU before attributing full-query coverage.

### 3. Chain incompatible specifications

Group identical partition/order specifications for shared organization and
evaluation. Different frame bounds may share sorting while using different
evaluators. Chain incompatible specifications, carrying prior results and every
still-needed input column through subsequent exchanges and sorts.

For `SUM(a) OVER (ORDER BY b ROWS UNBOUNDED PRECEDING)` and the analogous window
on `c`, sort by `b`, append the first result, then sort the entire extended table
by `c` and append the second. Restore requested output-column positions with a
final projection; do not restore an arbitrary original row order.

For different partition keys, rehash the entire extended rows. No synthetic row
identity or joining independently computed window outputs is needed. Initial
planning may conservatively repeat an exchange/sort; reuse requires established
compatible organization, not merely a shared column name.

Acceptance: distinct orderings with different per-row results, duplicate `a`
values, different partition specifications, hidden expression keys, and earlier
results retained across multiple stages. Compare complete tuples, not independent
column multisets, to catch misassociation. Test final explicit ORDER BY separately.

### 4. Small organization optimizations, after correctness

Consider reuse for compatible order prefixes. Sorting by `(b,c)` can serve a
window ordered by `b`, but that window's peers are still determined by `b` only.
Do not add a tie-breaker as an undocumented SQL requirement. Identical grouping
and sorting reuse can ship before prefix reuse.

## Completion

Supported ranking and cumulative ROWS functions execute over complete SQL
partitions. Multiple specifications preserve row count and association across
all stages. Window ORDER BY does not advertise a final SQL output-order guarantee.
Repeated payload movement is accepted and measured, not hidden behind a premature
permutation abstraction.
