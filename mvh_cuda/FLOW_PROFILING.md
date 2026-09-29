# Database-search flow profiling (no search optimization)

The known 689.98 s run used `rt-custom`, not the CUDA bucket matcher. This work
keeps that backend, the Marine inputs, 6,000,000 generated entries per batch and
host spectrum caching. The same counter implementation supports CUDA, triangles,
instanced triangles, spheres, and diagnostic modes (counts follow the primary
result, not the extra reference scoring).

## Exact pipeline

| File / function | Input → output | Unit |
|---|---|---|
| `src/database_search.cpp`: `MvhScanVector::searchDatabaseMvh`; `original/src/proteindatabase.cpp`: `getNextPeptide` | FASTA streaming, digestion and PTM enumeration → peptide objects, including repeated sequences | protein / peptide entry |
| `cuda/engine.cu`: `assignPeptides2Scans` | peptide masses, sorted precursor hypotheses, mass windows → candidate allocation | batch |
| `cuda/assignment.cuh`: `GetAllRangeFromMass`; `engine.cu`: `exclusiveOffsets` | mass-window ranges → counts and prefix offsets | peptide entry |
| `cuda/assignment.cuh`: `assignPeptides2Scans`; CUB `DeviceRadixSort::SortPairs` | ranges → `{peptideId, precursorId, scanId, charge}` records in stable scan order | association |
| `engine.cu`: `preprocessingMVH`; `cuda/preprocess.cuh`: `preprocessingMVH` | peptide text → neutral-loss-transformed device text and length; failures throw | peptide entry |
| `engine.cu`: `packScoringBatch`, `prepareScoringSpectra` | objects, spectra, current per-scan top list → flat scoring input, sequence IDs | batch / peptide / scan |
| `engine.cu`: `executeScoringBatch`; `assignment.cuh`: `setCandidateRanges` | sorted candidates → each scan's contiguous candidate slice | association / scan |
| `MVH_RT/gpu_bridge/{sphere_backend,triangle_backend,scene_resources}.cpp`: `prepare`, `build` | retained experimental peaks → persistent RT geometry / GAS / SBT | first batch / scan |
| `cuda/theoretical.cuh`: `markTheoreticalKeys`, `countTheoreticalIons`, `generateTheoreticalIons` | active `(peptide-entry ID, charge)` keys → optional cache | key, reused within batch |
| `cuda/scoring.cuh`: `scoreCandidate<Counter>`; CUDA `ScoreSequenceVsSpectrum`, RT `__raygen__camera` | association → speculative `Result` | association |
| `scoring.cuh`: `CalculateSequenceIons` | transformed text, precursor charge → b/y m/z values streamed to a sink | generation invocation / ion |
| `scoring.cuh`: `IonCounter::add/findNear`; RT `device.cu`: `RtCounter::add`; `custom_device.cu`: `SphereCounter::add/tracePeak` | in-range m/z → backend-selected class, histogram | fragment query |
| `scoring.cuh`: `scoreCandidate`, `lnCombin` | histogram + experimental class counts → MVH log-hypergeometric score | eligible association |
| `scoring.cuh`: kernel `scorePeptidesMVH`, `saveScoreSort` | speculative results + previous top-50 → ordered merge/accept/evict decisions | scan, association in original order |
| `scoring.cuh`: `KeepScoringEvent`, `gatherScoringEvents`; CUB `DeviceSelect::If` | full results → stable accepted/merged events | association/event |
| `engine.cu`: `restoreScoringResults`; `original/src/ms2scan.cpp`: `mergePeptide`, `saveScore` | downloaded events → host top candidates/protein merges | event / scan |
| `src/mvh_scan_vector.cpp`: `postMvh`; `app/runner.cpp`: `writePsms` | final scan top lists → PSM TSV | final retained candidate |

No XCorr or WDP calculation is called on this search path. The export writes the
MVH score slot. Explicit `--verify-cuda` additionally replays CPU scoring and is
not enabled in full performance captures.

## Filters and meanings

* Precursor assignment is the mass-window filter. It retains precursor hypotheses,
  including different charges and the original duplicate-window behavior.
* `scoreCandidate` rejects `scan.skip` before entering its fragment-processing path.
  An entered association may still issue zero queries. `total_associations_with_queries`
  distinguishes actual lookup work from merely entering the path.
* A theoretical m/z outside `[scan.lower, scan.upper]` is not queried. It is still
  counted as a generated/offered ion, and separately as outside the scan range.
* Invalid peptide text, missing residue/PTM masses, too-short sequence, or charge
  below 1 makes `CalculateSequenceIons` invalid. This is an error path, not an
  ordinary silent filter. Retention can mask it only if the original CPU would
  have merged that candidate. `total_invalid_ion_sequences` exposes these cases.
* MVH executes only when `matched > 0 && matched >= cfg.minMatched`, where
  `matched` is the **backend's positive-class** evidence, not the observer's
  geometric hit count.
* Duplicate merging occurs AFTER speculative GPU scoring. A merged candidate
  may already have executed all fragment lookups and MVH arithmetic. Therefore
  existing `calls/success/inrange/matched` logs undercount physical work; they
  are post-merge statistics. The new counters are recorded before that overwrite.
* Top retention accepts an eligible unmerged candidate when the current top list
  has fewer than 50 entries or its score is strictly larger than the last score.
  Ties at a full-list cutoff do not replace the last entry. `saveScoreSort` retains
  the existing ordering, and later candidates/batches can evict earlier insertions.

A geometric fragment hit here means **at least one retained, preprocessed
experimental peak** satisfies `abs(experimental_mz - theoretical_mz) < tolerance`.
It counts at most one hit per query, including class 0. It does not count raw peaks
removed during scan preprocessing. An observational binary search on the sorted
peak array tests this independently of the selected matcher and never feeds the
score. RT float geometry and class-0 policy can differ; consequently
`total_backend_scored_fragment_hits` is reported separately. The observer adds
work and its overhead must be measured, not assumed negligible.

## Counters

All global counters and aggregation arithmetic are 64-bit. In the table, `score`
means `scoring.cuh::scoreCandidate`, and `host` means the aggregation in
`engine.cu::executeScoringBatch`.

| Counter | Increment / precise meaning |
|---|---|
| `total_peptide_entries` | host `assignPeptides2Scans`: batch `peptides.size()`; no sequence deduplication |
| `total_precursor_associations` | host assignment: return of `exclusiveOffsets(counts, offsets)` |
| `total_associations_entering_fragment_stage` | score: one for every non-skipped association |
| `total_associations_rejected_before_fragment_stage` | score: `scan.skip` early return |
| `total_associations_with_queries` | score: `ions.predicted > 0` |
| `total_theoretical_fragment_ions_generated` | physical emissions from ALL calls to `CalculateSequenceIons`, including cache sizing and cache fill passes; not unique ions |
| `total_fragment_queries` | score: number of in-scan-range `Counter::add` calls |
| `total_fragment_hits` | observer: query has any experimental peak strictly within tolerance |
| `total_fragment_misses` | queries minus geometric hits |
| `total_backend_scored_fragment_hits` | score: original `ions.matched`, before merge overwrites |
| `total_associations_with_at_least_one_fragment_hit` | score: geometric hit count is nonzero |
| `total_associations_with_zero_fragment_hits` | entered minus associations with a geometric hit (includes zero-query cases) |
| `total_associations_entering_mvh_scoring` | score: original matched-count gate passed |
| `total_mvh_scores_computed` | score: after existing MVH arithmetic, same count as gate on a successful run |
| `total_candidates_retained_after_mvh` | host: downloaded `ResultAccepted` insertion events, including insertions later evicted |
| `total_final_psm_candidates` | runner: rows actually written by `writePsms` |
| `total_merged_candidate_events` | host: downloaded `ResultMerged` events |
| `total_ions_offered_to_associations` | score: all sink calls, including cached replay and out-of-range m/z |
| `total_ions_outside_scan_range` | offered minus queried |
| `total_invalid_ion_sequences` | score: invalid direct/cache generation result |
| `total_direct_generation_calls`, `total_direct_generated_ions` | uncached score path: call and actual emissions |
| `total_cache_count_generation_calls`, `total_cache_count_ions` | `theoretical.cuh::countCacheWork`: active keys and offsets representing the count pass, even if materialization is rejected |
| `total_cache_store_generation_calls`, `total_cache_stored_ions` | cache observer: valid active keys and offsets, only if fill pass executed |
| `total_theoretical_ion_generation_calls` | direct calls + cache-count calls + cache-store calls |

The global device buffer has 4096 striped shards (512 KiB with the current 16
metrics). Each scoring thread accumulates ion counts locally and adds totals once
per association; there are no per-fragment atomics or prints. It is allocated only
after both cache memory-admission decisions. The buffer is copied and aggregated
once per batch; production `Result` layout, candidate arrays and matching logic
are unchanged. Tests/diagnostic references pass a null observer when not primary.

## Theoretical ion reuse

`CalculateSequenceIons` is the actual generator. Without cache it runs once per
non-skipped association, regenerating the same entry's ions across scans. With
cache, it runs once for counting and once for storing each active valid
`(peptide-entry ID, precursor charge)` key. Charge values outside the cache stride
use direct generation. Cache scope is one batch, not a sequence-deduplicated global
library. Identical sequences in different peptide entries have distinct keys.

The stride is `min(maxCharge + 1, 9)`. Cache admission requires nonzero candidate
count, `keys < 32,000,000`, `keys * 20 < freeBytes/2`, then sufficient memory for
the ion array. A rejected fill can still incur a sizing-generation pass. These
conditions and memory checks are unchanged. Diagnostic allocations occur later.
Always report generation-call ratios from counters; do not assume one call per
peptide or one call per association when a cache is active.

## Timing contract

`flow_timings.tsv` reports accumulated host NVTX-scope wall time, call count,
average, and percentage of search time. Nested host intervals are inclusive and
must not be summed. `gpu/scoring_fused` uses existing CUDA events and their existing
completion wait. No additional synchronization is added inside the algorithm.

Direct generation, lookup and MVH arithmetic are interleaved inside one scoring
kernel/OptiX launch. They do not have independent CUDA event boundaries. Their
separate seconds are explicitly `NA`, not zero and not invented percentages.
Splitting kernels or replaying modified algorithms would violate this task's
constraints. Per-thread clock sums also cannot be presented as wall time. The
cache-generation kernels, if present, are separately visible in NSYS.

Use a COUNTERS-OFF reference run under NSYS for kernel, memcpy/memset and CUDA API
wait timing. A wait overlaps GPU work; never add it to kernel time. Kernel/memcpy
summaries include the application's preprocessing unless filtered to the search
interval. Host stages cover database generation, assignment, preprocessing,
packing, RT preparation, retention, compaction, download and restoration. The
report retains scope/kind labels to prevent false additive breakdowns.

## Builds and runs

Build on `rt-profiling` using CUDA 12.8 / OptiX 9. Keep the normal binary separate:

```bash
cmake -S mvh_cuda -B build/mvh_rt/flow_profile -DCMAKE_BUILD_TYPE=Release \
  -DMVH_CUDA_ENABLE_RT=ON -DMVH_ENABLE_PROFILING=ON \
  -DMVH_ENABLE_FLOW_COUNTERS=ON -DOPTIX_ROOT=/workspace/sipros/build/mvh_rt/optix_sdk
cmake --build build/mvh_rt/flow_profile -j4
```

Use the existing search CLI on that binary. It writes `flow_counters.tsv` and
`flow_timings.tsv` beside `run_summary.tsv` and the PSM output. Build
`build/mvh_rt/flow_reference` with `MVH_ENABLE_FLOW_COUNTERS=OFF` for timing and
output equivalence checks. This removes device counters/observer code at compile
time. Keep application NVTX ON for NSYS, or turn it OFF for a clean benchmark.

Retain input/binary/PTX hashes, parameters, completion status, and full output
hashes with each measurement. Never compare an NCU replay duration to normal wall
time. Do not interpret differences between runs as an algorithmic speedup.
