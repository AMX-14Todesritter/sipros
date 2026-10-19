# Matching module

This module matches a theoretical fragment m/z against one scan's **already
preprocessed, sorted** experimental peaks. It does not generate peptides or
fragments, select candidates, accumulate intensity classes, or calculate MVH.

## Contract and production integration

- Input: scan layout, experimental m/z arrays and `(scanId, mz)` queries.
- Output: scan-local peak index, or `NoPeak = -1`.
- CUDA/CPU bucket lookup uses `abs(query - peak) < tolerance`, selecting the
  nearest peak and retaining the first peak in array order on equal distance.
- A class-zero peak is a real hit. Class interpretation belongs to the scorer.
- Inputs must be finite and valid; low-level device calls trust scan IDs and
  buffer sizes. The replay importer validates these before uploading.
- Tolerance is a single absolute Da value. This does **not** implement ppm.

`matching.cuh::nearestPeak` is the shared host/device bucket lookup.
`launchCuda` executes matching-only queries on resident device arrays.
The original scorer calls this lookup through its class-returning compatibility
adapter, preserving the candidate-per-thread production layout.

RT sphere tracing is in `MVH_RT/gpu_bridge/sphere_matching.cuh`. Both the fused
production scorer and the matching-only raygen branch call the same helper.
Use `prepare(..., GeometryKind::Spheres, ...)` to build the index, then
`stageMatching(...)` to upload launch parameters. Repeated
`launchStagedMatching()` calls do not upload query data or launch parameters.
`launchMatching(...)` is a convenience operation combining staging and launch;
do not time that convenience operation as pure kernel execution.
`reset()` releases RT resources. RT resources are global, support one prepared
dataset at a time, and are not a concurrent multi-index API. Buffers must outlive
all launches. A scoring launch or reset invalidates matching parameter staging.

The current RT float geometry retains its existing precision and tie behavior;
it is not guaranteed identical to the double bucket reference at boundaries.
The replay report counts **peak-index** differences, including equal-class
peaks. It never silently corrects RT results with a CUDA fallback.

The CPU replay runs the extracted bucket algorithm with OpenMP. It is not a
timing of the complete original CPU scorer. Existing semantic tests separately
check compatibility with the original `PeakList::findNear` behavior.

## Replay executable

Build the existing `mvh_cuda` CMake project with `MVH_CUDA_ENABLE_RT=ON` for all
three backends, or OFF for CPU and CUDA. Target: `matching_benchmark`.
`mvh_matching` is a separate library; the replay executable does not link the
Sipros peptide-generation or MVH scoring library. Its layout types currently
reuse `types.cuh` to keep production adapters small.

```sh
build/mvh_rt/gpu_integration/bin/matching_benchmark \
  --peaks output/replay/peaks.tsv \
  --queries output/replay/queries.tsv \
  --tolerance 0.01 --threads 8 --repeats 11 \
  --output output/replay/matching.tsv
```

Both input files have two whitespace-separated columns, with no header:
`scan_id mz`. Blank lines and lines starting with `#` are ignored. IDs are
zero-based. Peaks within each scan must be sorted; their array positions are
the returned local peak IDs. Scans with no peaks may be represented by gaps in
the IDs before the highest populated scan. Query order is preserved.

Without input paths, the executable uses a deterministic synthetic fixture of
two scans and 100,000 queries. `--contract` runs a smaller fixture plus exact
CPU boundary/tie checks and exhaustive-reference validation; it fails if GPU
backends disagree on the non-boundary fixture. Ordinary replay reports
mismatches rather than rejecting approximate RT output.

## Timing interpretation

- Three warmup passes, then the requested number of timed repetitions.
- GPU: CUDA events around only the CUDA kernel / staged OptiX launch, on the
  default stream. Results stay in VRAM. Event synchronization and download are
  outside the elapsed device interval. This includes device-side OptiX raygen,
  traversal, hit/miss handling and result stores, not just RT-core activity.
- CPU: wall time for the corresponding parallel lookup loop and result stores.
- Query upload and parameter staging are outside matching timing. The reported
  upload time includes allocations and first CUDA initialization, so it is
  setup wall time, not an isolated PCIe bandwidth measurement.
- CPU/CUDA index build time is host layout/bucket preparation. RT build time
  includes geometry preparation, OptiX setup and acceleration-structure build;
  the common host preparation remains an additional prerequisite for RT.
- Min/median/max, queries/second, mismatch count and checksum are reported.
  Thread count/repeat count are printed to stdout; preserve the invocation and
  input hashes alongside reports when running research experiments.

The replay currently uses **one query per GPU thread**, not one candidate per
thread. It is a lookup microbenchmark, not a claim about fused production-kernel
speedup. Candidate-group replay and capture of real production query streams
are subsequent experiments; arbitrary synthetic queries do not substitute for
those workloads. No full-search speedup is inferred from this executable.

## Capturing a real workload

Add `--capture-matching` to an ordinary `sipros_mvh_cuda` invocation. The first
nonempty scoring batch is saved in `<run output>/matching_capture/`. Subsequent
batches are not captured; the ordinary search continues. Such a run includes
diagnostic GPU generation and disk I/O and is **not a performance measurement**.
With the switch absent, no capture files or diagnostic kernels are created.

The capture reuses `CalculateSequenceIons` and the actual cache selection and
charge stride chosen by the production scorer. A sink filters queries by the
same inclusive scan bounds before lookup. All candidate groups are preserved,
including empty/invalid/skipped groups and speculative candidates later rejected
by top filtering. Queries are never sorted or deduplicated. Chunked count/store
passes bound temporary GPU query memory; each chunk checks count/store agreement.

Version 1 uses little-endian binary arrays:

| File | Layout |
| --- | --- |
| `scans.bin` | 48-byte `<QQiiiidd` records: peak offset, hub offset, peak count, lowest/highest integer bucket, skip, lower/upper query bounds |
| `peaks.f64`, `classes.i32`, `hub.i16` | Actual preprocessed peak, intensity-class and bucket arrays |
| `candidates.i32x4` | Four int32 fields: peptide ID, precursor ID, scan index, precursor charge; file row is the candidate group ID |
| `query_offsets.u64` | N+1 cumulative offsets for N candidate groups |
| `query_mz.f64` | Concatenated query m/z values in production order |
| `validity.u8` | Per group: 0 invalid theory, 1 valid theory, 2 skipped scan |
| `scan_sources.tsv` | Scan index to original input file and scan number |
| `COMPLETE.json` | Version, counts, Da tolerance and scope; written after binary streams close |

`capture_io.py::Capture` provides memory-mapped access, including
`iter_group_queries(group_id)` without expanding the whole dataset in RAM.
The TSV benchmark importer is unchanged. `matching_replay` directly consumes
this grouped binary format with one candidate per thread/raygen.

```sh
python3 -B mvh_cuda/matching/verify_capture.py \
  output/<run>/matching_capture --write
```

This validates all file sizes, every group's offsets/references/status, scan
array spans, sampled query bounds, and SHA-256 of each saved file. It creates
`verification.json`; query-bound sampling is not an exhaustive numerical check.
Preserve the surrounding run manifest containing input and binary hashes, the
exact command and configuration alongside the capture. No source snapshots are
needed. The full batch may contain hundreds of millions of queries; binary
storage avoids the much larger expanded `(scan_id, mz)` text representation.

## Real grouped replay (CUDA and RT)

```sh
build/mvh_rt/gpu_integration/bin/matching_replay \
  --capture output/<run>/matching_capture \
  --output output/<new-replay-directory> \
  --chunk-groups 1000000 --warmup 2 --repeats 7
```

All scan indices are prepared once. Complete candidate groups are streamed in
original order; a group is never split across chunks. Both implementations
consume the same resident query m/z and group-offset arrays, emit a peak index
for every query, and perform no MVH/class accumulation. CUDA has one thread per
candidate and OptiX one raygen per candidate, matching the production grouping.

For each chunk, launch parameters and input arrays are uploaded before warmup
and timing. Backend order alternates by chunk/repetition. CUDA events measure
only each launch. Per-repeat chunk times are summed, then min/median/max of
these totals are reported. This is a **chunked replay**, not seven uninterrupted
full-batch launches: chunk residency and warmup can affect cache behavior.

`samples.tsv` preserves all timed samples. `comparison.tsv` reports matching
throughput. `result.json` separately reports RT construction (OptiX setup,
geometry and BVH), common array upload and query upload. CUDA reuses captured
bucket arrays, so CUDA index construction was performed at capture time and
is excluded. Array upload includes allocations; it is not a bandwidth test.
Free VRAM is measured at chunk boundaries, not continuously sampled peak usage.

Every final query result is downloaded and compared outside timing.
`peak_index_mismatches` counts differing hits/indices; `class_mismatches` counts
different scorer-visible classes, treating miss and class zero as unscored.
`mismatch_examples.tsv` records up to 20 examples. RT discrepancies are reported
without fallback. Any precision discrepancy must accompany speedup claims.

`--replicas R` is an explicitly synthetic capacity stress test based on the real
capture (default 1). Peak, class and bucket arrays are copied into R distinct
device ranges; each scan has a distinct handle/BVH. Scan references are remapped.
For each original candidate its R copies are interleaved, and each copy emits
the unchanged original query sequence. Total scans, indexed peaks, candidates
and queries all scale by R. `--chunk-groups` still limits total groups per launch.
The original capture files are never expanded or modified. Repetition count
remains independent of replication.

This is not R new biological samples and does not increase peaks per scan or
the number of peaks traversed by a single query. Per-scan BVHs remain independent.
It tests global resident index footprint and total workload; interleaving,
chunk boundaries and cache locality must be considered when interpreting it.

## Optional diagnostics and profiling

All variants compile the **same** matching and replay sources:

| Target | GPU code | Purpose |
| --- | --- | --- |
| `matching_replay` | NoopObserver, ordinary CUDA and RT libraries | Performance measurement |
| `matching_replay_diagnostics` | CountingObserver, separate CUDA library and OptiX PTX | Algorithmic-work diagnostics |
| `matching_replay_nvtx` | Ordinary CUDA/RT libraries; host NVTX ranges only | Nsight kernel/timeline profiling without GPU counting perturbation |

The NVTX target is available when CUDA's NVTX3 headers are installed. Diagnostic
counting still builds without NVTX headers, with a configure-time warning.
Neither diagnostic nor NVTX reports emit a speedup field or `comparison.tsv`;
their metadata says `performance_valid: false`. Use the ordinary executable's
unprofiled measurements for speedup. Nsight itself can perturb execution too.

```sh
build/mvh_rt/gpu_integration/bin/matching_replay_diagnostics \
  --capture output/<run>/matching_capture --output output/<new-diagnostics> \
  --chunk-groups 1000000 --warmup 0 --repeats 1
```

`diagnostics.tsv` contains CUDA bucket visits, empty buckets, queries visiting
only empty buckets, coarse integer-range rejections, peak checks, max checks,
and a checked-peaks/query histogram. Histogram bins are 0, 1, 2, 3-4, 5-8,
9-16, 17-32, 33-64, and >=65. Zero peak checks includes early rejects.
RT reports queries, actual `optixTrace` calls, no-handle queries, hits and misses.
**BVH node/intersection counts are unavailable**, not zero; these counters do not
claim to measure RT-core internal work. Peak hits include intensity class zero.

CountingObserver accumulates privately per thread/candidate, then stores one
Statistics record per candidate. No per-peak global atomics are used. Only the
last pass is downloaded and accumulated: warmups/repeats do not multiply counts.
Query, hit, histogram and trace conservation are checked. This instrumentation
changes register pressure and memory traffic, so do not use its occupancy,
instruction count or bandwidth as the ordinary kernel's hardware characteristics.

For hardware profiling, invoke `matching_replay_nvtx` under the installed Nsight
tool with the same capture/chunk arguments. NVTX ranges mark `matching/replay`,
`matching/rt_build`, `matching/chunk`, `matching/upload`, `matching/warmup`,
`matching/measured`, `matching/cuda`, `matching/rt`, and validation/download work.
Query ranges surround enqueue calls; CUDA/Nsight correlation identifies their
device work. Uploads, setup, validation and counters remain separately identifiable.

NoopObserver is selected at compile time. On the local sm_120 Release build,
the ordinary CUDA flat and grouped kernel instruction text was compared against
the pre-observer build and found identical. This is evidence for this build,
not a guarantee across all compilers or architectures. Output identity and
hand-counted diagnostic fixtures are covered by CTest.

## Output-layout control experiment

`--output-mode queries` (default) writes one scan-local int32 peak ID per
fragment. `--output-mode candidate` writes one 40-byte CandidateSummary per
candidate: raw peak-hit count plus nine class bins. Only `classes+1` bins are
active; the rest are zero. Positive intensity class c increments bin c-1;
misses and class-zero hits increment bin `classes`, as in production scoring.
The scored match count is the sum of the positive-class bins. No MVH probability
or ranking is calculated. Inputs, tolerance and candidate grouping are unchanged.

CUDA instantiates the same candidate lookup body with two compile-time output
policies. RT builds separate query/aggregate PTX modules to avoid adding
accumulator register state to the query-only module. Aggregate mode also reads
classes and performs additions; this is **not** a store-only ablation.

Outside measurement, aggregate replay runs the ordinary CUDA query kernel and
reduces its returned peak IDs on the host, checking each candidate's full summary
against both timed implementations. Aggregate reports have
`candidate_summary_mismatches`; per-query mismatch counts are null because the
RT aggregate interface does not return per-query peak IDs. Input query/reference
buffers are still allocated for this validation, outside timing.

For matched hardware comparisons, use the NVTX-only executable with
`--max-chunks 1 --profile-capture 1 --warmup 0 --repeats 1` and the desired output
mode. Profiling capture calls cudaProfilerStart/Stop only around the measured
CUDA/RT pair; setup, uploads and validation-reference launches are outside capture.
`--profile-capture 1` requires NVTX and a one-chunk limit. This selects a
representative subset, not the complete dataset. Use the same profiler settings
and subset for both modes; do not use profiler durations as performance results.

### CUDA without bin coarse screening

`matching_replay_no_bin` is a separately linked exhaustive CUDA control. It uses
one thread per candidate and reads every preprocessed experimental peak for each
fragment query, without reading `hub`, integer bucket bounds or exiting early.
Strict double-precision `abs(delta) < tolerance`, nearest-distance selection and
first-peak tie handling are preserved. Both `queries` and `candidate` output
modes are supported. The RT backend remains unchanged. `result.json` records
`cuda_search_method` (`linear_all_peaks` versus `integer_bins`). This target does
not change production scoring or the ordinary replay executable. Index upload
and RT build remain outside the reported matching kernel times; unused hub data
is still uploaded by the common replay setup.

The full real-input control outputs are under
`output/benchmarks/matching_no_bin_20261019/` (7 repeats, 2 warmups per chunk,
1,000,000 candidate groups per chunk, per-query output).

### Controlled generation / matching / MVH fusion experiment

`matching_fusion_replay` compares three compile-time GPU paths and equivalent
OpenMP CPU paths, using the same captured candidate order and spectra:

| Mode | Work inside timed compute |
| --- | --- |
| `precomputed` | Read captured theoretical m/z, match and accumulate classes |
| `generate_match` | Generate theoretical ions, match and accumulate classes |
| `generate_match_mvh` | Generate ions, match, accumulate and compute MVH |

All modes write the same `FusionOutput` (class bins, raw peak-hit count,
predicted/scored match counts, status and score). Modes without MVH leave score
zero; the MVH mode follows the original candidate scorer's insufficient/invalid
rules. No mode performs top-list retention, FDR or peptide generation from FASTA.
CUDA retains exact integer-bin lookup; RT uses sphere tracing followed by a
strict double-precision tolerance check on each returned peak, with no bin
prefilter. The postcheck is specific to fusion replay; existing production and
ordinary replay sphere paths are unchanged. No theoretical-ion cache is used by generation replay paths.
Each GPU kernel/raygen handles one candidate; CUDA uses 128-thread blocks.
CUDA configuration and RT launch configuration reside in read-only constant
memory, not per-thread copies. CUDA and RT modes are compiled separately so an
unused generation/MVH path cannot inflate the matching-only kernel's resources.

CPU and GPU use the shared `ion_math.cuh` and `fusion.cuh` algorithms. Host
floating-point contraction is disabled except explicit `std::fma` matching
explicit CUDA FMA. CPU is an algorithm-controlled baseline, not the original
Sipros CPU executable or an end-to-end CPU speedup claim. `--threads` controls a
static OpenMP candidate loop; dynamic team sizing is disabled and actual thread
count is reported. `--backends cpu` runs without initializing CUDA/OptiX and is
tested with `CUDA_VISIBLE_DEVICES` empty; the executable still needs its linked
runtime libraries installed. Thread pinning follows OpenMP/environment settings.

The capture now optionally includes `fusion_scans.bin`, `fusion_config.bin`,
`fusion_peptides.bin`, `fusion_texts.bin`, `fusion_ln.f64`, and `fusion_abi.json`.
These extend capture v1 without changing its original files. Native struct
layouts are checked against the ABI tag; use matching little-endian builds.
Existing captures without the extension must be recaptured. New captures remain
opt-in using `sipros_mvh_cuda --capture-matching`.

Example inside the existing container (repository `/workspace/sipros`):

```sh
export LD_LIBRARY_PATH=/workspace/sipros/build/mvh_rt/optix_runtime:/usr/local/cuda/lib64
build/mvh_rt/gpu_integration/bin/matching_fusion_replay \
  --capture output/benchmarks/matching_fusion_20261019/real_capture/matching_capture \
  --output output/benchmarks/fusion_repeat_t8 \
  --backends cpu,cuda,rt --threads 8 --chunk-groups 1000000 \
  --repeats 5 --warmup 1
```

Use `--backends cpu --threads 1` (or 4/8/32) and a new output directory for
CPU-only scaling. `--max-chunks N` selects an identical candidate prefix for
all modes/backends; zero means full capture. Output directories must be new.

Before timing, an OpenMP audit checks every regenerated host m/z bit-for-bit
against the captured GPU query stream, including query counts and validity.
Outside timing, each GPU candidate is compared against the CPU result; bins,
counts and status must be exact, score error at most `1e-10 * max(1, |score|)`.
CPU reference summary checksums must also agree across all three modes. By
default, a mismatch aborts and never writes `complete=true`. The original scoring contract separately checks
MVH against a hand-calculated example and the original CUDA scorer, including
invalid charge, skipped scan, insufficient matches and empty spectra.

CPU uses wall-clock time around its OpenMP compute loop. GPU uses CUDA events
around only the kernel/OptiX launch after parameters are staged. Repeats rotate
backend order. Per-chunk times are summed for each repeat; reported values are
medians of these totals. Setup, generation audit, uploads, BVH builds, validation
and file I/O are excluded, with setup timings recorded separately. `samples.tsv`
retains every measured chunk/repeat, `comparison.tsv` the aggregate values, and
`result.json` records successful validation. No profiler is active in these runs.
The equal output contract differs from earlier 4-byte/query or 40-byte/candidate
replays, so compare fusion modes within this harness rather than mixing runtimes.

Marine/E.coli validation exposed a float-geometry boundary false positive:
query `608.83258996688005`, peak `608.84259`, distance
`0.010000033119922591` Da versus tolerance `0.01` Da. Fusion RT now rejects
such returned peaks using `validateRtPeak`. This does not add a bin lookup or
exhaustive search. A postcheck alone is not a general proof against float
false negatives or nearest-peak ties; every dataset is still compared against
CPU/CUDA. The initial failure, diagnosis, corrected Marine run and a 293T
reference rerun using the same corrected executable are retained under
`output/benchmarks/matching_fusion_marine_ecoli_x1_20261019/`.

### RT postcheck ablation

`matching_fusion_replay --rt-postcheck off` selects three separately compiled
sphere PTX modules without the double-precision hit postcheck. The default is
`--rt-postcheck on`. No runtime branch is added to the timed shaders. Class
accumulation, candidate output layout and MVH work remain identical.

With postcheck off, RT discrepancies are counted outside timing rather than
aborting on the first one; CPU/CUDA discrepancies still abort. The first 20
RT examples per mode are written to `result_mismatches.tsv`. `result.json`
records `rt_postcheck`, `rt_mismatches_by_mode`, `results_equivalent`, and
`rt_equivalent_performance_comparison_valid`. In this ablation, `complete=true`
means all measurement and comparison work finished, not that RT results were
equivalent. A false equivalence flag must accompany any reported RT timing.
The common generation audit remains mandatory for both settings.

Full Marine/E.coli X1 on/off results using the same captured input and binary
are under `output/benchmarks/matching_fusion_rt_postcheck_ablation_20261019/`.
