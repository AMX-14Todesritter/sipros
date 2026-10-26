# Evaluation plan — agreed scope

## Baseline and output policy

Audit source in `/home/ams098z/projects/sipros`, branch `profiling_0.0`.
The chat's initial worktree at 12e9d25 lacks GPU implementations; it is not the
experimental baseline. Capture the actual commit, dirty diff and binary hash for
each experiment. Do not silently switch branches or reset others' changes.

All generated results and reports: `output/evaluation/`. Evaluation source:
`evaluation/`. No scientific matching/scoring algorithm changes are authorized.
Primary fused kernel includes ion generation or cache reads, matching, class
accumulation and MVH. Cache generation is timed separately when enabled. Existing
RT CUDA-event intervals also include launch-parameter upload/host submission gaps;
label them fused scoring event intervals until profiler timelines isolate kernels.

## Implementation matrix

| Experiment | CPU | cuda | rt-instanced | rt-custom |
|---|---|---|---|---|
| Correctness/PSM | deterministic reference coverage | yes | yes | yes |
| Batch sensitivity | no | yes | yes | yes |
| Real-world X1/X2/X3/Human/Marine | yes, feasibility assessed | yes | yes | yes |
| Real-world Soil | excluded | yes | yes | yes |
| Soil DB scaling | excluded | yes | yes | yes |
| Scan scaling | excluded | yes | yes | yes |
| GPU profiling | no | representative cases | representative cases | representative cases |
| MVH application timing | representative cases, no Soil | yes | yes | yes |

## Execution order and gates

1. Inventory datasets (paths/counts/hashes/precursors), implementations, hardware,
   build flags, timing boundaries, reusable scripts and missing measurements.
2. Build isolated Release binaries under output/evaluation/build; run relevant
   semantic contracts and a small nonempty CPU/CUDA/RT PSM comparison. Record
   all failures, including pre-existing provenance errors. No matching-only tests.
3. Real X1 deterministic scan subset pilot: validate repeatability, batch
   invariance and diagnostic-vs-normal results; five measured runs with warmup.
4. Recreate missing X2/X3/Human FT2 from retained RAW; verify scan counts and
   provenance. Search for existing merged files; otherwise create nested,
   source-stratified inputs and original-ID mappings with exact record retention.
5. Main real-world runs. Use resource pilots before Soil; record OOM without
   silently changing settings. Batch test uses Marine FASTA + Marine FT2, not the
   historical Marine FASTA + E.coli cross-workload.
6. Batch values 2M,4M,8M,10M,12M **generated peptide entries**. Identical total
   workload and fixed generation/cache/restoration policies. Record candidate
   counts and cache decisions, not only requested batch size.
7. Soil nested DB fractions 1,5,10,25,50,100%; inspect decoy families before
   deterministic sampling. Preserve target/decoy families where possible; record
   actual sizes, seeds and hashes. Same full Soil FT2, fixed batch policy.
8. E.coli scan points 5k,10k,20k,40k,60k,80k,100k,120k,140297; fixed E.coli
   FASTA, seeded nested source-stratified sampling. Remap scan numbers and retain
   source/original-ID map; never merge peaks. Inspect source composition.
9. Representative NSYS/NCU captures; separate construction, transfers, theory,
   scoring and retention. Exact commands and exposed counter names required.
10. Aggregate CSV/JSON, median/min/max/stdev and publication figures. Report
    verified observations separately from hypotheses and missing evidence.

## Correctness and measurement requirements

Do not infer biological accuracy from implementation consistency. Normalize
spectrum/candidate identities; compare scores, top-1, top-k, proteins and decoys.
Keep boundary/tie/duplicate/float mismatches visible. ACC-1 peak-ID diagnostics
must execute within the fused path; standalone matching benchmarks are deferred.
No FDR/Jaccard claims without a validated common downstream filter.

Timing runs: verification/flow counters off, warmup, at least five feasible
repeats, rotate backend order, exclusive suite lock; record background GPU load,
thermal/power limits and sampled memory limitations. Diagnostic counters run
separately and must be linked by workload hashes and output consistency.

All runs have unique output directories and manifests with status planned,
ready, running, completed, failed, blocked or invalidated. Never overwrite a
validated run. Preserve failed logs. Do not launch uncontrolled overlapping jobs.

## Resource risks

12 GiB laptop GPU and 24 GiB host RAM constrain batches and full Soil. Adaptive
ion caching can switch with batch size/backend memory consumption: record it,
do not assume kernel workloads are equivalent merely from candidate totals.
Full Soil runtime and output disk growth require pilot estimates. Desktop GPU
load is visible in WSL but its process list may be incomplete. CPU and GPU FT2
readers must be audited for identical primary precursor conventions.

## Audit and pilot observations

Initial source commit: `cfd214c13e2abf7d194bd128966eb0c1590e206d`.
The existing CPU-baseline provenance JSON has unresolved merge markers, retained
as a documented failure. Actual baseline source hashes and run diffs are captured.

X1 first256: CPU/CUDA/sphere retained PSMs agree; instanced differs on four
spectra. Instanced full candidate diagnostic: six score changes among candidates
scored by both and ten CUDA-only-scored candidates across the two batches.
Sphere candidate diagnostic found no changes on this subset, and strict CPU
verification passed. These do not prove complete peak-ID equivalence on all data.
Same-backend normal/diagnostic and 2-batch/16-batch retained PSMs agree.

Soil subsets use a seeded SHA256 threshold on unique target IDs and retain the
adjacent decoy record; counts are approximate requested fractions. Full audit
found 3,408,250 unique target/decoy families. Scan subsets use deterministic
per-source ranks and monotone proportional quotas, retaining original records
except remapping S-line scan IDs, with per-record original/remapped hashes.

## User amendment: fixed scan policy v2

This amendment supersedes all earlier dataset-to-spectrum mappings. Fixed
workloads pair Pan X1 with Ecoli/Human/Marine/Soil FASTA, never the native Human,
Marine or Soil FT2. Batch and Soil database scaling also hold Pan X1 fixed.
Pan X2/X3 participate only in the existing nested source-stratified scan-scaling
series (up to 140,297 records). CPU excludes Soil and both scaling experiments.
Earlier native-spectrum runs are preserved but not included in current findings.
