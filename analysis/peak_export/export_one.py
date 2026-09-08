#!/usr/bin/env python3
"""Build an isolated Regular MVH extractor and export one scan/candidate pair."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"Source version mismatch: expected one hook in {path}: {old!r}")
    path.write_text(text.replace(old, new, 1))


def prepare(work):
    source = work / "source"
    source.mkdir(parents=True, exist_ok=False)
    for name in ("src", "include", "MSToolkit"):
        shutil.copytree(ROOT / name, source / name,
                        ignore=shutil.ignore_patterns(".git", "lib", "*.o", "*.a"))
    shutil.copy2(ROOT / "openmp/main.cpp", source / "main.cpp")
    shutil.copy2(Path(__file__).with_name("export_hook.h"), source / "include/export_hook.h")
    scan = source / "src/ms2scan.cpp"
    mvh = source / "src/MVH.cpp"
    for path in (scan, mvh):
        path.write_text(path.read_text() + '\n')
        # Include after the original header, which defines the required types.
        original = '#include "ms2scan.h"' if path == scan else '#include "MVH.h"'
        replace_once(path, original, original + '\n#include "export_hook.h"')
    replace_once(scan,
                 'void MS2Scan::preprocessMvh(multimap<double, double> *pIntenSortedPeakPreData)\n{',
                 'void MS2Scan::preprocessMvh(multimap<double, double> *pIntenSortedPeakPreData)\n{\n\tPeakExport::raw(this);')
    replace_once(scan, 'if (MVH::ScoreSequenceVsSpectrum(peptidePtr->sNeutralLossPeptide, precursorCharge,',
                 'PeakExport::candidate(peptidePtr);\n\t\t\t\tif (MVH::ScoreSequenceVsSpectrum(peptidePtr->sNeutralLossPeptide, precursorCharge,')
    # Scope this hook to Regular's scoring method, never the SIP method.
    text = mvh.read_text()
    start = text.index('bool MVH::ScoreSequenceVsSpectrum(')
    pos = text.index('\tint totalPeaks = (int) seqIons->size();', start)
    mvh.write_text(text[:pos] + '\tPeakExport::theoretical(Spectrum, currentPeptide, precursorCharge, seqIons);\n' + text[pos:])
    # Smaller batches reduce latency/memory; selected scan is still scored through production code.
    replace_once(source / "include/ms2scanvector.h", '#define PEPTIDE_ARRAY_SIZE  2000000',
                 '#define PEPTIDE_ARRAY_SIZE  1000')
    replace_once(source / "src/ms2scanvector.cpp", '\tsearchDatabaseMvh();',
                 '\tsearchDatabaseMvh();\n\tstd::cerr << "No theoretical candidate exported for selected scan. Try another scan ID.\\n";\n\tstd::exit(3);')
    replace_once(source / "main.cpp", 'if (ProNovoConfig::getSearchType() == "SIP")',
                 'if (ProNovoConfig::getSearchType() != "Regular")')
    replace_once(source / "main.cpp", '\t\t\tpMainMS2ScanVector->startProcessingWdpSip();',
                 '\t\t\tstd::cerr << "Exporter requires Search_Type = Regular\\n"; std::exit(2);')
    (source / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.12)
project(SiprosPeakExport LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/MSToolkit/include)
add_subdirectory(MSToolkit)
file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_executable(sipros_peak_export ${SOURCES} main.cpp)
target_compile_definitions(sipros_peak_export PRIVATE GCC _FILE_OFFSET_BITS=64 _NOSQLITE)
target_compile_options(sipros_peak_export PRIVATE -ffast-math)
target_link_libraries(sipros_peak_export PRIVATE mstoolkit OpenMP::OpenMP_CXX)
''')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="FT2 or mzML input")
    parser.add_argument("--config", type=Path, required=True, help="Regular search configuration")
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--scan-id", type=int, help="Default: first preprocessed scan with >=7 raw peaks")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; refuses overwrite")
    parser.add_argument("--jobs", type=int, default=4, help="Compilation jobs (search always uses one thread)")
    args = parser.parse_args()
    for key in ("input", "config", "fasta"):
        path = getattr(args, key).resolve()
        if not path.is_file():
            parser.error(f"Missing {key}: {path}")
        setattr(args, key, path)
    if args.input.suffix.lower() not in (".ft2", ".mzml"):
        parser.error("Use an FT2 or mzML file, not a vendor RAW file")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not shutil.which("cmake"):
        parser.error("cmake not found; run inside the project development container")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = prepare(out / "build")
    binary_dir = out / "build/cmake"
    with (out / "build.log").open("w") as log:
        subprocess.run(["cmake", "-S", str(source), "-B", str(binary_dir), "-DCMAKE_BUILD_TYPE=Release"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(["cmake", "--build", str(binary_dir), "--target", "sipros_peak_export", "-j", str(args.jobs)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE", OMP_THREAD_LIMIT="1",
               SIPROS_PEAK_EXPORT_DIR=str(out))
    env.pop("SIPROS_PEAK_EXPORT_SCAN", None)
    if args.scan_id is not None:
        env["SIPROS_PEAK_EXPORT_SCAN"] = str(args.scan_id)
    command = [str(binary_dir / "sipros_peak_export"), "-f", str(args.input),
               "-c", str(args.config), "-fasta", str(args.fasta), "-o", str(out)]
    (out / "run.json").write_text(json.dumps({"command": command, "scan_id": args.scan_id,
        "selection": "First candidate reaching MVH scoring for selected scan; not best hit",
        "order": "Observed: before Sipros sort; theoretical: generation order"}, indent=2) + "\n")
    with (out / "run.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    for name in ("observed_unsorted.tsv", "theoretical_unsorted.tsv", "observed_mvh.tsv", "candidate.tsv"):
        if not (out / name).is_file():
            raise RuntimeError(f"Export incomplete: missing {name}; inspect {out / 'run.log'}")
    print(f"Export complete: {out}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Export failed: {exc}. Inspect build.log / run.log in the output directory.", file=sys.stderr)
        sys.exit(1)
