"""Check path policy and real CLI behavior from an unrelated working directory."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.output_paths import output_root, resolve_output


def check_policy():
    with patch.dict(os.environ, {"SIPROS_OUTPUT_ROOT": ""}):
        assert output_root() == ROOT / "output"
        path = resolve_output(None, "exports", "peak_export", "scan / 1004")
        assert path.parent == ROOT / "output/exports/peak_export"
        assert re.fullmatch(r"scan___1004_\d{8}T\d{6}_\d{6}Z", path.name)
        assert not path.exists(), "Path resolution must not create directories"
    with patch.dict(os.environ, {"SIPROS_OUTPUT_ROOT": "output/custom"}):
        assert output_root() == ROOT / "output/custom"
        assert resolve_output(Path("explicit/run"), "search", "mvh") == Path("explicit/run").resolve()
    with tempfile.TemporaryDirectory(prefix="sipros-path-policy-") as temporary:
        with patch.dict(os.environ, {"SIPROS_OUTPUT_ROOT": temporary}):
            assert output_root() == Path(temporary)


def check_cli(binary, component):
    data = ROOT / "mvh/tests/data"
    command = [str(binary.resolve()), "-f", str(data / "sample.ft2"),
               "-c", str(data / "search.cfg"), "-fasta", str(data / "proteins.fasta")]
    with tempfile.TemporaryDirectory(prefix="sipros-output-cli-") as temporary:
        work = Path(temporary)
        environment = os.environ.copy()
        environment["SIPROS_OUTPUT_ROOT"] = str(work / "automatic")

        def run(extra=(), success=True):
            result = subprocess.run(command + list(extra), cwd=work, env=environment,
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert (result.returncode == 0) == success, result.stdout
            return result

        run()
        run()
        runs = sorted((work / "automatic/search" / component).iterdir())
        assert len(runs) == 2, "Repeated runs must use distinct directories"
        reference = (runs[0] / "mvh_psms.tsv").read_bytes()
        assert reference == (runs[1] / "mvh_psms.tsv").read_bytes()
        assert all(re.fullmatch(r"sample_\d{8}T\d{6}_\d{6}Z", path.name) for path in runs)

        # Explicit output remains cwd-relative and takes precedence over the env var.
        explicit = work / "nested/explicit/run"
        run(["-o", "nested/explicit/run"])
        assert (explicit / "mvh_psms.tsv").read_bytes() == reference
        before = {p.name: p.read_bytes() for p in explicit.iterdir() if p.is_file()}
        run(["-o", str(explicit)], success=False)
        assert before == {p.name: p.read_bytes() for p in explicit.iterdir() if p.is_file()}
        run(["-o", ""], success=False)

        # Relative output roots use the project location, even from a different cwd.
        relative_root = work / "relative_root"
        environment["SIPROS_OUTPUT_ROOT"] = os.path.relpath(relative_root, ROOT)
        run()
        relative_runs = list((relative_root / "search" / component).iterdir())
        assert len(relative_runs) == 1
        assert (relative_runs[0] / "mvh_psms.tsv").read_bytes() == reference
    print(f"PASS: {component} default names, independent cwd, explicit override, nested parents, no overwrite, identical PSMs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--component", default="mvh")
    args = parser.parse_args()
    check_policy()
    if args.binary:
        check_cli(args.binary, args.component)
    print("PASS: shared Python path policy")
