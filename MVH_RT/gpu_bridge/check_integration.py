"""Check indexed CUDA/audit and bucket-free RT using the same scoring inputs."""
import argparse
import csv
from pathlib import Path
import re
import subprocess
import tempfile


def bucket_entries(log, field):
    values = [int(value) for value in re.findall(rf"{field}=(\d+)", log)]
    assert values, f"Missing {field} metric: {log}"
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    data = args.root / "mvh_cuda/tests/data"
    with tempfile.TemporaryDirectory(prefix="gpu_rt_bridge_") as temporary:
        root = Path(temporary)

        def run(name, mode, verify, batch, spectrum=None):
            output = root / name
            command = [str(args.binary), "-f", str(spectrum or data / "sample.ft2"),
                       "-c", str(data / "search.cfg"), "-fasta", str(data / "proteins.fasta"),
                       "-o", str(output), "--match-backend", mode,
                       "--peptide-batch-size", str(batch)]
            if verify:
                command.append("--verify-cuda")
            result = subprocess.run(command, capture_output=True, text=True)
            assert result.returncode == 0, (command, result.stdout, result.stderr)
            return output, result.stdout

        outputs = []
        for mode in ("cuda", "rt-audit", "rt-triangle", "rt-instanced"):
            device_buckets = mode in ("cuda", "rt-audit")
            for verify in (False, True):
                for batch in (2000000, 3):
                    name = f"{mode}_{verify}_{batch}"
                    output, log = run(name, mode, verify, batch)
                    outputs.append((output / "mvh_psms.tsv").read_bytes())
                    host_entries = bucket_entries(log, "host_bucket_entries")
                    device_entries = bucket_entries(log, "device_bucket_entries")
                    assert all((count > 0) == (device_buckets or verify) for count in host_entries), log
                    assert all((count > 0) == device_buckets for count in device_entries), log
                    if mode != "cuda":
                        assert log.count("[RT GPU setup]") == 1, log
                    if mode == "rt-audit":
                        differences = re.findall(r"differing_results=(\d+)", log)
                        assert differences and all(int(value) == 0 for value in differences), log
        assert len(set(outputs)) == 1, "Sample PSMs differ by backend, verification, or batch size"

        # Sphere matching uses a different class-priority rule. Compare its
        # batches to itself, not to the original nearest-mass scoring rule.
        sphere_outputs = []
        for batch in (2000000, 3):
            output, log = run(f"sphere_{batch}", "rt-custom", False, batch)
            sphere_outputs.append((output / "mvh_psms.tsv").read_bytes())
            assert all(count == 0 for count in bucket_entries(log, "host_bucket_entries")), log
            assert all(count == 0 for count in bucket_entries(log, "device_bucket_entries")), log
            assert log.count("[RT GPU setup]") == 1, log
        assert len(set(sphere_outputs)) == 1, "Sphere outputs changed with batch size"

        # Exercise an empty pre-index peak map: no PeakList may be dereferenced.
        skipped = root / "skipped.ft2"
        lines = (data / "sample.ft2").read_text().splitlines()
        headers = [line for line in lines if line and not line[0].isdigit()]
        skipped.write_text("\n".join(headers) + "\n100.0\t1000.0\n")
        for mode in ("rt-triangle", "rt-instanced", "rt-custom"):
            output, log = run(f"{mode}_skipped", mode, False, 3, skipped)
            assert bucket_entries(log, "host_bucket_entries") == [0], log
            with (output / "run_summary.tsv").open() as stream:
                summary = dict(list(csv.reader(stream, delimiter="\t"))[1:])
            assert summary["skipped_scan_count"] == "1", summary
            assert summary["retained_psm_count"] == "0", summary
    print("PASS: normal/verified modes, host/device bucket policy, tiny batches, identical PSMs, skipped RT scans")


if __name__ == "__main__":
    main()
