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

        def run(name, mode, verify, batch, spectrum=None, fasta=None, generation="cuda", restoration="final", group=8):
            output = root / name
            command = [str(args.binary), "-f", str(spectrum or data / "sample.ft2"),
                       "-c", str(data / "search.cfg"), "-fasta", str(fasta or data / "proteins.fasta"),
                       "-o", str(output), "--match-backend", mode,
                       "--peptide-batch-size", str(batch), "--peptide-generation", generation, "--result-restoration", restoration]
            if mode == "rt-custom":
                command.extend(["--rt-scan-group-size", str(group)])
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

        # Shared split-coordinate spheres use class priority 3 -> 2 -> 1.
        # Their independent CPU verifier checks the new rule.
        sphere_outputs = []
        for group, batch in ((1,2000000), (2,3), (8,2000000), (32,3),(128,2000000)):
            output, log = run(f"sphere_{group}_{batch}", "rt-custom", True, batch, group=group)
            sphere_outputs.append((output / "mvh_psms.tsv").read_bytes())
            assert all(count > 0 for count in bucket_entries(log, "host_bucket_entries")), log
            assert all(count == 0 for count in bucket_entries(log, "device_bucket_entries")), log
            assert log.count("[RT GPU setup]") == 1, log
        assert len(set(sphere_outputs)) == 1, "Sphere outputs changed with batch size"

        # A real shared launch spans several groups and a partial final group.
        multi = root / "multi.ft2"
        lines = (data / "sample.ft2").read_text().splitlines()
        header = [line for line in lines if line.startswith("H\t")]
        body = [line for line in lines if not line.startswith("H\t")]
        blocks = []
        for scan in range(35):
            blocks.extend("S\t" + str(2000+scan) + "\t" + line.split("\t",2)[2]
                          if line.startswith("S\t") else line for line in body)
        multi.write_text("\n".join(header+blocks)+"\n")
        multi_outputs = []
        for group, batch in ((1,2000000),(2,3),(8,2000000),(32,3),(128,2000000)):
            output, log = run(f"multi_{group}", "rt-custom", True, batch, spectrum=multi, group=group)
            multi_outputs.append((output / "mvh_psms.tsv").read_bytes())
            pairs = re.findall(r"\[RT shared tasks\] candidates=(\d+) tasks=(\d+)",log)
            assert pairs, log
            if group > 1:
                assert any(int(tasks)<int(candidates) for candidates,tasks in pairs), log
        assert len(set(multi_outputs)) == 1, "Shared multi-scan PSMs depend on K or batch size"

        # Repeat identical peptides under different protein names across batch
        # boundaries. Protein attribution must survive GPU-resident Top reuse.
        repeated = root / "repeated.fasta"
        sequence = "".join((data / "proteins.fasta").read_text().splitlines()[1:])
        repeated.write_text("".join(f">protein_{i}\n{sequence}\n" for i in range(4)))
        for mode in ("cuda", "rt-custom"):
            reference, _ = run(f"{mode}_repeat_reference", mode, False, 2000000, fasta=repeated)
            reference_psms = (reference / "mvh_psms.tsv").read_bytes()
            immediate, _ = run(f"{mode}_batch_restore", mode, False, 3, fasta=repeated, restoration="batch")
            assert (immediate / "mvh_psms.tsv").read_bytes() == reference_psms, "Final/batch restoration changed PSMs"

            cpu, _ = run(f"{mode}_cpu_generation", mode, False, 3, fasta=repeated, generation="cpu")
            assert (cpu / "mvh_psms.tsv").read_bytes() == reference_psms, "CPU/GPU generation changed PSMs"

            for verify in (False, True):
                output, _ = run(f"{mode}_repeat_{verify}", mode, verify, 3, fasta=repeated)
                assert (output / "mvh_psms.tsv").read_bytes() == reference_psms, \
                    "Cross-batch Top changed scores, ordering or protein merges"

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
