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

        def run(name, mode, verify, batch, spectrum=None, fasta=None, generation="cuda", restoration="final", group=10, candidate=False, hit_filter="scan"):
            output = root / name
            command = [str(args.binary), "-f", str(spectrum or data / "sample.ft2"),
                       "-c", str(data / "search.cfg"), "-fasta", str(fasta or data / "proteins.fasta"),
                       "-o", str(output), "--match-backend", mode,
                       "--candidate-batch-size" if candidate else "--peptide-batch-size", str(batch), "--peptide-generation", generation, "--result-restoration", restoration]
            if mode == "rt-custom":
                command.extend(["--rt-scan-group-size", str(group), "--rt-hit-filter", hit_filter])
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
        # Put 25 relevant scans and 10 unrelated precursor scans in mass order,
        # so K=10 produces a group containing both. Identical fragment peaks
        # ensure the unrelated geometry really triggers any-hit callbacks.
        filter_spectra = root / "filter_mixed_precursors.ft2"
        filter_blocks = []
        for scan in range(35):
            for line in body:
                if line.startswith("S\t"):
                    parts = line.split("\t")
                    parts[1] = str(3000+scan)
                    if scan >= 25: parts[2] = "10000.0"
                    line = "\t".join(parts)
                elif scan >= 25 and line.startswith("Z\t"):
                    line = "Z\t2\t20000.0\t2\t10000.0"
                filter_blocks.append(line)
        filter_spectra.write_text("\n".join(header+filter_blocks)+"\n")
        baseline, baseline_log = run("early_filter_off", "rt-custom", True, 2000000,
                                     spectrum=filter_spectra, group=10, hit_filter="off")
        filtered, filtered_log = run("early_filter_scan", "rt-custom", True, 2000000,
                                     spectrum=filter_spectra, group=10, hit_filter="scan")
        assert (baseline / "mvh_psms.tsv").read_bytes() == (filtered / "mvh_psms.tsv").read_bytes()
        def early_totals(log):
            rows = [dict((k,int(v)) for k,v in re.findall(r"(\w+)=(\d+)", line))
                    for line in log.splitlines() if line.startswith("[RT early filtering]")]
            assert rows, log
            assert all(r["raw_hits"] == r["unrelated_scan"] + r["stored_hits"] for r in rows)
            return {k:sum(r[k] for r in rows) for k in ("raw_hits", "unrelated_scan", "stored_hits")}
        before, after = early_totals(baseline_log), early_totals(filtered_log)
        assert before["raw_hits"] == after["raw_hits"] and before["unrelated_scan"] == 0
        assert after["unrelated_scan"] > 0 and after["stored_hits"] < before["stored_hits"]
        assert all(int(x)==0 for x in re.findall(r"\[RT hit filtering\].*?unrelated_scan=(\d+)",filtered_log))


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

        # Candidate budgets vary peptide counts but retain all associated scores,
        # duplicate precursor hypotheses, scan-local order, and protein attribution.
        for mode in ("cuda", "rt-custom"):
            reference, reference_log = run(f"{mode}_candidate_reference", mode, True, 2000000,
                                           spectrum=multi, fasta=repeated)
            expected = (reference / "mvh_psms.tsv").read_bytes()
            expected_candidates = sum(map(int, re.findall(r"\[CUDA assignment\].*?associations=(\d+)", reference_log)))
            all_generated = sum(map(int, re.findall(r"\[CUDA assignment\] generated=(\d+)", reference_log)))
            for limit, generation, restoration in ((50, "cuda", "final"), (100, "cpu", "batch")):
                output, log = run(f"{mode}_candidates_{limit}", mode, True, limit, spectrum=multi,
                                  fasta=repeated, generation=generation, restoration=restoration, candidate=True)
                assert (output / "mvh_psms.tsv").read_bytes() == expected, "Candidate batching changed Top/protein output"
                batches = re.findall(r"\[CUDA candidate batch\] peptides=(\d+) candidates=(\d+) limit=(\d+)", log)
                assert len(batches) > 1, log
                assert all(0 < int(count) <= limit and int(bound) == limit for _, count, bound in batches), log
                assert sum(int(count) for _, count, _ in batches) == expected_candidates, log
                totals = re.search(r"\[CUDA candidate batching\] generated=(\d+) unassigned=(\d+) candidates=(\d+) batches=(\d+)", log)
                assert totals and int(totals[1]) == all_generated and int(totals[3]) == expected_candidates, log
                assert sum(int(peptides) for peptides, _, _ in batches) + int(totals[2]) == all_generated, log
                with (output / "run_summary.tsv").open() as stream:
                    summary = dict(list(csv.reader(stream, delimiter="\t"))[1:])
                assert summary["batch_sizing"] == "candidates" and int(summary["candidate_batch_size"]) == limit
                assert summary["peptide_batch_size"] == "0"
            command = [str(args.binary), "-f", str(multi), "-c", str(data / "search.cfg"),
                       "-fasta", str(repeated), "-o", str(root / f"{mode}_too_small"),
                       "--match-backend", mode, "--candidate-batch-size", "1"]
            result = subprocess.run(command, capture_output=True, text=True)
            assert result.returncode != 0 and "One peptide produces" in result.stderr, result

        # Cross the 65,536-object counting page boundary, including many zero
        # association objects, without losing or duplicating protein metadata.
        page_fasta = root / "counting_pages.fasta"
        page_fasta.write_text("".join(f">page_protein_{i}\n{sequence}\n" for i in range(160)))
        reference, _ = run("pages_reference", "rt-custom", True, 2000000,
                           spectrum=multi, fasta=page_fasta)
        output, log = run("pages_candidates", "rt-custom", True, 1000,
                          spectrum=multi, fasta=page_fasta, candidate=True)
        assert (output / "mvh_psms.tsv").read_bytes() == (reference / "mvh_psms.tsv").read_bytes()
        generated = int(re.search(r"\[CUDA candidate batching\] generated=(\d+)", log)[1])
        assert generated > 65536, log

        # Default candidate mode must also handle a database with no associations.
        unrelated = root / "unrelated.fasta"
        unrelated.write_text(">unrelated\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")
        output, log = run("zero_candidates", "cuda", True, 200, fasta=unrelated, candidate=True)
        assert "candidates=0 batches=0" in log and "[CUDA assignment]" not in log, log
        assert int(re.search(r"\[CUDA candidate batching\] generated=(\d+)", log)[1]) > 0, log
        with (output / "run_summary.tsv").open() as stream:
            summary = dict(list(csv.reader(stream, delimiter="\t"))[1:])
        assert summary["retained_psm_count"] == "0"

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
