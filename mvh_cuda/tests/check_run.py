"""Container-only integration checks; no retained temporary build or output copies."""
import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--root',type=Path,required=True)
a=p.parse_args()
data=Path(__file__).resolve().parent/'data'
with tempfile.TemporaryDirectory(prefix='mvh-check-',dir=a.binary.parent.parent) as tmp:
    tmp=Path(tmp)
    def run(name, threads=1, spectrum=None, fasta=None, expect_ok=True, verify=True, batch=None):
        output=tmp/name
        cmd=[str(a.binary),'-f',str(spectrum or data/'sample.ft2'),'-c',str(data/'search.cfg'),
             '-fasta',str(fasta or data/'proteins.fasta'),'-o',str(output),'-t',str(threads)]
        if verify: cmd.append('--verify-cuda')
        if batch is not None: cmd += ['--peptide-batch-size',str(batch)]
        result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        assert (result.returncode==0)==expect_ok, result.stdout
        return output
    one=run('t1'); four=run('t4',4)
    normal=run('normal', verify=False)
    tiny=run('tiny_batches', batch=3)
    assert (one/'mvh_psms.tsv').read_bytes()==(tiny/'mvh_psms.tsv').read_bytes()
    run('bad_batch', batch=0, expect_ok=False)
    assert (one/'mvh_psms.tsv').read_bytes()==(normal/'mvh_psms.tsv').read_bytes()
    assert (one/'mvh_psms.tsv').read_bytes()==(four/'mvh_psms.tsv').read_bytes()
    with (one/'mvh_psms.tsv').open() as f: rows=list(csv.DictReader(f,delimiter='\t'))
    assert any(r['peptide']=='[LDNM~ATK]' for r in rows), rows
    # Existing output rejection must leave results untouched.
    before=(one/'mvh_psms.tsv').read_bytes()
    run('t1',expect_ok=False)
    assert before==(one/'mvh_psms.tsv').read_bytes()
    run('bad_threads',threads=0,expect_ok=False)
    unknown=tmp/'unmatched.fasta'; unknown.write_text('>unmatched\n'+'W'*40+'\n')
    no=run('unmatched',fasta=unknown)
    assert len((no/'mvh_psms.tsv').read_text().splitlines())==1
    skip=tmp/'skipped.ft2'
    lines=(data/'sample.ft2').read_text().splitlines()
    skip.write_text('\n'.join(x for x in lines if x and not x[0].isdigit())+'\n100.0\t1000.0\n')
    skipped=run('skipped',spectrum=skip)
    with (skipped/'run_summary.tsv').open() as f: summary=dict(list(csv.reader(f,delimiter='\t'))[1:])
    assert summary['skipped_scan_count']=='1',summary
    assert summary['retained_psm_count']=='0',summary
print('PASS: matching/PTM, normal=verified, t1=t4, unmatched, skipped, invalid threads, no overwrite')
