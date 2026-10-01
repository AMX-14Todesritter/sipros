"""Compare complete GPU-generated streams with the untouched CPU generator."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument('--binary', type=Path, required=True)
a = p.parse_args()
data = Path(__file__).parent / 'data'
base = (data / 'search.cfg').read_text()

def configuration(**values):
    text = base
    for key, value in values.items():
        text, count = re.subn(r'^' + re.escape(key) + r'\s*=.*$', key+' = '+str(value), text, flags=re.M)
        assert count == 1, key
    return text

with tempfile.TemporaryDirectory(prefix='sipros-generation-') as temporary:
    root = Path(temporary)
    cases = [
        ('blocks', configuration(Minimum_Peptide_Length=1, Maximum_Peptide_Length=128),
         ''.join(f'>protein_{i} description\nMKRANNQMKKQJX?\n' for i in range(270))),
        ('ptm_pages', configuration(Minimum_Peptide_Length=1, Maximum_Peptide_Length=128,
             **{'PTM{~}':'MNQ', 'PTM{!}':'NQ[]'}),
         ''.join(f'>rich_{i}\nM'+'NQ'*15+'\n' for i in range(8))),
        ('no_ptm', configuration(Minimum_Peptide_Length=1, Max_PTM_Count=0,
              Try_First_Methionine='false', Maximum_Missed_Cleavages=0),
         '>one\nMKR\n>two\nMNQK\n>single\nM\n'),
        ('cleavage', configuration(Minimum_Peptide_Length=1, Cleave_Before_Residues='AG',
               Maximum_Missed_Cleavages=5, SIP_Element='N'),
         '>one\nMKRANQKRAGKR\n>single\nM\n>next\nANNQK\n'),
        ('long', configuration(Minimum_Peptide_Length=1, Maximum_Peptide_Length=128, Max_PTM_Count=1),
         '>long\n'+'A'*127+'K\n>next\n'+'A'*130+'K\n'),
    ]
    for name, config, fasta in cases:
        cfg = root / (name+'.cfg'); cfg.write_text(config)
        db = root / (name+'.fasta'); db.write_text(fasta)
        result = subprocess.run([str(a.binary), str(cfg), str(db)], capture_output=True, text=True)
        assert result.returncode == 0, (name, result.stdout, result.stderr)
        assert 'PASS:' in result.stdout, result.stdout
        if name == 'ptm_pages':
            count = int(re.search(r'PASS: generated=(\d+)', result.stdout)[1])
            assert count > 65536, count
        print(name, result.stdout.splitlines()[-1])
    cfg = root / 'overflow.cfg'
    cfg.write_text(configuration(Minimum_Peptide_Length=1, Maximum_Peptide_Length=128, Max_PTM_Count=128))
    db = root / 'overflow.fasta'; db.write_text('>overflow\n'+'N'*128+'\n')
    result = subprocess.run([str(a.binary), str(cfg), str(db)], capture_output=True, text=True)
    assert result.returncode != 0 and 'PTM combination count overflow' in result.stderr, result
print('PASS: block/page boundaries, terminal/multiple PTMs, cleavage rules, isotope mass, lengths, overflow')
