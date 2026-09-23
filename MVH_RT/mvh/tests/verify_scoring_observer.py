"""The scoring copy must differ only by the observer include and invocation."""
from pathlib import Path

module = Path(__file__).resolve().parents[1]
original = (module / 'original/src/MVH.cpp').read_text()
actual = (module / 'src/mvh_scoring.cpp').read_text()
include = '#include "rt_match_observer.h"\n'
hook = ('\t// Observe generated ions without changing CPU matching or scoring.\n'
        '\tif (mvh_rt::matchObserver) {\n'
        '\t\tmvh_rt::matchObserver(Spectrum, *seqIons);\n'
        '\t}\n')
assert actual.count(include) == actual.count(hook) == 1
assert actual.replace(include, '', 1).replace(hook, '', 1) == original
assert actual.index(hook) < actual.index('bool MVH::ScoreSequenceVsSpectrumSIP(')
print('PASS: only observer include/call added; original CPU algorithm unchanged')
