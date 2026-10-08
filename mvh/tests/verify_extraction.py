"""Only class renaming is permitted in extracted original method definitions."""
import json
from pathlib import Path
from source_methods import methods
m=Path(__file__).resolve().parents[1]
spec=json.loads((m/'extraction.json').read_text())
old=spec['original_class'];new=spec['extracted_class']
original=methods((m/spec['source']).read_text(),old)
count=0
for filename,names in spec['files'].items():
    actual=methods((m/filename).read_text(),new)
    expected={n.replace(old,new):original[n].replace(old,new) for n in names}
    if filename == 'src/spectrum_input.cpp':
        # Authorized instrument-only FT2 policy; keep all other extracted methods checked.
        instrument = methods((m.parent/'src/ms2scanvector.cpp').read_text(), old)
        for name in ('ReadFT2File', 'loadFT2File', 'saveFT2Scan'):
            expected[name] = instrument[name].replace(old, new)
    if filename == 'src/database_search.cpp':
        # Authorized runtime generated-peptide batching; every other byte remains checked.
        body = actual['searchDatabaseMvh']
        body = body.replace('\n\tsize_t generatedPeptides = 0;', '')
        body = body.replace('if (++generatedPeptides >= size_t(peptideBatchSize)) {\n\t\t\t\tprocessPeptideArrayMvh(vpPeptideArray);\n\t\t\t\tgeneratedPeptides = 0;\n\t\t\t}',
                            'if (vpPeptideArray.size() >= PEPTIDE_ARRAY_SIZE)\n\t\t\t\tprocessPeptideArrayMvh(vpPeptideArray);')
        actual['searchDatabaseMvh'] = body
    assert actual==expected, 'Original methods changed: '+filename
    count+=len(names)
runner=(m/'app/runner.cpp').read_text()
assert 'access(' not in runner and 'template' not in runner
assert 'spectra.preProcessAllMs2Mvh();' in runner
assert 'spectra.searchDatabaseMvh();' in runner
print(f'PASS: {count} method definitions identical except class rename; direct calls')
