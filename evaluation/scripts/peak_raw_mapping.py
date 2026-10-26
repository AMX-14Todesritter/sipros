"""Map fused diagnostic retained-peak identities to original FT2 records."""
import argparse,csv,json,math
from pathlib import Path
from prepare import records
from run import save
p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);a=p.parse_args();m=json.loads(a.manifest.read_text());source=Path(m['arguments']['scans'])
for run in m['runs']:
 if run.get('status')!='completed' or run['warmup']:continue
 folder=a.manifest.parent/run['name']/'peak_audit';examples=folder/'peak_audit_examples.tsv'
 if not examples.is_file():continue
 wanted={}
 with examples.open() as f:
  for row in csv.DictReader(f,delimiter='\t'):
   for which in ('oracle','actual'):
    if int(row[which+'_retained_peak_id'])>=0:wanted.setdefault(row['scan_id'],set()).add(float(row[which+'_mz']))
 mapping={};copied=0
 with (folder/'original_mismatch_spectra.FT2').open('wb') as output:
  for rec in records(source):
   sid=rec.splitlines()[0].split()[1].decode()
   if sid not in wanted:continue
   output.write(rec);copied+=1;indices={mz:[] for mz in wanted[sid]};ordinal=0
   for line in rec.splitlines():
    if line and 48<=line[0]<=57:
     mz=float(line.split()[0])
     if mz in indices:indices[mz].append(ordinal)
     ordinal+=1
   for mz,index in indices.items():mapping[(sid,mz)]=index
 unresolved=0;duplicates=0;n=0
 with examples.open() as f,(folder/'peak_audit_original_ids.tsv').open('w') as out:
  reader=csv.DictReader(f,delimiter='\t');fields=list(reader.fieldnames)+['oracle_raw_peak_indices','actual_raw_peak_indices','classification','near_float32_tolerance_boundary']
  writer=csv.DictWriter(out,fieldnames=fields,delimiter='\t');writer.writeheader()
  for row in reader:
   q=float(row['theoretical_mz']);tol=float(row['tolerance']);left=int(row['oracle_retained_peak_id']);right=int(row['actual_retained_peak_id'])
   for which in ('oracle','actual'):
    ids=[] if int(row[which+'_retained_peak_id'])<0 else mapping.get((row['scan_id'],float(row[which+'_mz'])),[])
    if int(row[which+'_retained_peak_id'])>=0 and not ids:unresolved+=1
    duplicates+=len(ids)>1;row[which+'_raw_peak_indices']=json.dumps(ids)
   if left<0:reason='RT_or_CUDA_hit_outside_strict_double_tolerance'
   elif right<0:reason='missed_valid_peak'
   elif left!=right:
    reason='equal_distance_tie_selection' if abs(float(row['oracle_mz'])-q)==abs(float(row['actual_mz'])-q) else 'different_nearest_peak'
   else:reason='class_difference'
   scale=2**(math.floor(math.log2(abs(q))) if q else -126)*2**-23
   boundary=any(abs(abs(float(row[k])-q)-tol)<=4*scale for k in ('oracle_mz','actual_mz') if float(row[k])>=0)
   row.update(classification=reason,near_float32_tolerance_boundary=boundary);writer.writerow(row);n+=1
 save(folder/'raw_mapping_manifest.json',{'status':'completed' if not unresolved else 'invalidated','source_ft2':str(source),'example_rows':n,'original_spectrum_records_saved':copied,'unresolved_peak_references':unresolved,'duplicate_raw_mz_memberships':duplicates,'note':'Zero-based original peak-record ordinals; duplicate raw m/z memberships remain explicit. Float-boundary proximity is a diagnostic flag, not a proven causal explanation.'})
 print(run['backend'],n,'mapped examples;',unresolved,'unresolved',flush=True)
