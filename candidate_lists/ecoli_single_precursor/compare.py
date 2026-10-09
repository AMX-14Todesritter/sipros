from pathlib import Path
import csv,json,re
root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
source=root/'test_output/Pan_062822_X1iso5/ft/Pan_062822_X1iso5.FT2'
proton=1.007276466
tol=float(re.search(r'^Search_Mass_Tolerance_Parent_Ion\s*=\s*([\d.]+)',(root/'experiments/Regular.cfg').read_text(),re.M).group(1))
records=[];current=None
for line in source.open():
 if line.startswith('S\t'):
  w=line.split();current={'scan':int(w[1]),'mz':float(w[2]),'zlines':[]};records.append(current)
 elif line.startswith('Z\t') and current is not None:current['zlines'].append(line.split())
old=[];new=[];summary=[];invalid=[];multiple=0
for scan in records:
 s=scan['scan'];mz=scan['mz'];zl=scan['zlines'];multiple+=len(zl)>1
 firstz=int(zl[0][1]) if zl else 0;lastz=int(zl[-1][1]) if zl else 0
 before=[]
 if lastz>0:before.append((s,'primary',lastz,mz,lastz*(mz-proton)))
 for line in zl:
  for i in range(3,len(line)-1,2):
   z=int(line[i]);extra=float(line[i+1])
   if lastz<=0 or abs(extra*z-mz*lastz)>tol:before.append((s,'extra',z,extra,z*(extra-proton)))
 old.extend(before)
 if firstz>0 and mz>0:new.append((s,'primary',firstz,mz,firstz*(mz-proton)))
 else:invalid.append(s)
 summary.append((s,len(before),int(firstz>0 and mz>0),len(before)-int(firstz>0 and mz>0)))
headers=['scan_id','source','charge','mz','neutral_mass_Da']
for name,rows in [('old_precursors.tsv',old),('new_precursors.tsv',new)]:
 with (out/name).open('w') as f:
  writer=csv.writer(f,delimiter='\t');writer.writerow(headers)
  writer.writerows((a,b,c,f'{d:.6f}',f'{e:.9f}') for a,b,c,d,e in sorted(rows,key=lambda x:x[4]))
with (out/'per_scan_changes.tsv').open('w') as f:
 writer=csv.writer(f,delimiter='\t');writer.writerow(['scan_id','old_count','new_count','removed_count']);writer.writerows(summary)
result={'scan_file':str(source),'old_reference':'82522a4 FT2 precursor construction','new_rule':'S mz + first Z charge; positive primary required','proton_mass':proton,'parent_tolerance':tol,'scan_count':len(records),'old_precursor_count':len(old),'new_precursor_count':len(new),'removed_count':len(old)-len(new),'invalid_primary_scans':invalid,'multiple_z_scan_count':multiple,'changed_scan_count':sum(a!=b for _,a,b,_ in summary)}
(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
print('SCAN 1004 old:',[r for r in old if r[0]==1004]);print('SCAN 1004 new:',[r for r in new if r[0]==1004])
