"""Preserve the active measurement and stop the frozen study coordinator."""
import datetime,json,os,signal,time
from pathlib import Path
from scan_scaling_marine_nested import save,report
E=Path('/workspace/sipros/output/evaluation');D=E/'scan_scaling_marine_nested';p=D/'manifest.json'
m=json.loads(p.read_text());m.update(status='stopping_after_current',stop_requested_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save(p,m);report(m)
while Path('/proc/738/stat').exists():
 if Path('/proc/738/stat').read_text().split(') ',1)[1].startswith('Z'):break
 time.sleep(1)
m=json.loads(p.read_text())
for j in m['jobs']:
 if j['status']!='running':continue
 result=E/'results/scan_scaling'/j['id']/'manifest.json';n=json.loads(result.read_text());r=n['runs'][0];j['result_manifest']=str(result)
 if r['status']=='completed':
  t=r['totals'];j.update(status='completed',metrics={'fused_seconds':t['fused_device_seconds'],'rt_setup_seconds':r['rt_setup_seconds'],'complete_scoring_seconds':sum(t[k] for k in ('pack_seconds','gpu_service_seconds','replay_verify_seconds')),'search_seconds':float(r['summary']['search_seconds']),'device_peak_bytes':r['device_peak_bytes'],'host_peak_bytes':r['host_observed_hwm_bytes'],'candidates':int(t['candidates']),'psm_sha256':r['psm_sha256']})
 else:j.update(status=r['status'])
m.update(status='paused_by_user',stopped_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save(p,m);report(m)
os.kill(435,signal.SIGTERM);os.kill(435,signal.SIGCONT)
print('Paused; active result preserved; no further measurements dispatched',flush=True)
