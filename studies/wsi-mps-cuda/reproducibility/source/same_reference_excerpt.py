"""Source SAME-F: exact selected definitions. See docs/SOURCE_STATUS.md.
No historical dispatch or scientific execution is invoked on import.
"""
from pathlib import Path
import json, math, time
import numpy as np
import h5py


def gap_fields(F,E,old,af,ae,out):
 """Full E-F ordered read only for fields absent from exact legacy record.

 Does not recompute legacy global relative L2, MAE, cosine, max_abs or the four
 saved cosine quantiles. Norms reuse A-F/A-E records with exact input bindings.
 Same row-block float64 arithmetic and ceil(1%N) rule as accepted comparator.
 """
 out.mkdir(parents=True,exist_ok=False);begin=time.perf_counter();before={str(p):stat(p) for p in [F,E]}
 with h5py.File(F,'r') as a,h5py.File(E,'r') as b:
  coords=a['coords'][:];assert np.array_equal(coords,b['coords'][:])
  n,d=a['features'].shape;assert b['features'].shape==(n,d) and d==1536 and a['features'].dtype==b['features'].dtype==np.float32
  names=['SSE','absolute_L2','reference_norm','candidate_norm','relative_L2','cosine','MAE','RMSE','max_abs']
  fields={k:np.empty(n,dtype=np.float64) for k in names};columns=np.empty(n,dtype=np.int32);exact=np.empty(n,dtype=bool);location=None;max_seen=-1.
  for i in range(0,n,256):
   aa=a['features'][i:i+256].astype(np.float64);bb=b['features'][i:i+256].astype(np.float64)
   assert np.isfinite(aa).all() and np.isfinite(bb).all()
   diff=bb-aa;ad=np.abs(diff);sse=np.sum(diff*diff,axis=1);rn=np.sqrt(np.sum(aa*aa,axis=1));en=np.sqrt(np.sum(bb*bb,axis=1));dot=np.sum(aa*bb,axis=1)
   v=dict(SSE=sse,absolute_L2=np.sqrt(sse),reference_norm=rn,candidate_norm=en,
    relative_L2=np.divide(np.sqrt(sse),rn,out=np.full(len(aa),np.nan),where=rn>0),
    cosine=np.divide(dot,rn*en,out=np.full(len(aa),np.nan),where=(rn*en)>0),MAE=np.mean(ad,axis=1),RMSE=np.sqrt(sse/d),max_abs=np.max(ad,axis=1))
   for k,x in v.items():fields[k][i:i+len(aa)]=x
   columns[i:i+len(aa)]=np.argmax(ad,axis=1);exact[i:i+len(aa)]=np.all(diff==0,axis=1)
   rr,cc=np.unravel_index(int(np.argmax(ad)),ad.shape);value=float(ad[rr,cc])
   if value>max_seen:
    max_seen=value;location=dict(row=int(i+rr),column=int(cc),coord_xy=coords[i+rr].tolist(),reference_value=float(aa[rr,cc]),candidate_value=float(bb[rr,cc]),candidate_minus_reference=float(diff[rr,cc]),indexing='zero based row and feature column',tie_rule='first original row, then first feature column')
  assert max_seen==old['metrics']['max_abs'],'Legacy scalar does not match actual E/F; no replacement'
  # Existing exact scalar values remain the authority. Only missing fields filled.
  total=float(fields['SSE'].sum());order=np.lexsort((np.arange(n),-fields['SSE']));k=math.ceil(.01*n);selected=float(fields['SSE'][order[:k]].sum())
  def distribution(v):
   vv=v[np.isfinite(v)]
   return dict(defined=len(vv),undefined=len(v)-len(vv),quantiles=dict(zip(['min','p01','p05','median','p95','p99','max'],map(float,np.quantile(vv,[0,.01,.05,.5,.95,.99,1])))) if len(vv) else None)
  distributions={key:distribution(v) for key,v in fields.items()}
  reuse_quantiles={'min':'tile_min','p01':'tile_p01','p05':'tile_p05','median':'tile_median'}
  checks={}
  for q,legacy in reuse_quantiles.items():
   checks[q]=dict(legacy=old['metrics'][legacy],gap_scan=distributions['cosine']['quantiles'][q])
   assert math.isclose(checks[q]['legacy'],checks[q]['gap_scan'],rel_tol=0,abs_tol=3e-15)
   distributions['cosine']['quantiles'][q]=old['metrics'][legacy]
  def row(j):return dict(row=int(j),coord_xy=coords[j].tolist(),**{key:float(v[j]) if np.isfinite(v[j]) else None for key,v in fields.items()})
  metric=dict(reference=str(F),candidate=str(E),direction='E minus F; denominator=the exact canonical F used by A-F',shape=[n,d],features_storage_dtype='float32',coords_row_equal=True,
   relative_L2=old['metrics']['relative_l2'],MAE=old['metrics']['mae'],global_cosine=old['metrics']['global_cosine'],max_abs=old['metrics']['max_abs'],
   RMSE=math.sqrt(total/(n*d)),absolute_L2=math.sqrt(total),reference_norm=af['reference_norm'],candidate_norm=ae['reference_norm'],
   max_abs_location=location,tile_distributions=distributions,top_SSE_fraction=dict(N=n,k=k,k_over_N=k/n,SSE_total=total,selected_SSE=selected,contribution=selected/total if total>0 else None,rule='ceil(.01*N); descending SSE; ascending original row index at ties; zero total undefined'),
   highest_SSE_rows=list(map(row,order[:5])),lowest_cosine_rows=list(map(row,np.lexsort((np.arange(n),fields['cosine']))[:5])),
   preserved_legacy_scalar_fields=['relative_L2','MAE','global_cosine','max_abs','cosine min/p01/p05/median'],
   missing_field_completion='RMSE, complete position, row distributions, row SSE and tail; reference/candidate norms reused from exact A-F and A-E identities',
   legacy_cosine_quantile_consistency=checks,CPU_missing_field_wall_s=time.perf_counter()-begin)
  assert math.isclose(metric['absolute_L2']/af['reference_norm'],metric['relative_L2'],rel_tol=2e-13)
  assert math.isclose(float(np.sum(fields['reference_norm']**2)),af['reference_norm']**2,rel_tol=2e-13)
  assert math.isclose(float(np.sum(fields['candidate_norm']**2)),ae['reference_norm']**2,rel_tol=2e-13)
  np.savez_compressed(out/'Per_Tile_EF_Missing_Fields.npz',row=np.arange(n),coords=coords,max_abs_column=columns,features_exact_equal=exact,**fields)
  save(out/'Metrics.json',metric)
 for p in [F,E]:assert stat(p)==before[str(p)],'Input metadata changed during read'
 return metric

def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def stat(p):
 s=Path(p).stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino)
