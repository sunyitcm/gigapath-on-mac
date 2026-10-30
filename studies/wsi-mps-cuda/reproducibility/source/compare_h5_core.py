"""Source CUDA-COMPARE: exact selected definitions. See docs/SOURCE_STATUS.md.
No historical dispatch or scientific execution is invoked on import.
"""
from pathlib import Path
import argparse, hashlib, json, math, time


def compare(reference,candidate,destination,block=256,direction='candidate minus reference; relative L2 denominator=reference'):
 import h5py,numpy as np
 destination.mkdir(parents=True,exist_ok=False);begin=time.perf_counter()
 with h5py.File(reference,'r') as a,h5py.File(candidate,'r') as b:
  assert a['features'].shape==b['features'].shape
  n,d=a['features'].shape;assert n>0 and d>0
  assert a['features'].dtype==b['features'].dtype==np.float32
  assert a['coords'].dtype==b['coords'].dtype==np.int32 and a['coords'].shape==b['coords'].shape==(n,2)
  coords=a['coords'][:];assert np.array_equal(coords,b['coords'][:]),'Coordinates differ: no sorting/truncating/intersection'
  x,y=coords[:,0],coords[:,1]
  assert (coords>=0).all() and ((y[1:]>y[:-1])|((y[1:]==y[:-1])&(x[1:]>x[:-1]))).all()
  fields={k:np.empty(n,dtype=np.float64) for k in ['SSE','absolute_L2','reference_norm','candidate_norm','relative_L2','cosine','MAE','RMSE','max_abs']}
  exact=np.empty(n,dtype=bool);dot_total=refsq=candsq=abs_sum=0.;maxabs=0.;max_location=None;max_columns=np.empty(n,dtype=np.int32)
  for i in range(0,n,block):
   aa=a['features'][i:i+block];bb=b['features'][i:i+block]
   assert np.isfinite(aa).all() and np.isfinite(bb).all()
   exact[i:i+len(aa)]=np.all(aa==bb,axis=1)
   aa=aa.astype(np.float64);bb=bb.astype(np.float64);diff=bb-aa
   absdiff=np.abs(diff);max_columns[i:i+len(aa)]=np.argmax(absdiff,axis=1)
   rr,cc=np.unravel_index(int(np.argmax(absdiff)),absdiff.shape);val=float(absdiff[rr,cc])
   if max_location is None or val>maxabs:
    max_location={'row':int(i+rr),'column':int(cc),'coord_xy':coords[i+rr].tolist(),'reference_value':float(aa[rr,cc]),'candidate_value':float(bb[rr,cc]),'candidate_minus_reference':float(diff[rr,cc]),'indexing':'zero-based row/feature column','tie_rule':'first original row then first feature column, including zero-error case'}
   sse=np.sum(diff*diff,axis=1);ar=np.sum(aa*aa,axis=1);br=np.sum(bb*bb,axis=1);dot=np.sum(aa*bb,axis=1)
   rn=np.sqrt(ar);cn=np.sqrt(br);al=np.sqrt(sse)
   values={'SSE':sse,'absolute_L2':al,'reference_norm':rn,'candidate_norm':cn,
    'relative_L2':np.divide(al,rn,out=np.full(len(aa),np.nan),where=rn>0),
    'cosine':np.divide(dot,rn*cn,out=np.full(len(aa),np.nan),where=(rn*cn)>0),
    'MAE':np.mean(np.abs(diff),axis=1),'RMSE':np.sqrt(np.mean(diff*diff,axis=1)),'max_abs':np.max(np.abs(diff),axis=1)}
   for k,v in values.items():fields[k][i:i+len(aa)]=v
   refsq+=float(ar.sum());candsq+=float(br.sum());dot_total+=float(dot.sum());abs_sum+=float(np.abs(diff).sum());maxabs=max(maxabs,float(np.abs(diff).max()))
  sse=float(fields['SSE'].sum());k=math.ceil(.01*n)
  # Deterministic top-SSE: descending SSE, ascending row index breaks boundary ties; exactly k rows.
  order=np.lexsort((np.arange(n),-fields['SSE']));top=order[:k]
  finite=np.flatnonzero(np.isfinite(fields['cosine']));worst=finite[np.lexsort((finite,fields['cosine'][finite]))[:5]]
  def row(idx):
   def safe(v):return float(v) if np.isfinite(v) else None
   return {'row':int(idx),'coord_xy':coords[idx].tolist(),'features_exact_equal':bool(exact[idx]),**{key:safe(v[idx]) for key,v in fields.items()}}
  def distribution(v):
   vv=v[np.isfinite(v)]
   return {'defined':int(len(vv)),'undefined':int(len(v)-len(vv)),
    'quantiles':dict(zip(['min','p01','p05','median','p95','p99','max'],map(float,np.quantile(vv,[0,.01,.05,.5,.95,.99,1])))) if len(vv) else None}
  result={'reference':str(reference),'candidate':str(candidate),'direction':direction,
   'shape':[n,d],'features_storage_dtype':'float32','coords_row_equal':True,'features_exact_equal':bool(exact.all()),
   'H5_binary_equality':'NOT_TESTED_NOT_IMPLIED','MAE':abs_sum/(n*d),'RMSE':math.sqrt(sse/(n*d)),
   'max_abs':maxabs,'max_abs_location':max_location,'absolute_L2':math.sqrt(sse),'relative_L2':math.sqrt(sse/refsq) if refsq>0 else None,
   'global_cosine':dot_total/math.sqrt(refsq*candsq) if refsq>0 and candsq>0 else None,
   'reference_norm':math.sqrt(refsq),'candidate_norm':math.sqrt(candsq),
   'norm_ratio_candidate_over_reference':math.sqrt(candsq/refsq) if refsq>0 else None,
   'reference_zero_norm_rows':int((fields['reference_norm']==0).sum()),'candidate_zero_norm_rows':int((fields['candidate_norm']==0).sum()),
   'tile_distributions':{key:distribution(v) for key,v in fields.items()},'lowest_cosine_rows':list(map(row,worst)),
   'highest_SSE_rows':list(map(row,order[:5])),
   'top_SSE_fraction':{'N':n,'k':k,'k_over_N':k/n,'SSE_total':sse,'selected_SSE':float(fields['SSE'][top].sum()),
    'contribution':float(fields['SSE'][top].sum())/sse if sse>0 else None,
    'rule':'ceil(.01*N), descending SSE; row-index ascending boundary tie break; zero total SSE => undefined'},
   'interpretation':'Descriptive cross-session precision-policy comparison; no controlled speedup or scientific equivalence threshold',
   'CPU_analysis_wall_s':time.perf_counter()-begin}
  np.savez_compressed(destination/'Per_Tile_Error.npz',row=np.arange(n),coords=coords,features_exact_equal=exact,max_abs_column=max_columns,**fields)
  (destination/'Metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
  return result
