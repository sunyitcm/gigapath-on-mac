"""Retained HIST-COMPARE definitions; independent historical contract, not a new computation."""
import math
import numpy as np
import h5py
CHUNK_ROWS=512

def compare_h5(pa,pb,ra,rb):
    r={'slide':ra['slide'],'encoder':ra['encoder'],'dtype':ra['dtype'],
       'filename':ra['source_filename'],'coords_sha_equal':ra.get('coords_sha256')==rb.get('coords_sha256'),
       'features_sha_equal':ra.get('features_sha256')==rb.get('features_sha256'),
       'shape_equal':ra.get('features_shape')==rb.get('features_shape'),
       'features_dtype_equal':ra.get('features_dtype')==rb.get('features_dtype'),
       'M5_features_sha256':ra.get('features_sha256'),'M3_features_sha256':rb.get('features_sha256'),
       'M5_coords_sha256':ra.get('coords_sha256'),'M3_coords_sha256':rb.get('coords_sha256')}
    if ra['status']!='ok' or rb['status']!='ok':
        r['status']='FAIL_H5_STRUCTURE';return r
    if not (r['coords_sha_equal'] and r['shape_equal'] and r['features_dtype_equal']):
        r['status']='BLOCK_UNALIGNED';return r
    if r['features_sha_equal']:
        r.update({'status':'EXACT_STORED_FEATURE_EQUAL','global_cosine':1.0,
             'relative_l2':0.0,'mae':0.0,'max_abs':0.0,'tile_cos_p01':1.0,'tile_cos_p05':1.0,
             'tile_cos_median':1.0,'tile_cos_min':1.0})
        return r
    sa2=sb2=sab=sd2=sabs=0.;n=0;maxdiff=0.;tile_cos=[]
    with h5py.File(pa,'r') as h1,h5py.File(pb,'r') as h2:
        da,db=h1['features'],h2['features']
        for i in range(0,len(da),CHUNK_ROWS):
            a=np.asarray(da[i:i+CHUNK_ROWS],dtype=np.float64)
            b=np.asarray(db[i:i+CHUNK_ROWS],dtype=np.float64)
            d=a-b
            n+=d.size
            sa2+=float(np.sum(a*a));sb2+=float(np.sum(b*b));sab+=float(np.sum(a*b))
            sd2+=float(np.sum(d*d));sabs+=float(np.sum(np.abs(d)))
            if d.size: maxdiff=max(maxdiff,float(np.max(np.abs(d))))
            norms=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
            cos=np.divide(np.sum(a*b,axis=1),norms,out=np.full(len(a),np.nan),where=norms>0)
            tile_cos.extend(cos[np.isfinite(cos)].tolist())
    r.update({'status':'NUMERICALLY_COMPARED','global_cosine':sab/math.sqrt(sa2*sb2) if sa2*sb2 else None,
      'relative_l2':math.sqrt(sd2/sa2) if sa2 else None,'mae':sabs/n if n else None,'max_abs':maxdiff})
    if tile_cos:
        r.update({'tile_cos_p01':float(np.percentile(tile_cos,1)),
                  'tile_cos_p05':float(np.percentile(tile_cos,5)),
                  'tile_cos_median':float(np.median(tile_cos)),
                  'tile_cos_min':float(min(tile_cos))})
    return r
