"""Retained SSE-MATH definitions; independent historical contract, not a new computation."""
import numpy as np, math

def row_metrics(a,b):
 d=b-a;sse=np.einsum('ij,ij->i',d,d);an=np.sqrt(np.einsum('ij,ij->i',a,a));bn=np.sqrt(np.einsum('ij,ij->i',b,b))
 absolute=np.sqrt(sse);rel=np.full(len(a),np.nan);np.divide(absolute,an,out=rel,where=an!=0)
 ratio=np.full(len(a),np.nan);np.divide(bn,an,out=ratio,where=an!=0)
 cosine=np.full(len(a),np.nan);np.divide(np.einsum('ij,ij->i',a,b),an*bn,out=cosine,where=(an!=0)&(bn!=0))
 return sse,absolute,rel,an,bn,ratio,cosine

def rank_and_contribution(sse,fraction):
 n=len(sse);rows=np.arange(n);order=np.lexsort((rows,-sse));k=math.ceil(fraction*n);chosen=order[:k];total=float(np.sum(sse,dtype=np.float64));cut=float(sse[chosen[-1]])
 return {'fraction_planned':fraction,'k':k,'actual_k_over_N':k/n,'total_SSE':total,'selected_SSE_sum':float(np.sum(sse[chosen],dtype=np.float64)),
 'contribution':float(np.sum(sse[chosen],dtype=np.float64))/total if total else None,'status':'DEFINED' if total else 'UNDEFINED_ZERO_TOTAL_SSE',
 'cutoff_SSE':cut,'all_cutoff_ties':int((sse==cut).sum()),'selected_cutoff_ties':int((sse[chosen]==cut).sum()),'rows':chosen.tolist()}

def vector_metrics(a,b):
 d=b-a;an=float(np.linalg.norm(a));bn=float(np.linalg.norm(b));ab=float(np.linalg.norm(d))
 return dict(reference_norm=an,compared_norm=bn,norm_ratio=bn/an if an else None,absolute_L2=ab,relative_L2=ab/an if an else None,
  cosine=float(np.dot(a,b)/(an*bn)) if an and bn else None,MAE=float(np.mean(np.abs(d))),max_abs=float(np.max(np.abs(d))),elementwise_equal=bool(np.array_equal(a,b)))

def verify_coords(a,b):
 if a.shape!=b.shape or not np.array_equal(a,b):raise ValueError('COORDINATE_CONFLICT_NO_SORT_TRUNCATE_OR_INTERSECTION')
