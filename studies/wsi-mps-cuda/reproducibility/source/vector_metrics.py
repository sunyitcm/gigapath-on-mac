"""Source AGG-METRIC: exact selected definitions. See docs/SOURCE_STATUS.md.
No historical dispatch or scientific execution is invoked on import.
"""
import os


def metrics(candidate,reference):
    import numpy as np
    a=np.asarray(candidate,dtype=np.float64).ravel();b=np.asarray(reference,dtype=np.float64).ravel()
    if a.shape!=b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):raise ValueError('Invalid comparison input')
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b));d=a-b;dl=float(np.linalg.norm(d))
    return {'candidate_norm':na,'reference_norm':nb,'norm_ratio_candidate_over_reference':na/nb if nb else None,
            'absolute_L2':dl,'relative_L2_reference_denominator':dl/nb if nb else None,
            'cosine':float(a@b/(na*nb)) if na and nb else None,'MAE':float(np.mean(np.abs(d))),
            'max_abs':float(np.max(np.abs(d))),'elementwise_numerically_equal':bool(np.array_equal(candidate,reference))}

def precision_policy(torch):
    os.environ['NVIDIA_TF32_OVERRIDE']='0'
    torch.backends.fp32_precision='ieee'
    torch.backends.cuda.matmul.fp32_precision='ieee'
    torch.backends.cudnn.fp32_precision='ieee'
    torch.backends.cudnn.benchmark=False
    actual={'global_fp32_precision':torch.backends.fp32_precision,
            'cuda_matmul_fp32_precision':torch.backends.cuda.matmul.fp32_precision,
            'cudnn_fp32_precision':torch.backends.cudnn.fp32_precision,
            'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'NVIDIA_TF32_OVERRIDE':os.environ['NVIDIA_TF32_OVERRIDE']}
    if any(actual[k]!='ieee' for k in ['global_fp32_precision','cuda_matmul_fp32_precision','cudnn_fp32_precision']):
        raise RuntimeError('IEEE/TF32-disabled readback failed')
    return actual

import datetime as dt
def now(): return dt.datetime.now().astimezone().isoformat()
