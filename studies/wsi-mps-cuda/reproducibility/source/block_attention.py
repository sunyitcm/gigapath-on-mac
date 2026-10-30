# Legacy flash_attn_func interface adapted from GigaPath's vendored TorchScale
# component/flash_attention.py: Copyright (c) 2023 Microsoft, MIT license.
# The blocked recurrence is the retained project adapter, not the upstream CUDA kernel.
# Algorithm context: https://arxiv.org/abs/2205.14135 (not proof of historical copying).
# This particular archived adapter accepts CPU/MPS only; see docs/LICENSE_SCOPE.md.
"""Source BLOCK-FP32: exact selected definitions. See docs/SOURCE_STATUS.md.
No historical dispatch or scientific execution is invoked on import.
"""
import math
import torch
QUERY_CHUNK=64
KEY_CHUNK=128
COUNTERS={'attempted':0,'completed':0}

@torch.no_grad()
def block_attention(q, k, v, dropout=0.0, bias=None, softmax_scale=None,
                    is_causal=False, query_chunk=QUERY_CHUNK, key_chunk=KEY_CHUNK,
                    telemetry=None):
    COUNTERS['attempted'] += 1
    if dropout != 0.0 or bias is not None or is_causal:
        raise NotImplementedError('Only frozen inference: dropout=0, no bias/mask, noncausal')
    if q.device.type not in ('cpu','mps'):
        raise RuntimeError('Explicit CPU/MPS adapter only; no device fallback')
    if any(x.device != q.device for x in (k,v)):
        raise ValueError('Q/K/V devices must match')
    if any(x.dtype != torch.float32 for x in (q,k,v)):
        raise TypeError('This adapter contract is FP32 only')
    if any(x.requires_grad for x in (q,k,v)):
        raise ValueError('Training/autograd inputs not supported')
    if q.ndim!=4 or k.ndim!=4 or v.shape!=k.shape:
        raise ValueError('Expected Q[B,Q,H,D], K/V[B,K,H,D]')
    b,ql,h,d=q.shape
    if (k.shape[0],k.shape[2],k.shape[3])!=(b,h,d) or min(b,ql,k.shape[1],h,d)<=0:
        raise ValueError('Incompatible or empty Q/K/V shape')
    if not isinstance(query_chunk,int) or not isinstance(key_chunk,int) or min(query_chunk,key_chunk)<=0:
        raise ValueError('Positive integer block sizes required')
    scale=d**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale):raise ValueError('Non-finite softmax scale')
    qh=q.permute(0,2,1,3);kh=k.permute(0,2,1,3);vh=v.permute(0,2,1,3)
    result=torch.empty((b,ql,h,d),device=q.device,dtype=q.dtype)
    lse=torch.empty((b,h,ql),device=q.device,dtype=q.dtype)
    largest=0;blocks=0
    for qs in range(0,ql,query_chunk):
        qe=min(qs+query_chunk,ql);qn=qe-qs
        maximum=torch.full((b,h,qn),-float('inf'),device=q.device,dtype=q.dtype)
        denominator=torch.zeros((b,h,qn),device=q.device,dtype=q.dtype)
        numerator=torch.zeros((b,h,qn,d),device=q.device,dtype=q.dtype)
        for ks in range(0,k.shape[1],key_chunk):
            ke=min(ks+key_chunk,k.shape[1])
            scores=torch.matmul(qh[:,:,qs:qe,:],kh[:,:,ks:ke,:].transpose(-1,-2))*scale
            new_max=torch.maximum(maximum,scores.amax(dim=-1))
            correction=torch.exp(maximum-new_max)
            probabilities_unnormalized=torch.exp(scores-new_max.unsqueeze(-1))
            denominator=denominator*correction+probabilities_unnormalized.sum(dim=-1)
            numerator=numerator*correction.unsqueeze(-1)+torch.matmul(probabilities_unnormalized,vh[:,:,ks:ke,:])
            maximum=new_max
            largest=max(largest,scores.numel());blocks+=1
        result[:,qs:qe,:,:]=(numerator/denominator.unsqueeze(-1)).permute(0,2,1,3)
        lse[:,:,qs:qe]=maximum+torch.log(denominator)
    COUNTERS['completed'] += 1
    if telemetry is not None:
        telemetry.update(device=str(q.device),dtype=str(q.dtype),shape_q=list(q.shape),
                         shape_k=list(k.shape),query_chunk=query_chunk,key_chunk=key_chunk,
                         score_blocks=blocks,max_score_block_elements=largest,
                         max_score_block_bytes=largest*q.element_size(),
                         global_dense_matrix_allocated=False)
    return result,lse

def flash_attn_func(q,k,v,dropout=0.0,bias=None,softmax_scale=None,is_causal=False):
    """Original interface: output[B,Q,H,D], natural-log LSE[B,H,Q]."""
    return block_attention(q,k,v,dropout,bias,softmax_scale,is_causal)
