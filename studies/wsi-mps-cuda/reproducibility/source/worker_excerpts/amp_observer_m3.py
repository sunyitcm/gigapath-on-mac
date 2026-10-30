"""Source M3AMP-amp_observer.py: exact selected definitions. See docs/SOURCE_STATUS.md.
No historical dispatch or scientific execution is invoked on import.
"""
from pathlib import Path
import contextlib, datetime, importlib.util, inspect, json, os, resource, sys, time, traceback, hashlib
from source.timing import traverse, ResourceSampler


class WarmupObserver:
 def __init__(self,model,torch,job):
  self.torch=torch;self.job=Path(job);self.active=True;self.closed=False;self.layers={};self.handles=[]
  self.raw_calls={'warmup':{'attempted':0,'completed':0},'output':{'attempted':0,'completed':0}}
  self.record={'policy':'MPS_AMP_FP16','status':'PENDING','parameter_dtypes':sorted({str(p.dtype) for p in model.parameters()}),
   'floating_buffer_dtypes':sorted({str(b.dtype) for b in model.buffers() if b.is_floating_point()}),
   'fallback_environment':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),'warmup_batches':[],
   'first_batch_input_sha_scope':'Raw contiguous little-endian FP32 tensor bytes for first preprocessed submitted warmup batch only; not whole WSI byte proof',
   'extra_forwards':0,'formal_output_hooks_enabled':False}
  assert self.record['parameter_dtypes']==['torch.float32']
  assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
  warnings.filterwarnings('error',message=r'.*(fallback|fall back|falling back).*',category=UserWarning)
  modules=dict(model.named_modules())
  for name in ['patch_embed.proj','blocks.0.attn.qkv','blocks.0.mlp.fc1','norm']:
   if name in modules:
    def hook(module,args,out,key=name):
     values=[v for v in (out if isinstance(out,(tuple,list)) else [out]) if isinstance(v,torch.Tensor)]
     self.layers[key]=[{'dtype':str(v.dtype),'device':str(v.device)} for v in values]
     assert all(v.device.type=='mps' for v in values),'CPU layer output observed'
    self.handles.append(modules[name].register_forward_hook(hook))
  assert len(self.handles)>=2,'Representative layers unavailable; do not change architecture'
  self.record['representative_layers']=[n for n in ['patch_embed.proj','blocks.0.attn.qkv','blocks.0.mlp.fc1','norm'] if n in modules]
 def input(self,x):
  if self.active and 'first_batch_FP32_input' not in self.record:
   assert x.dtype==self.torch.float32 and x.device.type=='mps'
   cpu=x.detach().cpu().contiguous();raw=cpu.numpy().tobytes(order='C')
   self.record['first_batch_FP32_input']={'dtype':str(x.dtype),'device':str(x.device),'shape':list(x.shape),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'byteorder':'little'}
 def output(self,raw,out):
  if not self.active:return
  torch=self.torch
  row={'batch':len(self.record['warmup_batches'])+1,'autocast_enabled':torch.is_autocast_enabled('mps'),
   'autocast_dtype':str(torch.get_autocast_dtype('mps')),'raw_dtype':str(raw.dtype),'postprocess_dtype':str(out.dtype),
   'raw_device':str(raw.device),'postprocess_device':str(out.device),'layers':dict(self.layers)}
  self.record['warmup_batches'].append(row)
  assert row['autocast_enabled'] and row['autocast_dtype']=='torch.float16','Autocast context not enabled as requested'
  assert raw.device.type==out.device.type=='mps','CPU output/fallback observed'
  assert any(v['dtype']=='torch.float16' for values in self.layers.values() for v in values),'AMP did not take effect in representative eligible layers'
  self.layers={}
 def close(self,success=False):
  if self.closed:return
  for h in self.handles:h.remove()
  self.active=False;self.closed=True
  self.record.update(status='PASS_WARMUP_AMP_EFFECTIVE' if success else 'FAILED_OR_INCOMPLETE_WARMUP',
    completed_observed_batches=len(self.record['warmup_batches']),diagnostic_hooks_removed=True)
  if success:assert len(self.record['warmup_batches'])==10
  with (self.job/'AMP_Warmup_Diagnostics.json').open('x') as f:json.dump(self.record,f,indent=2)
