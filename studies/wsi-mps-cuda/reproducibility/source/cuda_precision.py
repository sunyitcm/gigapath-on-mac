"""Source CUDA-Core5-precision_policy.py: exact selected definitions. See docs/SOURCE_STATUS.md.
No historical dispatch or scientific execution is invoked on import.
"""
import inspect, sys
from pathlib import Path


class PrecisionPath:
 def __init__(self,torch,model,postprocess,sync,policy):
  assert policy in ('CUDA_EXPLICIT_FP16_AUTOCAST','CUDA_AMP_FP16')
  self.policy=policy;self.expected_input_dtype=torch.float16 if policy=='CUDA_EXPLICIT_FP16_AUTOCAST' else torch.float32
  self.torch=torch;self.model=model;self.post=postprocess;self.sync=sync
  self.observing=True;self.observations=[]
 def __call__(self,x):
  t=self.torch
  assert x.dtype==self.expected_input_dtype and x.device.type=='cuda'
  observe=self.observing and len(self.observations)<10
  with t.autocast(device_type='cuda',dtype=t.float16,enabled=True):
   raw=self.model(x)
   if observe:
    # Metadata only; no tensor copies, reductions, .item(), or new device synchronization.
    row={'warmup_call':len(self.observations)+1,'input_dtype':str(x.dtype),'input_device':str(x.device),
      'autocast_enabled':t.is_autocast_enabled('cuda'),'autocast_dtype':str(t.get_autocast_dtype('cuda')),
      'raw_model_return':tensor_metadata(raw)}
   out=self.post(raw)
   if observe:row['postprocess_return']=tensor_metadata(out)
  self.sync() # exactly the preserved batch-end synchronization, after original postprocess
  if observe:self.observations.append(row)
  return out
 def report(self):
  return {'policy':self.policy,'autocast_scope':'model and original postprocess',
   'parameter_dtype_checked_by_worker':'float16' if self.policy=='CUDA_EXPLICIT_FP16_AUTOCAST' else 'float32','warmup_metadata':self.observations,
   'storage_dtype':'float32','write_conversion':'out.float().cpu().numpy() then H5 float32; conversion does not restore precision',
   'internal_accumulator_dtype':'UNKNOWN','actual_kernel_dispatch':'UNKNOWN',
   'observation_scope':'First ten already planned warmup calls only; no extra forward'}

def tensor_metadata(value):
 if hasattr(value,'dtype') and hasattr(value,'shape'):
  return {'dtype':str(value.dtype),'device':str(value.device),'shape':list(value.shape)}
 if isinstance(value,(tuple,list)):return [tensor_metadata(x) for x in value]
 if isinstance(value,dict):return {str(k):tensor_metadata(v) for k,v in value.items()}
 raise TypeError('Unrecognized model return for bounded dtype observation')

def validate_observation(r):
 assert r['policy'] in ('CUDA_EXPLICIT_FP16_AUTOCAST','CUDA_AMP_FP16')
 expected='torch.float16' if r['policy']=='CUDA_EXPLICIT_FP16_AUTOCAST' else 'torch.float32'
 assert len(r['warmup_metadata'])==10
 def tensors(v):
  if isinstance(v,dict) and 'dtype' in v:yield v
  elif isinstance(v,dict):
   for x in v.values():yield from tensors(x)
  elif isinstance(v,list):
   for x in v:yield from tensors(x)
 for row in r['warmup_metadata']:
  assert row['input_dtype']==expected and row['input_device'].startswith('cuda')
  assert row['autocast_enabled'] is True and row['autocast_dtype']=='torch.float16'
  observed=list(tensors(row['raw_model_return']))+list(tensors(row['postprocess_return']))
  assert observed and all(x['device'].startswith('cuda') and x['dtype'] in ('torch.float16','torch.float32') for x in observed)

class RestoreObserver:
 """Observe successful Python load APIs only during loader; do not replace loaders.

 Reports unique actual weight paths of completed deserialization/restore API calls,
 not physical read/syscall count. Nested calls deduplicate paths. A meta-only load
 is excluded. No observed successful path is UNKNOWN, never the planned count.
 """
 def __init__(self,apis):
  self.apis={getattr(fn,'__func__',fn).__code__:(label,arg) for label,fn,arg in apis}
  self.active={};self.events=[];self.paths=set();self.previous=None
 @classmethod
 def for_libraries(cls,torch):
  import safetensors.torch
  from transformers import PreTrainedModel
  return cls([('torch.load',torch.load,'f'),('safetensors.torch.load_file',safetensors.torch.load_file,'filename'),
   ('transformers._load_pretrained_model',PreTrainedModel._load_pretrained_model,'checkpoint_files')])
 def _profile(self,frame,event,arg):
  spec=self.apis.get(frame.f_code)
  if spec is None:return
  if event=='call':
   label,key=spec;value=frame.f_locals.get(key)
   values=value if isinstance(value,(list,tuple)) else [value]
   paths=[str(Path(p).resolve()) for p in values if isinstance(p,(str,Path)) and Path(p).suffix in ('.bin','.pth','.pt','.safetensors')]
   meta=str(frame.f_locals.get('map_location',''))=='meta'
   row={'api':label,'actual_paths':paths,'meta_only':meta,'returned_non_null':False,'status':'ATTEMPTED'}
   self.events.append(row);self.active[id(frame)]=row
  elif event=='return' and id(frame) in self.active:
   row=self.active.pop(id(frame));row.update(returned_non_null=arg is not None,
    status='RETURNED_NON_NULL' if arg is not None else 'FAILED_OR_NULL_RETURN')
   if arg is not None and not row['meta_only']:self.paths.update(row['actual_paths'])
 def __enter__(self):
  self.previous=sys.getprofile()
  if self.previous is not None:raise RuntimeError('Existing load profiler would conflict; no silent replacement')
  sys.setprofile(self._profile);return self
 def __exit__(self,*exc):sys.setprofile(self.previous)
 def report(self):
  return {'successful_unique_weight_files':len(self.paths) if self.paths else None,
   'actual_successful_paths':sorted(self.paths),'events':self.events,
   'scope':'Passive current-thread Python load-call observations during model CPU loading only; not file-open count',
   'unobserved_is_UNKNOWN':True,'planned_count_substituted':False}
