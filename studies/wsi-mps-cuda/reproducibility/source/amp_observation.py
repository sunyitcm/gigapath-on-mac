"""Source AMP-amp_observation.py: exact selected definitions. See docs/SOURCE_STATUS.md.
No historical dispatch or scientific execution is invoked on import.
"""
import hashlib, inspect, sys


def metadata(v):
 if hasattr(v,'dtype'):return dict(dtype=str(v.dtype),device=str(v.device),shape=list(v.shape))
 if isinstance(v,(tuple,list)):return [metadata(x) for x in v]
 if isinstance(v,dict):return {str(k):metadata(x) for k,x in v.items()}
 return dict(type=type(v).__name__)

class WarmupObservation:
 def __init__(self,t,model):
  self.t=t;self.rows=[];self.operators=[];self.handles=[];self.call=0;self.baseline={n:(len(m._forward_hooks),len(m._forward_pre_hooks)) for n,m in model.named_modules()};self.model=model
  wanted=['patch_embed.proj','blocks.0.norm1','blocks.0.attn.qkv','blocks.0.attn.proj','blocks.0.mlp.fc1','norm']
  modules=dict(model.named_modules())
  for name in wanted:
   assert name in modules,name
   m=modules[name]
   source=inspect.getsource(type(m).forward)
   def hook(module,args,out,name=name,source=source):
    self.operators.append(dict(warmup_call=self.call,module=name,operator_module=type(module).__module__+'.'+type(module).__name__,
     forward_source_sha256=hashlib.sha256(source.encode()).hexdigest(),autocast_enabled=t.is_autocast_enabled('mps'),
     autocast_dtype=str(t.get_autocast_dtype('mps')),input=metadata(args),output=metadata(out)))
   self.handles.append(m.register_forward_hook(hook))
 def remove(self):
  for h in self.handles:h.remove()
  self.handles=[]
  assert self.baseline=={n:(len(m._forward_hooks),len(m._forward_pre_hooks)) for n,m in self.model.named_modules()}
 def validate(self):
  assert len(self.rows)==10 and len(self.operators)==60
  assert all(x['autocast_enabled'] and x['autocast_dtype']=='torch.float16' and x['input']['dtype']=='torch.float32' and x['input']['device'].startswith('mps') for x in self.rows)
  low=[x for x in self.operators if x['operator_module'].endswith(('.Linear','.Conv2d')) and x['output'].get('dtype')=='torch.float16' and x['output']['device'].startswith('mps') and x['autocast_enabled']]
  assert {x['warmup_call'] for x in low}==set(range(1,11)),'No representative low-precision operator evidence'
 def report(self):return dict(warmup_metadata=self.rows,representative_operator_modules=self.operators,diagnostic_hooks_removed=not self.handles,
  observation_scope='Metadata only, existing warmups; no tensor copies/reductions/sync in hooks, no extra forward. Conv2d/Linear/LayerNorm module boundaries; no kernel/accumulator claim.',
  actual_kernel_dispatch='UNKNOWN',internal_accumulator_dtype='UNKNOWN')

class CheckpointObserver:
 """Passive profile of the unmodified original torch.load call, no wrapper."""
 def __init__(self,t):self.code=t.load.__code__;self.paths=[];self.active={}
 def profile(self,frame,event,arg):
  if frame.f_code!=self.code:return
  if event=='call':self.active[id(frame)]=str(frame.f_locals.get('f'))
  elif event=='return' and id(frame) in self.active:
   path=self.active.pop(id(frame))
   if arg is not None:self.paths.append(path)
 def __enter__(self):
  self.previous=sys.getprofile();assert self.previous is None
  sys.setprofile(self.profile);return self
 def __exit__(self,*args):sys.setprofile(self.previous)
