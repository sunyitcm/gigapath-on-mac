"""Source mps_cross104-multi_stage_worker.py: exact selected definitions. See docs/SOURCE_STATUS.md.
No historical dispatch or scientific execution is invoked on import.
"""
import os, argparse, contextlib, importlib, json, sys, time, traceback
from pathlib import Path
# RT, verified runtime identity, observers and lifecycle binding belong to retained orchestration.
def verify_runtime():
    raise RuntimeError('Archival source excerpt: runtime/third-party dependencies are not bundled. No dispatch.')


@contextlib.contextmanager
def tokenizer_binding(transformers,snapshot,state):
 cls=transformers.PreTrainedTokenizerFast;descriptor=cls.__dict__.get('from_pretrained');original=cls.from_pretrained
 def local(_cls,name,*args,**kwargs):
  if str(name)!='MahmoodLab/TITAN':raise ValueError('Unexpected tokenizer target')
  kwargs['local_files_only']=True;state.setdefault('tokenizer_bindings',[]).append(dict(requested=str(name),resolved=str(snapshot)))
  return original(str(snapshot),*args,**kwargs)
 cls.from_pretrained=classmethod(local)
 try:yield
 finally:
  if descriptor is None:delattr(cls,'from_pretrained')
  else:cls.from_pretrained=descriptor

def run(stage,route,binding,output):
 verify_runtime();c=read(RT/'Contract.json');plan=c['stages'][stage];b=read(binding)
 if not c['allow_execution'] or route not in plan['routes']:raise PermissionError('Not an authorized route')
 device='mps' if route in ['A','T1'] else 'cuda';amp=route in ['C','T3','D_hybrid']
 if route in ['T2','T3'] and not plan['remote_paths_executable']:raise RuntimeError('Frozen TITAN remote dependencies missing; no model dispatch')
 out=Path(output);out.mkdir(parents=True,exist_ok=False)
 state=dict(status='STARTING',stage=stage,route=route,device=device,argv=sys.argv,pid=os.getpid(),started_at=now(),
   model_initialization_attempts=0,model_initialization_completions=0,checkpoint_restore_attempts=0,checkpoint_restore_completions=0,
   slide_forward_attempts=0,slide_forward_completions=0,accepted_outputs=0,tile_forward=0,warmup=0,results=[],phase_records=[],
   OS_exit='SEE_PARENT_WAIT',worker_self_exit=None,contract_SHA256=sha(RT/'Contract.json'),runtime_SHA256=sha(RT/'Runtime_Identity.json'),
   parameters_dtype='FP32',original_input_dtype='FP32',autocast='FP16' if amp else 'OFF')
 def save():state['updated_at']=now();atomic(out/'Worker_Receipt.json',state)
 save();monitor=None;torch=None;tworker=time.perf_counter()
 try:
  import torch,numpy as np,h5py,psutil,timm,importlib.metadata as md
  proc=psutil.Process();cpu_start=proc.cpu_times()
  if str(Path(sys.executable).absolute())!=b['python']:raise RuntimeError('Interpreter differs from binding')
  for pkg,version in b['exact_packages'].items():
   if md.version(pkg)!=version:raise RuntimeError('Frozen dependency changed: '+pkg)
  if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no CPU fallback')
  if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
  if device=='cuda':state['IEEE_TF32_readback']=precision_policy(torch)
  else:
   torch.backends.fp32_precision='ieee';state['IEEE_TF32_readback']={'global':torch.backends.fp32_precision,'MPS_fast_math':os.environ['PYTORCH_MPS_FAST_MATH'],'fallback':os.environ['PYTORCH_ENABLE_MPS_FALLBACK']}
  threads=c['TITAN_threads'] if stage=='stage3' else c['MPS_GigaPath_threads'] if device=='mps' else c['GigaPath_threads']
  torch.set_num_threads(threads)
  if stage=='stage3':torch.set_num_interop_threads(1)
  seed=c['seed_TITAN'] if stage=='stage3' else c['GigaPath_seed'];torch.manual_seed(seed);np.random.seed(seed)
  state['threads']=threads;state['seed']=seed;state['actual_packages']={k:md.version(k) for k in b['exact_packages']}
  available=psutil.virtual_memory().available
  if device=='cuda':
   limit=Path('/sys/fs/cgroup/memory.max').read_text().strip();current=int(Path('/sys/fs/cgroup/memory.current').read_text())
   if limit!='max':available=min(available,int(limit)-current)
   free,total=torch.cuda.mem_get_info();state['GPU_memory_before_model_bytes']={'free':free,'total':total}
   if free<8*1024**3:raise RuntimeError('GPU memory planning guard')
  state['available_host_memory_before_model_bytes']=available
  if available<(24 if plan['model']=='gigapath' else 8)*1024**3:raise RuntimeError('Host memory planning guard')
  monitor=Monitor(torch,psutil,device,out/'Resource_Samples.jsonl');monitor.start()
  def measured(name,fn,task=None):
   monitor.phase=name;u=proc.cpu_times();t=time.perf_counter()
   try:return fn()
   finally:
    wall=time.perf_counter()-t;v=proc.cpu_times();du,ds=v.user-u.user,v.system-u.system
    state['phase_records'].append(dict(phase=name,task_id=task,wall_s=wall,CPU_user_delta_s=du,CPU_system_delta_s=ds,CPU_avg_percent_one_core_100=100*(du+ds)/wall if wall else None))
    monitor.capture('BOUNDARY:'+name);save()
  attention_rows=[];layers=[];projection=[];handles=[]
  if plan['model']=='gigapath':
   source=RT/'sources'/('B' if route=='D_hybrid' else route);sys.path.insert(0,str(source));sys.path.insert(0,str(source/'gigapath'))
   import gigapath.slide_encoder
   from torchscale.component import multihead_attention as mh
   actual_attention=mh.flash_attn_func
   expected_module='torchscale.component.flash_attention' if route=='C' else 'torchscale.component.mps_block_attention'
   if actual_attention.__module__!=expected_module:raise RuntimeError('Attention route conflict')
   def observed_attention(q,k,v,*args,**kwargs):
    expected_dtype=torch.float16 if amp else torch.float32
    if any(x.dtype!=expected_dtype or x.device.type!=device for x in [q,k,v]):raise ValueError('Actual attention QKV violates frozen route')
    if route=='D_hybrid':
     from hybrid_boundary import fp32_attention_boundary
     result,lse=fp32_attention_boundary(actual_attention,q,k,v,*args,**kwargs)
    else:result,lse=actual_attention(q,k,v,*args,**kwargs)
    attention_rows.append(dict(QKV_dtype=str(q.dtype),attention_entry_dtype='torch.float32' if route=='D_hybrid' else str(q.dtype),attention_output_dtype=str(result.dtype),LSE_dtype=str(lse.dtype),device=str(q.device)))
    return result,lse
   mh.flash_attn_func=observed_attention
   state['attention_path']=dict(callable_module=actual_attention.__module__,source=str(source),block_algorithm_unchanged=True)
   cfg=c['GigaPath_model'];kwargs={k:cfg[k] for k in ['in_chans','global_pool','tile_size','slide_ngrids','max_wsi_size','dropout','drop_path_rate']}
   state['model_initialization_attempts']+=1;save()
   model=measured('model_construction_position_initialization',lambda:timm.create_model(cfg['model_arch'],pretrained=False,**kwargs))
   state['model_initialization_completions']+=1
  else:
   import transformers,safetensors.torch
   sys.path.insert(0,str(RT/'sources'))
   mod=importlib.import_module('titan_fixed.modeling_titan');cfgmod=importlib.import_module('titan_fixed.configuration_titan')
   config=cfgmod.TitanConfig.from_dict(read(RT/'titan_snapshot/config.json'))
   state['model_initialization_attempts']+=1;save()
   def create():
    with tokenizer_binding(transformers,RT/'titan_snapshot',state):return mod.Titan(config)
   model=measured('model_construction_fixed_tokenizer',create);state['model_initialization_completions']+=1
  weight=b['weight_paths'][plan['model']]
  if Path(weight).stat().st_size!=plan['weight']['bytes']:raise ValueError('Weight metadata conflict')
  state['checkpoint_restore_attempts']+=1;save()
  received=measured('checkpoint_restore',lambda:torch.load(weight,map_location='cpu',weights_only=True)['model'] if plan['model']=='gigapath' else safetensors.torch.load_file(weight,device='cpu'))
  state['checkpoint_restore_completions']+=1
  expected=model.state_dict();params=dict(model.named_parameters())
  gate=dict(missing=sorted(set(expected)-set(received)),unexpected=sorted(set(received)-set(expected)),
     wrong_shapes=[k for k in expected.keys()&received.keys() if expected[k].shape!=received[k].shape],
     wrong_dtype=[k for k,v in received.items() if v.is_floating_point() and v.dtype!=torch.float32],
     expected_keys=len(expected),received_keys=len(received),effective_parameter_elements=sum(x.numel() for x in params.values()))
  atomic(out/'Weight_Gate.json',gate)
  if any(gate[k] for k in ['missing','unexpected','wrong_shapes','wrong_dtype']):raise ValueError('Full effective weight mismatch')
  def load_values():
   model.load_state_dict(received,strict=True)
   unequal=[k for k,v in model.state_dict().items() if not torch.equal(v,received[k])]
   if unequal:raise ValueError('Post-load values differ')
   gate.update(strict=True,coverage=1.0,post_load_unequal_keys=[]);atomic(out/'Weight_Gate.json',gate)
  measured('strict_load_full_value_verification',load_values)
  del received,expected,params
  model.requires_grad_(False).eval()
  def migrate():
   model.to(device=device,dtype=torch.float32)
   (torch.cuda.synchronize if device=='cuda' else torch.mps.synchronize)()
  measured('device_migration_and_sync',migrate)
  if any(p.dtype!=torch.float32 or p.device.type!=device for p in model.parameters()):raise ValueError('Parameter dtype/device conflict')
  if plan['model']=='gigapath':
   if len(model.encoder.layers)!=12 or model.global_pool:raise ValueError('Depth/CLS conflict')
   for index,layer in enumerate(model.encoder.layers):
    def observer(module,args,result,i=index):
     value=result[0];layers.append(dict(index=i,shape=list(value.shape),dtype=str(value.dtype),device=str(value.device)));save()
    handles.append(layer.register_forward_hook(observer))
   handles.append(model.patch_embed.register_forward_hook(lambda m,a,v:projection.append(dict(input_dtype=str(a[0].dtype),output_dtype=str(v.dtype)))))
  else:
   from observation_and_save import register_observers,validate_observations
   def describe(v):
    vv=[v] if isinstance(v,torch.Tensor) else [x for x in v if isinstance(x,torch.Tensor)]
    return [dict(shape=list(x.shape),device=str(x.device),dtype=str(x.dtype)) for x in vv]
   handles+=register_observers(model.vision_encoder,state,describe,expected_blocks=6)
   for name,mod in model.vision_encoder.named_modules():
    if isinstance(mod,torch.nn.Linear):
     handles.append(mod.register_forward_hook(lambda m,a,v,n=name:projection.append(dict(module=n,input_dtype=str(a[0].dtype),output_dtype=str(v.dtype)))))
  inputmap={x['id']:x for x in plan['inputs']}
  with torch.no_grad():
   for task in plan['tasks']:
    while (out.parents[2]/'Pause_After_Task.json').exists():
     state['status']='PAUSED_AT_COMPLETE_TASK_BOUNDARY_MODEL_RESIDENT';save();time.sleep(5)
    ttask=time.perf_counter();task_cpu=proc.cpu_times()
    item=inputmap[task['input_id']];tid=task['id'];state['current_task']=tid;state['status']='READING_INPUT';save()
    layers.clear();attention_rows.clear();projection.clear();state['layer_observations']=[];state['observation_errors']=[]
    def load_h5():
     path=b['input_paths'][item['id']]
     expected_stat=b['input_stats'][item['id']]
     if stat_identity(path)!=expected_stat:raise ValueError('Accepted input metadata changed')
     with h5py.File(path,'r') as h:f=h['features'][:];co=h['coords'][:]
     if f.shape!=(item['N'],item['input_dim']) or f.dtype!=np.float32 or not np.isfinite(f).all():raise ValueError('Complete features rejected')
     if co.shape!=(item['N'],2) or co.dtype!=np.int32:raise ValueError('Complete coords rejected')
     if plan['model']=='gigapath':
      native=np.load(RT/item['coords_relative'],allow_pickle=False)
      if not np.array_equal((co.astype(np.int64)//item['actual_read_px'])*256,native) or np.any(co%item['actual_read_px']):raise ValueError('Frozen actual lattice conflict')
      return f,native.astype(np.float32)
     if np.any(co%item['actual_read_px']) or np.any(~np.any(f!=0,axis=1)):raise ValueError('TITAN spatial/foreground contract conflict')
     return f,co.astype(np.int64)
    features,coords=measured('H5_read_full_input_checks',load_h5,tid)
    def submit():return torch.from_numpy(features)[None].to(device),torch.from_numpy(coords)[None].to(device)
    xf,xc=measured('input_prepare_device_submit_call',submit,tid)
    peak={}
    if device=='cuda':
     peak['live_baseline_allocated_bytes']=torch.cuda.memory_allocated();peak['live_baseline_reserved_bytes']=torch.cuda.memory_reserved()
     torch.cuda.reset_peak_memory_stats();peak['reset_boundary']='after input submission, immediately before forward; live model/input baseline included'
    state['slide_forward_attempts']+=1;state['status']='RUNNING_FULL_SLIDE';save()
    def forward():
     with torch.autocast(device_type=device,dtype=torch.float16,enabled=amp):
      v=model(xf,xc,all_layer_embed=False) if plan['model']=='gigapath' else model.encode_slide_from_patch_features(xf,xc,item['actual_read_px'])
     (torch.cuda.synchronize if device=='cuda' else torch.mps.synchronize)()
     return v[0] if plan['model']=='gigapath' else v
    vector=measured('complete_forward_final_sync',forward,tid);state['slide_forward_completions']+=1
    if device=='cuda':peak.update(max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),max_memory_reserved_bytes=torch.cuda.max_memory_reserved())
    if vector.shape!=(1,768) or vector.device.type!=device:raise ValueError('Full output shape/device conflict')
    raw=measured('CPU_output_readback',lambda:vector.detach().cpu().numpy(),tid)
    if not np.isfinite(raw).all():raise ValueError('Nonfinite complete output')
    dest=out/(tid+'.npy')
    def serialize():
     with dest.open('xb') as f:np.save(f,raw.astype(np.float32),allow_pickle=False);f.flush();os.fsync(f.fileno())
    measured('output_save_close',serialize,tid)
    row=dict(task_id=tid,input_id=item['id'],repeat=task['repeat'],status='SAVED_PENDING_OBSERVATION_ACCEPTANCE',
      output=dest.name,bytes=dest.stat().st_size,shape=[1,768],raw_output_dtype=str(raw.dtype),storage_dtype='float32',
      full_input_tiles=item['N'],device_before_readback=str(vector.device),input_dtype='float32',parameters_dtype='float32',
      projection_dtype_observations=list(projection),attention_dtype_observations=list(attention_rows),layers=list(layers),CUDA_forward_allocator_peak=peak)
    row['sha256']=measured('output_SHA256',lambda:sha(dest),tid)
    state['results'].append(row);save() # durable vector before optional observation acceptance
    if plan['model']=='gigapath':
     if route=='D_hybrid' and (not any(x['output_dtype']=='torch.float16' for x in projection) or any(x['attention_entry_dtype']!='torch.float32' or x['attention_output_dtype']!='torch.float32' or x['LSE_dtype']!='torch.float32' for x in attention_rows)):raise ValueError('D_hybrid precision boundary not satisfied')
     if len(attention_rows)!=60 or [x['index'] for x in layers]!=list(range(12)):raise ValueError('Full layer/attention completeness')
     if any(x['shape']!=[1,item['N']+1,768] or not x['device'].startswith(device) for x in layers):raise ValueError('Complete token sequence/device failed')
     if not amp and (raw.dtype!=np.float32 or any(x['dtype']!='torch.float32' for x in layers)):raise ValueError('FP32 route output differs')
    else:
     if not amp:validate_observations(state,full_N=item['N'],device=str(vector.device),dtype='torch.float32')
     else:
      names=[x['name'] for x in state['layer_observations']]
      if any(names.count('blocks.modules_list.'+str(i))!=1 for i in range(6)):raise ValueError('TITAN complete block observations missing')
      if not any(x['output_dtype']=='torch.float16' for x in projection):raise ValueError('Autocast did not change observed projection precision')
     row['TITAN_layer_observations']=list(state['layer_observations'])
    row['status']='ACCEPTED';row['task_wall_s']=time.perf_counter()-ttask
    endcpu=proc.cpu_times();row['task_CPU_user_s']=endcpu.user-task_cpu.user;row['task_CPU_system_s']=endcpu.system-task_cpu.system
    row['task_CPU_average_percent_one_core_100']=100*(row['task_CPU_user_s']+row['task_CPU_system_s'])/row['task_wall_s']
    state['accepted_outputs']+=1;save()
    atomic(out/(tid+'.Completion.json'),row)
    if task==plan['tasks'][0]:atomic(out/'First_Full_Input_Acceptance.json',row)
    print(json.dumps(dict(task=tid,status='ACCEPTED',route=route)),flush=True)
    del features,coords,xf,xc,vector,raw
  state.update(status='COMPLETE',worker_self_exit=0)
 except BaseException as e:
  state.update(status='FAILED_STOPPED_NO_RETRY',worker_self_exit=1,error=repr(e),traceback=traceback.format_exc())
  if state['results'] and state['results'][-1]['status']!='ACCEPTED':state['results'][-1]['status']='FAILED_ADDITIONAL_CHECKS_VECTOR_PRESERVED'
  raise
 finally:
  state['finished_at']=now();state['worker_wall_s']=time.perf_counter()-tworker
  if monitor:
   try:state['resources']=monitor.finish()
   except Exception as observation_error:state['final_resource_observation_error']=repr(observation_error)
  save()
