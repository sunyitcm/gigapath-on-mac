"""Source CUDA-Core5-stage_worker.py: exact selected definitions. See docs/SOURCE_STATUS.md.
No historical dispatch or scientific execution is invoked on import.
"""
from pathlib import Path
import contextlib, datetime, importlib.util, inspect, json, os, resource, sys, time, traceback
from source.timing import traverse, ResourceSampler
from source.cuda_precision import PrecisionPath, RestoreObserver, tensor_metadata, validate_observation
def require_execution(*args, **kwargs):
    raise RuntimeError('Archival source excerpt: private orchestration is not distributed; use component APIs and documented contracts. No dispatch from this excerpt.')


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def real_worker(c,task,job,root):
 c=task_config(c,task)
 require_execution(c,root) # false flags stop before framework import or source loader
 assert c['batch']==8 and c['dtype']=='per_task_policy' and c['warmup_batches']==10
 assert c['inference_policy']=='DUAL_CUDA_FP16' and c['mode']=='FORMAL' and c['backend']=='cuda'
 policy=task['precision_policy'];assert policy in ('CUDA_EXPLICIT_FP16_AUTOCAST','CUDA_AMP_FP16')
 assert c['cpu_threads']==4 and c['cpu_interop_threads']==1 and c['seed']==20260826
 assert Path(sys.executable).resolve()==Path(c['python']).resolve() and sys.version_info[:3]==(3,11,16)
 package=Path(c['package']);sys.path.insert(0,str(package/'scripts'))
 r={'slide_id':task['slide_id'],'input_path':c['slide'],'accepted_slide_sha256':c['accepted_slide_sha256'],'task_id':task['task_id'],'platform':c['platform'],'mode':c['mode'],'attempt_id':task['attempt_id'],'started_utc':utc(),
    'precision_policy':policy,'model_load_attempts':0,'model_loads_completed':0,'checkpoint_weight_files_restored':None,
    'output_forward_attempts':0,'output_forward_completed':0,'warmup_forward_attempts':0,'warmup_forward_completed':0,
    'formal_cuda_execution':c['formal_cuda_execution'],'timing':{},'cache_state':'NOT_CONTROLLED'}
 timing=r['timing'];slide=None;chunks=[];coords=None;begin=time.perf_counter();cpu_begin=resource.getrusage(resource.RUSAGE_SELF);rc=1
 try:
  # Small source/config identity checks outside model/output measurement. Never hash weights/SVS here.
  for name,spec in c['sources'].items():assert sha_small(package/spec['package_relative'])==spec['sha256'],name
  from common import bundle,native_cache_bindings
  from smoke_worker import load_original
  import numpy as np,h5py,openslide,torch
  from PIL import Image
  data=bundle();model_id=task['model_id'];pin=data['models'][model_id]
  r['resource_identity_basis']='Accepted SVS/model receipt mapping plus current metadata precheck; no large rehash'
  r['cache_bindings']=native_cache_bindings(data,Path(c['hub']),model_id)
  torch.set_default_device('cpu');torch.set_default_dtype(torch.float32)
  torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20260826);torch.use_deterministic_algorithms(False)
  backend=c['backend']
  if backend=='cuda':
   torch.backends.fp32_precision='ieee';torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.fp32_precision='ieee';torch.backends.cudnn.benchmark=False
   assert all(x=='ieee' for x in [torch.backends.fp32_precision,torch.backends.cuda.matmul.fp32_precision,torch.backends.cudnn.fp32_precision])
   r['precision_control_readback']={'global_fp32_precision':torch.backends.fp32_precision,'cuda_matmul_fp32_precision':torch.backends.cuda.matmul.fp32_precision,'cudnn_fp32_precision':torch.backends.cudnn.fp32_precision,'cudnn_benchmark':torch.backends.cudnn.benchmark,'API_family':'fp32_precision; no legacy allow_tf32 calls'}
  sync=torch.cuda.synchronize if backend=='cuda' else torch.mps.synchronize
  encoders=load_original()
  ms=importlib.util.spec_from_file_location('stage_archived_extract',package/'sources/extract_mps.py')
  original=importlib.util.module_from_spec(ms);ms.loader.exec_module(original)
  spec=encoders.REGISTRY[pin['historical_model_id']]
  with torch.no_grad():
   t=time.perf_counter();r['model_load_attempts']=1
   observer=RestoreObserver.for_libraries(torch)
   with observer:
    if model_id=='titan_conchv1_5':model,titan=encoders.load_conch();del titan
    else:model=spec['load']()
   r['checkpoint_restore_observation']=observer.report()
   timing['model_CPU_load_s']=time.perf_counter()-t;r['model_loads_completed']=1
   r['checkpoint_weight_files_restored']=observer.report()['successful_unique_weight_files']
   assert r['checkpoint_weight_files_restored']==c['models'][model_id]['checkpoint_files_per_successful_load'], 'Restore observation incomplete/conflicting; stop before forward'
   assert not model.training and all(not p.is_floating_point() or p.dtype==torch.float32 for p in model.parameters())
   r['parameter_dtypes_actual']=sorted({str(p.dtype) for p in model.parameters()})
   r['floating_parameter_dtypes_actual']=sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})
   assert repr(spec['transform'])==data['inputs'][model_id]['transform_repr']
   forward_sha=__import__('hashlib').sha256(inspect.getsource(type(model).forward).encode()).hexdigest()
   assert forward_sha==c['accepted_forward_sha256'][model_id]
   r.update(actual_transform_repr=repr(spec['transform']),forward_source_sha256=forward_sha,model_revision=pin['revision'])
   t=time.perf_counter();model=model.to(backend);timing['model_migration_submit_wall_s']=time.perf_counter()-t
   if policy=='CUDA_EXPLICIT_FP16_AUTOCAST':
    conversion_start=time.perf_counter();model=model.half();timing['explicit_dtype_conversion_submit_wall_s']=time.perf_counter()-conversion_start
   else:timing['explicit_dtype_conversion_submit_wall_s']='NOT_APPLICABLE_NO_HALF'
   sync();timing['model_device_migration_sync_s']=time.perf_counter()-t
   r['migration_timing_scope']='Migration submission + explicit half submission if applicable + joint end sync; submit intervals are nested, not independent transfer/kernel durations'
   expected_dtype=torch.float16 if policy=='CUDA_EXPLICIT_FP16_AUTOCAST' else torch.float32
   assert all(not p.is_floating_point() or p.dtype==expected_dtype for p in model.parameters())
   from collections import Counter
   def dtype_distribution(items):
    result={}
    for item in items:
     key=str(item.dtype);row=result.setdefault(key,{'tensors':0,'elements':0});row['tensors']+=1;row['elements']+=item.numel()
    return result
   r['parameter_dtype_distribution_after_conversion']=dtype_distribution(model.parameters())
   r['buffer_dtype_distribution_after_conversion']=dtype_distribution(model.buffers())
   r['submitted_input_dtype']=str(expected_dtype)
   r['actual_parameter_dtypes_before_conversion']=r['parameter_dtypes_actual']
   r['parameter_dtypes_actual']=sorted({str(p.dtype) for p in model.parameters()})
   r['floating_parameter_dtypes_actual']=sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})
   t=time.perf_counter();slide=openslide.OpenSlide(c['slide']);mpp=float(slide.properties['openslide.mpp-x']);read_px=encoders.read_px(spec['win_px'],mpp)
   timing['slide_open_metadata_s']=time.perf_counter()-t
   t=time.perf_counter();coords=original.tile_coords(slide,read_px);timing['tissue_mask_coords_s']=time.perf_counter()-t
   n=len(coords);expected=c['models'][model_id]
   assert n==expected['tiles'] and (n+7)//8==expected['batches'] and (n%8 or 8)==expected['last_batch']
   assert n>=80 and read_px>0
   r.update(tiles=n,batches=(n+7)//8,last_batch=(n%8 or 8),slide_dimensions=list(slide.dimensions),read_px=read_px,mpp_x=mpp)
   pen={'replaced_tiles':0}
   def white(im):
    if original.C.PEN_FILTER and original.pen_reject(im):
     pen['replaced_tiles']+=1;return Image.new('RGB',(read_px,read_px),(255,255,255))
    return im
   precision=PrecisionPath(torch,model,spec['postprocess'],sync,policy)
   def forward(x):return precision(x)
   def check_x(x):assert x.dtype==expected_dtype and x.device.type==backend and not torch.is_autocast_enabled(backend)
   def check_out(out,size):assert out.dtype in (torch.float16,torch.float32) and list(out.shape)==[size,pin['out_dim']]
   def readback(out):
    value=out.float().cpu().numpy()
    if 'CPU_readback_dtype' not in r:r['CPU_readback_dtype']=str(value.dtype)
    return value
   ops={'read_rgb':lambda xy:slide.read_region(tuple(xy),0,(read_px,read_px)).convert('RGB'),
        'white_check_replace':white,'transform':spec['transform'],
        'stack_submit':lambda tiles:torch.stack(tiles).to(backend,dtype=expected_dtype),
        'validate_input':check_x,'forward_post_sync':forward,'validate_output':check_out,
        'readback':readback,'concatenate':np.concatenate}
   warm={'attempted':0,'completed':0};timing['short_warmup_started_utc']=utc();t=time.perf_counter()
   try:_,warm_detail=traverse(coords[:80],ops,warm,collect=False)
   finally:r.update(warmup_forward_attempts=warm['attempted'],warmup_forward_completed=warm['completed'])
   sync();timing['short_warmup_wall_s']=time.perf_counter()-t;timing['short_warmup_finished_utc']=utc()
   r['warmup_timing_detail']=warm_detail;r['warmup_pen_replaced_tiles']=pen['replaced_tiles'];pen['replaced_tiles']=0;assert warm['completed']==10
   r['precision_observation']=precision.report();validate_observation(r['precision_observation']);precision.observing=False
   counts={'attempted':0,'completed':0};resource_sampler=ResourceSampler(torch,backend)
   window=coords if c['mode']=='FORMAL' else coords[:160]
   profiler=contextlib.nullcontext()
   if c['mode']=='DIAGNOSTIC':
    activities=[torch.profiler.ProfilerActivity.CPU]
    if backend=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler=torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False,with_stack=False)
    def annotated(name,fn):
     def call(*args):
      with torch.profiler.record_function('stage:'+name):return fn(*args)
     return call
    ops={name:annotated(name,fn) for name,fn in ops.items()}
   timing['output_started_utc']=utc()
   with profiler as prof:
    try:
     features,stages=traverse(window,ops,counts,sampler=resource_sampler,chunk_sink=chunks,
      progress=lambda v:G.persist(job/'Progress.json',dict(phase=c['mode'],**v)))
    finally:r.update(output_forward_attempts=counts['attempted'],output_forward_completed=counts['completed'])
   timing['output_finished_utc']=utc();r['output_stages']=stages;r['resources']=resource_sampler.finish(stages['traversal_wall_s']);r['output_pen_replaced_tiles']=pen['replaced_tiles']
   if c['mode']=='DIAGNOSTIC':
    assert counts['completed']==20;prof.export_chrome_trace(str(job/'diagnostic_trace.json'))
    timing['diagnostic_window_wall_s']=stages['traversal_wall_s'];r['GPU_trace_scope']='CUDA Kineto if supported; MPS CPU trace only'
   else:
    assert counts['completed']==(n+7)//8;timing['one_output_pass_wall_s']=stages['traversal_wall_s']
    t=time.perf_counter();assert np.isfinite(features).all(), 'Nonfinite complete output: stop further dispatch'
    timing['output_CPU_finite_validation_s']=time.perf_counter()-t;r['complete_features_finite']=True
    t=time.perf_counter();dest=job/(task['task_id']+'.h5')
    with h5py.File(dest,'x') as f:
     f.create_dataset('features',data=features.astype(np.float32));f.create_dataset('coords',data=np.array(coords,dtype=np.int32))
     r['H5_features_dtype_actual']=str(f['features'].dtype);r['H5_coords_dtype_actual']=str(f['coords'].dtype)
     f.attrs.update(run_id=c['run_id'],task_id=task['task_id'],contract_sha256=c['contract_sha256'],platform=c['platform'],repeat=task['repeat'],warmup_batches=10,inference_policy=policy,parameter_dtype=str(expected_dtype),submitted_input_dtype=str(expected_dtype),storage_dtype='float32',autocast_dtype='float16')
    timing['H5_write_close_s']=time.perf_counter()-t;r['output_h5']=dest.name
   rc=0;r['status']='COMPLETED_NOT_YET_OUTPUT_ACCEPTED'
 except Exception as e:
  r.update(status='FAILED_NO_AUTOMATIC_RETRY',error=str(e),traceback=traceback.format_exc());traceback.print_exc()
  if c['mode']=='FORMAL' and chunks:
   try:
    partial=np.concatenate(chunks)
    with h5py.File(job/'partial_features_NOT_ACCEPTED.h5','x') as f:
     f.create_dataset('features',data=partial.astype(np.float32));f.create_dataset('coords',data=np.array(coords[:len(partial)],dtype=np.int32))
     f.attrs['status']='FAILED_PARTIAL_NOT_ACCEPTED_NO_TILE_RESUME'
    r['preserved_partial_rows']=len(partial)
   except Exception as pe:r['partial_preservation_error']=str(pe)
 finally:
  if slide is not None:
   try:slide.close()
   except Exception as ce:r['slide_close_error']=str(ce);rc=1
  if 'observer' in locals():r['checkpoint_restore_observation']=observer.report()
  if 'precision' in locals():r['precision_observation']=precision.report()
  if 'resource_sampler' in locals() and 'resources' not in r:r['resource_samples_before_failure']=resource_sampler.samples
  u=resource.getrusage(resource.RUSAGE_SELF)
  r.update(worker_self_exit_code=rc,worker_wall_s=time.perf_counter()-begin,finished_utc=utc(),OS_exit_code='PENDING_PARENT_WAIT',
           worker_CPU_user_delta_s=u.ru_utime-cpu_begin.ru_utime,worker_CPU_system_delta_s=u.ru_stime-cpu_begin.ru_stime,
           process_lifetime_rss_highwater_bytes=int(u.ru_maxrss*(1 if sys.platform=='darwin' else 1024)))
  G.persist(job/'Worker_Receipt.json',r,True)
 return rc
