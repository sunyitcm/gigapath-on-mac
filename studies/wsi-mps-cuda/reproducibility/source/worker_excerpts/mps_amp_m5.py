"""Source AMP-worker.py: exact selected definitions. See docs/SOURCE_STATUS.md.
No historical dispatch or scientific execution is invoked on import.
"""
from pathlib import Path
import sys, json, os, time, resource, traceback, importlib.util, inspect, hashlib
from source.timing import traverse, ResourceSampler
from source.amp_observation import WarmupObservation, CheckpointObserver, metadata


def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
 root=Path(sys.argv[1]).resolve();binding_path=Path(sys.argv[2]).resolve();task_id=sys.argv[3];job=Path(sys.argv[4]).resolve()
 c=json.loads((root/'Contract.json').read_text());b=json.loads(binding_path.read_text());task=next(x for x in c['tasks'] if x['id']==task_id);s=c['slides'][task['WSI']]
 r=dict(task=task,model_load_attempts=0,model_loads_completed=0,warmup_forward_attempts=0,warmup_forward_completed=0,output_forward_attempts=0,output_forward_completed=0,timing={},pid=os.getpid(),sys_executable=sys.executable,cache_state='NOT_CONTROLLED')
 begin=time.perf_counter();cpu0=resource.getrusage(resource.RUSAGE_SELF);rc=1;slide=None;obs=None;chunks=[];coords=None
 try:
  assert b['platform']==task['platform'] and Path(sys.executable).resolve()==Path(b['python']).resolve()
  assert os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0' and sys.platform=='darwin'
  assert json.loads((job/'Preflight.json').read_text())['status']=='PASS_NO_MODEL_PREFLIGHT'
  assert c['batch']==8 and c['warmup_batches']==10 and c['model']=='gigapath'
  sys.path.insert(0,str(root/'original/scripts'))
  from common import bundle,native_cache_bindings
  from smoke_worker import load_original
  import torch,numpy as np,h5py,openslide
  from PIL import Image
  assert torch.backends.mps.is_available() and torch.amp.autocast_mode.is_autocast_available('mps')
  torch.set_default_device('cpu');torch.set_default_dtype(torch.float32);torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20260826);torch.use_deterministic_algorithms(False)
  r['cache_bindings']=native_cache_bindings(bundle(),Path(b['hub']),'gigapath')
  encoders=load_original();spec=encoders.REGISTRY['gigapath']
  ms=importlib.util.spec_from_file_location('amp_original_extract',root/'original/sources/extract_mps.py');original=importlib.util.module_from_spec(ms);ms.loader.exec_module(original)
  assert repr(spec['transform'])==bundle()['inputs']['gigapath']['transform_repr']
  r['transform']=repr(spec['transform']);r['Original_postprocess_source']=inspect.getsource(spec['postprocess'])
  with torch.no_grad():
   t=time.perf_counter();r['model_load_attempts']=1
   with CheckpointObserver(torch) as restore:model=spec['load']()
   r['timing']['model_CPU_load_s']=time.perf_counter()-t;r['model_loads_completed']=1;r['successful_torch_load_paths']=restore.paths
   assert len(restore.paths)==1 and Path(restore.paths[0]).resolve()==Path(r['cache_bindings'][0]['resolved_path']).resolve()
   assert not model.training
   r['parameter_dtypes']=sorted({str(p.dtype) for p in model.parameters()});r['buffer_dtypes']=sorted({str(p.dtype) for p in model.buffers()})
   assert all(not x.is_floating_point() or x.dtype==torch.float32 for x in list(model.parameters())+list(model.buffers()))
   r['forward_source_sha256']=hashlib.sha256(inspect.getsource(type(model).forward).encode()).hexdigest();assert r['forward_source_sha256']==c['accepted_forward_sha256']
   t=time.perf_counter();model=model.to('mps');torch.mps.synchronize();r['timing']['model_device_migration_sync_s']=time.perf_counter()-t
   assert all(p.device.type=='mps' for p in model.parameters())
   t=time.perf_counter();slide=openslide.OpenSlide(b['slides'][task['WSI']]);mpp=float(slide.properties['openslide.mpp-x']);px=encoders.read_px(spec['win_px'],mpp);r['timing']['slide_open_metadata_s']=time.perf_counter()-t
   t=time.perf_counter();coords=original.tile_coords(slide,px);r['timing']['tissue_mask_coords_s']=time.perf_counter()-t
   expected=np.load(root/s['coords_file'],allow_pickle=False)
   assert len(coords)==s['N'] and px==s['read_px'] and np.array_equal(np.asarray(coords,dtype=np.int32),expected),'Exact coords/order gate failed'
   r.update(tiles=len(coords),read_px=px,mpp_x=mpp,slide_dimensions=list(slide.dimensions),coords_exact_equal=True)
   pen={'replaced_tiles':0};obs=WarmupObservation(torch,model)
   def white(im):
    if original.C.PEN_FILTER and original.pen_reject(im):pen['replaced_tiles']+=1;return Image.new('RGB',(px,px),(255,255,255))
    return im
   def amp(x):
    with torch.autocast('mps',dtype=torch.float16):
     raw=model(x);out=spec['postprocess'](raw)
    torch.mps.synchronize();return out
   def warm_amp(x):
    obs.call+=1
    with torch.autocast('mps',dtype=torch.float16):
     active=torch.is_autocast_enabled('mps');dtype=str(torch.get_autocast_dtype('mps'));raw=model(x)
     rawmeta=metadata(raw);out=spec['postprocess'](raw)
     obs.rows.append(dict(warmup_call=obs.call,input=metadata(x),autocast_enabled=active,autocast_dtype=dtype,raw=rawmeta,post=metadata(out),postprocess_autocast_enabled=torch.is_autocast_enabled('mps')))
    torch.mps.synchronize();return out
   def check_x(x):assert x.dtype==torch.float32 and x.device.type=='mps' and not torch.is_autocast_enabled('mps')
   def check_out(out,n):assert out.dtype in (torch.float16,torch.float32) and out.device.type=='mps' and list(out.shape)==[n,1536]
   ops=dict(read_rgb=lambda xy:slide.read_region(tuple(xy),0,(px,px)).convert('RGB'),white_check_replace=white,transform=spec['transform'],stack_submit=lambda tiles:torch.stack(tiles).to('mps',dtype=torch.float32),validate_input=check_x,forward_post_sync=warm_amp,validate_output=check_out,readback=lambda out:out.float().cpu().numpy(),concatenate=np.concatenate)
   warm={'attempted':0,'completed':0};t=time.perf_counter()
   try:_,detail=traverse(coords[:80],ops,warm,collect=False)
   finally:
    r.update(warmup_forward_attempts=warm['attempted'],warmup_forward_completed=warm['completed']);obs.remove()
   r['timing']['short_warmup_wall_s']=time.perf_counter()-t;r['warmup_timing_detail']=detail;r['warmup_pen_replaced_tiles']=pen['replaced_tiles'];pen['replaced_tiles']=0
   assert warm['completed']==10;obs.validate();r['precision_observation']=obs.report()
   assert all(x['postprocess_autocast_enabled'] for x in obs.rows)
   ops['forward_post_sync']=amp;r['formal_output_has_diagnostic_hooks']=False
   counts={'attempted':0,'completed':0};sampler=ResourceSampler(torch,'mps')
   try:features,stages=traverse(coords,ops,counts,sampler=sampler,chunk_sink=chunks,progress=lambda row:save(job/'Progress.json',row))
   finally:r.update(output_forward_attempts=counts['attempted'],output_forward_completed=counts['completed'])
   r['output_stages']=stages;r['resources']=sampler.finish(stages['traversal_wall_s']);r['output_pen_replaced_tiles']=pen['replaced_tiles']
   assert counts['completed']==s['batches'] and np.isfinite(features).all() and features.dtype==np.float32
   assert all(p.dtype==torch.float32 for p in model.parameters())
   t=time.perf_counter()
   with h5py.File(job/(task_id+'.h5'),'x') as f:
    f.create_dataset('features',data=features.astype(np.float32));f.create_dataset('coords',data=np.asarray(coords,dtype=np.int32))
    f.attrs.update(run_id=c['run_id'],task_id=task_id,platform=b['platform'],repeat=1,inference_policy='MPS_AMP_FP16',parameter_dtype='float32',input_dtype='float32',warmup_batches=10,contract_sha256=hashlib.sha256((root/'Contract.json').read_bytes()).hexdigest())
   r['timing']['H5_write_close_s']=time.perf_counter()-t;r['output_h5']=task_id+'.h5';r['status']='COMPLETED_PENDING_PARENT_ACCEPTANCE';rc=0
 except Exception as e:r.update(status='FAILED_NO_AUTOMATIC_RETRY',error=str(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:
  if obs is not None:obs.remove();r['precision_observation']=obs.report()
  if slide is not None:slide.close()
  u=resource.getrusage(resource.RUSAGE_SELF)
  r.update(worker_self_exit_code=rc,OS_exit_code='PENDING_PARENT_WAIT',worker_wall_s=time.perf_counter()-begin,worker_CPU_user_delta_s=u.ru_utime-cpu0.ru_utime,worker_CPU_system_delta_s=u.ru_stime-cpu0.ru_stime,process_lifetime_rss_highwater_bytes=u.ru_maxrss)
  save(job/'Worker_Receipt.json',r)
 return rc
