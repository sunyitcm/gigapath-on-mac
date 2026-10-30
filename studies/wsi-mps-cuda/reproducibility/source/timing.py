"""Source C8-stage_common.py: exact selected definitions. See docs/SOURCE_STATUS.md.
No historical dispatch or scientific execution is invoked on import.
"""
from pathlib import Path
import os, resource, sys, time


PHASES=('rgb_read_wall_s','white_check_replace_wall_s','transform_wall_s',
        'batch_stack_device_submit_wall_s','forward_postprocess_end_sync_wall_s',
        'cpu_readback_collect_concat_wall_s')

class Ledger:
 def __init__(self,clock=time.perf_counter):self.clock=clock;self.values={k:0.0 for k in PHASES}
 def call(self,key,fn,*a):
  begin=self.clock()
  try:return fn(*a)
  finally:self.values[key]+=self.clock()-begin

def traverse(coords,ops,counts,*,collect=True,progress=None,sampler=None,clock=time.perf_counter,chunk_sink=None):
 """Original serial batch loop; added timers never call synchronize themselves."""
 ledger=Ledger(clock);chunks=chunk_sink if chunk_sink is not None else [];begin=clock();n=len(coords)
 for i in range(0,n,8):
  tiles=[]
  for xy in coords[i:i+8]:
   rgb=ledger.call(PHASES[0],ops['read_rgb'],xy)
   rgb=ledger.call(PHASES[1],ops['white_check_replace'],rgb)
   tiles.append(ledger.call(PHASES[2],ops['transform'],rgb))
  x=ledger.call(PHASES[3],ops['stack_submit'],tiles)
  ops['validate_input'](x)
  counts['attempted']+=1
  out=ledger.call(PHASES[4],ops['forward_post_sync'],x)
  counts['completed']+=1
  ops['validate_output'](out,len(tiles))
  def collect_out():
   value=ops['readback'](out)
   if collect:chunks.append(value)
  ledger.call(PHASES[5],collect_out)
  batch=counts['completed']
  if batch==1 or batch%50==0 or i+8>=n:
   if sampler:sampler.sample(batch)
   if progress:progress({'completed_tiles':min(i+8,n),'batches_completed':batch,**counts})
 features=None
 if collect:features=ledger.call(PHASES[5],ops['concatenate'],chunks)
 wall=clock()-begin
 attributed=sum(ledger.values.values())
 return features,{**ledger.values,'traversal_wall_s':wall,
   'exclusive_stage_sum_s':attributed,'unattributed_wall_s':wall-attributed,
   'accounting_status':'OK' if wall>=attributed else 'NEGATIVE_RESIDUAL_REQUIRES_REVIEW',
   'unattributed_not_forced_to_zero':True}

class ResourceSampler:
 """Fixed pre/end + output batches 1,50,100,...,last; no background sampler."""
 def __init__(self,torch,backend):
  self.torch=torch;self.backend=backend;self.samples=[];self.cpu0=resource.getrusage(resource.RUSAGE_SELF)
  if sys.platform=='darwin':
   import psutil
   self.process=psutil.Process(os.getpid())
  else:self.process=None
  if backend=='cuda':torch.cuda.reset_peak_memory_stats() # counters only; no cache clearing
  self.sample(0)
 def sample(self,batch):
  u=resource.getrusage(resource.RUSAGE_SELF)
  row={'batch':batch,'monotonic_s':time.perf_counter(),'cpu_user_s':u.ru_utime,'cpu_system_s':u.ru_stime,
       'process_lifetime_rss_highwater_bytes':int(u.ru_maxrss*(1 if sys.platform=='darwin' else 1024))}
  if self.process:row.update(process_current_rss_bytes=self.process.memory_info().rss,rss_source='psutil.memory_info.rss')
  else:
   values=Path('/proc/self/status').read_text().splitlines()
   row.update(process_current_rss_bytes=next(int(x.split()[1])*1024 for x in values if x.startswith('VmRSS:')),rss_source='/proc/self/status VmRSS')
  if self.backend=='cuda':
   row.update(cuda_allocated_bytes=self.torch.cuda.memory_allocated(),cuda_reserved_bytes=self.torch.cuda.memory_reserved())
  else:row.update(mps_allocated_bytes=self.torch.mps.current_allocated_memory(),mps_driver_bytes=self.torch.mps.driver_allocated_memory())
  self.samples.append(row)
 def finish(self,wall):
  self.sample('END');u=resource.getrusage(resource.RUSAGE_SELF)
  du=u.ru_utime-self.cpu0.ru_utime;ds=u.ru_stime-self.cpu0.ru_stime
  v={'samples':self.samples,'cpu_user_delta_s':du,'cpu_system_delta_s':ds,
     'CPU_percent_one_logical_CPU_100':100*(du+ds)/wall if wall>0 else None,
     'CPU_delta_scope':'sampler creation through finish; narrowly encloses output wall; query overhead included',
     'process_highwater_scope':'OS process-lifetime HWM, not resettable output-only RSS peak',
     'process_sampled_max_rss_bytes':max(r['process_current_rss_bytes'] for r in self.samples),
     'device_sample_maxima':{k:max(r[k] for r in self.samples if k in r) for k in self.samples[-1] if k.startswith(('mps_','cuda_'))},
     'GPU_utilization':'NOT_MEASURED','GPU_busy_time':'NOT_MEASURED'}
  if self.backend=='cuda':v.update(cuda_output_allocator_max_allocated_bytes=self.torch.cuda.max_memory_allocated(),cuda_output_allocator_max_reserved_bytes=self.torch.cuda.max_memory_reserved())
  return v
