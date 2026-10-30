"""Source cuda_gap16-monitor.py: exact selected definitions. See docs/SOURCE_STATUS.md.
No historical dispatch or scientific execution is invoked on import.
"""
import os, resource, sys, time, threading, json, platform
from source.vector_metrics import now
from pathlib import Path


class Monitor:
 def __init__(self,torch,psutil,device,path):
  self.torch,self.proc,self.device,self.path=torch,psutil.Process(),device,path
  self.rows=[];self.phase='INITIALIZATION';self.stop=threading.Event();self.lock=threading.Lock()
  self.start_cpu=self.proc.cpu_times();self.start_wall=time.perf_counter()
 def capture(self,label=None):
  with self.lock:
   r=dict(at=now(),phase=label or self.phase,RSS_bytes=self.proc.memory_info().rss)
   raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
   r['process_lifetime_RSS_high_water_raw']=raw;r['high_water_raw_unit']='bytes' if platform.system()=='Darwin' else 'KiB'
   r['process_lifetime_RSS_high_water_bytes']=int(raw*(1 if platform.system()=='Darwin' else 1024))
   cpu=self.proc.cpu_times();r.update(CPU_user_s=cpu.user,CPU_system_s=cpu.system)
   if self.device=='cuda':r.update(CUDA_allocated_bytes=self.torch.cuda.memory_allocated(),CUDA_reserved_bytes=self.torch.cuda.memory_reserved())
   else:r.update(MPS_allocated_bytes=self.torch.mps.current_allocated_memory(),MPS_driver_bytes=self.torch.mps.driver_allocated_memory())
   self.rows.append(r)
   with self.path.open('a') as f:f.write(json.dumps(r)+'\n')
   return r
 def loop(self):
  while not self.stop.wait(5):self.capture()
 def start(self):
  self.capture('START');self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
 def finish(self):
  self.stop.set();self.thread.join();self.capture('END')
  cpu=self.proc.cpu_times();wall=time.perf_counter()-self.start_wall
  u,s=cpu.user-self.start_cpu.user,cpu.system-self.start_cpu.system
  keys=['RSS_bytes','process_lifetime_RSS_high_water_bytes','MPS_allocated_bytes','MPS_driver_bytes','CUDA_allocated_bytes','CUDA_reserved_bytes']
  return dict(interval_s=5,observations=len(self.rows),first_at=self.rows[0]['at'],last_at=self.rows[-1]['at'],
    sampled_max_bytes={k:max(x[k] for x in self.rows if k in x) for k in keys if any(k in x for x in self.rows)},
    CPU_user_delta_s=u,CPU_system_delta_s=s,CPU_interval_wall_s=wall,CPU_average_percent_one_logical_core_100=100*(u+s)/wall,
    scope='Sampler start to finish; includes loading and execution; ru_maxrss is separate process-lifetime high-water, not sampled max',
    continuous_physical_peak='NOT_MEASURED',GPU_utilization='NOT_MEASURED',monitoring_overhead='NOT_SEPARATELY_MEASURED',
    overlapping_memory_scopes_not_summed=True)
