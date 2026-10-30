"""New public scalar reader/plotter; not the historical model/array analysis.
Only explicitly named published CSV/JSON files are accepted. No H5/NPY/NPZ.
"""
import argparse,csv,json,math,statistics
from pathlib import Path
NAMES=['C8_Output_Times_27.csv','C8_Time_Resource_Summary_9.csv','C8_Stage_Shares_9.csv','C8_Output_Stage_Raw_27.csv','Propagation_135.csv','Same_H5_MPS_minus_CUDA_70.json','Same_Input_Repeats_140.json','Seven_Source_Input_Relationships_105.csv','Mac_AMP_Same_Reference_4.json','Core5_Tile_And_Mean_75.json','Core5_CUDA_Timing_Resources_30.csv']
MODELS=['gigapath','virchow2','titan_conchv1_5'];LABELS=['GigaPath','Virchow2','CONCH v1.5'];PLATFORMS=['M5 Max','M3 Ultra','RTX 5090 CUDA'];COLORS=['#176b91','#d27a21','#555b68']
def read_scalars(study):
    d={}
    for name in NAMES:
        p=study/'data'/name
        if p.is_symlink() or p.stat().st_size>512*1024:raise ValueError('Expected a bounded published scalar file')
        with p.open() as f:d[name]=json.load(f) if p.suffix=='.json' else list(csv.DictReader(f))
    return d

def check(d):
    t=d['C8_Output_Times_27.csv'];s=d['C8_Time_Resource_Summary_9.csv'];stage=d['C8_Stage_Shares_9.csv'];raw=d['C8_Output_Stage_Raw_27.csv'];errors=[]
    fields=['rgb_read_wall_s','white_check_replace_wall_s','transform_wall_s','batch_stack_device_submit_wall_s','forward_postprocess_end_sync_wall_s','cpu_readback_collect_concat_wall_s','unattributed_wall_s']
    assert len(t)==27 and len(s)==9 and len(stage)==9
    for row in s:
        z=[x for x in t if x['platform']==row['platform'] and x['model']==row['model']]
        assert len(z)==3
        if not math.isclose(statistics.median(float(x['output_wall_s']) for x in z),float(row['median_s']),rel_tol=1e-14):errors.append('C8 median mismatch')
    for row in stage:
        z=[x for x in raw if x['platform']==row['platform'] and x['model']==row['model']]
        den=sum(float(x['output_wall_s']) for x in z)
        if not math.isclose(den,float(row['denominator']),rel_tol=2e-14):errors.append('C8 denominator mismatch')
        for f in fields:
            expected=100*sum(float(x[f]) for x in z)/den
            if not math.isclose(expected,float(row[f]),rel_tol=2e-13,abs_tol=1e-11):errors.append('C8 stage share mismatch: '+f)
    mac=d['Mac_AMP_Same_Reference_4.json']
    for x in mac:
        a=x['A_minus_F'];e=x['E_minus_F'];ratio=a['relative_L2']/e['relative_L2'] if e['relative_L2'] else None
        assert a['reference_norm']==e['reference_norm'] or math.isclose(a['reference_norm'],e['reference_norm'],rel_tol=2e-13)
        assert ratio is None or math.isclose(ratio,x['relative_L2_A_over_E'],rel_tol=1e-14)
    rep=d['Same_Input_Repeats_140.json'];cross=d['Same_H5_MPS_minus_CUDA_70.json']
    assert len(rep)==140 and len({x['physical_pair_id'] for x in rep})==130
    assert len(cross)==70 and len({x['physical_pair_id'] for x in cross})==65
    assert all(x['metrics']['arrays_exact_equal'] for x in rep)
    assert not any(x['metrics']['arrays_exact_equal'] for x in cross)
    assert len(d['Seven_Source_Input_Relationships_105.csv'])==105 and len(d['Propagation_135.csv'])==135
    if errors:raise ValueError(errors)
    return dict(status='PASS_PUBLISHED_SCALAR_CHECKS',C8_workers=27,repeat_rows=140,unique_repeat_pairs=130,backend_rows=70,unique_backend_pairs=65,reference_direction='candidate minus named reference; no directional metric inversion',Mac_AMP_ratio_checks=4,model_runs=0,feature_array_reads=0,not_validated=['historical process state','model execution','array metrics recomputation','pixel-identical figure rendering'])

def figures(d,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
    points={}
    def save(fig,n,caption):
        fig.text(.03,.012,caption,fontsize=8,va='bottom');fig.savefig(out/f'Figure_{n}.png',dpi=200,bbox_inches='tight',pad_inches=.15,metadata={'Title':f'Scalar regeneration of snapshot Figure {n}','Software':'Matplotlib; public scalar-only helper'});plt.close(fig)
    t=d['C8_Output_Times_27.csv'];s=d['C8_Time_Resource_Summary_9.csv'];st=d['C8_Stage_Shares_9.csv']
    fig,(a,b)=plt.subplots(1,2,figsize=(13,5),gridspec_kw={'width_ratios':[1,1.4]})
    for pi,p in enumerate(PLATFORMS):
        for mi,m in enumerate(MODELS):
            z=[x for x in t if x['platform']==p and x['model']==m];x=mi+(pi-1)*.23;v=[float(r['output_wall_s']) for r in z]
            a.scatter([x]*3,v,s=20,color=COLORS[pi]);a.plot([x-.07,x+.07],[statistics.median(v)]*2,color=COLORS[pi],label=p if mi==0 else None)
    a.set_xticks(range(3),LABELS);a.set_ylabel('Complete output wall (s)');a.set_title('C8 / FP32 / B8 / 10 discarded warm-up batches');a.legend(frameon=False,fontsize=9)
    fields=['rgb_read_wall_s','white_check_replace_wall_s','transform_wall_s','batch_stack_device_submit_wall_s','forward_postprocess_end_sync_wall_s','cpu_readback_collect_concat_wall_s','unattributed_wall_s'];labels=['RGB','Check/white','Transform','Submit','Forward/post/sync','Readback/concat','Unattributed'];cs=['#2f789a','#92c4c6','#d4b46a','#9c8cb9','#d87a59','#d9a3b4','#d7dadd']
    z=[next(x for x in st if x['platform']==p and x['model']==m) for p in PLATFORMS for m in MODELS];left=[0.]*9
    for f,l,c in zip(fields,labels,cs):
        v=[float(x[f]) for x in z];b.barh(range(9),v,left=left,label=l,color=c);left=[x+y for x,y in zip(left,v)]
    b.set_yticks(range(9),[p.replace('RTX 5090 CUDA','CUDA')+' / '+l for p in PLATFORMS for l in LABELS]);b.invert_yaxis();b.set_xlim(0,100);b.set_xlabel('Sum of matched stage walls / sum of output walls (%)');b.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.15),fontsize=8)
    fig.subplots_adjust(bottom=.25,wspace=.7);points['Figure_3']={'times':t,'stage_shares':st};save(fig,3,'One WSI; three independent workers/arm. Host-wall intervals include waits; no GPU saturation or physical-peak inference.')
    def logpoints(ax,x,values,**kwargs):
        positive=[v for v in values if v>0];zero=sum(v==0 for v in values)
        if any(v<0 for v in values):raise ValueError('Negative nonnegative metric')
        ax.scatter([x]*len(positive),positive,**kwargs)
        if zero:ax.text(x,.02,f'zero: {zero}',transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
        ax.set_yscale('log')
    par=d['Propagation_135.csv'];cross=d['Same_H5_MPS_minus_CUDA_70.json'];fig,axs=plt.subplots(2,2,figsize=(12,7))
    for ri,m in enumerate(['gigapath','titan_conchv1_5']):
        for pi,cand in enumerate(['E','P']):
            z=[x for x in par if x['model']==m and x['candidate']==cand and x['reference']=='F']
            for si,field in enumerate(['tile_relative_L2','fixed_mean_metric_relative_L2','CUDA_learned_relative_L2']):
                logpoints(axs[ri,0],pi+(si-1)*.14,[float(x[field]) for x in z],color=COLORS[si],marker=['o','s','^'][si],label=['Tile','FP64 equal mean','Frozen CUDA aggregator'][si] if pi==0 else None)
        axs[ri,0].set_xticks([0,1],['CUDA explicit FP16','CUDA AMP']);axs[ri,0].set_ylabel('Relative L2 / saved CUDA FP32 norm');axs[ri,0].set_title(('GigaPath' if ri==0 else 'CONCH → TITAN')+': source substitution')
        if ri==0:axs[ri,0].legend(frameon=False)
        for si,src in enumerate(['F','M5F','M3F','E','P','M5E','M3E']):
            z=[x for x in cross if x['model']==m and x['source']==src];logpoints(axs[ri,1],si,[x['metrics']['relative_L2'] for x in z],facecolors='none' if src=='M3F' and ri==0 else COLORS[0],edgecolors=COLORS[0])
        axs[ri,1].set_xticks(range(7),['F','M5F','M3F','E','P','M5E','M3E']);axs[ri,1].set_ylabel('MPS − CUDA / CUDA vector norm');axs[ri,1].set_title('Fixed complete input + geometry: backend substitution')
    fig.subplots_adjust(bottom=.14,hspace=.45,wspace=.4);points['Figure_5']={'propagation_rows':par,'backend_rows':cross};save(fig,5,'Five WSIs/source. Hollow GigaPath M3F symbols are equal-input aliases. Independent spaces; tiles are not patients. Zero values are annotated, never replaced by an arbitrary error.')
    mac=d['Mac_AMP_Same_Reference_4.json'];cuda=d['Core5_Tile_And_Mean_75.json'];resources=d['Core5_CUDA_Timing_Resources_30.csv'];fig,axs=plt.subplots(2,2,figsize=(12,7))
    for key,c,off,label in [('E_minus_F',COLORS[1],-.15,'Historical explicit FP16'),('A_minus_F',COLORS[0],.15,'MPS AMP')]:
        axs[0,0].bar([i+off for i in range(4)],[r[key]['relative_L2'] for r in mac],width=.28,color=c,label=label)
        axs[0,1].bar([i+off for i in range(4)],[100*r[key]['top_SSE_fraction']['contribution'] for r in mac],width=.28,color=c)
    for ax in axs[0]:ax.set_xticks(range(4),['C8 M5','C8 M3','23 M5','23 M3'])
    axs[0,0].legend(frameon=False);axs[0,0].set_ylabel('Relative L2 / same per-Mac FP32 norm');axs[0,1].set_ylabel('Top ceil(1%N) tile SSE share (%)');axs[0,0].set_title('GigaPath / two WSIs / AMP r1');axs[0,1].set_title('Amplitude and residual concentration differ')
    for mi,m in enumerate(MODELS):
        for axis,c,off in [('CUDA_explicit_vs_saved_FP32',COLORS[1],-.12),('CUDA_AMP_vs_saved_FP32',COLORS[0],.12)]:
            z=[x['relative_L2'] for x in cuda if x['model_id']==m and x['comparison_axis']==axis];assert len(z)==5;logpoints(axs[1,0],mi+off,z,color=c)
        for policy,c,off in [('CUDA_EXPLICIT_FP16_AUTOCAST',COLORS[1],-.15),('CUDA_AMP_FP16',COLORS[0],.15)]:
            z=[int(x['CUDA_allocator_max_allocated_bytes'])/2**30 for x in resources if x['model']==m and x['policy']==policy];assert len(z)==5 and min(z)==max(z);axs[1,1].bar(mi+off,z[0],width=.28,color=c)
    for ax in axs[1]:ax.set_xticks(range(3),LABELS)
    axs[1,0].set_ylabel('Relative L2 / saved CUDA FP32 norm');axs[1,0].set_title('CUDA / five WSIs / r1 per policy');axs[1,1].set_ylabel('CUDA output allocator max allocated (GiB)');axs[1,1].set_title('API maxima; equal across five WSIs per arm')
    fig.subplots_adjust(bottom=.14,hspace=.45,wspace=.4);points['Figure_6']={'mac_same_reference':mac,'cuda_tile':cuda,'cuda_resources':resources};save(fig,6,'Orange: explicit half; blue: AMP. Cross-session FP32 references. No controlled speedup/accuracy claim; MPS and CUDA resource APIs are not interchangeable.')
    (out/'Figure_Data.json').write_text(json.dumps(points,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--study-root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--output',type=Path);p.add_argument('--execute-scalars',action='store_true');p.add_argument('--plots',action='store_true');a=p.parse_args()
    if not a.execute_scalars:
        print(json.dumps(dict(status='PLAN_ONLY',allowed_scalar_inputs=NAMES,models=0,array_reads=0)));return
    if a.output is None:p.error('new --output directory required')
    if a.output.exists():raise FileExistsError('Refuse overwrite')
    d=read_scalars(a.study_root);r=check(d);a.output.mkdir(parents=True);(a.output/'Scalar_Checks.json').write_text(json.dumps(r,indent=2)+'\n')
    if a.plots:figures(d,a.output)
    print(json.dumps(r))
if __name__=='__main__':main()
