"""New public CLI for the retained comparator. Not executed on arrays in this supplement."""
import argparse, json, sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path);p.add_argument('--candidate',type=Path);p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:
        print(json.dumps(dict(status='PLAN_ONLY',direction='candidate minus reference',relative_L2_denominator='reference Frobenius norm',block=256)));return
    if sys.flags.optimize:raise RuntimeError('Do not use -O: retained comparator uses assertions')
    if any(x is None for x in [a.reference,a.candidate,a.output]):p.error('reference/candidate/output required for execution')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from source.compare_h5_core import compare
    compare(a.reference,a.candidate,a.output,block=256)
if __name__=='__main__':main()
