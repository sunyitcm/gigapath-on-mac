"""New stdlib-only planner; no framework import, data scan, or dispatch."""
import argparse,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    if a.config.suffix!='.json' or a.config.stat().st_size>128*1024:raise ValueError('Bounded JSON configuration required')
    c=json.loads(a.config.read_text())
    print(json.dumps(dict(status='PLAN_ONLY',execution_enabled=False,configuration=c,model_loads=0,forwards=0,feature_array_reads=0),indent=2))
if __name__=='__main__':main()
