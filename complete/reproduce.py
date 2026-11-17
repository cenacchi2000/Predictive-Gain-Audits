from pathlib import Path
import argparse,shutil,subprocess,sys,os
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output).resolve()
if out.exists():raise SystemExit('Choose a new output directory to preserve records.')
root=Path(__file__).resolve().parent
shutil.copytree(root/'rebuttal/code',out/'code');(out/'results').mkdir()
env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
for name in ['matched_categorical.py','natural_sensitivity.py','continuous_witness.py','verify_new.py']:
 subprocess.run([sys.executable,str(out/'code'/name)],check=True,env=env)
print(out/'results')
