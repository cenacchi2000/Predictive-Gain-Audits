"""Losslessly rebuild gate/test NPZ records from deduplicated arrays in reviewer ZIP."""
from pathlib import Path
import json,hashlib,io
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'array_manifest.json').read_text())
for relative,keys in manifest.items():
 target=(ROOT/'results'/relative).resolve()
 if not target.is_relative_to((ROOT/'results').resolve()):raise ValueError('Invalid output path')
 data={}
 for name,digest in keys.items():
  if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid digest')
  b=(ROOT/'arrays'/(digest+'.npy')).read_bytes()
  if hashlib.sha256(b).hexdigest()!=digest:raise ValueError('Array checksum mismatch')
  data[name]=np.load(io.BytesIO(b),allow_pickle=False)
 target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**data)
print('Restored',len(manifest),'gate/test and stress records. Run python code/summarize.py next.')
