"""Maintainer helper: update hashes after a reviewed source/documentation edit."""
from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parents[1]
names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0')
names=sorted({n for n in names if n and n!='MANIFEST.json' and (root/n).is_file()})
assert not any(Path(n).suffix in {'.pt','.pth','.ckpt','.safetensors','.npz','.pyc'} for n in names)
out={'sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},
     'description':'Exact public release files; research checkpoints and calibration coefficients excluded.'}
(root/'MANIFEST.json').write_text(json.dumps(out,indent=2)+'\n')
print('Updated',len(names),'file hashes')
