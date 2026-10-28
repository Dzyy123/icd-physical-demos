"""Create an isolated demo environment, including on systems without ensurepip."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, urllib.request, venv

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-dir',default='.venv')
    args=parser.parse_args()
    if sys.version_info < (3,10):parser.error('Python 3.10 or newer is required; Linux 3.10/3.11 is recommended.')
    target=Path(args.env_dir).resolve()
    if target.exists() and any(target.iterdir()) and not (target/'pyvenv.cfg').exists():
        parser.error('Refusing to overwrite a non-venv directory: '+str(target))
    env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'}
    # ensurepip is intentionally not required. System-site packages stay disabled.
    venv.EnvBuilder(with_pip=False,system_site_packages=False,symlinks=os.name!='nt').create(target)
    py=target/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    subprocess.run([str(py),'-c','import sys,site; assert sys.prefix!=sys.base_prefix; assert not site.ENABLE_USER_SITE'],env=env,check=True)
    result=subprocess.run([str(py),'-m','pip','--version'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if result.returncode:
        url='https://bootstrap.pypa.io/get-pip.py'
        script=target/'get-pip.py'
        print('Installing pip inside the isolated environment from '+url,flush=True)
        try:
            with urllib.request.urlopen(url,timeout=60) as response:data=response.read()
        except Exception as exc:
            parser.exit(1,'Could not download the official pip bootstrap script: '+str(exc)+'\nCheck HTTPS access to bootstrap.pypa.io; no system packages were changed.\n')
        script.write_bytes(data)
        (target/'bootstrap-source.json').write_text(json.dumps({'url':url,'sha256':hashlib.sha256(data).hexdigest()},indent=2))
        subprocess.run([str(py),str(script)],env=env,check=True)
    subprocess.run([str(py),'-m','pip','install','-r',str(ROOT/'requirements.txt')],env=env,check=True)
    subprocess.run([str(py),'-m','pip','check'],env=env,check=True)
    print('Ready. Activate this environment, then run: python demo.py verify',flush=True)

if __name__=='__main__':main()
