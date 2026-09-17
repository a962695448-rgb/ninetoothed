"""Prepare a verified baseline checkout and the recorded source overlay."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, shutil
parser=argparse.ArgumentParser()
parser.add_argument('--baseline',type=Path,required=True)
parser.add_argument('--work',type=Path,required=True)
args=parser.parse_args();evidence=Path(__file__).resolve().parent
args.work.mkdir(exist_ok=False)
KEY='nine'
control=args.work/'control'/KEY;candidate=args.work/'candidate'/KEY
entries=json.loads((evidence/(KEY+'-base-tree.json')).read_text())['tree']
for entry in entries:
 if entry['type']!='blob':continue
 name=PurePosixPath(entry['path']);assert not name.is_absolute() and '..' not in name.parts
 source=args.baseline/entry['path'];data=source.read_bytes()
 assert hashlib.sha1(b'blob '+str(len(data)).encode()+bytes([0])+data).hexdigest()==entry['sha'],entry['path']
 target=control/entry['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
shutil.copytree(control,candidate)
for source in (evidence/'source').rglob('*'):
 if source.is_file():
  target=candidate/''/source.relative_to(evidence/'source');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
(args.work/'results').mkdir()
manifest={str(f.relative_to(args.work)):hashlib.sha256(f.read_bytes()).hexdigest()
 for folder in (control,candidate) for f in sorted(folder.rglob('*')) if f.is_file()}
(args.work/'INPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print('Verified baseline and candidate prepared:',args.work)
