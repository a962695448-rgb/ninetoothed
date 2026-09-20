"""Recover the frozen candidate and tools after verifying the package hash."""
import base64
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

here = Path(__file__).resolve().parent
receipt = json.loads((here / 'PACKAGE_RECEIPT.json').read_text())
data = base64.b64decode((here / 'INPUTS.zip.base64').read_bytes(), validate=True)
assert len(data) == receipt['bytes'] and hashlib.sha256(data).hexdigest() == receipt['sha256']
out = Path(sys.argv[1])
out.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(data)) as z:
    manifest = json.loads(z.read('INPUT_MANIFEST.json'))
    assert set(z.namelist()) == set(manifest) | {'INPUT_MANIFEST.json'}
    for name in z.namelist():
        p = PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts
        value = z.read(name)
        if name in manifest:
            assert hashlib.sha256(value).hexdigest() == manifest[name]
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
print('VERIFIED INPUTS', len(manifest))
