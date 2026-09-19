"""Recover the exact frozen inputs from their verified base64 ZIP."""

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
assert len(data) == receipt['package_bytes']
assert hashlib.sha256(data).hexdigest() == receipt['package_sha256']
destination = Path(sys.argv[1])
destination.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    manifest = json.loads(archive.read('INPUT_MANIFEST.json'))
    assert set(archive.namelist()) == set(manifest) | {'INPUT_MANIFEST.json'}
    for name in archive.namelist():
        relative = PurePosixPath(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        value = archive.read(name)
        if name in manifest:
            assert hashlib.sha256(value).hexdigest() == manifest[name], name
        output = destination / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(value)
print('VERIFIED INPUTS', len(manifest))
