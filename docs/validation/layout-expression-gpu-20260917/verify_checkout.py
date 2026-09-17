"""Check every submitted file against both Git and recorded GPU-input hashes."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
args = parser.parse_args()
manifest = json.loads((Path(__file__).parent / "SOURCE_MANIFEST.json").read_text())
for name, expected in manifest.items():
    relative = PurePosixPath(name)
    assert not relative.is_absolute() and ".." not in relative.parts, name
    data = (args.source / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected["sha256"], name
    blob = b"blob " + str(len(data)).encode() + b"\0" + data
    assert hashlib.sha1(blob).hexdigest() == expected["git_blob_sha"], name
print("VERIFIED", len(manifest), "submitted files")
