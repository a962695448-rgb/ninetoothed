import hashlib, importlib.util, json
from pathlib import Path
import ninetoothed
from ninetoothed.interpreter import failure
base = Path(ninetoothed.__file__).parent
assert "site-packages" in str(base)
assert all(importlib.util.find_spec(name) is None for name in ("torch", "triton"))
source = Path(__file__).parent / "source/src/ninetoothed"
count = 0
for path in source.rglob("*.py"):
    target = base / path.relative_to(source)
    assert target.read_bytes() == path.read_bytes(), str(path)
    count += 1
print(json.dumps({"installed_path": str(base), "files_verified": count, "torch": False, "triton": False}))
