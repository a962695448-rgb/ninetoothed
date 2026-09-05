# CPU evidence file mapping

The successful CPU-contract run is `pytest.log`, `pytest.xml`, and `manifest.json`.
The earlier broad selection intentionally remains archived with its dependency
setup errors. Its original manifest is `discovery_manifest.json`; the manifest's
artifact `pytest.log` is stored as `discovery.log`, and `pytest.xml` is stored as
`discovery.xml`. Their original SHA-256 values and bytes are unchanged.

The two runs use source commit `5b377252cc4452b5ccc48c46ff1ae07a4e5e0e8a`.
They cover different test selections. Neither establishes an A100 result or a
full-repository pass. Absolute paths in manifests describe the original runs;
use a new output directory and record new metadata when reproducing elsewhere.
