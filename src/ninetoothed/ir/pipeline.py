import hashlib
import pathlib

from ninetoothed.generation import CodeGenerator, cache_source
from ninetoothed.ir.passes.ast_to_mid import ASTToMidIRPass
from ninetoothed.ir.passes.mid_to_mlir import MidIRToMLIRPass

CACHE_DIR = pathlib.Path.home() / ".ninetoothed"


class _MLIRHandle:
    """Handle for MLIR pipeline results.

    Attributes:
        mid_func: The MidFunction IR (for dump inspection).
        mlir_text: The MLIR module string.
        mlir_file: Path to the cached .mlir file.
    """

    def __init__(self, mid_func, mlir_text, mlir_file):
        self.mid_func = mid_func
        self.mlir_text = mlir_text
        self.mlir_file = mlir_file

    def dump(self):
        """Dump the Mid IR in native format."""
        return self.mid_func.dump()

    def dump_mlir(self):
        """Return the MLIR text."""
        return self.mlir_text


class IRPipeline:
    """Dual-mode pipeline: legacy (Triton Python) or MLIR."""

    def __init__(
        self,
        func,
        use_mlir=False,
        caller="torch",
        kernel_name=None,
        num_warps=None,
        num_stages=None,
        max_num_configs=None,
        _prettify=False,
    ):
        self.func = func
        self.use_mlir = use_mlir
        self.caller = caller
        self.kernel_name = kernel_name or func.__name__
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.max_num_configs = max_num_configs
        self._prettify = _prettify

    def run(self):
        if self.use_mlir:
            return self._run_mlir_pipeline()
        return self._run_legacy_pipeline()

    def _run_legacy_pipeline(self):
        """Run existing CodeGenerator pipeline, return _Handle."""
        from ninetoothed.jit import JIT
        from ninetoothed.utils import calculate_default_configs

        default_num_warps, default_num_stages = calculate_default_configs()
        num_warps = self.num_warps or default_num_warps
        num_stages = self.num_stages or default_num_stages

        code_generator = CodeGenerator()
        source_file = code_generator(
            self.func,
            self.caller,
            self.kernel_name,
            num_warps,
            num_stages,
            self.max_num_configs,
            self._prettify,
        )

        from ninetoothed.jit import import_from_path
        module = import_from_path(source_file, source_file)
        module_vars = vars(module)

        from ninetoothed.jit import _Handle
        return _Handle(
            module_vars[self.kernel_name],
            module_vars[code_generator.launch_func_name],
            source_file,
        )

    def _run_mlir_pipeline(self):
        """Run AST -> Mid IR -> MLIR pipeline, return _MLIRHandle."""
        ast_pass = ASTToMidIRPass()
        mid_func = ast_pass.transform(self.func)
        mlir_pass = MidIRToMLIRPass()
        mlir_text = mlir_pass.transform(mid_func)

        mlir_file = self._cache_mlir(mlir_text)

        return _MLIRHandle(mid_func, mlir_text, mlir_file)

    @staticmethod
    def _cache_mlir(mlir_text):
        CACHE_DIR.mkdir(exist_ok=True)
        digest = hashlib.sha256(mlir_text.encode("utf-8")).hexdigest()
        cache_file = CACHE_DIR / f"{digest}.mlir"
        if not cache_file.exists():
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(mlir_text)
        return str(cache_file)
