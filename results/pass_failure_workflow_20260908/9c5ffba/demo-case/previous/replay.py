"""Replay this exact CPU case using an installed NineToothed checkout."""
from pathlib import Path
from ninetoothed.interpreter import interpret_program
from ninetoothed.interpreter.debugger import load_reproducer
program, inputs, options = load_reproducer(Path(__file__).parent)
result = interpret_program(program, inputs, **options)
print({name: (value.shape, str(value.dtype)) for name, value in result.outputs.items()})
