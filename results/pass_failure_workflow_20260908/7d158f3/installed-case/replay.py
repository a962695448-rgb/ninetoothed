"""Verify the recorded differential failure using installed NineToothed."""
from pathlib import Path
from ninetoothed.interpreter.failure import replay_failure
replay_failure(Path(__file__).parent)
