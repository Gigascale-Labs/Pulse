"""
Pulse persona discovery pipeline.
"""

from pulse.config import PipelineConfig
from pulse.pipeline import run_pipeline

__all__ = [
    "PipelineConfig",
    "run_pipeline",
]

'''
This lets Python users do:

'from pulse import PipelineConfig, run_pipeline'

'''