from .core import *


def __getattr__(name):
    if name == "DFusePipeline":
        from .pipelines.dfuse import DFusePipeline
        return DFusePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
