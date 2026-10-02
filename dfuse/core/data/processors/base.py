from typing import Any, Dict, Protocol, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from dfuse.core.data.video_dataset import ProcessorContext


class BaseProcessor(Protocol):
    """Processor protocol for VideoDataset.

    Implementations should take a ProcessorContext and an RNG and return a
    free-form dictionary containing the processed sample.
    """

    def process(self, context: "ProcessorContext") -> Dict[str, Any]: ...
