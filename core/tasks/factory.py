"""Which task computes a node: one place for the engine and the worker processes.

The node's role (vars.py) chooses the family; within it, the node's type and mode choose the
class. Source and preview tasks are imported only when asked for: worker processes never
need them, and they bring the input stack with them.
"""

from typing import Any, Dict, Optional

from ...vars import ROLE_MEASURE, ROLE_PREVIEW, ROLE_SOURCE, ROLE_SPLITTER, node_is_native, node_role
from .base import ACEAudioTask


def create_task(node_id: str, category: str, node_type: str, params: Optional[Dict[str, Any]], block_size: int) -> ACEAudioTask:
    """The task for one node; a plain ACEAudioTask (silence) for a kind without one."""
    params = params if params is not None else {}
    role = node_role(category, node_type)
    if node_is_native(category, node_type):
        return ACEAudioTask(node_id, params, block_size)  # it computes in Blender: never compiled
    if role == ROLE_MEASURE:
        from .stream import create_stream_task

        task = create_stream_task(node_id, node_type, params, block_size)
        if task is not None:
            return task
    elif role == ROLE_SPLITTER:
        from .delta import FrequencySplitterTask

        return FrequencySplitterTask(node_id, params, block_size)
    elif role == ROLE_SOURCE:
        from .source import SourceTask

        return SourceTask(node_id, params, block_size)
    elif role == ROLE_PREVIEW:
        from .preview import PreviewSinkTask

        return PreviewSinkTask(node_id, params, block_size)
    return ACEAudioTask(node_id, params, block_size)
