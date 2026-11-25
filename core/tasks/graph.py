"""Per-source runtime graph and sink bindings."""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Callable
from ... import utils
from .base import ACEAudioTask


class ACEAudioGraph:
    """Complete DSP processing chain for a single audio source."""
    def __init__(self, source_context: Any, proxy_name: str):
        self.source_context = source_context
        self.proxy_name = proxy_name
        self.tasks: Dict[str, ACEAudioTask] = {}
        self.output_attr_keys: List[str] = []
        self.sink_bindings_by_attr: Dict[str, List["ACESinkBinding"]] = {}
        self.parallel_graph_uid: str = ""
        self.parallel_ir: Dict[str, Any] = {}
        self.parallel_ir_digest: str = ""
        self.parallel_offload_attr_keys: List[str] = []
        self.parallel_output_shards: List[List[str]] = []
        self.parallel_local_attr_keys: List[str] = []
        self.manual_preview_task_ids: List[str] = []
        self.auto_preview_task_ids: List[str] = []
        self.auto_preview_passthrough_specs: List[Tuple[str, int]] = []
        self.preview_task_ids: List[str] = []
        self.source_task_ids: List[str] = []
        self.live_source_task_ids: List[str] = []
        self.file_source_task_ids: List[str] = []
        self.block_size: int = 0
        self.source_uid: str = ""
        self.scheduler_mode: str = "legacy"
        self.timeslice_mode: str = "fixed"
        self.execution_order: List[str] = []
        self.eval_plan_cache: Dict[Tuple[str, ...], List[str]] = {}
        self.output_slot_specs: List[Tuple[str, str, int]] = []
        self.output_slot_specs_by_subset: Dict[Tuple[str, ...], List[Tuple[str, str, int]]] = {}
        self.proxy_lookup_fn: Optional[Callable[[str], Any]] = None

    def get_proxy_object(self):
        resolver = getattr(self, "proxy_lookup_fn", None)
        if callable(resolver):
            try:
                return resolver(str(self.proxy_name or ""))
            except Exception as exc:
                utils.note_swallowed("graph.ACEAudioGraph.get_proxy_object", exc)
                return None
        return None


@dataclass(frozen=True)
class ACESinkBinding:
    """Concrete main-thread write target resolved at compile time."""

    sink_kind: str
    target_object_name: str
    target_data_name: str
    attr_type: str = "FLOAT"
