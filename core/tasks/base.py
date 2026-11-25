"""Task base class and frame evaluation."""

import threading
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
import numpy as np
from ... import utils
from .chunk import AudioChunk


# Thread-local flag set by PreviewSinkTask.process() to mark the Layer-3 preview
# call path. _masked_mono_signal() checks this and logs an ERROR on violation since
# calling iFFT reconstruction from Layer 3 would contaminate the time-domain output.
_LAYER3_THREAD_LOCAL: threading.local = threading.local()


class TaskConnection(NamedTuple):
    task: 'ACEAudioTask'
    socket_index: int


class ACEAudioTask:
    """Base class for all DSP processing nodes in the ACE pipeline.

    Subclasses implement ``process(inputs)`` to transform one or more
    ``AudioChunk`` inputs into output values.  Each task owns its internal
    filter state, silence buffer, and block-size configuration so that
    state is preserved across frames without external management.

    Subclasses should NOT import or call bpy — tasks are evaluated on
    Stage-B worker processes that are intentionally bpy-free.
    """
    def __init__(self, node_id: str, parameters: Dict[str, Any], block_size: int):
        self.node_id = node_id
        self.parameters = parameters
        self.block_size = block_size
        self.engine_ref = None
        self.undo_generation: int = 0
        self.resolve_block_size_fn = None
        
        self.inputs: Dict[int, TaskConnection] = {} 
        self.cached_output: Any = None
        self.last_frame_processed = -1
        self._current_chunk: Optional[AudioChunk] = None
        self._current_chunk_frame: int = -1
        
        self.silence_buffer = np.zeros(block_size, dtype=np.float32)
        
        self.source_context = None
        self.scene_fps: float = 24.0
        self.scene_frame_start: int = 1
        self.engine_settings = None

    def connect_input(self, input_idx: int, target_task: 'ACEAudioTask', target_output_idx: int):
        self.inputs[input_idx] = TaskConnection(target_task, target_output_idx)

    def _evaluate_frame_iterative(self, frame_index: int):
        """
        Evaluate upstream dependencies for this task without Python recursion.
        Keeps per-task frame caching semantics unchanged.
        """
        frame_idx = int(frame_index)
        if int(getattr(self, "last_frame_processed", -1)) == frame_idx:
            return

        stack: List[Tuple['ACEAudioTask', bool]] = [(self, False)]
        queued: set[int] = set()

        while stack:
            task, ready = stack.pop()
            if task is None:
                continue
            if int(getattr(task, "last_frame_processed", -1)) == frame_idx:
                continue

            if ready:
                setattr(task, "_current_frame_request", frame_idx)
                input_data: Dict[int, Any] = {}
                for in_sock_idx, connection in (getattr(task, "inputs", {}) or {}).items():
                    upstream = getattr(connection, "task", None)
                    socket_idx = int(getattr(connection, "socket_index", 0) or 0)
                    if upstream is None:
                        input_data[in_sock_idx] = None
                        continue
                    if int(getattr(upstream, "last_frame_processed", -1)) != frame_idx:
                        try:
                            input_data[in_sock_idx] = upstream.get_output(frame_idx, socket_idx)
                        except Exception as exc:
                            utils.note_swallowed("audio_tasks_registry.ACEAudioTask._evaluate_frame_iterative", exc)
                            input_data[in_sock_idx] = None
                        continue
                    extract_fn = getattr(upstream, "_extract_result", None)
                    if callable(extract_fn):
                        with utils.logged_errors("audio_tasks_registry.ACEAudioTask._evaluate_frame_iterative#2"):
                            input_data[in_sock_idx] = extract_fn(socket_idx)
                            continue
                    try:
                        input_data[in_sock_idx] = upstream.get_output(frame_idx, socket_idx)
                    except Exception as exc:
                        utils.note_swallowed("audio_tasks_registry.ACEAudioTask._evaluate_frame_iterative#3", exc)
                        input_data[in_sock_idx] = None

                try:
                    task.cached_output = task.process(input_data)
                    task._tag_output_frame(task.cached_output, frame_idx)
                    task.last_frame_processed = frame_idx
                except Exception as exc:
                    utils.note_swallowed("audio_tasks_registry.ACEAudioTask._evaluate_frame_iterative#4", exc)
                    continue
                continue

            task_key = id(task)
            if task_key in queued:
                # Cycle safeguard: process with whatever upstream data is currently available.
                stack.append((task, True))
                continue
            queued.add(task_key)
            stack.append((task, True))
            for connection in (getattr(task, "inputs", {}) or {}).values():
                upstream = getattr(connection, "task", None)
                if upstream is None:
                    continue
                if int(getattr(upstream, "last_frame_processed", -1)) == frame_idx:
                    continue
                stack.append((upstream, False))

    def get_output(self, frame_index: int, output_socket_idx: int = 0):
        frame_idx = int(frame_index)
        if self.last_frame_processed == frame_idx:
            return self._extract_result(output_socket_idx)

        self._evaluate_frame_iterative(frame_idx)

        # Fallback: preserve legacy recursive behavior for edge cases where
        # iterative scheduling could not complete this task.
        if self.last_frame_processed != frame_idx:
            self._current_frame_request = frame_idx
            input_data = {}
            for in_sock_idx, connection in self.inputs.items():
                input_data[in_sock_idx] = connection.task.get_output(frame_idx, connection.socket_index)
            self.cached_output = self.process(input_data)
            self._tag_output_frame(self.cached_output, frame_idx)
            self.last_frame_processed = frame_idx

        return self._extract_result(output_socket_idx)

    def _extract_result(self, socket_idx):
        if self.cached_output is None:
            return self.silence_buffer

        if isinstance(self.cached_output, tuple):
            if len(self.cached_output) == 0:
                return self.silence_buffer
            
            if 0 <= socket_idx < len(self.cached_output):
                val = self.cached_output[socket_idx]
                return val if val is not None else self.silence_buffer
            
            return self.silence_buffer

        return self.cached_output

    def _tag_output_frame(self, value, frame_index: int):
        current_chunk = self._tag_output_frame_recursive(value, frame_index)
        self._current_chunk = current_chunk
        self._current_chunk_frame = int(frame_index) if current_chunk is not None else -1

    def _tag_output_frame_recursive(self, value, frame_index: int) -> Optional[AudioChunk]:
        if isinstance(value, AudioChunk):
            value.frame_index = frame_index
            return value
        if isinstance(value, (list, tuple)):
            for v in value:
                chunk = self._tag_output_frame_recursive(v, frame_index)
                if chunk is not None:
                    return chunk
        return None

    def process(self, inputs: Dict[int, Any]):
        """Override this."""
        return self.silence_buffer
