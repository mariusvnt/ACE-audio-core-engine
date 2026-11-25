"""Proxy mesh attribute sinks: schema and per-frame writes."""

import bpy # type: ignore
import numpy as np
from typing import Dict, Set, Any
from ..id_management import ACEIdManager
from ..abstract_tree import ACEAbstractTree
from ... import utils


class AttributeSinks:
    def _cache_output_defs(self):
        """Precompute attribute slugs and Blender data types for all writeable outputs.

        Iterates over stream, source_specs, util_rolling_avg, and stream_accumulator
        nodes in the abstract tree.
        For each non-bundle output socket, builds a dict entry keyed by
        ``"<node_id>:<socket_idx>"`` containing the derived attribute name slug,
        Blender attribute type, and any user-supplied Attribute ID overrides.
        Results are stored in self.output_defs and consumed by _select_attr_name()
        and ensure_attributes_exist().
        """
        if not self.abstract_tree:
            return

        for node in self.abstract_tree.nodes.values():
            is_stream = node.category == 'stream'
            is_source_specs = node.type == 'src_specs' and node.category == 'utilities'
            is_rolling_avg = node.type == 'util_rolling_avg' and node.category == 'utilities'
            is_stream_accumulator = node.type == 'stream_accumulator' and node.category == 'utilities'
            if not (is_stream or is_source_specs or is_rolling_avg or is_stream_accumulator):
                continue

            for sock in node.outputs:
                # Skip bundle passthroughs on source specs to avoid creating useless attributes
                if getattr(sock, "data_type", "") in ("NodeSocketBundle", "BUNDLE", "BUNDLE_SOCKET"):
                    continue
                # Key is unique per node instance and socket index
                key = f"{node.id}:{sock.socket_idx}"
                node_slug = ACEIdManager._clean_string(node.id if node.id else node.type)  # type: ignore
                socket_slug = ACEIdManager._clean_string(sock.name or sock.id or f"out{sock.socket_idx}")  # type: ignore
                slug = f"{node_slug}_{socket_slug or sock.socket_idx}"
                attr_type = self._resolve_attr_type(sock.data_type)
                params = node.parameters if isinstance(getattr(node, "parameters", {}), dict) else {}
                attr_id = params.get(f"attr_id_out_{sock.socket_idx}")
                attr_custom = bool(params.get(f"attr_id_custom_out_{sock.socket_idx}", False))
                self.output_defs[key] = {
                    "slug": slug,
                    "type": attr_type,
                    "node_id": node.id,
                    "socket_idx": sock.socket_idx,
                    "output_name": sock.name,
                    "attr_id": attr_id,
                    "attr_custom": attr_custom,
                    "frontend_uid": str(getattr(node, "frontend_uid", "") or ""),
                    "frontend_kind": str(getattr(node, "frontend_kind", "") or ""),
                }

    @staticmethod
    def _resolve_attr_type(sock_type: str) -> str:
        """Map a GN socket data type string to a Blender mesh attribute type.

        Args:
            sock_type: A socket type string such as ``'VECTOR'``, ``'RGBA'``, or
                a NodeSocket class name like ``'NodeSocketVector'``.

        Returns:
            One of ``'FLOAT'``, ``'FLOAT_VECTOR'``, or ``'FLOAT_COLOR'``.
        """
        attr_type = 'FLOAT'
        if sock_type in ('VECTOR', 'NodeSocketVector'):
            attr_type = 'FLOAT_VECTOR'
        elif sock_type in ('RGBA', 'NodeSocketColor'):
            attr_type = 'FLOAT_COLOR'
        return attr_type

    def _select_attr_name(self, proxy_name: str, info: Dict[str, Any]) -> str:
        """Choose the mesh attribute name for a given output definition.

        When the user has explicitly set a custom Attribute ID on the node socket,
        that string is sanitised and returned (with the ACE prefix prepended if
        absent).  Otherwise a compact auto-generated name is derived via
        ACEIdManager.generate_compact_attr_name().

        Args:
            proxy_name: The proxy object name; used as the source hint in the
                auto-generated name.
            info: An output definition dict from self.output_defs containing
                at minimum ``node_id``, ``socket_idx``, ``output_name``,
                ``attr_id``, and ``attr_custom``.

        Returns:
            A Blender-compatible attribute name string (max 63 characters).
        """
        custom = info.get("attr_id")
        custom_is_explicit = bool(info.get("attr_custom", False))
        if custom_is_explicit and isinstance(custom, str) and custom.strip():
            name = custom.strip()
            if not name.startswith(ACEIdManager.PREFIX):
                name = ACEIdManager.PREFIX + ACEIdManager._clean_string(name)
            return name[:63]
        return ACEIdManager.generate_compact_attr_name(
            proxy_name,
            str(info.get("node_id", "") or ""),
            str(info.get("output_name", "") or ""),
            int(info.get("socket_idx", 0) or 0),
        )

    def ensure_attributes_exist(
        self,
        proxy: bpy.types.Object,
        abstract_tree: ACEAbstractTree | None = None,
        active_keys: Set[str] | None = None,
        prune_existing: bool = False,
    ):
        """Create or repair named point-domain attributes on the proxy mesh.

        Iterates over the output definitions produced by _cache_output_defs() and
        ensures that each required attribute exists with the correct Blender data
        type.  Attributes with a mismatched type are removed and re-created.

        When active_keys is an empty set and prune_existing is True all ACE
        attributes on the mesh are removed (compile-time cleanup path).  When
        active_keys is None a full sync is performed, which also prunes any ACE
        attributes not in the current required set.

        Flushes the per-mesh RNA handle cache and last-written value cache
        whenever the attribute layout changes so stale handles are not used.

        Args:
            proxy: The proxy bpy.types.Object whose mesh attributes to manage.
            abstract_tree: Optional replacement abstract tree; triggers a call to
                set_abstract_tree() when different from the current one.
            active_keys: Subset of output_defs keys to sync.  None means all keys
                (full sync with pruning).  Empty set with prune_existing=True
                removes all ACE attributes.
            prune_existing: When True, also remove ACE attributes not in the
                required set even for a scoped (non-None active_keys) sync.
        """
        utils.assert_main_thread("ensure_attributes_exist")
        mesh = getattr(proxy, "data", None)
        if not mesh:
            return

        if abstract_tree and abstract_tree is not self.abstract_tree:
            self.set_abstract_tree(abstract_tree)
        attrs_changed = False

        # Optional strict mode: empty active set means no ACE attributes should remain.
        if active_keys is not None and len(active_keys) == 0:
            if prune_existing:
                for attr in list(mesh.attributes):
                    if attr.name.startswith(ACEIdManager.PREFIX):
                        mesh.attributes.remove(attr)
                        attrs_changed = True
            if attrs_changed:
                self._attr_cache.pop(mesh.name, None)
                self._last_values.pop(mesh.name, None)
                self._write_buffer_cache.pop(mesh.name, None)
            return

        required: Dict[str, str] = {}
        keys = self.output_defs.keys() if active_keys is None else active_keys
        for attr_key in keys:
            info = self.output_defs.get(attr_key)
            if not info:
                continue
            name = self._select_attr_name(proxy.name, info)
            required[name] = info["type"]
            attr = mesh.attributes.get(name)
            if attr and attr.data_type == info["type"]:
                continue
            if attr:
                mesh.attributes.remove(attr)
                attrs_changed = True
            mesh.attributes.new(name=name, type=info["type"], domain='POINT')
            attrs_changed = True

        # Full syncs always prune, and compile-time scoped syncs may opt-in.
        if active_keys is None or prune_existing:
            for attr in list(mesh.attributes):
                if attr.name.startswith(ACEIdManager.PREFIX) and attr.name not in required:
                    mesh.attributes.remove(attr)
                    attrs_changed = True

        if attrs_changed:
            # Attribute layout changed: cached RNA handles and last written values
            # may no longer map to valid sockets, so flush per-mesh caches.
            self._attr_cache.pop(mesh.name, None)
            self._last_values.pop(mesh.name, None)
            self._write_buffer_cache.pop(mesh.name, None)

    def _get_write_buffer(self, mesh_name: str, attr_name: str, length: int) -> np.ndarray:
        """Return a reusable float32 NumPy buffer for bulk attribute writes.

        Caches one buffer per (mesh_name, attr_name, length) combination.
        A new array is allocated only when the cache entry is missing or the
        stored array's length does not match, avoiding per-frame allocations on
        steady-state writes.

        Args:
            mesh_name: Blender mesh data-block name used as the cache namespace.
            attr_name: Attribute name used as the secondary cache key.
            length: Required number of float32 elements in the buffer.

        Returns:
            A float32 NumPy array of shape (length,).
        """
        mesh_buffers = self._write_buffer_cache.setdefault(mesh_name, {})
        key = f"{attr_name}:{int(length)}"
        arr = mesh_buffers.get(key)
        if arr is None or int(arr.shape[0]) != int(length):
            arr = np.empty(int(length), dtype=np.float32)
            mesh_buffers[key] = arr
        return arr

    @staticmethod
    def _zero_value_for_type(attr_type: str):
        if attr_type == "FLOAT_VECTOR":
            return [0.0, 0.0, 0.0]
        if attr_type == "FLOAT_COLOR":
            return [0.0, 0.0, 0.0, 1.0]
        return 0.0

    @classmethod
    def _values_equal(cls, prev, curr) -> bool:
        if isinstance(prev, (int, float)) and isinstance(curr, (int, float)):
            return abs(float(prev) - float(curr)) <= 1e-6
        if isinstance(prev, np.ndarray):
            prev = prev.tolist()
        if isinstance(curr, np.ndarray):
            curr = curr.tolist()
        if isinstance(prev, (list, tuple)) and isinstance(curr, (list, tuple)):
            try:
                return bool(np.allclose(np.asarray(prev, dtype=np.float32), np.asarray(curr, dtype=np.float32), atol=1e-6))
            except Exception as exc:
                utils.note_swallowed("proxy_management.cls._values_equal", exc)
                return False
        return prev == curr

    def update_proxy_object(self, proxy: bpy.types.Object, results: Dict[str, Any], active_keys: Set[str] | None = None):
        """Write DSP result values into the proxy mesh's named attributes.

        Skips writes for float attributes whose value has not changed by more than
        1e-6 since the last call (delta suppression) to reduce Blender depsgraph
        dirtying overhead.  Transient pulse attributes (onset, flux) are exempt
        from delta suppression because their peak value and a subsequent 0.0 reset
        can arrive on consecutive frames with identical ``prev`` state.

        Falls back to writing zero values when the primary write raises an
        exception, to keep downstream GN graphs in a defined state.

        Uses foreach_set() with a reusable write buffer for multi-element meshes,
        and direct index assignment for the common single-vertex proxy case.

        Args:
            proxy: The proxy bpy.types.Object to write attribute values into.
            results: Mapping of ``"<node_id>:<socket_idx>"`` keys to result values
                (float, list, tuple, or ndarray depending on attribute type).
            active_keys: Optional subset of keys to process.  None means all keys
                present in results.
        """
        utils.assert_main_thread("update_proxy_object")
        mesh = getattr(proxy, "data", None)
        if not mesh:
            return

        keys = results.keys() if active_keys is None else active_keys

        for attr_key in keys:
            if attr_key not in results:
                continue
            value = results[attr_key]
            info = self.output_defs.get(attr_key)
            if not info:
                continue
            attr_name = self._select_attr_name(proxy.name, info)
            attr = None
            cache = self._attr_cache.setdefault(mesh.name, {})
            if attr_name in cache:
                attr = cache[attr_name]
            else:
                attr = mesh.attributes.get(attr_name)
                if attr:
                    cache[attr_name] = attr
            if not attr or not attr.data:
                continue
            last_vals = self._last_values.setdefault(mesh.name, {})
            dtype = attr.data_type
            # Transient pulse attributes (onset, flux) must never be delta-skipped:
            # stale-reuse can hold the previous value (0.0) while the correct peak
            # value arrives one frame later with the same prev=0.0, causing the write
            # to be suppressed.  These attributes reset every frame by design.
            _output_name = str(info.get("output_name", "") or "").lower()
            _is_transient = _output_name in ("onset", "spectral flux", "flux")
            if dtype == 'FLOAT':
                try:
                    val = float(value)
                except Exception as exc:
                    utils.note_swallowed("proxy_management.self.update_proxy_object", exc)
                    continue
                prev = last_vals.get(attr_name)
                if not _is_transient and isinstance(prev, float) and abs(prev - val) <= 1e-6:
                    continue
                last_vals[attr_name] = val
            elif dtype in ('FLOAT_VECTOR', 'FLOAT_COLOR'):
                arr = None
                if hasattr(value, "tolist"):
                    arr = np.asarray(value.tolist(), dtype=np.float32)
                elif isinstance(value, (list, tuple, np.ndarray)):
                    arr = np.asarray(value, dtype=np.float32)
                if arr is None:
                    continue
                target_len = 4 if dtype == 'FLOAT_COLOR' else 3
                flat = np.zeros(target_len, dtype=np.float32)
                flat[: min(target_len, arr.size)] = arr.ravel()[:target_len]
                prev = last_vals.get(attr_name)
                if isinstance(prev, np.ndarray) and prev.shape == flat.shape and np.allclose(prev, flat, atol=1e-6):
                    continue
                last_vals[attr_name] = flat
            else:
                continue

            dtype = attr.data_type
            count = len(attr.data)
            try:
                if dtype == 'FLOAT':
                    if count == 1:
                        attr.data[0].value = float(val)
                    else:
                        buf = self._get_write_buffer(mesh.name, attr_name, count)
                        buf.fill(float(val))
                        attr.data.foreach_set("value", buf)
                elif dtype == 'FLOAT_VECTOR':
                    vec = last_vals.get(attr_name)
                    if vec is None:
                        continue
                    if count == 1:
                        attr.data[0].vector = tuple(float(x) for x in vec.tolist())
                    else:
                        buf = self._get_write_buffer(mesh.name, attr_name, count * 3)
                        np.copyto(buf.reshape(count, 3), vec)
                        attr.data.foreach_set("vector", buf)
                elif dtype == 'FLOAT_COLOR':
                    col = last_vals.get(attr_name)
                    if col is None:
                        continue
                    if count == 1:
                        attr.data[0].color = tuple(float(x) for x in col.tolist())
                    else:
                        buf = self._get_write_buffer(mesh.name, attr_name, count * 4)
                        np.copyto(buf.reshape(count, 4), col)
                        attr.data.foreach_set("color", buf)
            except Exception as exc:
                utils.note_swallowed("proxy_management.self.update_proxy_object#2", exc)
                # Fallback to zeroing on failure
                with utils.logged_errors("proxy_management.self.update_proxy_object#3"):
                    if dtype == 'FLOAT':
                        if count == 1:
                            attr.data[0].value = 0.0
                        else:
                            buf = self._get_write_buffer(mesh.name, attr_name, count)
                            buf.fill(0.0)
                            attr.data.foreach_set("value", buf)
                    elif dtype == 'FLOAT_VECTOR':
                        if count == 1:
                            attr.data[0].vector = (0.0, 0.0, 0.0)
                        else:
                            buf = self._get_write_buffer(mesh.name, attr_name, count * 3)
                            buf.fill(0.0)
                            attr.data.foreach_set("vector", buf)
                    elif dtype == 'FLOAT_COLOR':
                        if count == 1:
                            attr.data[0].color = (0.0, 0.0, 0.0, 1.0)
                        else:
                            buf = self._get_write_buffer(mesh.name, attr_name, count * 4)
                            buf.fill(0.0)
                            buf.reshape(count, 4)[:, 3] = 1.0
                            attr.data.foreach_set("color", buf)
