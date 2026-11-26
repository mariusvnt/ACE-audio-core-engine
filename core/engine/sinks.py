"""Sink bindings and auto attribute names for proxy and object-property outputs."""

import bpy
import json
from typing import Any, Dict, List, Optional
from ... import utils
from ..abstract_tree import ACEAbstractNode, ACEAbstractTree
from ..tasks.graph import ACEAudioGraph, ACESinkBinding
from ..id_management import ACEIdManager
from .common import FRONTEND_KIND_SHADER


class SinkResolver:
    """Sink bindings and auto attribute names for proxy and object-property outputs."""

    def __init__(self, engine):
        self.engine = engine

    @staticmethod
    def _build_auto_attr_name(
        proxy_name: str,
        node_id: str,
        output_name: str,
        socket_idx: int = 0,
        topology_sig: str = "",
    ) -> str:
        hashed_node_id = f"{node_id}|{topology_sig}" if topology_sig else node_id
        return ACEIdManager.generate_compact_attr_name(proxy_name, hashed_node_id, output_name, socket_idx)[:63]

    @staticmethod
    def _build_source_proxy_lookup(scene) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        ace = getattr(scene, "audiocore", None)
        for src in getattr(ace, "audio_sources", []) or []:
            sid = getattr(src, "stable_id", "")
            proxy_name = getattr(src, "proxy_object_name", "")
            if proxy_name:
                lookup[proxy_name] = proxy_name
            if sid and proxy_name:
                lookup[sid] = proxy_name
        return lookup

    @staticmethod
    def _lookup_proxy_object(proxy_name: str):
        if not proxy_name:
            return None
        try:
            return bpy.data.objects.get(str(proxy_name))
        except Exception as exc:
            utils.note_swallowed("sinks.SinkResolver._lookup_proxy_object", exc)
            return None

    @staticmethod
    def _resolve_proxy_name_for_output_node(
        node: Optional[ACEAbstractNode],
        source_lookup: Dict[str, str],
        unique_proxies: list[str] | tuple[str, ...],
    ) -> str:
        if node is None:
            return ""
        for root in sorted(set(getattr(node, "root_sources", set()) or set())):
            proxy_name = str(source_lookup.get(str(root), "") or "")
            if proxy_name:
                return proxy_name
        if len(unique_proxies) == 1:
            return str(unique_proxies[0] or "")
        return ""

    @staticmethod
    def _resolve_sink_data_name(proxy_name: str, node_id: str, output_name: str, socket_idx: int, explicit_name: str = "") -> str:
        explicit = str(explicit_name or "").strip()
        if explicit:
            if not explicit.startswith(ACEIdManager.PREFIX):
                explicit = ACEIdManager.PREFIX + ACEIdManager._clean_string(explicit)
            return explicit[:63]
        return ACEIdManager.generate_compact_attr_name(proxy_name, node_id, output_name, socket_idx)

    def _resolve_sink_bindings_for_graph(
        self,
        graph: ACEAudioGraph,
        abstract_tree: ACEAbstractTree,
        source_uid: str,
        proxy_name: str,
        single_source_mode: bool,
    ) -> Dict[str, List[ACESinkBinding]]:
        bindings_by_attr: Dict[str, List[ACESinkBinding]] = {}
        source_aliases = {str(v) for v in (source_uid, proxy_name) if str(v)}
        consumer_intents = list(getattr(abstract_tree, "consumer_intents", []) or [])
        frontends = dict(getattr(abstract_tree, "frontends", {}) or {})

        for attr_key in list(getattr(graph, "output_attr_keys", []) or []):
            info = self.engine.writer.output_defs.get(attr_key, {}) if self.engine.writer else {}
            node_id = str(info.get("node_id", "") or "")
            socket_idx = int(info.get("socket_idx", 0) or 0)
            node = abstract_tree.nodes.get(node_id)
            if node is None:
                bindings_by_attr[attr_key] = []
                continue
            frontend_uid = str(getattr(node, "frontend_uid", "") or info.get("frontend_uid", "") or "")
            frontend_kind = str(getattr(node, "frontend_kind", "") or "")
            output_name = str(info.get("output_name", "") or "")
            attr_type = str(info.get("type", "FLOAT") or "FLOAT")
            resolved: list[ACESinkBinding] = []

            for intent in consumer_intents:
                if str(getattr(intent, "node_id", "") or "") != node_id:
                    continue
                if int(getattr(intent, "socket_idx", 0) or 0) != socket_idx:
                    continue
                intent_frontend_uid = str(getattr(intent, "frontend_uid", "") or "")
                if intent_frontend_uid and frontend_uid and intent_frontend_uid != frontend_uid:
                    continue
                source_token = str(getattr(intent, "source_token", "") or "")
                if (not single_source_mode) and source_token and source_token not in source_aliases:
                    continue
                sink_kind = str(getattr(intent, "sink_kind", "") or "")
                explicit_name = str(getattr(intent, "explicit_name", "") or "")
                sink_name = self._resolve_sink_data_name(
                    proxy_name,
                    node_id,
                    str(getattr(intent, "output_name", "") or output_name),
                    socket_idx,
                    explicit_name=explicit_name,
                )
                if sink_kind == "PROXY_ATTRIBUTE":
                    resolved.append(
                        ACESinkBinding(
                            sink_kind="PROXY_ATTRIBUTE",
                            target_object_name=proxy_name,
                            target_data_name=sink_name,
                            attr_type=attr_type,
                        )
                    )
                elif sink_kind == "OBJECT_PROPERTY":
                    consumers = tuple(str(v) for v in (getattr(intent, "consumer_keys", ()) or ()) if str(v))
                    if not consumers and frontend_uid in frontends:
                        consumers = tuple(str(v) for v in (frontends[frontend_uid].owner_keys or ()) if str(v))
                    for object_name in consumers:
                        resolved.append(
                            ACESinkBinding(
                                sink_kind="OBJECT_PROPERTY",
                                target_object_name=str(object_name),
                                target_data_name=sink_name,
                                attr_type=attr_type,
                            )
                        )

            if not resolved:
                sink_name = self._resolve_sink_data_name(proxy_name, node_id, output_name, socket_idx)
                if frontend_kind == FRONTEND_KIND_SHADER:
                    consumer_names = ()
                    if frontend_uid in frontends:
                        consumer_names = tuple(str(v) for v in (frontends[frontend_uid].owner_keys or ()) if str(v))
                    for object_name in consumer_names:
                        resolved.append(
                            ACESinkBinding(
                                sink_kind="OBJECT_PROPERTY",
                                target_object_name=object_name,
                                target_data_name=sink_name,
                                attr_type=attr_type,
                            )
                        )
                else:
                    resolved.append(
                        ACESinkBinding(
                            sink_kind="PROXY_ATTRIBUTE",
                            target_object_name=proxy_name,
                            target_data_name=sink_name,
                            attr_type=attr_type,
                        )
                    )

            deduped: list[ACESinkBinding] = []
            seen: set[tuple[str, str, str, str]] = set()
            for binding in resolved:
                key = (
                    str(getattr(binding, "sink_kind", "") or ""),
                    str(getattr(binding, "target_object_name", "") or ""),
                    str(getattr(binding, "target_data_name", "") or ""),
                    str(getattr(binding, "attr_type", "") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(binding)
            bindings_by_attr[attr_key] = deduped
        return bindings_by_attr

    def _sync_auto_attr_ids(
        self,
        bl_tree: bpy.types.NodeTree,
        abstract_tree: ACEAbstractTree,
        scene,
    ) -> tuple[bool, set[str]]:
        changed = False
        changed_node_ids: set[str] = set()
        source_lookup = self._build_source_proxy_lookup(scene)
        unique_proxies = list(dict.fromkeys(source_lookup.values()))
        incoming_by_target: Dict[str, list[tuple[str, int, int]]] = {}
        for link in (getattr(abstract_tree, "links", []) or []):
            source_node_id = str(getattr(link, "source_node_id", "") or "")
            target_node_id = str(getattr(link, "target_node_id", "") or "")
            if not source_node_id or not target_node_id:
                continue
            incoming_by_target.setdefault(target_node_id, []).append(
                (
                    source_node_id,
                    int(getattr(link, "source_socket_idx", 0) or 0),
                    int(getattr(link, "target_socket_idx", 0) or 0),
                )
            )

        for bl_node in getattr(bl_tree, "nodes", []):
            ace_id = ""
            try:
                ace_id = bl_node.get("ace_id", "")
            except Exception as exc:
                utils.note_swallowed("sinks.SinkResolver._sync_auto_attr_ids", exc)
                ace_id = ""
            if not ace_id:
                continue

            abs_node = abstract_tree.nodes.get(ace_id)
            if not abs_node:
                continue

            is_stream = abs_node.category == "stream"
            is_stream_accumulator = abs_node.type == "stream_accumulator" and abs_node.category == "utilities"
            if not (is_stream or is_stream_accumulator):
                continue

            proxy_name = self._resolve_proxy_name_for_output_node(abs_node, source_lookup, unique_proxies)
            if not proxy_name:
                continue

            root_sig = json.dumps(sorted(set(getattr(abs_node, "root_sources", set()) or set())))
            upstream_sig = json.dumps(sorted(incoming_by_target.get(abs_node.id, [])))
            topology_sig = f"{root_sig}|{upstream_sig}"

            desired_map: Dict[str, str] = {}
            socket_map: Dict[str, str] = {}
            if is_stream and len(abs_node.outputs) > 0:
                for out_idx, out_sock in enumerate(abs_node.outputs):
                    desired_map[str(out_idx)] = self._build_auto_attr_name(
                        proxy_name,
                        abs_node.id,
                        out_sock.name,
                        out_idx,
                        topology_sig=topology_sig,
                    )
                    socket_name = "Attribute ID" if out_idx == 0 else f"Attribute ID {int(out_idx + 1)}"
                    socket_map[socket_name] = str(out_idx)
            elif is_stream_accumulator and len(abs_node.outputs) > 0:
                desired_map["0"] = self._build_auto_attr_name(
                    proxy_name,
                    abs_node.id,
                    abs_node.outputs[0].name,
                    0,
                    topology_sig=topology_sig,
                )
                socket_map["Attribute ID"] = "0"

            if not desired_map:
                continue

            with utils.logged_errors("sinks.SinkResolver._sync_auto_attr_ids#2"):
                node_tree = getattr(bl_node, "node_tree", None)
                if node_tree is not None:
                    for item in getattr(node_tree.interface, "items_tree", []):
                        if (
                            getattr(item, "in_out", None) == "INPUT"
                            and getattr(item, "name", "") in socket_map
                            and hasattr(item, "hide_value")
                        ):
                            item.hide_value = False
            for sock in getattr(bl_node, "inputs", []):
                if getattr(sock, "name", "") in socket_map:
                    with utils.logged_errors("sinks.SinkResolver._sync_auto_attr_ids#3"):
                        sock.hide_value = False

            socket_bindings: list[tuple[Any, str, str]] = []
            for sock in getattr(bl_node, "inputs", []):
                socket_key = socket_map.get(getattr(sock, "name", ""))
                if socket_key is None:
                    continue
                current = str(getattr(sock, "default_value", "") or "")
                target = desired_map.get(socket_key, "")
                socket_bindings.append((sock, current, target))

            for sock, current, target in socket_bindings:
                if not target or current == target:
                    continue
                with utils.logged_errors("sinks.SinkResolver._sync_auto_attr_ids#4"):
                    sock.default_value = target
                    changed = True
                    changed_node_ids.add(str(abs_node.id))

            with utils.logged_errors("sinks.SinkResolver._sync_auto_attr_ids#5"):
                if str(bl_node.get("ace_auto_attr_topology_json", "")) != topology_sig:
                    bl_node["ace_auto_attr_topology_json"] = topology_sig
                    changed = True
        return changed, changed_node_ids

    @staticmethod
    def _attribute_socket_to_output_index(sock_name: str) -> Optional[int]:
        name = str(sock_name or "").strip()
        if name == "Attribute ID":
            return 0
        if name.startswith("Attribute ID "):
            suffix = name[len("Attribute ID "):].strip()
            if suffix.isdigit():
                idx = int(suffix) - 1
                if idx >= 0:
                    return int(idx)
        return None

    def _refresh_abstract_attr_params_from_tree(
        self,
        abstract_tree: ACEAbstractTree,
        bl_tree: bpy.types.NodeTree,
        node_ids: set[str],
    ):
        if not node_ids:
            return
        node_id_set = {str(nid) for nid in node_ids if nid}
        if not node_id_set:
            return

        for bl_node in getattr(bl_tree, "nodes", []):
            ace_id = ""
            try:
                ace_id = str(bl_node.get("ace_id", "") or "")
            except Exception as exc:
                utils.note_swallowed("sinks.SinkResolver._refresh_abstract_attr_params_from_tree", exc)
                ace_id = ""
            if not ace_id or ace_id not in node_id_set:
                continue

            abs_node = abstract_tree.nodes.get(ace_id)
            if abs_node is None:
                continue

            params = dict(getattr(abs_node, "parameters", {}) or {})
            for idx, sock in enumerate(getattr(bl_node, "inputs", [])):
                if bool(getattr(sock, "is_linked", False)):
                    params.pop(f"in_{idx}", None)
                else:
                    val = getattr(sock, "default_value", None)
                    if hasattr(val, "to_tuple"):
                        val = val.to_tuple()
                    elif hasattr(val, "to_list"):
                        val = val.to_list()
                    if val is not None:
                        params[f"in_{idx}"] = val
                    else:
                        params.pop(f"in_{idx}", None)

                out_idx = self._attribute_socket_to_output_index(str(getattr(sock, "name", "") or ""))
                if out_idx is None:
                    continue
                raw_val = getattr(sock, "default_value", None)
                attr_name = str(raw_val).strip() if isinstance(raw_val, str) else ""
                if attr_name:
                    params[f"attr_id_out_{out_idx}"] = attr_name
                else:
                    params.pop(f"attr_id_out_{out_idx}", None)
                params[f"attr_id_custom_out_{out_idx}"] = bool(attr_name)

            abs_node.parameters = params

    def _sync_shader_output_ids(
        self,
        bl_tree: bpy.types.NodeTree,
        abstract_tree: ACEAbstractTree,
        scene,
    ) -> tuple[bool, set[str]]:
        """Best-effort shader-side output-name sync for editable ID sockets."""
        utils.assert_main_thread("_sync_shader_output_ids")
        changed = False
        changed_node_ids: set[str] = set()
        source_lookup = self._build_source_proxy_lookup(scene)
        unique_proxies = list(dict.fromkeys(source_lookup.values()))
        shader_tree_mutations_unsafe = not self.engine._structural_tree_writes_allowed(
            scene,
            tree=bl_tree,
            frontend_kind=FRONTEND_KIND_SHADER,
        )
        shader_factory = None
        try:
            from ...nodes.shader.factory import ACEShaderNodeFactory as _ACEShaderNodeFactory
            shader_factory = _ACEShaderNodeFactory
        except Exception as exc:
            utils.note_swallowed("sinks.SinkResolver._sync_shader_output_ids", exc)
            shader_factory = None
        if shader_factory is not None and not shader_tree_mutations_unsafe:
            with utils.logged_errors("sinks.SinkResolver._sync_shader_output_ids#2"):
                if shader_factory.cleanup_tree_helper_nodes(bl_tree):
                    changed = True

        incoming_by_target: Dict[str, list[tuple[str, int, int]]] = {}
        for link in (getattr(abstract_tree, "links", []) or []):
            source_node_id = str(getattr(link, "source_node_id", "") or "")
            target_node_id = str(getattr(link, "target_node_id", "") or "")
            if not source_node_id or not target_node_id:
                continue
            incoming_by_target.setdefault(target_node_id, []).append(
                (
                    source_node_id,
                    int(getattr(link, "source_socket_idx", 0) or 0),
                    int(getattr(link, "target_socket_idx", 0) or 0),
                )
            )

        for bl_node in getattr(bl_tree, "nodes", []):
            try:
                ace_id = str(bl_node.get("ace_id", "") or "")
            except Exception as exc:
                utils.note_swallowed("sinks.SinkResolver._sync_shader_output_ids#3", exc)
                ace_id = ""
            if not ace_id:
                continue
            abs_node = abstract_tree.nodes.get(ace_id)
            if abs_node is None:
                continue
            is_stream = abs_node.category == "stream"
            is_rolling_avg = abs_node.type == "util_rolling_avg" and abs_node.category == "utilities"
            is_stream_accumulator = abs_node.type == "stream_accumulator" and abs_node.category == "utilities"
            if not (is_stream or is_rolling_avg or is_stream_accumulator):
                continue

            root_sig = json.dumps(sorted(set(getattr(abs_node, "root_sources", set()) or set())))
            upstream_sig = json.dumps(sorted(incoming_by_target.get(abs_node.id, [])))
            topology_sig = f"{root_sig}|{upstream_sig}"
            proxy_name = self._resolve_proxy_name_for_output_node(abs_node, source_lookup, unique_proxies)

            desired_map: Dict[str, str] = {}
            socket_map: Dict[str, str] = {}
            if proxy_name:
                # One property per node instance: no topology signature, so the name matches the other sync paths.
                if is_stream and len(abs_node.outputs) > 0:
                    for out_idx, out_sock in enumerate(abs_node.outputs):
                        desired_map[str(out_idx)] = self._build_auto_attr_name(
                            proxy_name,
                            abs_node.id,
                            out_sock.name,
                            out_idx,
                            topology_sig="",
                        )
                        socket_name = "Attribute ID" if out_idx == 0 else f"Attribute ID {int(out_idx + 1)}"
                        socket_map[socket_name] = str(out_idx)
                elif is_rolling_avg and len(abs_node.outputs) > 0:
                    desired_map["0"] = self._build_auto_attr_name(
                        proxy_name,
                        abs_node.id,
                        abs_node.outputs[0].name,
                        0,
                        topology_sig="",
                    )
                    socket_map["Attribute ID"] = "0"
                elif is_stream_accumulator and len(abs_node.outputs) > 0:
                    desired_map["0"] = self._build_auto_attr_name(
                        proxy_name,
                        abs_node.id,
                        abs_node.outputs[0].name,
                        0,
                        topology_sig="",
                    )
                    socket_map["Attribute ID"] = "0"

            if desired_map:
                for sock in getattr(bl_node, "inputs", []):
                    socket_key = socket_map.get(str(getattr(sock, "name", "") or ""))
                    if socket_key is None:
                        continue
                    target = desired_map.get(socket_key, "")
                    current = str(getattr(sock, "default_value", "") or "")
                    if target and target != current:
                        with utils.logged_errors("sinks.SinkResolver._sync_shader_output_ids#4"):
                            sock.default_value = target
                            changed = True
                            changed_node_ids.add(str(abs_node.id))
                with utils.logged_errors("sinks.SinkResolver._sync_shader_output_ids#5"):
                    if str(bl_node.get("ace_auto_attr_topology_json", "")) != topology_sig:
                        bl_node["ace_auto_attr_topology_json"] = topology_sig
                        changed = True
            # Only a full render (use_lock_interface off) is unsafe; viewport renders re-sync every frame.
            if shader_factory is not None and not shader_tree_mutations_unsafe:
                with utils.logged_errors("sinks.SinkResolver._sync_shader_output_ids#6"):
                    _names_by_idx = {int(k): str(v) for k, v in desired_map.items() if v} or None
                    if shader_factory.sync_node_instance_group_bindings(
                        bl_node,
                        proxy_name=proxy_name,
                        scene=scene,
                        attr_names_by_idx=_names_by_idx,
                    ):
                        changed = True
                        changed_node_ids.add(str(abs_node.id))
            # Store the names so consumer intents match the helper Attribute nodes.
            if desired_map:
                params = dict(getattr(abs_node, "parameters", {}) or {})
                for out_key, attr_name in desired_map.items():
                    if attr_name:
                        params[f"attr_id_out_{out_key}"] = attr_name
                        params[f"attr_id_custom_out_{out_key}"] = True
                abs_node.parameters = params
                changed = True
                changed_node_ids.add(str(abs_node.id))
        return changed, changed_node_ids
