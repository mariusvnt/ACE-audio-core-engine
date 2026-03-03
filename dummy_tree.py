import bpy  # type: ignore
import hashlib
import json
from collections import deque
from typing import Optional, Any

from .vars import ACE_NODES
from .abstract_tree import (
    ACEAbstractTree,
    ACEAbstractNode,
    ACEAbstractSocket,
    ACEAbstractLink,
    ACEAbstractOutputLink,
)
from .id_management import ACEIdManager
from . import utils

class ACEDummyNodeFactory:
    """
    Creates and Manages Blender Node Groups based on ACE_NODES configuration.
    Uses Blender 5.0+ Bundle nodes for internal logic.
    """
    
    @staticmethod
    def get_group_name(category: str, node_type: str) -> str:
        # Internal datablocks are hidden (leading ".") so the default Group list
        # does not duplicate the curated ACE add menu entries.
        return f".ACE_{category}_{node_type}"

    _NODES_CONTENT_HASH: str = ""

    @staticmethod
    def _ace_nodes_hash() -> str:
        """Compute a stable hash of ACE_NODES to detect registry changes."""
        try:
            serialized = json.dumps(ACE_NODES, sort_keys=True, default=str).encode("utf-8")
            return hashlib.blake2b(serialized, digest_size=8).hexdigest()
        except Exception:
            return ""

    @staticmethod
    def ensure_node_groups_exist():
        """
        Iterates ACE_NODES and creates missing node groups.
        A content hash of ACE_NODES short-circuits the O(N) scan when the
        registry hasn't changed since the last invocation.
        """
        try:
            _ = bpy.data.node_groups
        except AttributeError:
            utils.log("Node groups unavailable in current context; skipping ACE node group creation.", "WARNING")
            return
        current_hash = ACEDummyNodeFactory._ace_nodes_hash()
        if current_hash and current_hash == ACEDummyNodeFactory._NODES_CONTENT_HASH:
            return  # Registry unchanged — skip the full O(N) group scan.
        # 1. Standard Nodes
        for cat_key, category in ACE_NODES.items():
            for type_key, node_def in category['nodes'].items():
                group_name = ACEDummyNodeFactory.get_group_name(cat_key, type_key)
                ng = bpy.data.node_groups.get(group_name)

                if ng:
                    if ACEDummyNodeFactory._node_group_needs_refresh(ng, type_key):
                        ng.clear()
                        ng.interface.clear()
                        ACEDummyNodeFactory._build_group_content(ng, node_def, cat_key, type_key)
                    ACEDummyNodeFactory._ensure_attr_interface_visible(ng, cat_key, type_key)
                    continue
                else:
                    ng = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')
                    ACEDummyNodeFactory._build_group_content(ng, node_def, cat_key, type_key)

        # 2. Dynamic Splitter default variant (2 bands) is ensured separately to keep sockets consistent.
        try:
            from .dynamic_audio_operators import ACESplitterManager
            ACESplitterManager.refresh_existing_variants()
        except Exception:
            pass

        # Cache the hash so subsequent calls with an unchanged registry skip the scan.
        if current_hash:
            ACEDummyNodeFactory._NODES_CONTENT_HASH = current_hash

    @staticmethod
    def rebuild_node_group(cat_key: str, type_key: str):
        """Force-updates a specific Node Group definition."""
        group_name = ACEDummyNodeFactory.get_group_name(cat_key, type_key)
        
        try:
            node_def = ACE_NODES[cat_key]['nodes'][type_key]
        except KeyError:
            return

        ng = bpy.data.node_groups.get(group_name)
        if ng:
            ng.clear()             
            ng.interface.clear()   
        else:
            ng = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')
        
        ACEDummyNodeFactory._build_group_content(ng, node_def, cat_key, type_key)

    @staticmethod
    def prune_orphaned_ace_nodes(tree, scene) -> int:
        """
        Remove nodes from *tree* whose 'ace_id' property no longer corresponds
        to any active audio source in the scene.  Called once per compile so
        the node graph stays clean when sources are deleted.

        Returns the number of nodes removed.
        """
        if tree is None:
            return 0
        ace = getattr(scene, "audiocore", None)
        if ace is None:
            return 0
        live_source_ids: set[str] = set()
        for src in getattr(ace, "audio_sources", []):
            sid = getattr(src, "stable_id", "")
            if sid:
                live_source_ids.add(sid)
        removed = 0
        try:
            for node in list(getattr(tree, "nodes", [])):
                ace_id = getattr(node, "get", lambda k, d=None: None)("ace_id")
                if not ace_id:
                    continue
                # Source management nodes carry the source stable_id as their ace_id.
                from .id_management import ACEIdManager
                parsed = ACEIdManager.parse_node_id(ace_id)
                if parsed and parsed.get("category") == "source_mngmt" and parsed.get("type") == "src":
                    if ace_id not in live_source_ids:
                        tree.nodes.remove(node)
                        removed += 1
        except Exception as exc:
            utils.log(f"ACE: orphan pruning failed: {exc}", "WARNING")
        return removed

    @staticmethod
    def _node_group_needs_refresh(ng, node_type: str) -> bool:
        try:
            inputs = {item.name for item in ng.interface.items_tree if getattr(item, "in_out", None) == 'INPUT'}
            outputs = {item.name for item in ng.interface.items_tree if getattr(item, "in_out", None) == 'OUTPUT'}
        except Exception:
            return True

        if node_type == "src":
            required_inputs = {"Proxy Object"}
            required_outputs = {"Audio"}
            return not required_inputs.issubset(inputs) or not required_outputs.issubset(outputs)
        if node_type == "src_specs":
            required_inputs = {"Audio", "BPF Attribute ID", "Time Signature Attribute ID", "Beat Phase Attribute ID"}
            required_outputs = {"BPF", "Time Signature", "Beat Phase", "Audio"}
            return not required_inputs.issubset(inputs) or not required_outputs.issubset(outputs)
        if node_type == "scanner":
            required_inputs = {"Audio", "Latency Attribute ID"}
            required_outputs = {"Latency ms", "Audio"}
            return not required_inputs.issubset(inputs) or not required_outputs.issubset(outputs)
        if node_type == "preview_sink":
            # Sink: one Audio input, no outputs required.
            return "Audio" not in inputs
        if node_type == "audio_math":
            # Verify sockets follow the "Audio" / "Audio N" naming convention.
            # Checking merely for non-empty sets misses cases where sockets were
            # manually renamed, which silently breaks the audio math task wiring.
            has_audio_in = any(n == "Audio" or n.startswith("Audio ") for n in inputs)
            has_audio_out = any(n == "Audio" or n.startswith("Audio ") for n in outputs)
            return not has_audio_in or not has_audio_out
        return False

    @staticmethod
    def _attr_socket_names(category: str, node_type: str) -> tuple[str, ...]:
        if category == "stream":
            return ("Attribute ID",)
        if category == "utilities" and node_type == "src_specs":
            return ("BPF Attribute ID", "Time Signature Attribute ID", "Beat Phase Attribute ID")
        if category == "utilities" and node_type == "scanner":
            return ("Latency Attribute ID",)
        return ()

    @staticmethod
    def _ensure_attr_interface_visible(ng, category: str, node_type: str):
        for attr_name in ACEDummyNodeFactory._attr_socket_names(category, node_type):
            item = next(
                (
                    it for it in getattr(ng.interface, "items_tree", [])
                    if getattr(it, "in_out", None) == 'INPUT' and getattr(it, "name", "") == attr_name
                ),
                None,
            )
            if item is not None and hasattr(item, "hide_value"):
                try:
                    item.hide_value = False
                except Exception:
                    pass

    @staticmethod
    def _build_group_content(ng, node_def, category, node_type):
        # --- 1. INTERFACE ---
        inputs = node_def.get('inputs', [])
        for sock_type_str, sock_name in inputs:
            ng.interface.new_socket(sock_name, in_out='INPUT', socket_type=sock_type_str)

        outputs = node_def.get('outputs', [])
        for sock_type_str, sock_name in outputs:
            ng.interface.new_socket(sock_name, in_out='OUTPUT', socket_type=sock_type_str)
        
        # --- 2. INTERNALS SWITCH ---
        ACEDummyNodeFactory._create_dummy_internals(ng, inputs, outputs, category, node_type)
        ACEDummyNodeFactory._ensure_attr_interface_visible(ng, category, node_type)

    @staticmethod
    def _create_dummy_internals(ng, inputs_def, outputs_def, category, node_type):
        """
        Constructs the internal node graph logic using Bundle nodes.
        """
        input_node = ng.nodes.new('NodeGroupInput')
        output_node = ng.nodes.new('NodeGroupOutput')
        input_node.location = (-600, 0)
        output_node.location = (600, 0)

        # =====================================================================
        # CASE A: SOURCE MANAGEMENT
        # =====================================================================
        if category == 'source_mngmt':

            if node_type == 'src':
                # Logic: Object Info (Proxy) -> Combine Bundle -> Group Output

                # 1. Expose Object Socket on Group Input
                if not any(s.name == "Proxy Object" for s in ng.interface.items_tree):
                    ng.interface.new_socket("Proxy Object", in_out='INPUT', socket_type='NodeSocketObject')

                # 2. Object Info
                obj_info = ng.nodes.new('GeometryNodeObjectInfo')
                obj_info.location = (-200, 0)
                obj_info.transform_space = 'ORIGINAL'
                obj_info.inputs['As Instance'].default_value = False

                # Link Input (Proxy Object) -> Object Info
                proxy_sock_idx = len(inputs_def)
                ng.links.new(input_node.outputs[proxy_sock_idx], obj_info.inputs[0])

                # 3. Combine Bundle (Blender 5.0)
                bundle_node = ng.nodes.new('NodeCombineBundle')
                bundle_node.bundle_items.new('GEOMETRY', "Proxy Geometry")
                bundle_node.location = (200, 0)

                ng.links.new(obj_info.outputs['Geometry'], bundle_node.inputs["Proxy Geometry"])
                ng.links.new(bundle_node.outputs[0], output_node.inputs[0])

        # =====================================================================
        # CASE B: DELTA (Audio -> Audio)
        # =====================================================================
        elif category == 'delta':
            # Logic: Use Join Bundle to drive outputs
            
            if len(inputs_def) > 0:
                source_bundle = input_node.outputs[0] 
                
                for i in range(len(outputs_def)):
                    # Connect Input -> Join Bundle
                    ng.links.new(source_bundle, output_node.inputs[i])

        # =====================================================================
        # CASE C: STREAM (Audio -> Float/Vector)
        # =====================================================================
        elif category == 'stream':
            # Logic: Separate Bundle -> Sample Index (on Geometry) -> Output
            
            # 1. Attribute ID Input
            if not any(s.name == "Attribute ID" for s in ng.interface.items_tree):
                ng.interface.new_socket("Attribute ID", in_out='INPUT', socket_type='NodeSocketString')
                ng.interface.items_tree["Attribute ID"].default_value = "ACE_undefined"

            attr_id_idx = len(inputs_def)

            # 2. Nodes
            # [Separate Bundle] (Extract Geometry from Bundle)
            sep_bundle = ng.nodes.new('NodeSeparateBundle') 
            sep_bundle.bundle_items.new('GEOMETRY', "Proxy Geometry")
            sep_bundle.location = (-200, 0)
            
            # [Named Attribute] (Read data)
            named_attr = ng.nodes.new('GeometryNodeInputNamedAttribute')
            named_attr.location = (0, 150)
            
            target_type = 'FLOAT'
            if len(outputs_def) > 0 and 'Vector' in outputs_def[0][0]:
                target_type = 'FLOAT_VECTOR'
                named_attr.data_type = 'FLOAT_VECTOR'
            else:
                named_attr.data_type = 'FLOAT'

            # [Sample Index]
            sample_idx = ng.nodes.new('GeometryNodeSampleIndex')
            sample_idx.location = (200, 0)
            sample_idx.data_type = target_type
            sample_idx.domain = 'POINT'

            # 3. Wiring
            
            # A. Group Input (Bundle) -> Separate Bundle
            if len(inputs_def) > 0:
                ng.links.new(input_node.outputs[0], sep_bundle.inputs[0])
            
            # B. Separate Bundle (Geometry) -> Sample Index (Geometry)
            # Assuming output 0 of Separate Bundle is the geometry content
            ng.links.new(sep_bundle.outputs[0], sample_idx.inputs['Geometry'])
            
            # C. Group Input (Attribute ID) -> Named Attribute (Name)
            ng.links.new(input_node.outputs[attr_id_idx], named_attr.inputs['Name'])
            
            # D. Named Attribute -> Sample Index (Value)
            ng.links.new(named_attr.outputs[0], sample_idx.inputs['Value'])
            
            # E. Sample Index (Value) -> Group Output
            if len(outputs_def) > 0:
                ng.links.new(sample_idx.outputs['Value'], output_node.inputs[0])

        # =====================================================================
        # CASE D: UTILITIES
        # =====================================================================
        elif category == 'utilities':

            if node_type == 'src_specs':
                # Facade-only: expose attribute-ID input sockets for routing/params.
                # Runtime Layer 2 computes real metadata and writes proxy attributes.
                attr_inputs = (
                    ("BPF Attribute ID", "ACE_undefined_bpf"),
                    ("Time Signature Attribute ID", "ACE_undefined_timesig"),
                    ("Beat Phase Attribute ID", "ACE_undefined_beatphase"),
                )
                for sock_name, default_name in attr_inputs:
                    if not any(s.name == sock_name for s in ng.interface.items_tree):
                        ng.interface.new_socket(sock_name, in_out='INPUT', socket_type='NodeSocketString')
                        ng.interface.items_tree[sock_name].default_value = default_name
                # Audio passthrough: connect bundle input to the last output socket.
                # Float/vector metadata outputs are left unconnected (default zero) —
                # facade groups must not compute.
                if len(inputs_def) > 0 and len(outputs_def) > 0:
                    audio_out_idx = len(outputs_def) - 1
                    ng.links.new(input_node.outputs[0], output_node.inputs[audio_out_idx])

            elif node_type == 'scanner':
                # Facade-only: latency is resolved by runtime commits, not GN internals.
                attr_name = "Latency Attribute ID"
                if not any(s.name == attr_name for s in ng.interface.items_tree):
                    ng.interface.new_socket(attr_name, in_out='INPUT', socket_type='NodeSocketString')
                    ng.interface.items_tree[attr_name].default_value = "ACE_undefined_latency"
                # Audio passthrough to the second output (index 1); latency float stays at default.
                if len(inputs_def) > 0 and len(output_node.inputs) > 1:
                    ng.links.new(input_node.outputs[0], output_node.inputs[1])

            elif node_type == 'preview_sink':
                # Monitoring sink: one Bundle Audio input, zero outputs.
                # No internal wiring needed — the socket interface alone defines the contract.
                # The facade must accept but not forward the bundle (it is a pure sink).
                pass

            elif node_type == 'audio_math':
                # Transparent passthrough facade: input bundle i → output bundle i.
                # The actual add/subtract/multiply logic runs in the Layer 2 backend.
                # This wiring lets the user see the audio signal flow visually.
                for i in range(min(len(inputs_def), len(outputs_def))):
                    ng.links.new(input_node.outputs[i], output_node.inputs[i])

        # =====================================================================
        # FALLBACK (unknown node type)
        # =====================================================================
        else:
            for i, _ in enumerate(inputs_def):
                reroute = ng.nodes.new('NodeReroute')
                reroute.location = (0, -i * 100)
                ng.links.new(input_node.outputs[i], reroute.inputs[0])
                if i < len(outputs_def):
                    ng.links.new(reroute.outputs[0], output_node.inputs[i])


class ACEDummyTreeReader:
    """
    Parses the active Blender Node Tree into an ACEAbstractTree.
    """
    
    def __init__(self, blender_node_tree):
        self.bl_tree = blender_node_tree
        self.abstract_tree = ACEAbstractTree()
        # Track which source roots have already visited a Blender node
        self.visited_blender_nodes: dict[Any, set[str]] = {}
        # Track ACE ID ownership inside this read pass so duplicated Blender
        # nodes (Shift+D) cannot collapse to one abstract/task node.
        self._ace_id_owner: dict[str, int] = {}
        self._node_id_by_ptr: dict[int, str] = {}

    @staticmethod
    def _node_pointer(node) -> int:
        try:
            return int(node.as_pointer())
        except Exception:
            return id(node)

    def _reserve_unique_ace_id(
        self,
        bl_node,
        ace_id: str,
        category: str,
        node_type: str,
        deterministic_seed: str = "",
    ) -> str:
        if not ace_id:
            return ""
        node_ptr = self._node_pointer(bl_node)
        cached = self._node_id_by_ptr.get(node_ptr)
        if cached:
            return cached
        owner_ptr = self._ace_id_owner.get(ace_id)
        if owner_ptr is None or owner_ptr == node_ptr:
            self._ace_id_owner[ace_id] = node_ptr
            self._node_id_by_ptr[node_ptr] = ace_id
            return ace_id

        existing_ids = list(self._ace_id_owner.keys())
        seed_payload = str(deterministic_seed or f"{category}:{node_type}")

        try:
            new_id = ACEIdManager.generate_node_id(
                category,
                node_type,
                existing_ids=existing_ids,
                deterministic_seed=seed_payload,
            )
        except Exception:
            return ace_id

        self._ace_id_owner[new_id] = node_ptr
        self._node_id_by_ptr[node_ptr] = new_id
        # Persist the disambiguated ID back into the node's custom property so
        # that duplicate nodes have stable unique IDs on the next session/reload.
        try:
            bl_node["ace_id"] = new_id
        except Exception:
            pass
        return new_id

    def _resolve_node_kind(self, bl_node) -> tuple[str, str]:
        if getattr(bl_node, "type", "") != 'GROUP':
            return "", ""
        node_tree = getattr(bl_node, "node_tree", None)
        group_name = getattr(node_tree, "name", "") if node_tree else ""
        if not group_name:
            return "", ""

        resolved_cat = ""
        resolved_type = ""
        norm_group_name = group_name.lstrip(".")
        clean_group_name = ACEIdManager._clean_string(norm_group_name).lower()
        for cat_key, category in ACE_NODES.items():
            for type_key in category.get("nodes", {}).keys():
                canonical = ACEDummyNodeFactory.get_group_name(cat_key, type_key)
                norm_canonical = canonical.lstrip(".")
                clean_canonical = ACEIdManager._clean_string(norm_canonical).lower()
                if (
                    group_name == canonical
                    or norm_group_name == norm_canonical
                    or (clean_group_name and clean_group_name == clean_canonical)
                ):
                    resolved_cat = cat_key
                    resolved_type = type_key
                    break
                if (
                    cat_key == "delta"
                    and type_key == "frequency_splitter"
                    and (
                        group_name.startswith(canonical + "_")
                        or norm_group_name.startswith(norm_canonical + "_")
                        or (clean_group_name and clean_group_name.startswith(clean_canonical))
                    )
                ):
                    resolved_cat = cat_key
                    resolved_type = type_key
                    break
                if (
                    cat_key == "utilities"
                    and type_key == "audio_math"
                    and (
                        group_name.startswith(canonical + "_")
                        or norm_group_name.startswith(norm_canonical + "_")
                        or (clean_group_name and clean_group_name.startswith(clean_canonical))
                    )
                ):
                    resolved_cat = cat_key
                    resolved_type = type_key
                    break
            if resolved_cat:
                break
        return resolved_cat, resolved_type

    def _node_kind_token(self, bl_node) -> str:
        cat, typ = self._resolve_node_kind(bl_node)
        if cat and typ:
            return f"{cat}:{typ}"
        return str(getattr(bl_node, "type", "") or "node")

    @staticmethod
    def _socket_signature(sockets) -> list[tuple[int, str, str, str, int]]:
        sig: list[tuple[int, str, str, str, int]] = []
        for idx, sock in enumerate(getattr(sockets, "__iter__", lambda: [])()):
            sock_identifier = str(getattr(sock, "identifier", "") or "")
            sock_name = str(getattr(sock, "name", "") or "")
            sock_type = str(
                getattr(sock, "bl_socket_idname", "")
                or getattr(sock, "type", "")
                or type(sock).__name__
            )
            linked = int(bool(getattr(sock, "is_linked", False)))
            sig.append((int(idx), sock_identifier, sock_name, sock_type, linked))
        return sig

    @staticmethod
    def _socket_index_for(node, sock, is_input: bool) -> int:
        try:
            idx_val = getattr(sock, "index")
            if idx_val is not None:
                return int(idx_val)
        except Exception:
            pass
        try:
            seq = getattr(node, "inputs" if is_input else "outputs", [])
            for idx, ref in enumerate(seq):
                if ref == sock:
                    return int(idx)
        except Exception:
            pass
        return 0

    def _topology_signature(self, bl_node) -> dict[str, list[tuple[int, int, str]]]:
        incoming: list[tuple[int, int, str]] = []
        outgoing: list[tuple[int, int, str]] = []
        try:
            for in_sock in getattr(bl_node, "inputs", []):
                in_idx = self._socket_index_for(bl_node, in_sock, is_input=True)
                for link in getattr(in_sock, "links", []):
                    src_node = getattr(link, "from_node", None)
                    src_sock = getattr(link, "from_socket", None)
                    if src_node is None or src_sock is None:
                        continue
                    src_idx = self._socket_index_for(src_node, src_sock, is_input=False)
                    incoming.append((int(in_idx), int(src_idx), self._node_kind_token(src_node)))
        except Exception:
            pass
        try:
            for out_sock in getattr(bl_node, "outputs", []):
                out_idx = self._socket_index_for(bl_node, out_sock, is_input=False)
                for link in getattr(out_sock, "links", []):
                    dst_node = getattr(link, "to_node", None)
                    dst_sock = getattr(link, "to_socket", None)
                    if dst_node is None or dst_sock is None:
                        continue
                    dst_idx = self._socket_index_for(dst_node, dst_sock, is_input=True)
                    outgoing.append((int(out_idx), int(dst_idx), self._node_kind_token(dst_node)))
        except Exception:
            pass
        return {
            "incoming": sorted(set(incoming)),
            "outgoing": sorted(set(outgoing)),
        }

    def _build_deterministic_node_seed(
        self,
        bl_node,
        category: str,
        node_type: str,
        root_id: str = "",
    ) -> str:
        source_context = ""
        if category == "source_mngmt" and node_type == "src":
            try:
                source_context = str(bl_node.get("ace_source_id", "") or "")
            except Exception:
                source_context = ""
        payload = {
            "category": str(category),
            "type": str(node_type),
            "socket_signature": {
                "inputs": self._socket_signature(getattr(bl_node, "inputs", [])),
                "outputs": self._socket_signature(getattr(bl_node, "outputs", [])),
            },
            "topology_context": self._topology_signature(bl_node),
            "source_context": str(source_context),
            "root_context": str(root_id or ""),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def build(self) -> ACEAbstractTree:
        if not self.bl_tree: return self.abstract_tree
        sources = self._find_source_nodes()
        if sources:
            self._traverse_from_sources(sources)
        self._validate_cross_source_links()
        return self.abstract_tree

    def _validate_cross_source_links(self):
        """
        Warn when a non-AudioMath node receives inputs from distinct root sources.
        Cross-source mixing is only valid through AudioMathTask; any other target
        produces silently undefined analysis output because the audio chains are
        never synchronized.
        """
        for link in self.abstract_tree.links:
            src_node = self.abstract_tree.nodes.get(link.source_node_id)
            tgt_node = self.abstract_tree.nodes.get(link.target_node_id)
            if src_node is None or tgt_node is None:
                continue
            src_roots = getattr(src_node, "root_sources", None) or set()
            tgt_roots = getattr(tgt_node, "root_sources", None) or set()
            if not src_roots or not tgt_roots:
                continue
            if src_roots.issubset(tgt_roots):
                continue
            # Cross-source link detected.
            if getattr(tgt_node, "type", "") == "audio_math":
                continue
            utils.log(
                f"ACE: Cross-source link '{src_node.name}' → '{tgt_node.name}': "
                f"only AudioMath can receive mixed sources. "
                f"Node '{tgt_node.name}' will produce undefined output.",
                "WARNING",
            )

    def _ensure_ace_id(self, bl_node, root_id: str = "") -> str:
        node_ptr = self._node_pointer(bl_node)
        cached = self._node_id_by_ptr.get(node_ptr)
        if cached:
            return cached

        resolved_cat, resolved_type = self._resolve_node_kind(bl_node)
        if not resolved_cat:
            return ""

        if resolved_cat == "source_mngmt" and resolved_type == "src":
            try:
                source_id = str(bl_node.get("ace_source_id", "") or "")
            except Exception:
                source_id = ""
            if source_id and ACEIdManager.parse_node_id(source_id):
                return self._reserve_unique_ace_id(
                    bl_node,
                    source_id,
                    resolved_cat,
                    resolved_type,
                    deterministic_seed=f"source:{source_id}",
                )

        seed = self._build_deterministic_node_seed(
            bl_node,
            resolved_cat,
            resolved_type,
            root_id=str(root_id or ""),
        )
        try:
            generated = ACEIdManager.generate_node_id(
                resolved_cat,
                resolved_type,
                existing_ids=list(self._ace_id_owner.keys()),
                deterministic_seed=seed,
            )
        except Exception:
            return ""
        return self._reserve_unique_ace_id(
            bl_node,
            generated,
            resolved_cat,
            resolved_type,
            deterministic_seed=seed,
        )

    def _find_source_nodes(self):
        sources = []
        for node in self.bl_tree.nodes:
            root_id = self._resolve_root_id(node)
            ace_id = self._ensure_ace_id(node, root_id=root_id)
            if not ace_id: continue
            
            parsed = ACEIdManager.parse_node_id(ace_id)
            if parsed and parsed['category'] == 'source_mngmt' and parsed['type'] == 'src':
                # Ignore ad-hoc source nodes; only managed placeholders carry a source binding.
                source_id = ""
                try:
                    source_id = str(node.get("ace_source_id", "") or "")
                except Exception:
                    source_id = ""
                if not source_id or not ACEIdManager.parse_node_id(source_id):
                    continue
                sources.append(node)
        return sources

    def _traverse_from_sources(self, start_nodes):
        queue = deque()
        for src_node in start_nodes:
            root_id = self._resolve_root_id(src_node)
            queue.append((src_node, root_id, "", 0, None))

        visited_states: set[tuple[int, str, str, int, str]] = set()
        link_keys: set[tuple[str, str, int, int]] = {
            (l.source_node_id, l.target_node_id, int(l.source_socket_idx), int(l.target_socket_idx))
            for l in self.abstract_tree.links
        }
        output_link_keys: set[tuple[str, int, int]] = {
            (
                str(l.source_node_id),
                int(l.source_socket_idx),
                int(l.target_socket_idx),
            )
            for l in getattr(self.abstract_tree, "group_output_links", []) or []
        }

        while queue:
            current_bl_node, root_id, prev_ace_id, prev_output_idx, incoming_socket = queue.popleft()

            state_key = (
                id(current_bl_node),
                root_id or "",
                prev_ace_id or "",
                int(prev_output_idx) if prev_output_idx is not None else 0,
                self._socket_state_key(incoming_socket),
            )
            if state_key in visited_states:
                continue
            visited_states.add(state_key)

            abstract_node = self._parse_node(current_bl_node, root_id)
            current_ace_id = abstract_node.id if abstract_node else ""

            if abstract_node:
                # Merge roots if node already exists
                if abstract_node.id in self.abstract_tree.nodes:
                    self.abstract_tree.nodes[abstract_node.id].root_sources.update(abstract_node.root_sources)
                else:
                    self.abstract_tree.add_node(abstract_node)

            # Capture ACE outputs connected to Group Output so the runtime can
            # build default preview sinks without requiring manual nodes.
            if (
                getattr(current_bl_node, "type", "") == "GROUP_OUTPUT"
                and prev_ace_id
                and incoming_socket is not None
            ):
                target_socket_idx = self._resolve_target_socket_idx(current_bl_node, incoming_socket)
                if target_socket_idx is None:
                    target_socket_idx = 0
                link_key = (str(prev_ace_id), int(prev_output_idx), int(target_socket_idx))
                if link_key not in output_link_keys:
                    output_link_keys.add(link_key)
                    target_socket_name = str(getattr(incoming_socket, "name", "") or "")
                    self.abstract_tree.add_group_output_link(
                        ACEAbstractOutputLink(
                            source_node_id=str(prev_ace_id),
                            source_socket_idx=int(prev_output_idx),
                            target_socket_idx=int(target_socket_idx),
                            target_socket_name=target_socket_name,
                        )
                    )

            # If we reached an ACE node after traversing non-ACE nodes, bridge the link.
            if current_ace_id and prev_ace_id and incoming_socket is not None and current_ace_id != prev_ace_id:
                target_socket_idx = self._resolve_target_socket_idx(current_bl_node, incoming_socket)
                if target_socket_idx is None:
                    try:
                        if len(getattr(current_bl_node, "inputs", [])) > 0:
                            target_socket_idx = 0
                    except Exception:
                        target_socket_idx = None

                if target_socket_idx is not None:
                    link_key = (prev_ace_id, current_ace_id, int(prev_output_idx), int(target_socket_idx))
                    if link_key not in link_keys:
                        link_keys.add(link_key)
                        self.abstract_tree.links.append(ACEAbstractLink(
                            source_node_id=prev_ace_id,
                            target_node_id=current_ace_id,
                            source_socket_idx=int(prev_output_idx),
                            target_socket_idx=int(target_socket_idx),
                        ))

            carry_ace_id = current_ace_id or prev_ace_id
            for output_idx, socket in enumerate(getattr(current_bl_node, "outputs", [])):
                if not getattr(socket, "is_linked", False):
                    continue
                next_output_idx = output_idx if current_ace_id else prev_output_idx
                for link in getattr(socket, "links", []):
                    target_node = getattr(link, "to_node", None)
                    if not target_node:
                        continue
                    queue.append((target_node, root_id, carry_ace_id, next_output_idx, getattr(link, "to_socket", None)))

    @staticmethod
    def _socket_state_key(sock) -> str:
        if sock is None:
            return "NONE"
        identifier = str(getattr(sock, "identifier", "") or "")
        if identifier:
            return f"id:{identifier}"
        name = str(getattr(sock, "name", "") or "")
        if name:
            return f"name:{name}"
        try:
            return f"idx:{int(getattr(sock, 'index'))}"
        except Exception:
            return f"obj:{id(sock)}"

    @staticmethod
    def _resolve_target_socket_idx(target_node, to_socket) -> Optional[int]:
        target_socket_idx = None
        try:
            target_socket_idx = to_socket.index  # type: ignore[attr-defined]
        except Exception:
            target_socket_idx = None
        if target_socket_idx is not None:
            return target_socket_idx

        try:
            target_socket_idx = next((i for i, s in enumerate(target_node.inputs) if s == to_socket), None)
        except Exception:
            target_socket_idx = None
        if target_socket_idx is not None:
            return target_socket_idx

        sock_name = ""
        try:
            sock_name = getattr(to_socket, "name", "") or ""
        except Exception:
            sock_name = ""
        if sock_name:
            try:
                target_socket_idx = next(
                    (i for i, s in enumerate(target_node.inputs) if getattr(s, "name", "") == sock_name),
                    None,
                )
            except Exception:
                target_socket_idx = None
        if target_socket_idx is not None:
            return target_socket_idx

        sock_identifier = ""
        try:
            sock_identifier = getattr(to_socket, "identifier", "") or ""
        except Exception:
            sock_identifier = ""
        if sock_identifier:
            try:
                target_socket_idx = next(
                    (i for i, s in enumerate(target_node.inputs) if getattr(s, "identifier", "") == sock_identifier),
                    None,
                )
            except Exception:
                target_socket_idx = None
        return target_socket_idx

    def _resolve_root_id(self, bl_node) -> str:
        """
        Try to derive a stable source identifier from the source node:
        prefer explicit ace_source_id, then the proxy object name, else the node ace_id.
        """
        root_id = ""
        try:
            root_id = bl_node.get("ace_source_id", "")
        except Exception:
            root_id = ""
        if not root_id:
            try:
                proxy_sock = next((s for s in bl_node.inputs if getattr(s, "name", "") == "Proxy Object"), None)
                if proxy_sock and getattr(proxy_sock, "default_value", None):
                    proxy_obj = proxy_sock.default_value
                    root_id = getattr(proxy_obj, "name", "") or ""
            except Exception:
                root_id = ""
        return root_id

    @staticmethod
    def _attribute_socket_to_output_index(sock_name: str) -> Optional[int]:
        mapping = {
            "Attribute ID": 0,
            "BPF Attribute ID": 0,
            "Time Signature Attribute ID": 1,
            "Beat Phase Attribute ID": 2,
            "Latency Attribute ID": 0,
        }
        return mapping.get(sock_name)

    def _parse_node(self, bl_node, root_id: str) -> Optional[ACEAbstractNode]:
        ace_id = self._ensure_ace_id(bl_node, root_id=root_id)
        if not ace_id:
            return None
        parsed = ACEIdManager.parse_node_id(ace_id)
        if not parsed: return None
        
        cat, typ = parsed['category'], parsed['type']
        
        # Splitter Handling
        if cat == 'delta' and typ == 'frequency_splitter':
            input_count = len(bl_node.inputs)
            output_count = len(bl_node.outputs)
            input_defs = [('NodeSocketBundle', 'Audio In')]
            for k in range(input_count - 1):
                input_defs.append(('NodeSocketFloat', f"Split Freq {k+1}"))
            output_defs = [('NodeSocketBundle', 'Band')] * output_count
        elif cat == 'utilities' and typ == 'audio_math':
            input_count = len(bl_node.inputs)
            output_count = len(bl_node.outputs)
            input_defs = [('NodeSocketBundle', f"Audio {k+1}") for k in range(input_count)]
            output_defs = [('NodeSocketBundle', 'Audio')] * max(output_count, 1)
        else:
            try:
                node_def = ACE_NODES[cat]['nodes'][typ]
                input_defs = node_def.get('inputs', [])
                output_defs = node_def.get('outputs', [])
            except KeyError:
                return None

        # Build Inputs
        inputs = []
        params = {}
        
        for i, sock in enumerate(bl_node.inputs):
            # Safe type check
            bl_type_str = "NodeSocketFloat"
            if i < len(input_defs):
                bl_type_str = input_defs[i][0]
            elif "Attribute ID" in str(getattr(sock, "name", "") or ""):
                bl_type_str = "NodeSocketString"
            
            inputs.append(ACEAbstractSocket(
                id=f"{ace_id}_in_{i}",
                name=sock.name,
                socket_idx=i,
                is_output=False,
                data_type=bl_type_str,
                default_value=getattr(sock, 'default_value', None)
            ))
            
            # Param Capture
            if not sock.is_linked:
                val = getattr(sock, 'default_value', None)
                if hasattr(val, "to_tuple"): val = val.to_tuple()
                elif hasattr(val, "to_list"): val = val.to_list()
                if val is not None:
                     params[f"in_{i}"] = val

            attr_out_idx = self._attribute_socket_to_output_index(sock.name)
            if attr_out_idx is not None:
                val = getattr(sock, 'default_value', None)
                if isinstance(val, str):
                    attr_name = val.strip()
                    if attr_name:
                        params[f"attr_id_out_{attr_out_idx}"] = attr_name
                    is_custom = bool(attr_name)
                    params[f"attr_id_custom_out_{attr_out_idx}"] = bool(is_custom)

        # Custom parameters (e.g., Audio Math operation)
        op_value = bl_node.get("ace_op")
        if isinstance(op_value, str):
            params["op"] = op_value
        preview_enabled = bl_node.get("ace_preview_enabled", True)
        params["preview_enabled"] = bool(preview_enabled)

        # Build Outputs
        outputs = []
        for i, sock in enumerate(bl_node.outputs):
            bl_type_str = "NodeSocketFloat"
            if i < len(output_defs):
                bl_type_str = output_defs[i][0]
                
            outputs.append(ACEAbstractSocket(
                id=f"{ace_id}_out_{i}",
                name=sock.name,
                socket_idx=i,
                is_output=True,
                data_type=bl_type_str
            ))
            
        return ACEAbstractNode(
            id=ace_id,
            name=bl_node.name,
            type=typ,
            category=cat,
            inputs=inputs,
            outputs=outputs,
            parameters=params,
            root_sources={root_id} if root_id else set()
        )

# --- MODULE REGISTRATION ---

def register():
    try:
        ACEDummyNodeFactory.ensure_node_groups_exist()
    except AttributeError:
        # Restricted context (e.g., background handler). Skip building for now.
        utils.log("Skipping node group creation due to restricted data access.", "WARNING")

def unregister():
    pass
