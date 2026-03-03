import bpy # type: ignore
import numpy as np
from typing import Dict, Set, Any

from .id_management import ACEIdManager
from .abstract_tree import ACEAbstractTree
from .dummy_tree import ACEDummyNodeFactory

class ACEProxyManager:
    """
    Manages proxy objects (creation, placement) and acts as the attribute writer.
    A single class keeps lifecycle and write-path coherent.
    """

    # "Pro" name for the invisible collection
    COLLECTION_NAME = "ACE_Internals"
    SONAR_OBJECT_NAME = "ACE_Sonar"
    SONAR_MODIFIER_NAME = "ACE_Sonar_Mod"
    SONAR_TREE_NAME = "ACE_Sonar_Tree"

    def __init__(self, abstract_tree: ACEAbstractTree | None = None):
        self.abstract_tree = abstract_tree
        self.output_defs: Dict[str, Dict[str, str]] = {}
        self._attr_cache: Dict[str, Dict[str, Any]] = {}
        self._last_values: Dict[str, Dict[str, Any]] = {}
        self._write_buffer_cache: Dict[str, Dict[str, np.ndarray]] = {}
        if abstract_tree:
            self._cache_output_defs()

    # =========================================================================
    # 1. COLLECTION MANAGEMENT
    # =========================================================================

    @staticmethod
    def _resolve_scene(scene=None):
        if scene is not None:
            return scene
        active_scene = getattr(getattr(bpy, "context", None), "scene", None)
        if active_scene is not None:
            return active_scene
        scenes = list(getattr(bpy.data, "scenes", []) or [])
        return scenes[0] if scenes else None

    @staticmethod
    def _ensure_internal_collection(scene=None) -> bpy.types.Collection:
        """
        Creates or retrieves the hidden collection for proxy objects.
        Configures it to be blue and invisible.
        """
        if ACEProxyManager.COLLECTION_NAME in bpy.data.collections:
            col = bpy.data.collections[ACEProxyManager.COLLECTION_NAME]
        else:
            col = bpy.data.collections.new(ACEProxyManager.COLLECTION_NAME)
        target_scene = ACEProxyManager._resolve_scene(scene)
        if target_scene:
            children = target_scene.collection.children
            if children.get(col.name) is None:
                children.link(col)

        if col.color_tag != 'COLOR_05':
            col.color_tag = 'COLOR_05'

        if not col.hide_viewport:
            col.hide_viewport = True
        if not col.hide_render:
            col.hide_render = True

        return col

    # =========================================================================
    # 2. PROXY OBJECT LIFECYCLE
    # =========================================================================

    @staticmethod
    def ensure_proxy_for_source(source_prop, scene=None) -> bpy.types.Object:
        """
        Creates or Retrieves the single-vertex proxy mesh for an Audio Source.
        Updates the source_prop.proxy_object_name reference.
        """
        try:
            ACEIdManager.ensure_source_uid(source_prop)
        except Exception:
            pass
        stable_id = getattr(source_prop, "stable_id", "") or ""
        source_name = getattr(source_prop, "name", "") or "Source"
        proxy_name = ACEIdManager.generate_source_proxy_name(
            source_name,
            stable_seed=stable_id or source_name,
        )

        existing_ref = getattr(source_prop, "proxy_object_name", "") or ""
        if existing_ref:
            existing_obj = bpy.data.objects.get(existing_ref)
            if existing_obj and existing_obj.type == 'MESH':
                if existing_obj.name != proxy_name:
                    existing_obj.name = proxy_name
                if source_prop.proxy_object_name != existing_obj.name:
                    source_prop.proxy_object_name = existing_obj.name
                ACEProxyManager._move_to_internal_collection(existing_obj, scene=scene)
                return existing_obj

        if proxy_name in bpy.data.objects:
            obj = bpy.data.objects[proxy_name]
            if obj.type == 'MESH':
                ACEProxyManager._move_to_internal_collection(obj, scene=scene)
                if source_prop.proxy_object_name != obj.name:
                    source_prop.proxy_object_name = obj.name
                return obj

        mesh = bpy.data.meshes.new(f"{proxy_name}_Mesh")
        obj = bpy.data.objects.new(proxy_name, mesh)

        target_col = ACEProxyManager._ensure_internal_collection(scene=scene)
        target_col.objects.link(obj)

        mesh.from_pydata([(0,0,0)], [], [])

        obj.lock_location = (True, True, True)
        obj.lock_rotation = (True, True, True)
        obj.lock_scale = (True, True, True)

        source_prop.proxy_object_name = obj.name

        return obj

    @staticmethod
    def _move_to_internal_collection(obj, scene=None):
        """Helper to ensure existing proxies are moved to the hidden collection."""
        target_col = ACEProxyManager._ensure_internal_collection(scene=scene)
        if obj.name in target_col.objects:
            return

        target_col.objects.link(obj)
        for col in obj.users_collection:
            if col != target_col:
                col.objects.unlink(obj)

    @staticmethod
    def delete_proxy_for_source(source_prop):
        """
        Removes the proxy object and its mesh data.
        """
        if not source_prop.proxy_object_name:
            return

        obj = bpy.data.objects.get(source_prop.proxy_object_name)
        if obj:
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh and mesh.users == 0:
                bpy.data.meshes.remove(mesh, do_unlink=True)

        source_prop.proxy_object_name = ""

    # =========================================================================
    # 3. ATTRIBUTE SYNCHRONIZATION / WRITING
    # =========================================================================

    def set_abstract_tree(self, abstract_tree: ACEAbstractTree):
        self.abstract_tree = abstract_tree
        self.output_defs.clear()
        self._cache_output_defs()

    def _cache_output_defs(self):
        """Precompute attribute slugs and data types for all stream/spec outputs."""
        if not self.abstract_tree:
            return

        for node in self.abstract_tree.nodes.values():
            is_stream = node.category == 'stream'
            is_source_specs = node.type == 'src_specs' and node.category == 'utilities'
            is_scanner = node.type == 'scanner' and node.category == 'utilities'
            if not (is_stream or is_source_specs or is_scanner):
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
                }

    @staticmethod
    def _resolve_attr_type(sock_type: str) -> str:
        """Map socket data type to a Blender attribute type."""
        attr_type = 'FLOAT'
        if sock_type in ('VECTOR', 'NodeSocketVector'):
            attr_type = 'FLOAT_VECTOR'
        elif sock_type in ('RGBA', 'NodeSocketColor'):
            attr_type = 'FLOAT_COLOR'
        return attr_type

    def _select_attr_name(self, proxy_name: str, info: Dict[str, Any]) -> str:
        """
        Choose the attribute name for a given output, respecting user-provided Attribute ID.
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
        """Create (or fix type of) attributes on the proxy mesh for every output node."""
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
        mesh_buffers = self._write_buffer_cache.setdefault(mesh_name, {})
        key = f"{attr_name}:{int(length)}"
        arr = mesh_buffers.get(key)
        if arr is None or int(arr.shape[0]) != int(length):
            arr = np.empty(int(length), dtype=np.float32)
            mesh_buffers[key] = arr
        return arr

    def clear_runtime_caches(self):
        self._attr_cache.clear()
        self._last_values.clear()
        self._write_buffer_cache.clear()

    def update_proxy_object(self, proxy: bpy.types.Object, results: Dict[str, Any], active_keys: Set[str] | None = None):
        """Write current results dict into the proxy's attributes."""
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
                except Exception:
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
                        buf.reshape(count, 3)[:] = vec
                        attr.data.foreach_set("vector", buf)
                elif dtype == 'FLOAT_COLOR':
                    col = last_vals.get(attr_name)
                    if col is None:
                        continue
                    if count == 1:
                        attr.data[0].color = tuple(float(x) for x in col.tolist())
                    else:
                        buf = self._get_write_buffer(mesh.name, attr_name, count * 4)
                        buf.reshape(count, 4)[:] = col
                        attr.data.foreach_set("color", buf)
            except Exception:
                # Fallback to zeroing on failure
                try:
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
                except Exception:
                    pass

    # =========================================================================
    # 4. SONAR OBJECT / DEFAULT TREE
    # =========================================================================
    @staticmethod
    def _cube_vertices_faces(size: float = 2.0):
        h = size * 0.5
        verts = [
            (-h, -h, -h),
            (h, -h, -h),
            (h, h, -h),
            (-h, h, -h),
            (-h, -h, h),
            (h, -h, h),
            (h, h, h),
            (-h, h, h),
        ]
        faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 4, 7, 3),
            (1, 5, 6, 2),
            (0, 1, 5, 4),
            (3, 2, 6, 7),
        ]
        return verts, faces

    @staticmethod
    def _ensure_cube_mesh(mesh: bpy.types.Mesh):
        verts, faces = ACEProxyManager._cube_vertices_faces()
        try:
            mesh.clear_geometry()
        except Exception:
            pass
        mesh.from_pydata(verts, [], faces)
        mesh.update()

    @staticmethod
    def ensure_sonar_setup(scene, sources):
        """
        Create or refresh the ACE_Sonar object and its Geometry Nodes modifier.
        """
        if scene is None:
            return None

        obj = bpy.data.objects.get(ACEProxyManager.SONAR_OBJECT_NAME)
        created = False
        # Only create the sonar object once there is at least one source
        if obj is None:
            if not sources:
                return None
            # Create the design placeholder as a cube mesh.
            mesh = bpy.data.meshes.new(ACEProxyManager.SONAR_OBJECT_NAME)
            ACEProxyManager._ensure_cube_mesh(mesh)
            obj = bpy.data.objects.new(ACEProxyManager.SONAR_OBJECT_NAME, mesh)
            scene.collection.objects.link(obj)
            created = True

        mod = obj.modifiers.get(ACEProxyManager.SONAR_MODIFIER_NAME)
        if mod is None or mod.type != 'NODES':
            if mod and mod.type != 'NODES':
                obj.modifiers.remove(mod)
            mod = obj.modifiers.new(ACEProxyManager.SONAR_MODIFIER_NAME, 'NODES')

        tree = bpy.data.node_groups.get(ACEProxyManager.SONAR_TREE_NAME)
        if tree is None and sources:
            tree = bpy.data.node_groups.new(ACEProxyManager.SONAR_TREE_NAME, 'GeometryNodeTree')
        if tree is not None:
            try:
                tree.is_modifier = True
            except Exception:
                pass
        mod.node_group = tree

        if tree:
            ACEProxyManager._sync_sonar_tree(tree, sources or [])
        try:
            ace = getattr(scene, "audiocore", None)
            if ace and not getattr(ace, "gn_target_object", None):
                ace.gn_target_object = obj
        except Exception:
            pass
        if created:
            try:
                view_layer = None
                if scene is not None:
                    view_layers = list(getattr(scene, "view_layers", []) or [])
                    if view_layers:
                        view_layer = view_layers[0]
                if view_layer is None:
                    view_layer = getattr(getattr(bpy, "context", None), "view_layer", None)
                if view_layer is not None:
                    for o in view_layer.objects:
                        try:
                            o.select_set(False)
                        except TypeError:
                            o.select_set(False, view_layer=view_layer)
                    try:
                        obj.select_set(True)
                    except TypeError:
                        obj.select_set(True, view_layer=view_layer)
                    view_layer.objects.active = obj
            except Exception:
                pass
        return obj

    @staticmethod
    def _sync_sonar_tree(tree: bpy.types.NodeTree, sources):
        """
        Incrementally updates the sonar GN tree to include one managed ACE source
        node per audio source without touching user processing nodes.
        """
        if tree is None:
            return

        try:
            ACEDummyNodeFactory.ensure_node_groups_exist()
        except Exception:
            pass

        base_group_name = ACEDummyNodeFactory.get_group_name('source_mngmt', 'src')
        base_group = bpy.data.node_groups.get(base_group_name)

        # Ensure mandatory IO nodes exist
        input_node = next((n for n in tree.nodes if n.type == 'GROUP_INPUT'), None)
        if input_node is None:
            input_node = tree.nodes.new('NodeGroupInput')
            input_node.location = (-400, 0)
        output_node = next((n for n in tree.nodes if n.type == 'GROUP_OUTPUT'), None)
        if output_node is None:
            output_node = tree.nodes.new('NodeGroupOutput')
            output_node.location = (400, 0)

        # Ensure default outputs exist.
        if hasattr(tree, "interface"):
            output_names = {
                item.name
                for item in tree.interface.items_tree
                if getattr(item, "in_out", None) == 'OUTPUT'
            }
            if "Geometry" not in output_names:
                try:
                    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
                except Exception:
                    pass
            if "Audio" not in output_names:
                try:
                    tree.interface.new_socket("Audio", in_out='OUTPUT', socket_type='NodeSocketBundle')
                except Exception:
                    pass
        geom_out_sock = next((s for s in output_node.inputs if s.name == "Geometry"), None)
        if "Geometry" in input_node.outputs and geom_out_sock and not geom_out_sock.is_linked:
            try:
                tree.links.new(input_node.outputs["Geometry"], geom_out_sock)
            except Exception:
                pass

        desired_sources: list[tuple[str, str, Any]] = []
        existing_ids: list[str] = [
            node.get('ace_source_id')
            for node in tree.nodes
            if getattr(node, "get", None) and node.get('ace_source_id')
        ]
        for src in sources or []:
            try:
                src_id = ACEIdManager.ensure_source_uid(src, existing_ids=existing_ids)
            except Exception:
                src_id = getattr(src, "stable_id", "") or getattr(src, "name", "Source")
            if src_id and src_id not in existing_ids:
                existing_ids.append(src_id)
            desired_sources.append((src_id, getattr(src, "name", "Source"), src))

        def _is_managed_source_node(node: bpy.types.Node) -> bool:
            if node.type != 'GROUP':
                return False
            try:
                if node.get("ace_source_id"):
                    return True
            except Exception:
                pass

            ace_id = ""
            try:
                ace_id = node.get("ace_id", "")
            except Exception:
                ace_id = ""
            parsed = ACEIdManager.parse_node_id(ace_id) if ace_id else None
            if not parsed:
                return False
            if parsed.get("category") != "source_mngmt" or parsed.get("type") != "src":
                return False

            node_tree = getattr(node, "node_tree", None)
            if node_tree is None:
                return False
            return bool(base_group and node_tree == base_group)

        managed_nodes: list[bpy.types.Node] = [n for n in tree.nodes if _is_managed_source_node(n)]

        node_by_id: Dict[str, bpy.types.Node] = {}
        unassigned_nodes: list[bpy.types.Node] = []
        seen_unassigned_names: Set[str] = set()
        duplicate_nodes: list[bpy.types.Node] = []
        for node in managed_nodes:
            src_id = ""
            try:
                src_id = str(node.get("ace_source_id", "") or "")
            except Exception:
                src_id = ""
            if src_id:
                if src_id in node_by_id:
                    duplicate_nodes.append(node)
                    continue
                node_by_id[src_id] = node
                continue

            source_name = ""
            try:
                source_name = str(node.get("ace_source_name", "") or node.label or "")
            except Exception:
                source_name = ""
            if source_name and source_name in seen_unassigned_names:
                duplicate_nodes.append(node)
                continue
            if source_name:
                seen_unassigned_names.add(source_name)
            unassigned_nodes.append(node)

        for node in duplicate_nodes:
            try:
                tree.nodes.remove(node)
            except Exception:
                pass

        desired_ids = {sid for sid, _, _ in desired_sources if sid}
        desired_names = {name for _, name, _ in desired_sources}

        # Remove stale managed source nodes only.
        for node_id, node in list(node_by_id.items()):
            if node_id in desired_ids:
                continue
            try:
                tree.nodes.remove(node)
            except Exception:
                pass
            node_by_id.pop(node_id, None)

        reusable_nodes: list[bpy.types.Node] = []
        for node in unassigned_nodes:
            source_name = ""
            try:
                source_name = str(node.get("ace_source_name", "") or node.label or "")
            except Exception:
                source_name = ""
            if source_name and source_name not in desired_names:
                try:
                    tree.nodes.remove(node)
                except Exception:
                    pass
                continue
            reusable_nodes.append(node)

        def _take_reusable(preferred_name: str) -> bpy.types.Node | None:
            for i, node in enumerate(reusable_nodes):
                source_name = ""
                try:
                    source_name = str(node.get("ace_source_name", "") or node.label or "")
                except Exception:
                    source_name = ""
                if source_name == preferred_name:
                    return reusable_nodes.pop(i)
            if reusable_nodes:
                return reusable_nodes.pop(0)
            return None

        # Add or update source placeholders for current sources.
        for idx, (src_id, src_name, src) in enumerate(desired_sources):
            node = node_by_id.get(src_id)
            created_node = False
            if node is None:
                node = _take_reusable(src_name)
            if node is None:
                node = tree.nodes.new('GeometryNodeGroup')
                node.location = (0, -220 * idx)
                created_node = True

            if base_group and node.node_tree != base_group:
                node.node_tree = base_group
            elif not node.node_tree:
                node.node_tree = base_group

            try:
                node['ace_id'] = src_id
            except Exception:
                pass
            try:
                node['ace_source_id'] = src_id
            except Exception:
                pass
            try:
                node['ace_source_name'] = src_name
            except Exception:
                pass
            node.label = src_name
            node.name = src_name
            node_by_id[src_id] = node

            if created_node:
                try:
                    node.location.y = -220 * idx
                except Exception:
                    pass

            try:
                proxy_obj = bpy.data.objects.get(getattr(src, "proxy_object_name", ""))
                proxy_socket = next((s for s in node.inputs if getattr(s, "name", "") == "Proxy Object"), None)
                if proxy_obj and proxy_socket:
                    proxy_socket.default_value = proxy_obj
            except Exception:
                pass

        for node in reusable_nodes:
            try:
                tree.nodes.remove(node)
            except Exception:
                pass

        # Preserve user-authored Group Output sockets and links.
        # Geometry passthrough above remains guaranteed for the sonar object.

    # =========================================================================
    # 5. UTILS
    # =========================================================================
    @staticmethod
    def get_proxy_object(source_prop):
        """Safe getter"""
        if not source_prop.proxy_object_name:
            return None
        return bpy.data.objects.get(source_prop.proxy_object_name)


    
