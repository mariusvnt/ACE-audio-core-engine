import bpy # type: ignore
import numpy as np
from typing import Dict, Set, Any

from .id_management import ACEIdManager
from .abstract_tree import ACEAbstractTree
from .dummy_tree import ACEDummyNodeFactory

class ACEProxyManager:
    """Manage proxy objects and write DSP results as Blender mesh attributes.

    A proxy object is a single-vertex mesh stored in the hidden ACE_Internals
    collection.  Stream analysis results (floats, vectors, colors) are written
    to named point-domain attributes on the proxy mesh each frame so Geometry
    Nodes can read them via Named Attribute nodes.

    The manager caches attribute RNA handles and last-written values to avoid
    redundant Blender API calls on frames where values have not changed.
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
        """Return a valid Blender scene, falling back to bpy.context then bpy.data.

        Args:
            scene: A Blender scene to return directly when provided.

        Returns:
            The first available Blender scene, or None if Blender data is not
            accessible in the current context.
        """
        if scene is not None:
            return scene
        active_scene = getattr(getattr(bpy, "context", None), "scene", None)
        if active_scene is not None:
            return active_scene
        scenes = list(getattr(bpy.data, "scenes", []) or [])
        return scenes[0] if scenes else None

    @staticmethod
    def _ensure_internal_collection(scene=None) -> bpy.types.Collection:
        """Create or retrieve the hidden ACE_Internals collection.

        Links the collection into the target scene if not already present, marks
        it blue (COLOR_05) for visual distinction, and hides it from both viewport
        and render so proxy objects never appear in the user's scene.

        Args:
            scene: Optional Blender scene to link the collection into.  Resolved
                via _resolve_scene() when None.

        Returns:
            The ACE_Internals bpy.types.Collection.
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
        """Create or retrieve the single-vertex proxy mesh for an audio source.

        Derives the proxy name deterministically from the source name and
        stable_id so the same proxy is found across sessions without relying
        on Python object identity.  Moves the proxy into the hidden
        ACE_Internals collection if it is not there already.

        Args:
            source_prop: An ACEAudioSource property-group instance.
            scene: Optional Blender scene; resolved via _resolve_scene() when None.

        Returns:
            The proxy Blender Object.  Also writes source_prop.proxy_object_name.
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
        """Move an object into ACE_Internals, unlinking it from all other collections.

        Args:
            obj: The bpy.types.Object to relocate.
            scene: Optional Blender scene; forwarded to _ensure_internal_collection().
        """
        target_col = ACEProxyManager._ensure_internal_collection(scene=scene)
        if obj.name in target_col.objects:
            return

        target_col.objects.link(obj)
        for col in obj.users_collection:
            if col != target_col:
                col.objects.unlink(obj)

    @staticmethod
    def delete_proxy_for_source(source_prop):
        """Remove the proxy object and its orphaned mesh data block from Blender.

        Clears source_prop.proxy_object_name on completion so subsequent calls
        are no-ops.  Safe to call when no proxy has been created yet.

        Args:
            source_prop: An ACEAudioSource property-group instance whose
                proxy_object_name identifies the object to delete.
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
        """Replace the active abstract tree and rebuild the output definition cache.

        Clears all previously cached output definitions before re-running
        _cache_output_defs() so stale attribute slugs from the old tree are not
        carried forward.

        Args:
            abstract_tree: The new ACEAbstractTree to use for output definitions.
        """
        self.abstract_tree = abstract_tree
        self.output_defs.clear()
        self._cache_output_defs()

    def _cache_output_defs(self):
        """Precompute attribute slugs and Blender data types for all writeable outputs.

        Iterates over stream, source_specs, and scanner nodes in the abstract tree.
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

    def clear_runtime_caches(self):
        """Discard all cached RNA handles, last-written values, and write buffers.

        Call this after the proxy object is replaced or after the attribute layout
        is known to have changed outside the manager's control (e.g., manual
        attribute deletion in the Properties editor) to force a full cold-path
        read on the next update_proxy_object() call.
        """
        self._attr_cache.clear()
        self._last_values.clear()
        self._write_buffer_cache.clear()

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
        """Return vertex and face lists for an axis-aligned cube.

        Args:
            size: Full edge length of the cube.  Default is 2.0 (unit cube
                spanning -1 to +1 on each axis).

        Returns:
            A (verts, faces) tuple where verts is a list of 8 (x, y, z) triples
            and faces is a list of 6 quad index tuples.
        """
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


    
