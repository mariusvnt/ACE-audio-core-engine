"""ACE_Sonar: the object and trees that hold managed source nodes."""

import bpy # type: ignore
from typing import Dict, Any
from ..id_management import ACEIdManager
from ...nodes.frontend import FRONTEND_KIND_GEOMETRY, FRONTEND_KIND_SHADER, ensure_frontend_uid
from ...nodes.geometry.factory import ACEDummyNodeFactory
from ... import utils


class Sonar:
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

    @classmethod
    def _ensure_cube_mesh(cls, mesh: bpy.types.Mesh):
        verts, faces = cls._cube_vertices_faces()
        with utils.logged_errors("proxy_management.cls._ensure_cube_mesh"):
            mesh.clear_geometry()
        mesh.from_pydata(verts, [], faces)
        mesh.update()

    @classmethod
    def ensure_sonar_setup(cls, scene, sources):
        """
        Create or refresh the ACE_Sonar object and its Geometry Nodes modifier.
        """
        if scene is None:
            return None

        obj = bpy.data.objects.get(cls.SONAR_OBJECT_NAME)
        created = False
        # Only create the sonar object once there is at least one source
        if obj is None:
            if not sources:
                return None
            # Create the design placeholder as a cube mesh.
            mesh = bpy.data.meshes.new(cls.SONAR_OBJECT_NAME)
            cls._ensure_cube_mesh(mesh)
            obj = bpy.data.objects.new(cls.SONAR_OBJECT_NAME, mesh)
            scene.collection.objects.link(obj)
            created = True

        mod = obj.modifiers.get(cls.SONAR_MODIFIER_NAME)
        if mod is None or mod.type != 'NODES':
            if mod and mod.type != 'NODES':
                obj.modifiers.remove(mod)
            mod = obj.modifiers.new(cls.SONAR_MODIFIER_NAME, 'NODES')

        tree = bpy.data.node_groups.get(cls.SONAR_TREE_NAME)
        if tree is None and sources:
            tree = bpy.data.node_groups.new(cls.SONAR_TREE_NAME, 'GeometryNodeTree')
        if tree is not None:
            with utils.logged_errors("proxy_management.cls.ensure_sonar_setup"):
                tree.is_modifier = True
        mod.node_group = tree

        if tree:
            cls._sync_sonar_tree(tree, sources or [])
        cls.ensure_sonar_shader_setup(scene, sources, obj=obj)
        with utils.logged_errors("proxy_management.cls.ensure_sonar_setup#2"):
            ace = getattr(scene, "audiocore", None)
            if ace and not getattr(ace, "gn_target_object", None):
                ace.gn_target_object = obj
        if created:
            with utils.logged_errors("proxy_management.cls.ensure_sonar_setup#3"):
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
        return obj

    @classmethod
    def ensure_sonar_shader_setup(cls, scene, sources, obj=None):
        """Create or refresh a managed shader frontend material on ACE_Sonar."""
        if scene is None:
            return None

        if obj is None:
            obj = bpy.data.objects.get(cls.SONAR_OBJECT_NAME)
        if obj is None:
            return None

        material = None
        for slot in getattr(obj, "material_slots", []) or []:
            slot_mat = getattr(slot, "material", None)
            if slot_mat is None:
                continue
            if str(getattr(slot_mat, "name", "") or "") == cls.SONAR_SHADER_MATERIAL_NAME:
                material = slot_mat
                break

        if material is None and sources:
            material = bpy.data.materials.get(cls.SONAR_SHADER_MATERIAL_NAME)
            if material is None:
                material = bpy.data.materials.new(cls.SONAR_SHADER_MATERIAL_NAME)

            obj_materials = getattr(getattr(obj, "data", None), "materials", None)
            linked = False
            if obj_materials is not None:
                for candidate in obj_materials:
                    if candidate == material:
                        linked = True
                        break
                if not linked:
                    with utils.logged_errors("proxy_management.cls.ensure_sonar_shader_setup"):
                        obj_materials.append(material)
                        linked = True
            if not linked:
                with utils.logged_errors("proxy_management.cls.ensure_sonar_shader_setup#2"):
                    obj.active_material = material

        if material is None:
            return None

        # Blender 5 materials always have a node tree (Material.use_nodes is deprecated).
        tree = getattr(material, "node_tree", None)
        if tree is None:
            return None

        with utils.logged_errors("proxy_management.cls.ensure_sonar_shader_setup#4"):
            cls.sync_managed_source_tree(tree, sources or [], scene=scene)
        return tree

    @staticmethod
    def _frontend_kind_for_tree(tree) -> str:
        tree_type = str(getattr(tree, "bl_idname", "") or "")
        if tree_type == "ShaderNodeTree":
            return FRONTEND_KIND_SHADER
        return FRONTEND_KIND_GEOMETRY

    @classmethod
    def _source_group_factory_for_tree(cls, tree):
        frontend_kind = cls._frontend_kind_for_tree(tree)
        if frontend_kind == FRONTEND_KIND_SHADER:
            from ...nodes.shader.factory import ACEShaderNodeFactory
            return ACEShaderNodeFactory
        return ACEDummyNodeFactory

    @classmethod
    def _source_group_node_type_for_tree(cls, tree) -> str:
        return "ShaderNodeGroup" if cls._frontend_kind_for_tree(tree) == FRONTEND_KIND_SHADER else "GeometryNodeGroup"

    @staticmethod
    def _managed_source_ace_id(frontend_uid: str, source_id: str, existing_ids=None) -> str:
        return ACEIdManager.generate_node_id(
            "source_mngmt",
            "src",
            existing_ids=list(existing_ids or []),
            deterministic_seed=f"{frontend_uid}:source:{source_id}",
        )

    @classmethod
    def sync_managed_source_tree(cls, tree: bpy.types.NodeTree, sources, scene=None):
        """Sync managed ACE source placeholders in any supported top-level tree."""
        utils.assert_main_thread("sync_managed_source_tree")
        if tree is None:
            return
        if (
            cls._frontend_kind_for_tree(tree) == FRONTEND_KIND_SHADER
            and cls._shader_tree_writes_are_unsafe(scene)
        ):
            cls._defer_shader_tree_sync(tree, scene)
            return

        factory_cls = cls._source_group_factory_for_tree(tree)
        with utils.logged_errors("proxy_management.cls.sync_managed_source_tree"):
            factory_cls.ensure_node_groups_exist()

        frontend_kind = cls._frontend_kind_for_tree(tree)
        frontend_uid = ensure_frontend_uid(tree, frontend_kind, [])
        base_group_name = factory_cls.get_group_name("source_mngmt", "src")
        base_group = bpy.data.node_groups.get(base_group_name)
        group_node_type = cls._source_group_node_type_for_tree(tree)

        desired_sources: list[tuple[str, str, Any]] = []
        existing_source_ids: list[str] = [
            str(node.get("ace_source_id") or "")
            for node in getattr(tree, "nodes", [])
            if getattr(node, "get", None) and node.get("ace_source_id")
        ]
        for src in sources or []:
            try:
                src_id = ACEIdManager.ensure_source_uid(src, existing_ids=existing_source_ids)
            except Exception as exc:
                utils.note_swallowed("proxy_management.cls.sync_managed_source_tree#2", exc)
                src_id = getattr(src, "stable_id", "") or getattr(src, "name", "Source")
            if src_id and src_id not in existing_source_ids:
                existing_source_ids.append(src_id)
            desired_sources.append((str(src_id or ""), getattr(src, "name", "Source"), src))

        def _is_managed_source_node(node: bpy.types.Node) -> bool:
            if getattr(node, "type", "") != "GROUP":
                return False
            with utils.logged_errors("proxy_management.cls.sync_managed_source_tree._is_managed_source_node"):
                if node.get("ace_source_id"):
                    return True
            ace_id = ""
            try:
                ace_id = str(node.get("ace_id", "") or "")
            except Exception as exc:
                utils.note_swallowed("proxy_management.cls.sync_managed_source_tree._is_managed_source_node#2", exc)
                ace_id = ""
            parsed = ACEIdManager.parse_node_id(ace_id) if ace_id else None
            if not parsed:
                return False
            if parsed.get("category") != "source_mngmt" or parsed.get("type") != "src":
                return False
            node_tree = getattr(node, "node_tree", None)
            return bool(base_group and node_tree == base_group)

        managed_nodes: list[bpy.types.Node] = [n for n in getattr(tree, "nodes", []) if _is_managed_source_node(n)]
        node_by_source_id: Dict[str, bpy.types.Node] = {}
        unassigned_nodes: list[bpy.types.Node] = []
        duplicate_nodes: list[bpy.types.Node] = []
        for node in managed_nodes:
            try:
                src_id = str(node.get("ace_source_id", "") or "")
            except Exception as exc:
                utils.note_swallowed("proxy_management.cls.sync_managed_source_tree#3", exc)
                src_id = ""
            if src_id:
                if src_id in node_by_source_id:
                    duplicate_nodes.append(node)
                else:
                    node_by_source_id[src_id] = node
            else:
                unassigned_nodes.append(node)
        for node in duplicate_nodes:
            with utils.logged_errors("proxy_management.cls.sync_managed_source_tree#4"):
                tree.nodes.remove(node)

        desired_ids = {src_id for src_id, _src_name, _src in desired_sources if src_id}
        for src_id, node in list(node_by_source_id.items()):
            if src_id in desired_ids:
                continue
            with utils.logged_errors("proxy_management.cls.sync_managed_source_tree#5"):
                tree.nodes.remove(node)
            node_by_source_id.pop(src_id, None)

        def _take_reusable() -> bpy.types.Node | None:
            if not unassigned_nodes:
                return None
            return unassigned_nodes.pop(0)

        existing_node_ids = [
            str(node.get("ace_id") or "")
            for node in getattr(tree, "nodes", [])
            if getattr(node, "get", None) and node.get("ace_id")
        ]

        for idx, (src_id, src_name, src) in enumerate(desired_sources):
            node = node_by_source_id.get(src_id)
            created_node = False
            if node is None:
                node = _take_reusable()
            if node is None:
                node = tree.nodes.new(group_node_type)
                created_node = True
            if base_group and getattr(node, "node_tree", None) != base_group:
                node.node_tree = base_group
            elif not getattr(node, "node_tree", None):
                node.node_tree = base_group

            ace_id = cls._managed_source_ace_id(frontend_uid, src_id, existing_ids=existing_node_ids)
            if ace_id and ace_id not in existing_node_ids:
                existing_node_ids.append(ace_id)
            with utils.logged_errors("proxy_management.cls.sync_managed_source_tree#6"):
                node["ace_id"] = ace_id
            with utils.logged_errors("proxy_management.cls.sync_managed_source_tree#7"):
                node["ace_source_id"] = src_id
            with utils.logged_errors("proxy_management.cls.sync_managed_source_tree#8"):
                node["ace_source_name"] = src_name
            node.label = src_name
            node.name = src_name
            if created_node:
                with utils.logged_errors("proxy_management.cls.sync_managed_source_tree#9"):
                    node.location = (0, -220 * idx)

            if frontend_kind == FRONTEND_KIND_GEOMETRY:
                with utils.logged_errors("proxy_management.cls.sync_managed_source_tree#10"):
                    proxy_obj = bpy.data.objects.get(getattr(src, "proxy_object_name", ""))
                    proxy_socket = next((s for s in node.inputs if getattr(s, "name", "") == "Proxy Object"), None)
                    if proxy_obj and proxy_socket:
                        proxy_socket.default_value = proxy_obj
            elif frontend_kind == FRONTEND_KIND_SHADER:
                with utils.logged_errors("proxy_management.cls.sync_managed_source_tree#11"):
                    proxy_name = str(getattr(src, "proxy_object_name", "") or "")
                    proxy_socket = next((s for s in node.inputs if getattr(s, "name", "") == "Proxy Object"), None)
                    if proxy_socket and proxy_name:
                        try:
                            current = getattr(proxy_socket, "default_value", None)
                        except Exception as exc:
                            utils.note_swallowed("proxy_management.cls.sync_managed_source_tree#12", exc)
                            current = None
                        if isinstance(current, str):
                            proxy_socket.default_value = proxy_name

        for node in unassigned_nodes:
            with utils.logged_errors("proxy_management.cls.sync_managed_source_tree#13"):
                tree.nodes.remove(node)

    @classmethod
    def _sync_sonar_tree(cls, tree: bpy.types.NodeTree, sources):
        """
        Incrementally updates the sonar GN tree to include one managed ACE source
        node per audio source without touching user processing nodes.
        """
        utils.assert_main_thread("_sync_sonar_tree")
        if tree is None:
            return

        with utils.logged_errors("proxy_management.cls._sync_sonar_tree"):
            ACEDummyNodeFactory.ensure_node_groups_exist()

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
                with utils.logged_errors("proxy_management.cls._sync_sonar_tree#2"):
                    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
            if "Audio" not in output_names:
                with utils.logged_errors("proxy_management.cls._sync_sonar_tree#3"):
                    tree.interface.new_socket("Audio", in_out='OUTPUT', socket_type='NodeSocketBundle')
        geom_out_sock = next((s for s in output_node.inputs if s.name == "Geometry"), None)
        if "Geometry" in input_node.outputs and geom_out_sock and not geom_out_sock.is_linked:
            with utils.logged_errors("proxy_management.cls._sync_sonar_tree#4"):
                tree.links.new(input_node.outputs["Geometry"], geom_out_sock)
        cls.sync_managed_source_tree(tree, sources or [])
