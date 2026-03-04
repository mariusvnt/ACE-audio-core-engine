"""
Blender Shader node group management for ACE frontends.
"""

import hashlib
import json

import bpy  # type: ignore

from ..vars import ACE_NODES
from .. import utils
from .dummy_tree import ACEDummyTreeReader, FRONTEND_KIND_SHADER


class ACEShaderNodeFactory:
    """Create hidden ShaderNodeTree ACE groups for shader frontends."""

    _NODES_CONTENT_HASH: str = ""

    @staticmethod
    def get_group_name(category: str, node_type: str) -> str:
        return f".ACE_SHADER_{category}_{node_type}"

    @staticmethod
    def _ace_nodes_hash() -> str:
        try:
            serialized = json.dumps(
                {"kind": "shader", "nodes": ACE_NODES},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
            return hashlib.blake2b(serialized, digest_size=8).hexdigest()
        except Exception:
            return ""

    @staticmethod
    def _shader_socket_type(sock_type: str) -> str:
        mapping = {
            "NodeSocketBundle": "NodeSocketFloat",
            "NodeSocketObject": "NodeSocketString",
            "NodeSocketInt": "NodeSocketFloat",
        }
        return mapping.get(str(sock_type or ""), str(sock_type or "NodeSocketFloat"))

    @classmethod
    def _new_socket(cls, ng, sock_name: str, in_out: str, socket_type: str):
        mapped = cls._shader_socket_type(socket_type)
        try:
            return ng.interface.new_socket(sock_name, in_out=in_out, socket_type=mapped)
        except Exception:
            try:
                return ng.interface.new_socket(sock_name, in_out=in_out, socket_type="NodeSocketFloat")
            except Exception:
                return None

    @classmethod
    def _build_group_content(cls, ng, node_def, category: str, node_type: str):
        for sock_type_str, sock_name in node_def.get("inputs", []):
            cls._new_socket(ng, sock_name, "INPUT", sock_type_str)
        for sock_type_str, sock_name in node_def.get("outputs", []):
            cls._new_socket(ng, sock_name, "OUTPUT", sock_type_str)

        if category == "stream":
            cls._new_socket(ng, "Attribute ID", "INPUT", "NodeSocketString")
        elif category == "utilities" and node_type == "src_specs":
            cls._new_socket(ng, "Frames per Beat Attribute ID", "INPUT", "NodeSocketString")
            cls._new_socket(ng, "Beat Phase Attribute ID", "INPUT", "NodeSocketString")
        elif category == "utilities" and node_type == "util_rolling_avg":
            cls._new_socket(ng, "Attribute ID", "INPUT", "NodeSocketString")
        elif category == "utilities" and node_type == "stream_accumulator":
            cls._new_socket(ng, "Attribute ID", "INPUT", "NodeSocketString")

        try:
            input_node = next((n for n in ng.nodes if n.type == "GROUP_INPUT"), None)
            if input_node is None:
                input_node = ng.nodes.new("NodeGroupInput")
                input_node.location = (-300, 0)
            output_node = next((n for n in ng.nodes if n.type == "GROUP_OUTPUT"), None)
            if output_node is None:
                output_node = ng.nodes.new("NodeGroupOutput")
                output_node.location = (300, 0)
        except Exception:
            pass

    @classmethod
    def rebuild_node_group(cls, cat_key: str, type_key: str):
        """Force-rebuild a single shader ACE node group from ACE_NODES."""
        try:
            node_def = ACE_NODES[cat_key]["nodes"][type_key]
        except KeyError:
            return

        group_name = cls.get_group_name(cat_key, type_key)
        ng = bpy.data.node_groups.get(group_name)
        if ng is not None and getattr(ng, "bl_idname", "") != "ShaderNodeTree":
            try:
                bpy.data.node_groups.remove(ng)
            except Exception:
                ng = None
            else:
                ng = None
        if ng is None:
            ng = bpy.data.node_groups.new(group_name, "ShaderNodeTree")
        else:
            ng.clear()
            ng.interface.clear()
        cls._build_group_content(ng, node_def, cat_key, type_key)

    @classmethod
    def ensure_node_groups_exist(cls):
        try:
            _ = bpy.data.node_groups
        except AttributeError:
            utils.log("Shader node groups unavailable in current context; skipping build.", "WARNING")
            return
        current_hash = cls._ace_nodes_hash()
        if current_hash and current_hash == cls._NODES_CONTENT_HASH:
            return
        for cat_key, category in ACE_NODES.items():
            for type_key, node_def in category.get("nodes", {}).items():
                group_name = cls.get_group_name(cat_key, type_key)
                ng = bpy.data.node_groups.get(group_name)
                if ng is not None and getattr(ng, "bl_idname", "") != "ShaderNodeTree":
                    try:
                        bpy.data.node_groups.remove(ng)
                    except Exception:
                        ng = None
                    else:
                        ng = None
                if ng is None:
                    ng = bpy.data.node_groups.new(group_name, "ShaderNodeTree")
                else:
                    ng.clear()
                    ng.interface.clear()
                cls._build_group_content(ng, node_def, cat_key, type_key)
        try:
            from .dynamic_audio_operators import ACESplitterManager, ACEAudioMathManager
            ACESplitterManager.refresh_existing_variants("ShaderNodeTree")
            ACEAudioMathManager.refresh_existing_variants("ShaderNodeTree")
        except Exception:
            pass
        if current_hash:
            cls._NODES_CONTENT_HASH = current_hash
            ACEShaderTreeReader._GROUP_NAME_LOOKUP.clear()
            ACEShaderTreeReader._ACE_NODES_LOOKUP_HASH = ""


class ACEShaderTreeReader(ACEDummyTreeReader):
    """Shader frontend reader sharing the GN reader contract."""

    FACTORY_CLS = ACEShaderNodeFactory
    _GROUP_NAME_LOOKUP: dict[str, tuple[str, str]] = {}
    _ACE_NODES_LOOKUP_HASH: str = ""

    def __init__(
        self,
        blender_node_tree,
        frontend_uid: str = "",
        frontend_kind: str = FRONTEND_KIND_SHADER,
        reserved_node_ids: set[str] | None = None,
        consumer_keys: tuple[str, ...] = (),
    ):
        super().__init__(
            blender_node_tree,
            frontend_uid=frontend_uid,
            frontend_kind=frontend_kind,
            reserved_node_ids=reserved_node_ids,
            consumer_keys=consumer_keys,
        )


def register():
    try:
        ACEShaderNodeFactory.ensure_node_groups_exist()
    except AttributeError:
        utils.log("Skipping shader node group creation due to restricted data access.", "WARNING")


def unregister():
    pass
