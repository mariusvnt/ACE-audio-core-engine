"""Reads an object-bound shader frontend into an abstract graph."""

from ... import utils
from ..frontend import FRONTEND_KIND_SHADER
from ..geometry.reader import ACEDummyTreeReader
from .factory import ACEShaderNodeFactory


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

    @staticmethod
    def is_managed_tree(tree) -> bool:
        """Return True when *tree* is a shader frontend containing ACE nodes."""
        if tree is None or str(getattr(tree, "bl_idname", "") or "") != "ShaderNodeTree":
            return False
        with utils.logged_errors("shader_tree.ACEShaderTreeReader.is_managed_tree"):
            if bool(tree.get("ace_frontend_uid", "")):
                return True
        for node in getattr(tree, "nodes", []) or []:
            with utils.logged_errors("shader_tree.ACEShaderTreeReader.is_managed_tree#2"):
                if bool(node.get("ace_id", "")):
                    return True
            with utils.logged_errors("shader_tree.ACEShaderTreeReader.is_managed_tree#3"):
                if bool(node.get("ace_source_id", "")):
                    return True
            if str(getattr(node, "type", "") or "") != "GROUP":
                continue
            group = getattr(node, "node_tree", None)
            group_name = str(getattr(group, "name", "") or "")
            norm_group_name = group_name.lstrip(".")
            if (
                group_name.startswith(".ACE_SHADER_")
                or norm_group_name.startswith("ACE_SHADER_")
                or group_name.startswith(".ACE_")
                or norm_group_name.startswith("ACE_")
            ):
                return True
        return False
