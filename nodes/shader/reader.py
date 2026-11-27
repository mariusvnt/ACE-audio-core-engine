"""Reads a shader frontend into an abstract graph."""

from ..frontend import FRONTEND_KIND_SHADER
from ..geometry.reader import ACEDummyTreeReader


class ACEShaderTreeReader(ACEDummyTreeReader):
    """Shader frontend reader sharing the GN reader contract."""

    def __init__(self, blender_node_tree, frontend_uid: str = "", frontend_kind: str = FRONTEND_KIND_SHADER, **kwargs):
        super().__init__(blender_node_tree, frontend_uid=frontend_uid, frontend_kind=frontend_kind, **kwargs)

    @staticmethod
    def is_managed_tree(tree) -> bool:
        """Return True when *tree* is a shader frontend containing ACE nodes."""
        if tree is None or str(getattr(tree, "bl_idname", "") or "") != "ShaderNodeTree":
            return False
        from ..definitions import tree_uses_ace

        return tree_uses_ace(tree)
