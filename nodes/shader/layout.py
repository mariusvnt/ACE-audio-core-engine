"""Placement of the helper attribute nodes beside each ACE shader node."""

from ... import utils
from ...core.id_management import ACEIdManager


class HelperNodeLayout:
    @classmethod
    def _helper_node_name(cls, ace_id: str, output_idx: int) -> str:
        return f".ACE_HELPER_{ACEIdManager._clean_string(str(ace_id or 'node'))[:24]}_{int(output_idx)}"

    @classmethod
    def _helper_frame_name(cls, ace_id: str) -> str:
        return f".ACE_HELPER_FRAME_{ACEIdManager._clean_string(str(ace_id or 'node'))[:24]}"

    @classmethod
    def _coerce_xy(cls, raw_value) -> tuple[float, float]:
        if raw_value is None:
            return 0.0, 0.0
        with utils.logged_errors("layout.HelperNodeLayout._coerce_xy"):
            return (
                float(getattr(raw_value, "x")),
                float(getattr(raw_value, "y")),
            )
        try:
            return float(raw_value[0]), float(raw_value[1])
        except Exception as exc:
            utils.note_swallowed("layout.HelperNodeLayout._coerce_xy#2", exc)
            return 0.0, 0.0

    @classmethod
    def _node_local_location(cls, node) -> tuple[float, float]:
        return cls._coerce_xy(getattr(node, "location", None))

    @classmethod
    def _node_dimensions(cls, node) -> tuple[float, float]:
        return cls._coerce_xy(getattr(node, "dimensions", None))

    @classmethod
    def _node_height(cls, node, fallback: float = 0.0) -> float:
        _width, height = cls._node_dimensions(node)
        resolved = abs(float(height))
        if resolved > 1.0:
            return resolved
        return float(fallback)

    @classmethod
    def _node_absolute_location(cls, node) -> tuple[float, float]:
        if node is None:
            return 0.0, 0.0
        abs_loc = getattr(node, "location_absolute", None)
        if abs_loc is not None:
            abs_x, abs_y = cls._coerce_xy(abs_loc)
            if abs_x or abs_y or getattr(node, "parent", None) is not None:
                return abs_x, abs_y
        x, y = cls._node_local_location(node)
        visited: set[int] = set()
        parent = getattr(node, "parent", None)
        while parent is not None:
            ptr = id(parent)
            if ptr in visited:
                break
            visited.add(ptr)
            parent_x, parent_y = cls._node_local_location(parent)
            x += parent_x
            y += parent_y
            parent = getattr(parent, "parent", None)
        return x, y

    @classmethod
    def _set_node_location(cls, node, location: tuple[float, float]) -> bool:
        if node is None:
            return False
        target_x, target_y = float(location[0]), float(location[1])
        current_x, current_y = cls._node_local_location(node)
        if abs(current_x - target_x) <= 0.5 and abs(current_y - target_y) <= 0.5:
            return False
        try:
            node.location = (target_x, target_y)
            return True
        except Exception as exc:
            utils.note_swallowed("layout.HelperNodeLayout._set_node_location", exc)
            return False

    @staticmethod
    def _is_frame_node(node) -> bool:
        if node is None:
            return False
        return bool(
            str(getattr(node, "bl_idname", "") or "") == "NodeFrame"
            or str(getattr(node, "type", "") or "") == "FRAME"
        )

    @classmethod
    def _ensure_helper_node_compact_display(cls, helper_node) -> bool:
        if helper_node is None:
            return False
        changed = False
        if hasattr(helper_node, "hide"):
            with utils.logged_errors("layout.HelperNodeLayout._ensure_helper_node_compact_display"):
                if not bool(getattr(helper_node, "hide", False)):
                    helper_node.hide = True
                    changed = True
        if hasattr(helper_node, "width_hidden"):
            with utils.logged_errors("layout.HelperNodeLayout._ensure_helper_node_compact_display#2"):
                current = float(getattr(helper_node, "width_hidden", 0.0) or 0.0)
                target = float(cls._HELPER_HIDDEN_WIDTH)
                if abs(current - target) > 0.5:
                    helper_node.width_hidden = target
                    changed = True
        return changed

    @classmethod
    def _parent_node_preserve_absolute(cls, node, parent) -> bool:
        if node is None:
            return False
        if getattr(node, "parent", None) == parent:
            return False
        abs_x, abs_y = cls._node_absolute_location(node)
        try:
            node.parent = parent
        except Exception as exc:
            utils.note_swallowed("layout.HelperNodeLayout._parent_node_preserve_absolute", exc)
            return False
        if parent is None:
            cls._set_node_location(node, (abs_x, abs_y))
            return True
        parent_abs_x, parent_abs_y = cls._node_absolute_location(parent)
        cls._set_node_location(node, (abs_x - parent_abs_x, abs_y - parent_abs_y))
        return True

    @classmethod
    def _find_helper_frame_node(cls, tree, ace_id: str):
        target_owner = str(ace_id or "")
        for helper_node in getattr(tree, "nodes", []) or []:
            if not cls._is_frame_node(helper_node):
                continue
            try:
                owner = str(helper_node.get(cls._HELPER_FRAME_OWNER_KEY, "") or "")
            except Exception as exc:
                utils.note_swallowed("layout.HelperNodeLayout._find_helper_frame_node", exc)
                owner = ""
            if owner == target_owner:
                return helper_node
        return None

    @classmethod
    def _ensure_helper_frame(cls, tree, owner_node, ace_id: str):
        if tree is None or owner_node is None:
            return None, False

        changed = False
        owner_parent = getattr(owner_node, "parent", None)
        if cls._is_frame_node(owner_parent):
            try:
                parent_owner = str(owner_parent.get(cls._HELPER_FRAME_OWNER_KEY, "") or "")
            except Exception as exc:
                utils.note_swallowed("layout.HelperNodeLayout._ensure_helper_frame", exc)
                parent_owner = ""
            if not parent_owner or parent_owner == str(ace_id or ""):
                return owner_parent, False

        frame_node = cls._find_helper_frame_node(tree, ace_id)
        created = False
        if frame_node is None:
            try:
                frame_node = tree.nodes.new("NodeFrame")
            except Exception as exc:
                utils.note_swallowed("layout.HelperNodeLayout._ensure_helper_frame#2", exc)
                return None, False
            created = True
            changed = True
            cls._set_node_location(frame_node, cls._node_absolute_location(owner_node))

        with utils.logged_errors("layout.HelperNodeLayout._ensure_helper_frame#3"):
            if str(frame_node.get(cls._HELPER_FRAME_OWNER_KEY, "") or "") != str(ace_id or ""):
                frame_node[cls._HELPER_FRAME_OWNER_KEY] = str(ace_id or "")
                changed = True
        with utils.logged_errors("layout.HelperNodeLayout._ensure_helper_frame#4"):
            if str(frame_node.get(cls._HELPER_FRAME_KIND_KEY, "") or "") != "ATTRIBUTE_CLUSTER":
                frame_node[cls._HELPER_FRAME_KIND_KEY] = "ATTRIBUTE_CLUSTER"
                changed = True
        frame_name = cls._helper_frame_name(ace_id)
        with utils.logged_errors("layout.HelperNodeLayout._ensure_helper_frame#5"):
            if str(getattr(frame_node, "name", "") or "") != frame_name:
                frame_node.name = frame_name
                changed = True
        with utils.logged_errors("layout.HelperNodeLayout._ensure_helper_frame#6"):
            if str(getattr(frame_node, "label", "") or "") != "":
                frame_node.label = ""
                changed = True
        if hasattr(frame_node, "shrink"):
            with utils.logged_errors("layout.HelperNodeLayout._ensure_helper_frame#7"):
                if not bool(getattr(frame_node, "shrink", False)):
                    frame_node.shrink = True
                    changed = True
        if created or getattr(owner_node, "parent", None) != frame_node:
            if cls._parent_node_preserve_absolute(owner_node, frame_node):
                changed = True
        return frame_node, changed

    @classmethod
    def _sync_helper_node_layout(cls, tree, owner_node, helper_node, ace_id: str, output_idx: int) -> bool:
        if tree is None or owner_node is None or helper_node is None:
            return False
        changed = False
        container = getattr(owner_node, "parent", None)

        frame_node, frame_changed = cls._ensure_helper_frame(tree, owner_node, ace_id)
        if frame_node is not None:
            container = frame_node
            if frame_changed:
                changed = True

        if cls._parent_node_preserve_absolute(helper_node, container):
            changed = True

        if cls._ensure_helper_node_compact_display(helper_node):
            changed = True

        if container is not None and getattr(owner_node, "parent", None) == container:
            owner_x, owner_y = cls._node_local_location(owner_node)
        else:
            owner_x, owner_y = cls._node_absolute_location(owner_node)
        owner_height = cls._node_height(owner_node, cls._HELPER_FALLBACK_OWNER_HEIGHT)

        helper_target = (
            float(owner_x),
            float(owner_y) - owner_height - cls._HELPER_VERTICAL_GAP - (int(output_idx) * cls._HELPER_VERTICAL_STEP),
        )
        if cls._set_node_location(helper_node, helper_target):
            changed = True
        return changed
