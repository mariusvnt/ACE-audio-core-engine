"""Frontend kinds and frontend identity shared by geometry and shader trees."""

import hashlib
from ..core.id_management import ACEIdManager
from .. import utils


FRONTEND_KIND_GEOMETRY = "GEOMETRY_NODES"


FRONTEND_KIND_SHADER = "SHADER_NODES"


def _tree_pointer(tree) -> int:
    try:
        return int(tree.as_pointer())
    except Exception as exc:
        utils.note_swallowed("frontend._tree_pointer", exc)
        return id(tree)


def ensure_frontend_uid(tree, frontend_kind: str, existing_uids) -> str:
    """Guarantee a persistent, unique frontend UID on a top-level node tree."""
    if tree is None:
        return ""
    existing = {str(v) for v in (existing_uids or []) if str(v)}
    current_uid = ""
    try:
        current_uid = str(tree.get("ace_frontend_uid", "") or "")
    except Exception as exc:
        utils.note_swallowed("frontend.ensure_frontend_uid", exc)
        current_uid = ""
    if current_uid and current_uid not in existing:
        return current_uid

    kind_token = ACEIdManager._clean_string(str(frontend_kind or "frontend")).lower()[:10] or "frontend"
    tree_name = str(getattr(tree, "name", "") or "Tree")
    tree_type = str(getattr(tree, "bl_idname", "") or "")
    max_attempts = max(len(existing) + 64, 256)
    for salt in range(max_attempts):
        suffix = hashlib.blake2b(
            f"{frontend_kind}:{tree_type}:{tree_name}:{salt}".encode("utf-8"),
            digest_size=4,
        ).hexdigest()[:6]
        candidate = f"ACE_FRONTEND_{kind_token}_{suffix}"
        if candidate in existing:
            continue
        with utils.logged_errors("frontend.ensure_frontend_uid#2"):
            tree["ace_frontend_uid"] = candidate
        return candidate
    return current_uid
