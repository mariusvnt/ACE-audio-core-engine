"""Who an ACE node is, read from the node without storing anything on it.

- **Kind** (category, type): from the group the node uses. An ACE group either
  has the canonical name of its kind (``.ACE_stream_loudness``,
  ``.ACE_SHADER_stream_loudness`` and variants such as ``..._4``) or carries
  ``ace_kind`` (the shader sink and source groups, whose names are their own).
- **Id**: ``ACE_<category>_<type>_<hash>`` of where the node is used: the root
  tree (and the values its owner binds), the groups entered, and the node's
  name. A tree is identified by its session uid, which survives undo but not a
  reload; ids live only as long as the session, the node's name is what lasts.
- **Sink names**: what an output node writes to, as the user's tree holds them:
  the Attribute ID sockets in Geometry Nodes, the node's sink group in shaders.
"""

import hashlib

from ..vars import ACE_NODES
from .. import utils

PREFIX = "ACE_"
KIND_KEY = "ace_kind"
SINK_NAMES_KEY = "ace_sink_names"
SOURCE_ID_KEY = "ace_source_id"
# What a new stream group's Attribute ID sockets hold until the add menu names them.
PLACEHOLDER_NAMES = frozenset({"ACE_undefined"} | {f"ACE_undefined_{i}" for i in range(2, 9)})

_GROUP_PREFIXES = (".ACE_SHADER_", ".ACE_")
_canonical: dict[str, tuple[str, str]] = {}


def _canonical_kinds() -> dict[str, tuple[str, str]]:
    if not _canonical:
        for category, cat_def in ACE_NODES.items():
            for node_type in cat_def.get("nodes", {}):
                _canonical[f"{category}_{node_type}"] = (category, node_type)
    return _canonical


def _valid_kind(category: str, node_type: str) -> bool:
    return node_type in ACE_NODES.get(category, {}).get("nodes", {})


def group_kind(group) -> tuple[str, str]:
    """(category, type) of an ACE group, ("", "") for any other group."""
    if group is None:
        return "", ""
    try:
        raw = str(group.get(KIND_KEY, "") or "")
    except Exception as exc:
        utils.note_swallowed("identity.group_kind", exc)
        raw = ""
    if raw:
        category, _, node_type = raw.partition(":")
        if _valid_kind(category, node_type):
            return category, node_type
    name = str(getattr(group, "name", "") or "")
    if not name.startswith("."):
        name = "." + name
    for prefix in _GROUP_PREFIXES:
        if not name.startswith(prefix):
            continue
        body = name[len(prefix):]
        kinds = _canonical_kinds()
        if body in kinds:
            return kinds[body]
        # A variant (".ACE_delta_frequency_splitter_4", a copy's ".001"): the longest kind it extends.
        best = ""
        for token in kinds:
            if len(token) > len(best) and body.startswith(token) and body[len(token)] in "_.":
                best = token
        if best:
            return kinds[best]
    return "", ""


def node_kind(node) -> tuple[str, str]:
    """(category, type) of an ACE node, ("", "") for any other node."""
    if str(getattr(node, "type", "") or "") != "GROUP":
        return "", ""
    return group_kind(getattr(node, "node_tree", None))


def is_ace_node(node) -> bool:
    return bool(node_kind(node)[0])


def is_ace_group(tree) -> bool:
    """True for a group ACE makes and owns (its internals, never a user tree)."""
    return bool(str(getattr(tree, "name", "") or "").startswith(".ACE_") or group_kind(tree)[0])


def tree_key(tree) -> str:
    """A key for *tree* that is unique and stable for the session, undo included."""
    uid = getattr(tree, "session_uid", None)
    if uid:
        return str(uid)
    with utils.logged_errors("identity.tree_key"):
        return f"p{int(tree.as_pointer())}"
    return f"o{id(tree)}"


def instance_node_id(category: str, node_type: str, root_key: str, chain, name: str) -> str:
    """The id of one use of an ACE node: its root (owner tree and bindings), the groups entered, its name."""
    path = "/".join([*map(str, chain or ()), str(name)])
    digest = hashlib.blake2b(f"{root_key}:{path}".encode("utf-8"), digest_size=4).hexdigest()
    return f"{PREFIX}{category}_{node_type}_{digest}"


def node_id(node, tree=None) -> str:
    """The id of the node used directly by its own tree, "" when it is not an ACE node."""
    category, node_type = node_kind(node)
    if not category:
        return ""
    tree = tree if tree is not None else getattr(node, "id_data", None)
    return instance_node_id(category, node_type, tree_key(tree), (), getattr(node, "name", ""))


def find_node(tree, wanted_id: str):
    """The node of *tree* whose id is *wanted_id*, or None."""
    if tree is None or not wanted_id:
        return None
    for node in getattr(tree, "nodes", []) or []:
        if node_id(node, tree) == wanted_id:
            return node
    return None


def ace_nodes(tree):
    """(node, category, type) for each ACE node of *tree*."""
    for node in getattr(tree, "nodes", []) or []:
        category, node_type = node_kind(node)
        if category:
            yield node, category, node_type


# Sink names


def is_output_kind(category: str, node_type: str) -> bool:
    """Nodes whose values ACE writes for Blender to read."""
    return category == "stream" or (category, node_type) in (
        ("utilities", "util_rolling_avg"),
        ("utilities", "stream_accumulator"),
    )


def value_outputs(category: str, node_type: str) -> list[tuple[int, str]]:
    """(output index, output name) of each value an output node writes."""
    if not is_output_kind(category, node_type):
        return []
    outputs = ACE_NODES[category]["nodes"][node_type].get("outputs", [])
    return [(idx, str(name)) for idx, (sock_type, name) in enumerate(outputs) if "Bundle" not in str(sock_type)]


def attribute_socket_name(slot: int) -> str:
    """The Geometry Nodes socket that holds the name of a node's *slot*-th value."""
    return "Attribute ID" if slot == 0 else f"Attribute ID {slot + 1}"


def usable_sink_name(name) -> bool:
    name = str(name or "").strip()
    return bool(name) and name not in PLACEHOLDER_NAMES


def sink_names(node) -> dict[int, str]:
    """Output index -> sink name, as the user's tree holds them; placeholders left out."""
    category, node_type = node_kind(node)
    outputs = value_outputs(category, node_type)
    if not outputs:
        return {}
    names: dict[int, str] = {}
    group = getattr(node, "node_tree", None)
    stored = None
    with utils.logged_errors("identity.sink_names"):
        stored = group.get(SINK_NAMES_KEY) if group is not None else None
    if stored is not None and str(getattr(group, "bl_idname", "") or "") == "ShaderNodeTree":
        stored = [str(v) for v in stored]
        for slot, (out_idx, _name) in enumerate(outputs):
            if slot < len(stored) and usable_sink_name(stored[slot]):
                names[out_idx] = stored[slot].strip()
        return names
    inputs = getattr(node, "inputs", None)
    for slot, (out_idx, _name) in enumerate(outputs):
        sock = inputs.get(attribute_socket_name(slot)) if inputs is not None and hasattr(inputs, "get") else None
        if sock is None:
            sock = next((s for s in inputs or [] if getattr(s, "name", "") == attribute_socket_name(slot)), None)
        value = getattr(sock, "default_value", "") if sock is not None and not getattr(sock, "is_linked", False) else ""
        if isinstance(value, str) and usable_sink_name(value):
            names[out_idx] = value.strip()
    return names


_METRIC_TOKENS = {
    "rms": "rms", "loudness": "loud", "brightness": "bright", "pitch": "pitch", "transient": "trans",
    "width": "width", "balance": "bal", "stereo": "stereo", "rollingavg": "avg", "accumulated": "acc",
}


def new_sink_name(output_name: str, taken) -> str:
    """A fresh ``ACE_<metric>_<6 hex>`` name, not in *taken*."""
    import secrets

    clean = "".join(ch for ch in str(output_name or "") if ch.isalnum()).lower()
    metric = _METRIC_TOKENS.get(clean, clean[:6] or "val")
    while True:
        name = f"{PREFIX}{metric}_{secrets.token_hex(3)}"
        if name not in taken:
            return name


def source_tempo_props(source_id: str) -> tuple[str, str]:
    """The scene properties that hold a source's BPM and Beats per Bar for shaders."""
    token = str(source_id or "").rsplit("_", 1)[-1] or "src"
    return f"{PREFIX}bpm_{token}", f"{PREFIX}bpb_{token}"


def parsed_node(node) -> dict | None:
    """{"category", "type"} of an ACE node, as ACEIdManager.parse_node_id gives them, or None."""
    category, node_type = node_kind(node)
    return {"category": category, "type": node_type} if category else None
