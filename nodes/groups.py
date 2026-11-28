"""The shared groups ACE nodes use, found by what they are rather than by their names.

Each node kind has one shared group per tree type, its *template*: the add menu, Blender's
link-drag search and Add > Group all add it. Its name is the node's own ("Envelope",
"Loudness"), with " (Shader)" in shader trees: groups of both types share one name space. A
visible name is what lets Blender's search and its Group menu offer the node.

A template carries ``ace_kind`` and ``ace_template``, so it is found by its kind: a group of
the user's that has the same name is never taken for it (Blender then names ACE's "Loudness.001").
The groups a node gets once placed are hidden (a leading "."): a splitter's variant per band
count, a shader node's own sink group, a shader Source's group per source.
"""

import bpy  # type: ignore

from ..vars import node_def
from .identity import KIND_KEY, group_kind

TEMPLATE_KEY = "ace_template"
SHADER_SUFFIX = " (Shader)"
GEOMETRY = "GeometryNodeTree"
SHADER = "ShaderNodeTree"


def display_name(category: str, node_type: str, tree_type: str) -> str:
    label = str(node_def(category, node_type).get("label", "") or node_type)
    return f"{label}{SHADER_SUFFIX}" if tree_type == SHADER else label


def legacy_name(category: str, node_type: str, tree_type: str) -> str:
    """The hidden name a template had before it was named after its node (still the variants' prefix)."""
    return f".ACE_SHADER_{category}_{node_type}" if tree_type == SHADER else f".ACE_{category}_{node_type}"


def is_template(group) -> bool:
    return bool(group is not None and group.get(TEMPLATE_KEY, False))


def find_template(category: str, node_type: str, tree_type: str):
    """The local template group of a kind for *tree_type*, or None.

    A file saved before templates were named after their node holds it under its legacy name:
    that group is the template (tag_template names and marks it).
    """
    kind = (str(category), str(node_type))
    for group in bpy.data.node_groups:
        if group.library is None and group.bl_idname == tree_type and is_template(group) and group_kind(group) == kind:
            return group
    legacy = bpy.data.node_groups.get(legacy_name(category, node_type, tree_type))
    if legacy is not None and legacy.library is None and legacy.bl_idname == tree_type:
        return legacy
    return None


def tag_template(group, category: str, node_type: str, tree_type: str) -> None:
    """Mark *group* as the template of its kind and give it the node's name, unless it has it already."""
    kind = f"{category}:{node_type}"
    if group.get(KIND_KEY) != kind:
        group[KIND_KEY] = kind
    if not is_template(group):
        group[TEMPLATE_KEY] = True
    name = display_name(category, node_type, tree_type)
    current = str(group.name)
    taken_by_us = current == name or (current.startswith(name + ".") and current[len(name) + 1:].isdigit())
    if not taken_by_us:
        group.name = name


def new_template(category: str, node_type: str, tree_type: str):
    group = bpy.data.node_groups.new(display_name(category, node_type, tree_type), tree_type)
    tag_template(group, category, node_type, tree_type)
    return group


def tag_owned(group, category: str, node_type: str) -> None:
    """Mark a hidden group a placed node uses (a splitter variant) with its kind."""
    kind = f"{category}:{node_type}"
    if group.get(KIND_KEY) != kind:
        group[KIND_KEY] = kind
