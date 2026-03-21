"""
Structure of the 'AbstractTree' object, which is an abstract representation of
scene frontends.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import List, Dict, Any


WRITEABLE_OUTPUT_UTILITY_NODE_TYPES = frozenset({
    "src_specs",
    "util_rolling_avg",
    "stream_accumulator",
    "logic",
})


def ace_node_type_is_writeable_output(category: str, node_type: str) -> bool:
    category_str = str(category or "")
    node_type_str = str(node_type or "")
    return bool(
        category_str == "stream"
        or (
            category_str == "utilities"
            and node_type_str in WRITEABLE_OUTPUT_UTILITY_NODE_TYPES
        )
    )


def ace_node_is_writeable_output(node) -> bool:
    if node is None:
        return False
    return ace_node_type_is_writeable_output(
        getattr(node, "category", ""),
        getattr(node, "type", ""),
    )

@dataclass
class ACEAbstractSocket:
    """
    Represents an input or output connection point.
    """
    id: str  # Unique ID for the socket (useful for UI mapping)
    name: str # The internal name (e.g., "Value", "Vector")
    socket_idx: int
    is_output: bool
    data_type: str
    
    # Critical for inputs that have no link connected:
    default_value: Any = None 


@dataclass
class ACEFrontendRef:
    """Reference to a top-level ACE authoring frontend in Blender."""

    uid: str
    kind: str
    tree_name: str
    tree_bl_idname: str
    owner_keys: tuple[str, ...] = ()
    material_name: str = ""
    tree_ptr: int = 0
    discovery_index: int = 0


@dataclass
class ACEConsumerIntent:
    """Frontend-local declaration of who consumes a node output."""

    frontend_uid: str
    node_id: str
    socket_idx: int
    sink_kind: str
    source_token: str = ""
    consumer_keys: tuple[str, ...] = ()
    output_name: str = ""
    explicit_name: str = ""


@dataclass
class ACEAbstractNode:
    """
    Represents a processing unit.
    """
    id: str
    name: str
    category: str
    type: str
    
    # Stores internal settings (e.g., Math Node operation: "MULTIPLY")
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    inputs: List[ACEAbstractSocket] = field(default_factory=list)
    outputs: List[ACEAbstractSocket] = field(default_factory=list)
    # Identifiers of source roots (proxy/stable ids) that feed this node.
    root_sources: set[str] = field(default_factory=set)
    frontend_uid: str = ""
    frontend_kind: str = ""
    source_binding_id: str = ""

@dataclass
class ACEAbstractLink:
    """
    Represents data flow between two sockets.
    """
    source_node_id: str
    target_node_id: str
    
    source_socket_idx: int
    target_socket_idx: int

@dataclass
class ACEAbstractOutputLink:
    """
    Represents an ACE output socket that is connected to the GN Group Output.
    Used to derive default preview-monitor branches.
    """
    source_node_id: str
    source_socket_idx: int
    target_socket_idx: int
    target_socket_name: str = ""
    frontend_uid: str = ""
    frontend_kind: str = ""

@dataclass
class ACEAbstractTree:
    """
    The complete graph container.
    """
    nodes: Dict[str, ACEAbstractNode] = field(default_factory=dict)
    links: List[ACEAbstractLink] = field(default_factory=list)
    group_output_links: List[ACEAbstractOutputLink] = field(default_factory=list)
    frontends: Dict[str, ACEFrontendRef] = field(default_factory=dict)
    consumer_intents: List[ACEConsumerIntent] = field(default_factory=list)

    def to_dict(self):
        """Helper to serialize the tree for sending to the backend."""
        return dataclasses.asdict(self)
        
    def add_node(self, node: ACEAbstractNode):
        self.nodes[node.id] = node

    def add_link(self, link: ACEAbstractLink):
        self.links.append(link)

    def add_group_output_link(self, link: ACEAbstractOutputLink):
        self.group_output_links.append(link)

    def add_frontend(self, frontend: ACEFrontendRef):
        self.frontends[frontend.uid] = frontend

    def add_consumer_intent(self, intent: ACEConsumerIntent):
        self.consumer_intents.append(intent)

    def merge_from(self, other_tree: "ACEAbstractTree"):
        """Deterministically merge another partial frontend tree into this one."""
        for frontend_uid, frontend in (getattr(other_tree, "frontends", {}) or {}).items():
            if frontend_uid in self.frontends:
                raise ValueError(f"Duplicate frontend UID during merge: {frontend_uid}")
            self.frontends[frontend_uid] = frontend
        for node_id, node in (getattr(other_tree, "nodes", {}) or {}).items():
            if node_id in self.nodes:
                raise ValueError(f"Duplicate node ID during merge: {node_id}")
            self.nodes[node_id] = node
        self.links.extend(list(getattr(other_tree, "links", []) or []))
        self.group_output_links.extend(list(getattr(other_tree, "group_output_links", []) or []))
        self.consumer_intents.extend(list(getattr(other_tree, "consumer_intents", []) or []))
