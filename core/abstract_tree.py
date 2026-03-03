"""
Structure of the 'AbstractTree' object, which is an abstract representation of a node tree.
It is used as a common language between the blender dummy tree, the backend tree and the audio engine.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any

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

@dataclass
class ACEAbstractTree:
    """
    The complete graph container.
    """
    nodes: Dict[str, ACEAbstractNode] = field(default_factory=dict)
    links: List[ACEAbstractLink] = field(default_factory=list)
    group_output_links: List[ACEAbstractOutputLink] = field(default_factory=list)

    def to_dict(self):
        """Helper to serialize the tree for sending to the backend."""
        import dataclasses
        return dataclasses.asdict(self)
        
    def add_node(self, node: ACEAbstractNode):
        self.nodes[node.id] = node

    def add_link(self, link: ACEAbstractLink):
        self.links.append(link)

    def add_group_output_link(self, link: ACEAbstractOutputLink):
        self.group_output_links.append(link)
