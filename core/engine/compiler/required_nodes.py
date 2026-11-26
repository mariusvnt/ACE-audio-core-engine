"""Nodes each source graph needs: outputs, consumers and their upstream chain."""

from typing import Dict, Optional
from ....properties import ACEAudioSource
from ...abstract_tree import ACEAbstractNode, ACEAbstractTree


class RequiredNodes:
    @staticmethod
    def _refresh_consumer_intents_from_nodes(abstract_tree: ACEAbstractTree):
        for intent in getattr(abstract_tree, "consumer_intents", []) or []:
            node = abstract_tree.nodes.get(str(getattr(intent, "node_id", "") or ""))
            if node is None:
                continue
            socket_idx = int(getattr(intent, "socket_idx", 0) or 0)
            outputs = list(getattr(node, "outputs", []) or [])
            if 0 <= socket_idx < len(outputs):
                intent.output_name = str(getattr(outputs[socket_idx], "name", "") or "")
            params = dict(getattr(node, "parameters", {}) or {})
            explicit_name = str(params.get(f"attr_id_out_{socket_idx}", "") or "")
            intent.explicit_name = explicit_name

    @staticmethod
    def _is_output_node(node: ACEAbstractNode) -> bool:
        is_value_utility = bool(
            node
            and node.type in {'util_rolling_avg', 'stream_accumulator'}
            and node.category == 'utilities'
        )
        return bool(
            node
            and (
                node.category == 'stream'
                or is_value_utility
                or (node.category == 'utilities' and node.type == 'preview_sink')
            )
        )

    @staticmethod
    def _node_matches_source(node: ACEAbstractNode, source_uid: str, proxy_name: str, single_source_mode: bool) -> bool:
        if single_source_mode:
            return True
        roots = {str(v) for v in set(getattr(node, "root_sources", set()) or set()) if str(v)}
        if not roots:
            return True
        source_aliases = {str(v) for v in (source_uid, proxy_name) if str(v)}
        if not source_aliases:
            return False
        return bool(roots.intersection(source_aliases))

    @staticmethod
    def _prune_ambiguous_required_node_ids(
        abstract_tree: ACEAbstractTree,
        required_node_ids: set[str],
        source_uid: str,
        proxy_name: str,
        single_source_mode: bool,
    ) -> set[str]:
        if single_source_mode:
            return set(required_node_ids)
        aliases = {str(v) for v in (source_uid, proxy_name) if str(v)}
        if not aliases:
            return set()
        reverse_adj: Dict[str, set[str]] = {}
        for link in getattr(abstract_tree, "links", []) or []:
            src_id = str(getattr(link, "source_node_id", "") or "")
            dst_id = str(getattr(link, "target_node_id", "") or "")
            if src_id and dst_id:
                reverse_adj.setdefault(dst_id, set()).add(src_id)

        ambiguous: set[str] = set()
        for node_id in set(required_node_ids):
            node = abstract_tree.nodes.get(node_id)
            if node is None:
                continue
            roots = {str(v) for v in set(getattr(node, "root_sources", set()) or set()) if str(v)}
            if not roots or roots.issubset(aliases) or not roots.intersection(aliases):
                continue
            upstream_roots: set[tuple[str, ...]] = set()
            for upstream_id in reverse_adj.get(str(node_id), set()):
                if upstream_id not in required_node_ids:
                    continue
                upstream_node = abstract_tree.nodes.get(str(upstream_id))
                if upstream_node is None:
                    continue
                uroots = sorted(
                    str(v)
                    for v in set(getattr(upstream_node, "root_sources", set()) or set())
                    if str(v)
                )
                if uroots:
                    upstream_roots.add(tuple(uroots))
            if len(upstream_roots) >= 2:
                ambiguous.add(str(node_id))

        if not ambiguous:
            return set(required_node_ids)

        pruned_adj: Dict[str, set[str]] = {}
        for link in getattr(abstract_tree, "links", []) or []:
            src_id = str(getattr(link, "source_node_id", "") or "")
            dst_id = str(getattr(link, "target_node_id", "") or "")
            if not src_id or not dst_id:
                continue
            if src_id in ambiguous or dst_id in ambiguous:
                continue
            if src_id not in required_node_ids or dst_id not in required_node_ids:
                continue
            pruned_adj.setdefault(dst_id, set()).add(src_id)

        seed_nodes = [
            node_id
            for node_id in required_node_ids
            if node_id not in ambiguous
        ]
        kept: set[str] = set()
        stack = list(seed_nodes)
        while stack:
            current = stack.pop()
            if current in kept:
                continue
            kept.add(current)
            for upstream in pruned_adj.get(current, set()):
                if upstream not in kept:
                    stack.append(upstream)
        return kept

    @staticmethod
    def _is_bundle_output_socket(node: Optional[ACEAbstractNode], socket_idx: int) -> bool:
        if node is None:
            return False
        outputs = getattr(node, "outputs", []) or []
        if socket_idx < 0 or socket_idx >= len(outputs):
            return False
        data_type = str(getattr(outputs[socket_idx], "data_type", "") or "")
        return "Bundle" in data_type

    def _group_output_preview_specs_for_source(
        self,
        abstract_tree: ACEAbstractTree,
        source_uid: str,
        proxy_name: str,
        single_source_mode: bool,
    ) -> list[tuple[str, int]]:
        specs: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for out_link in (getattr(abstract_tree, "group_output_links", []) or []):
            node_id = str(getattr(out_link, "source_node_id", "") or "")
            sock_idx = int(getattr(out_link, "source_socket_idx", 0) or 0)
            if not node_id:
                continue
            target_name = str(getattr(out_link, "target_socket_name", "") or "").strip().lower()
            target_idx = int(getattr(out_link, "target_socket_idx", 0) or 0)
            # Auto-preview only taps the intended GN Group Output "Audio" bundle.
            if target_name:
                if target_name != "audio":
                    continue
            elif target_idx != 1:
                continue
            node = abstract_tree.nodes.get(node_id)
            if node is None:
                continue
            if not self._is_bundle_output_socket(node, sock_idx):
                continue
            if not self._node_matches_source(node, source_uid, proxy_name, single_source_mode):
                continue
            key = (node_id, sock_idx)
            if key in seen:
                continue
            seen.add(key)
            specs.append(key)
        return specs

    def _compute_required_node_ids_for_source(
        self,
        abstract_tree: ACEAbstractTree,
        source_uid: str,
        proxy_name: str,
        single_source_mode: bool,
        scheduler_mode: str = "legacy",
    ) -> set[str]:
        if not abstract_tree:
            return set()

        seed_nodes: set[str] = set()
        for node in abstract_tree.nodes.values():
            if not self._is_output_node(node):
                continue
            if self._node_matches_source(node, source_uid, proxy_name, single_source_mode):
                seed_nodes.add(node.id)
        if str(scheduler_mode or "legacy").strip().lower() == "legacy":
            for node in abstract_tree.nodes.values():
                # Legacy mode keeps source nodes active each frame even when no
                # stream/spec branch requests them.
                if not (node.category == 'source_mngmt' and node.type == 'src'):
                    continue
                if self._node_matches_source(node, source_uid, proxy_name, single_source_mode):
                    seed_nodes.add(node.id)
        for node_id, _sock_idx in self._group_output_preview_specs_for_source(
            abstract_tree,
            source_uid,
            proxy_name,
            single_source_mode,
        ):
            seed_nodes.add(node_id)

        if not seed_nodes:
            return set()

        reverse_adj: Dict[str, set[str]] = {}
        for link in abstract_tree.links:
            reverse_adj.setdefault(link.target_node_id, set()).add(link.source_node_id)

        required: set[str] = set()
        stack = list(seed_nodes)
        while stack:
            current = stack.pop()
            if current in required:
                continue
            required.add(current)
            for upstream in reverse_adj.get(current, set()):
                if upstream not in required:
                    stack.append(upstream)
        return self._prune_ambiguous_required_node_ids(
            abstract_tree,
            required,
            source_uid=source_uid,
            proxy_name=proxy_name,
            single_source_mode=single_source_mode,
        )

    @staticmethod
    def _resolve_source_context_for_node(
        node: Optional[ACEAbstractNode],
        source_lookup: Dict[str, ACEAudioSource],
        default_source: Optional[ACEAudioSource] = None,
    ) -> Optional[ACEAudioSource]:
        if node is None:
            return default_source

        tokens: list[str] = []
        if getattr(node, "id", ""):
            tokens.append(str(node.id))
        for root in sorted(getattr(node, "root_sources", set()) or set()):
            tokens.append(str(root))

        for token in tokens:
            src = source_lookup.get(token)
            if src is not None:
                return src
        return default_source
