import hashlib
import re
from typing import Dict, Optional

from .. import utils
from ..vars import ACE_PROXY_PREFIX


class ACEIdManager:
    """Central utility for generating and managing standardised ACE identifiers."""
    PREFIX = ACE_PROXY_PREFIX

    
    @staticmethod
    def generate_node_id(
        category_key: str,
        node_type_key: str,
        existing_ids: Optional[list[str]] = None,
        deterministic_seed: str = "",
    ) -> str:
        """Generate a stable, unique ACE node ID."""
        from ..vars import ACE_NODES

        if category_key not in ACE_NODES:
            raise ValueError(f"Invalid Node Category: '{category_key}'. Available: {list(ACE_NODES.keys())}")
        
        # Access nested 'nodes' dictionary
        category_data = ACE_NODES[category_key]
        if node_type_key not in category_data['nodes']:
            raise ValueError(f"Invalid Node Type: '{node_type_key}' for Category '{category_key}'.")

        existing_ids = {str(v) for v in (existing_ids or []) if str(v)}
        seed_prefix = str(deterministic_seed or f"{category_key}:{node_type_key}")
        max_attempts = max(len(existing_ids) + 64, 256)
        for salt in range(max_attempts):
            suffix = hashlib.blake2b(f"{seed_prefix}:{salt}".encode("utf-8"), digest_size=4).hexdigest()[:6]
            candidate = f"{ACEIdManager.PREFIX}{category_key}_{node_type_key}_{suffix}"
            if candidate not in existing_ids:
                return candidate
        raise RuntimeError(f"Could not allocate deterministic ID for '{category_key}:{node_type_key}'")

    @staticmethod
    def parse_node_id(node_id: str) -> Optional[Dict[str, str]]:
        """Parse an ACE node ID string into its component parts."""
        if not node_id.startswith(ACEIdManager.PREFIX):
            return None
        payload = node_id[len(ACEIdManager.PREFIX):]
        try:
            body, uid = payload.rsplit("_", 1)
        except ValueError:
            return None
        if not body or not uid:
            return None

        # Robust parsing against category/type keys that themselves contain "_".
        try:
            from ..vars import ACE_NODES
        except Exception as exc:
            utils.note_swallowed("id_management.ACEIdManager.parse_node_id", exc)
            ACE_NODES = {}

        matches: list[tuple[str, str]] = []
        for category, cat_def in ACE_NODES.items():
            nodes = cat_def.get("nodes", {}) if isinstance(cat_def, dict) else {}
            for type_key in nodes.keys():
                if body == f"{category}_{type_key}":
                    matches.append((category, type_key))

        if not matches:
            return None
        # Prefer the most specific (longest) exact match in case of rare collisions.
        category, type_key = sorted(matches, key=lambda ct: (len(ct[0]), len(ct[1])), reverse=True)[0]
        return {
            "category": category,
            "type": type_key,
            "uuid": uid,
        }

    @staticmethod
    def ensure_source_uid(source_prop, existing_ids: Optional[list[str]] = None) -> str:
        """Guarantee that an audio source has a stable internal identifier."""
        sid = str(getattr(source_prop, "stable_id", "") or "") if source_prop else ""
        if sid:
            parsed = ACEIdManager.parse_node_id(sid)
            if (
                parsed
                and parsed.get("category") == "source_mngmt"
                and parsed.get("type") == "src"
            ):
                return sid

        # An invalid persisted ID gets a valid source ID so traversal does not prune the node.
        legacy_seed = sid
        existing = [str(v) for v in (existing_ids or []) if str(v) and str(v) != sid]
        if source_prop and not legacy_seed:
            source_name = str(getattr(source_prop, "name", "") or "")
            source_type = str(getattr(source_prop, "source_type", "") or "")
            source_path = str(getattr(source_prop, "filepath", "") or "")
            source_flow = str(getattr(source_prop, "flow_identifier", "") or "")
            legacy_seed = "|".join(
                token for token in (source_type, source_name, source_path, source_flow) if token
            )

        sid = ACEIdManager.generate_node_id(
            "source_mngmt",
            "src",
            existing_ids=existing,
            deterministic_seed=f"source:{legacy_seed or 'default'}",
        )
        with utils.logged_errors("id_management.ACEIdManager.ensure_source_uid"):
            source_prop.stable_id = sid  # type: ignore[attr-defined]
        return sid


    @staticmethod
    def generate_source_proxy_name(source_name: str, stable_seed: Optional[str] = None) -> str:
        """Generate a deterministic proxy object name from the user-facing source name."""
        raw_name = (source_name or "").strip()
        if not raw_name:
            raw_name = "Source"

        # Normalize path-like names and optional extension from file-based labels.
        base_name = raw_name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if "." in base_name:
            stem = base_name.rsplit(".", 1)[0]
            if stem:
                base_name = stem

        if base_name.startswith(ACEIdManager.PREFIX):
            base_name = base_name[len(ACEIdManager.PREFIX):]
        clean_name = ACEIdManager._clean_string(base_name) or "Source"
        short_name = clean_name[:24]

        seed = (stable_seed or raw_name or "source").strip()
        name_hash = hashlib.blake2b(seed.encode("utf-8"), digest_size=4).hexdigest()[:6]
        return f"{ACEIdManager.PREFIX}{short_name}_{name_hash}"[:63]

    @staticmethod
    def _compact_metric_token(metric: str) -> str:
        clean = ACEIdManager._clean_string(metric).lower()
        aliases = {
            "rms": "rms",
            "centroid": "cent",
            "spectralcentroid": "cent",
            "flux": "flux",
            "spectralflux": "flux",
            "pitch": "pitch",
            "onset": "onset",
            "zcr": "zcr",
            "zerocrossingrate": "zcr",
            "stereowidth": "width",
            "stereobalance": "bal",
            "stereovector": "svec",
            "spectralvector": "spvec",
            "bpf": "bpf",
            "timesignature": "tsig",
            "beatphase": "beat",
        }
        if clean in aliases:
            return aliases[clean]
        if not clean:
            return "val"
        return clean[:6]

    @staticmethod
    def generate_compact_attr_name(proxy_name: str, node_id: str, output_name: str = "", socket_idx: int = 0) -> str:
        """Generate a short deterministic mesh attribute key."""
        source_id = proxy_name
        if source_id.startswith(ACEIdManager.PREFIX):
            source_id = source_id[len(ACEIdManager.PREFIX):]
        source_hint = ACEIdManager._clean_string(source_id.split("_", 1)[0]).lower()[:4] or "src"

        metric_raw = output_name or ""
        if not metric_raw:
            parsed = ACEIdManager.parse_node_id(node_id)
            if parsed:
                metric_raw = parsed.get("type", "")
        metric_hint = ACEIdManager._compact_metric_token(metric_raw)

        short_hash = hashlib.blake2b(f"{node_id}:{int(socket_idx)}".encode("utf-8"), digest_size=4).hexdigest()[:6]
        return f"{ACEIdManager.PREFIX}{metric_hint}_{source_hint}{short_hash}"[:63]

    @staticmethod
    def generate_error_state_attr_name(proxy_name: str, node_id: str) -> str:
        """Generate the attribute name for an error-state signal output."""
        source_id = proxy_name
        if source_id.startswith(ACEIdManager.PREFIX):
            source_id = source_id[len(ACEIdManager.PREFIX):]
        source_hint = ACEIdManager._clean_string(source_id.split("_", 1)[0]).lower()[:4] or "src"
        short_hash = hashlib.blake2b(
            f"{node_id}:errst".encode("utf-8"), digest_size=4
        ).hexdigest()[:6]
        return f"{ACEIdManager.PREFIX}errst_{source_hint}{short_hash}"[:63]

    # HELPERS

    @staticmethod
    def _clean_string(input_str: str) -> str:
        return re.sub(r'[^a-zA-Z0-9]', '', input_str)

