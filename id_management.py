import hashlib
import re
from typing import Dict, Optional

from .vars import ACE_PROXY_PREFIX


class ACEIdManager:
    """
    Central utility to generate and manage standardized IDs 
    compatible with the ACE_NODES structure and Blender naming limits.
    """
    PREFIX = ACE_PROXY_PREFIX

    # -------------------------------------------------------------------------
    # 1. NODE ID GENERATION (Deterministic hash-seeded IDs)
    # -------------------------------------------------------------------------
    
    @staticmethod
    def generate_node_id(
        category_key: str,
        node_type_key: str,
        existing_ids: Optional[list[str]] = None,
        deterministic_seed: str = "",
    ) -> str:
        """
        Generates a stable, unique ID for a Node.
        Validates that the category and type exist in ACE_NODES.

        Format: ACE_{Category}_{Type}_{ShortHash}
        Example: ACE_stream_rms_a19c3f
        """
        # 1. Importing the node registry
        from .vars import ACE_NODES

        # 2. Validation against ACE_NODES structure
        if category_key not in ACE_NODES:
            raise ValueError(f"Invalid Node Category: '{category_key}'. Available: {list(ACE_NODES.keys())}")
        
        # Access nested 'nodes' dictionary
        category_data = ACE_NODES[category_key]
        if node_type_key not in category_data['nodes']:
            raise ValueError(f"Invalid Node Type: '{node_type_key}' for Category '{category_key}'.")

        # 3. Deterministic construction with collision stepping.
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
        """
        Utility to extract info from an ID. 
        Returns None if ID is not an ACE ID.
        """
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
            from .vars import ACE_NODES
        except Exception:
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
        """
        Guarantee that an audio source has a stable internal ID.
        """
        sid = getattr(source_prop, "stable_id", "") if source_prop else ""
        if sid:
            return sid
        sid = ACEIdManager.generate_node_id("source_mngmt", "src", existing_ids=existing_ids)
        try:
            source_prop.stable_id = sid  # type: ignore[attr-defined]
        except Exception:
            pass
        return sid

    # -------------------------------------------------------------------------
    # 2. PROXY OBJECT NAMING (Audio Sources)
    # -------------------------------------------------------------------------

    @staticmethod
    def generate_source_proxy_name(source_name: str, stable_seed: Optional[str] = None) -> str:
        """
        Generates a deterministic proxy object name from the user-facing source
        name while keeping uniqueness stable via an optional seed.
        """
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

    # -------------------------------------------------------------------------
    # 3. ATTRIBUTE NAMING (Result Data)
    # -------------------------------------------------------------------------
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
        """
        Short deterministic attribute key.
        Example: ACE_rms_guit1a2
        """
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

        short_hash = hashlib.blake2b(f"{node_id}:{int(socket_idx)}".encode("utf-8"), digest_size=4).hexdigest()[:3]
        return f"{ACEIdManager.PREFIX}{metric_hint}_{source_hint}{short_hash}"[:63]

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    @staticmethod
    def _clean_string(input_str: str) -> str:
        """
        Removes non-alphanumeric characters for Blender compatibility.
        Important: This removes underscores too, which is helpful for the ID 
        parsing logic (ACE_Cat_Type_UUID) to ensure 'Cat' and 'Type' are atomic.
        """
        return re.sub(r'[^a-zA-Z0-9]', '', input_str)


