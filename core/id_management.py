import hashlib
import re
from typing import Dict, Optional

from .. import utils
from ..vars import ACE_PROXY_PREFIX


class ACEIdManager:
    """Central utility for generating and managing standardised ACE identifiers.

    All IDs follow the pattern ``ACE_<category>_<type>_<hex6>`` (node IDs) or
    ``ACE_<name>_<hex6>`` (proxy/attribute names), keeping all keys within
    Blender's 63-character attribute name limit.  The hex suffix is derived
    from a blake2b-4 digest so IDs are deterministic given the same seed and
    collision-free by stepping the salt on collision.
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
        """Generate a stable, unique ACE node ID.

        Validates *category_key* and *node_type_key* against ACE_NODES before
        generating so callers receive a clear ValueError on mismatches rather
        than a silent bad ID.  Steps a salt counter until a candidate is not
        in *existing_ids*.

        Format: ``ACE_<category>_<type>_<hex6>``
        Example: ``ACE_stream_rms_a19c3f``

        Args:
            category_key: ACE node category (e.g. "stream").
            node_type_key: ACE node type within that category (e.g. "rms").
            existing_ids: Collection of IDs already in use; collision avoidance
                steps the salt until a free candidate is found.
            deterministic_seed: Optional stable string used as the hash input
                base.  Defaults to "category:type".

        Returns:
            A unique ACE node ID string.

        Raises:
            ValueError: If *category_key* or *node_type_key* are not in ACE_NODES.
            RuntimeError: If no collision-free candidate can be generated within
                the allowed attempt count.
        """
        # 1. Importing the node registry
        from ..vars import ACE_NODES

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
        """Parse an ACE node ID string into its component parts.

        Strips the ACE prefix and resolves the category/type pair by comparing
        the body against all known ACE_NODES keys.  Prefers the most specific
        (longest) match to handle rare cases where a category or type key
        contains an underscore.

        Args:
            node_id: The full ACE ID string to parse.

        Returns:
            A dict with keys "category", "type", and "uuid", or None if the
            string is not a recognised ACE ID.
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
        """Guarantee that an audio source has a stable internal identifier.

        Returns the existing stable_id when it is a valid source ACE ID.
        Otherwise generates a replacement ID via generate_node_id and writes
        it back to source_prop.stable_id.

        Args:
            source_prop: An ACEAudioSource property-group instance.  None is
                accepted and returns an empty string.
            existing_ids: IDs already in use; forwarded to generate_node_id for
                collision avoidance when a new ID must be allocated.

        Returns:
            The stable ID string assigned to *source_prop*.
        """
        sid = str(getattr(source_prop, "stable_id", "") or "") if source_prop else ""
        if sid:
            parsed = ACEIdManager.parse_node_id(sid)
            if (
                parsed
                and parsed.get("category") == "source_mngmt"
                and parsed.get("type") == "src"
            ):
                return sid

        # Legacy migration path: replace invalid persisted IDs with a valid
        # source ACE ID so source nodes are not pruned during graph traversal.
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

    # -------------------------------------------------------------------------
    # 2. PROXY OBJECT NAMING (Audio Sources)
    # -------------------------------------------------------------------------

    @staticmethod
    def generate_source_proxy_name(source_name: str, stable_seed: Optional[str] = None) -> str:
        """Generate a deterministic proxy object name from the user-facing source name.

        Normalises path separators and file extensions, strips the ACE prefix
        if already present, and appends a 6-character blake2b suffix for
        uniqueness.  Result is capped at 63 characters for Blender compatibility.

        Args:
            source_name: Human-readable source label or file path.
            stable_seed: Optional stable seed string used for the hash.
                Defaults to *source_name* when not provided.

        Returns:
            A proxy object name string with the ACE prefix.
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
        """Generate a short deterministic mesh attribute key.

        Combines a metric hint (derived from the output name or node type) with
        a 4-character source hint and a 6-character hash of node_id and
        socket_idx.  Result is capped at 63 characters.

        Format: ``ACE_<metric>_<srchint><hash6>``
        Example: ``ACE_rms_guit1a2``

        Args:
            proxy_name: The proxy object name (used for the source hint).
            node_id: The ACE node ID (hashed with socket_idx for uniqueness).
            output_name: Human-readable output socket name; used to derive the
                metric token when provided.
            socket_idx: Output socket index; combined with node_id in the hash.

        Returns:
            A compact mesh attribute name string with the ACE prefix.
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

        short_hash = hashlib.blake2b(f"{node_id}:{int(socket_idx)}".encode("utf-8"), digest_size=4).hexdigest()[:6]
        return f"{ACEIdManager.PREFIX}{metric_hint}_{source_hint}{short_hash}"[:63]

    @staticmethod
    def generate_error_state_attr_name(proxy_name: str, node_id: str) -> str:
        """Generate the attribute name for an error-state signal output.

        The GN proxy attribute named by this function should be written with
        ``-1.0`` when the associated source is in an unrecoverable error state,
        and ``0.0`` otherwise.  This lets GN-side drivers detect broken sources
        deterministically without polling ``ace_source_error``.

        Format: ``ACE_errst_<srchint><hash6>``
        Example: ``ACE_errst_guit1a2b3c``

        Args:
            proxy_name: The proxy object name (used for the source hint).
            node_id: The ACE node ID (hashed with "errst" suffix for uniqueness).

        Returns:
            A compact mesh attribute name string with the ACE prefix.
        """
        source_id = proxy_name
        if source_id.startswith(ACEIdManager.PREFIX):
            source_id = source_id[len(ACEIdManager.PREFIX):]
        source_hint = ACEIdManager._clean_string(source_id.split("_", 1)[0]).lower()[:4] or "src"
        short_hash = hashlib.blake2b(
            f"{node_id}:errst".encode("utf-8"), digest_size=4
        ).hexdigest()[:6]
        return f"{ACEIdManager.PREFIX}errst_{source_hint}{short_hash}"[:63]

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    @staticmethod
    def _clean_string(input_str: str) -> str:
        """Remove all non-alphanumeric characters from *input_str*.

        Underscores are intentionally removed so that category and type tokens
        embedded in the ID format (``ACE_Cat_Type_UUID``) remain atomic — a
        type key containing "_" would otherwise break the parse_node_id rsplit
        logic.

        Args:
            input_str: The raw string to sanitise.

        Returns:
            A string containing only [a-zA-Z0-9] characters.
        """
        return re.sub(r'[^a-zA-Z0-9]', '', input_str)

