"""Live input over UDP: flow identifiers and packet reads."""

import socket
from typing import Optional, Tuple
from ... import utils
import numpy as np


class UdpInput:
    # Internal helpers (live IO)
    @staticmethod
    def _parse_udp_flow_identifier(flow_id: str):
        raw = (flow_id or "").strip()
        lower = raw.lower()
        if lower.startswith("udp://"):
            raw = raw[6:]
        elif lower.startswith("udp:"):
            raw = raw[4:]
        raw = raw.strip()
        if not raw:
            return None

        host = "0.0.0.0"
        port_str = ""
        channels = 2
        sample_rate = 44100

        if "@" in raw:
            raw, meta = raw.split("@", 1)
            meta = meta.strip().lower()
            if "x" in meta:
                sr_part, ch_part = meta.split("x", 1)
                if sr_part.strip().isdigit():
                    sample_rate = int(sr_part.strip())
                if ch_part.strip().isdigit():
                    channels = int(ch_part.strip())
            elif meta.isdigit():
                sample_rate = int(meta)

        parts = [p.strip() for p in raw.split(":") if p.strip()]
        if len(parts) == 1:
            port_str = parts[0]
        elif len(parts) == 2:
            if parts[0].isdigit():
                port_str = parts[0]
                if parts[1].isdigit():
                    channels = int(parts[1])
            else:
                host, port_str = parts[0], parts[1]
        elif len(parts) >= 3:
            if parts[0].isdigit():
                port_str = parts[0]
                if parts[1].isdigit():
                    channels = int(parts[1])
                if parts[2].isdigit():
                    sample_rate = int(parts[2])
            else:
                host, port_str = parts[0], parts[1]
                if parts[2].isdigit():
                    channels = int(parts[2])
                if len(parts) >= 4 and parts[3].isdigit():
                    sample_rate = int(parts[3])

        if not port_str.isdigit():
            return None
        port = int(port_str)
        if port < 1 or port > 65535:
            return None
        channels = max(1, min(int(channels), 8))
        sample_rate = max(8000, min(int(sample_rate), 192000))

        return {
            "host": host,
            "port": port,
            "channels": channels,
            "sample_rate": sample_rate,
        }

    @staticmethod
    def _parse_daw_flow_identifier(flow_id: str):
        payload = (flow_id or "").strip()
        if not payload:
            # Default DAW bridge: UDP stereo float stream on a conventional local port.
            return "daw_udp", {"host": "0.0.0.0", "port": 7001, "channels": 2, "sample_rate": 44100}

        lower = payload.lower()
        if lower.startswith("udp:") or lower.startswith("udp://"):
            cfg = UdpInput._parse_udp_flow_identifier(payload)
            if cfg is None:
                return "invalid", {"error": "Invalid DAW UDP format. Use daw:udp:<port>[:channels[:sample_rate]]"}
            return "daw_udp", cfg

        if lower.startswith("sd:"):
            device_token = payload[3:].strip()
            return "daw_sounddevice", {"device": device_token or None, "channels": 2}

        if lower.startswith("sounddevice:"):
            device_token = payload.split(":", 1)[1].strip()
            return "daw_sounddevice", {"device": device_token or None, "channels": 2}

        # Bare numeric/host payloads map to UDP by default for DAW emitters.
        cfg = UdpInput._parse_udp_flow_identifier(f"udp:{payload}")
        if cfg is not None:
            return "daw_udp", cfg

        # Fallback for loopback-capable DAW drivers exposed through sounddevice.
        return "daw_sounddevice", {"device": payload, "channels": 2}

    @staticmethod
    def _parse_instrument_flow_identifier(flow_id: str):
        payload = (flow_id or "").strip()
        if not payload:
            return "instrument_sounddevice", {"device": None, "channels": 1}

        lower = payload.lower()
        if lower.startswith("udp:") or lower.startswith("udp://"):
            cfg = UdpInput._parse_udp_flow_identifier(payload)
            if cfg is None:
                return "invalid", {"error": "Invalid instrument UDP format. Use instrument:udp:<port>[:channels[:sample_rate]]"}
            if int(cfg.get("channels", 1) or 1) < 1:
                cfg["channels"] = 1
            return "instrument_udp", cfg

        device_token = payload
        channels = 1
        if lower.startswith("sd:"):
            device_token = payload[3:].strip()
        elif lower.startswith("sounddevice:"):
            device_token = payload.split(":", 1)[1].strip()

        if "@" in device_token:
            dev, meta = device_token.split("@", 1)
            device_token = dev.strip()
            if meta.strip().isdigit():
                channels = int(meta.strip())

        channels = max(1, min(channels, 8))
        return "instrument_sounddevice", {"device": device_token or None, "channels": channels}

    def _open_live_udp(self, cfg: dict, kind: str = "udp_stream") -> bool:
        """Bind a non-blocking UDP socket for float32 audio streaming."""
        try:
            host = str(cfg.get("host", "0.0.0.0") or "0.0.0.0")
            port = int(cfg.get("port", 0) or 0)
            channels = int(cfg.get("channels", 2) or 2)
            sample_rate = int(cfg.get("sample_rate", 44100) or 44100)
            if port <= 0:
                raise ValueError("Invalid UDP port")

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setblocking(False)
            sock.bind((host, port))
            self._live_udp_socket = sock
            self._live_udp_host = host
            self._live_udp_port = port
            self._live_udp_channels = max(1, channels)
            self._live_udp_sample_rate = max(8000, sample_rate)
            self.sample_rate = self._live_udp_sample_rate
            self.channels = self._live_udp_channels
            self.kind = kind
            self._eof = False
            self.last_error = ""
            self._last_live_block = None
            self._live_capture_sample_cursor = 0
            self._live_last_capture_sample = 0
            self._live_last_capture_block_samples = 0
            return True
        except Exception as e:
            utils.note_swallowed("udp.UdpInput._open_live_udp", e)
            self._live_udp_socket = None
            self.kind = "live_stub"
            self.last_error = f"UDP live stream failed to open: {e}"
            return False

    def _read_live_udp_block(self, block_size: int) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], int]]:
        """Drain the non-blocking UDP socket and return the freshest packet."""
        sock = self._live_udp_socket
        if sock is None:
            return None

        packet = None
        packet_frames = 0
        consumed_samples = 0
        channels = max(1, int(self._live_udp_channels))
        while True:
            try:
                data, _addr = sock.recvfrom(262144)
                if data:
                    raw = np.frombuffer(data, dtype=np.float32)
                    if raw.size <= 0:
                        continue
                    if channels == 1:
                        frames = int(raw.size)
                    else:
                        frames = int(raw.size // channels)
                    if frames <= 0:
                        continue
                    consumed_samples += frames
                    packet = raw
                    packet_frames = frames
            except BlockingIOError:
                break
            except Exception as exc:
                utils.note_swallowed("udp.UdpInput._read_live_udp_block", exc)
                self.last_error = "UDP live stream read failed"
                break

        if packet is None:
            if self._last_live_block:
                mono_prev, stereo_prev, sr_prev = self._last_live_block
                return mono_prev, stereo_prev, sr_prev
            return None

        raw = packet
        if raw.size == 0:
            return None
        if consumed_samples > 0:
            self._live_capture_sample_cursor += int(consumed_samples)
            self._live_last_capture_sample = int(self._live_capture_sample_cursor)
        self._live_last_capture_block_samples = max(int(packet_frames), 1)
        mono = None
        stereo = None
        if channels == 1:
            mono = raw
        else:
            frames = raw.size // channels
            if frames <= 0:
                return None
            raw = raw[:frames * channels]
            arr = raw.reshape(frames, channels)
            mono = np.mean(arr, axis=1)
            if channels >= 2:
                stereo = arr[:, :2].T.astype(np.float32, copy=False)

        if mono.shape[0] < block_size:
            padded = np.zeros(block_size, dtype=np.float32)
            padded[:mono.shape[0]] = mono
            mono = padded
            if stereo is not None:
                padded_stereo = np.zeros((stereo.shape[0], block_size), dtype=np.float32)
                padded_stereo[:, :stereo.shape[1]] = stereo
                stereo = padded_stereo
        elif mono.shape[0] > block_size:
            mono = mono[:block_size]
            if stereo is not None:
                stereo = stereo[:, :block_size]

        sr = int(self._live_udp_sample_rate or self.sample_rate or 44100)
        self.sample_rate = sr
        self.channels = channels
        self._last_live_block = (
            mono.astype(np.float32, copy=False),
            stereo.astype(np.float32, copy=False) if stereo is not None else None,
            sr,
        )
        return mono.astype(np.float32, copy=False), stereo, sr
