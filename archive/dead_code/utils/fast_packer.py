
import msgpack
import logging
import numpy as np
from datetime import datetime
from typing import Any

logger = logging.getLogger("FastPacker")

class FastPacker:
    """
    High-Performance Binary Serializer (Phase 2).
    Uses MessagePack for compact, zero-latency IPC.
    """

    @staticmethod
    def _default_encoder(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.timestamp()
        return str(obj)

    @staticmethod
    def pack(data: Any) -> bytes:
        """
        Serializes data to optimized binary format.
        """
        try:
            return msgpack.packb(data, default=FastPacker._default_encoder, use_bin_type=True)
        except Exception as e:
            logger.error(f"Packing error: {e}")
            return b""

    @staticmethod
    def unpack(data: bytes) -> Any:
        """
        Deserializes binary data to python object.
        """
        try:
            return msgpack.unpackb(data, raw=False, strict_map_key=False)
        except Exception as e:
            logger.error(f"Unpacking error: {e}")
            return None
