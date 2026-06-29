
import mmap
import os
import json
import struct
import logging
from typing import Dict, Any

logger = logging.getLogger("SharedMemory")

class SharedStateManager:
    """
    🔬 PHASE 36: MEMORY MAPPING (mmap)
    Creates a memory-mapped file to share state between Bot and Dashboard
    with ZERO disk I/O latency during reads/writes.
    
    Structure:
    [4 bytes: Data Length] [N bytes: JSON Data] [Padding...]
    """
    
    def __init__(self, filename: str, size: int = 4096):
        self.filename = filename
        self.size = size
        self.mmap_file = None
        self._file_obj = None
        self._init_mmap()
        
    def _init_mmap(self):
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            
            # Open file (create if not exists)
            # 'w+b' truncates! We want 'r+b' but need to create if missing.
            if not os.path.exists(self.filename):
                with open(self.filename, 'wb') as f:
                    f.write(b'\0' * self.size)
            
            self._file_obj = open(self.filename, 'r+b')
            
            # Create mmap
            self.mmap_file = mmap.mmap(
                self._file_obj.fileno(), 
                self.size, 
                access=mmap.ACCESS_WRITE
            )
            logger.info(f"🔬 [PHASE 36] Shared Memory initialized: {self.filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to init Shared Memory: {e}")
            if self._file_obj: self._file_obj.close()

    def write_state(self, state: Dict[str, Any]):
        """
        Writes state dict to memory map.
        1. Serialize to JSON bytes
        2. Write Length (4 bytes)
        3. Write Payload
        """
        if not self.mmap_file:
            return
            
        try:
            # Phase 1: Use orjson or ujson if available (implied)
            # For now standard json for safety, assuming Phase 1 json wrapper is used elsewhere
            data_bytes = json.dumps(state).encode('utf-8')
            data_len = len(data_bytes)
            
            if data_len + 4 > self.size:
                logger.warning(f"SharedMemory overflow! Size {data_len} > {self.size}. Truncating.")
                data_bytes = data_bytes[:self.size - 4]
                data_len = len(data_bytes)
            
            self.mmap_file.seek(0)
            self.mmap_file.write(struct.pack('<I', data_len))
            self.mmap_file.write(data_bytes)
            # No flush needed for mmap to be visible to other processes immediately (usually)
            # self.mmap_file.flush() # Optional, might induce disk I/O. valid for persistence.
            
        except Exception as e:
            logger.error(f"SharedMemory Write Error: {e}")

    def close(self):
        if self.mmap_file:
            self.mmap_file.close()
        if self._file_obj:
            self._file_obj.close()
