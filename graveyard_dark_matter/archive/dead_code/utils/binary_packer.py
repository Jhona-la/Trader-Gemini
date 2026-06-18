
import struct
from datetime import datetime

class BinaryPacker:
    """
    🔬 PHASE 26: STRUCT PACKING
    Provides ultra-fast, zero-overhead binary serialization for core events.
    Used for writing high-frequency audit logs to disk or shared memory.
    
    Format:
    - timestamp (8 bytes, double)
    - type_id (1 byte, char)
    - symbol_id (2 bytes, unsigned short) - Requires intern mapping
    - price (8 bytes, double)
    - quantity (8 bytes, double)
    - side (1 byte, char)
    
    Total: ~28 bytes per trade event (vs ~200 bytes JSON)
    """
    
    # Pre-compiled struct formats
    # d = double (8)
    # H = unsigned short (2)
    # B = unsigned char (1)
    # 16s = 16 char string
    
    # Tick: Timestamp(d), Price(d), Volume(d)
    _FMT_TICK = struct.Struct('!ddd') 
    
    # Trade: Timestamp(d), Price(d), Qty(d), Side(B), SymbolID(H)
    _FMT_TRADE = struct.Struct('!dddBH')
    
    @staticmethod
    def pack_tick(timestamp: float, price: float, volume: float) -> bytes:
        return BinaryPacker._FMT_TICK.pack(timestamp, price, volume)
        
    @staticmethod
    def unpack_tick(buffer: bytes):
        return BinaryPacker._FMT_TICK.unpack(buffer)

    @staticmethod
    def pack_trade(timestamp: float, price: float, quantity: float, side_int: int, symbol_id: int) -> bytes:
        """
        Packs a trade event into 28 bytes.
        side_int: 0=Buy, 1=Sell
        symbol_id: Integer mapped ID for symbol (managed externally)
        """
        return BinaryPacker._FMT_TRADE.pack(timestamp, price, quantity, side_int, symbol_id)

    @staticmethod
    def unpack_trade(buffer: bytes):
        return BinaryPacker._FMT_TRADE.unpack(buffer)
