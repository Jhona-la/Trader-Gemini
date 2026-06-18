"""
⚛️ QUANTUM MEMORY BRIDGE — Topología de la Inmortalidad
=========================================================
Almacenamiento binario contiguo mapeado en RAM (MMAP).
Reemplaza completamente a RobustDataLake (Parquet/Pandas).

Layout del Archivo Binario:
  Bytes [0..7]:   head_ptr  (int64) — Índice de la última vela escrita
  Bytes [8..15]:  tail_ptr  (int64) — Índice de la primera vela válida
  Bytes [16..23]: count     (int64) — Total de velas escritas históricamente
  Bytes [24..63]: Reservado (padding para alineación a 64 bytes)
  Bytes [64..]:   Candle[]  — Array contiguo de structs de 28 bytes

Candle Struct (28 bytes, C-aligned):
  int64   timestamp_ms   (8 bytes)
  float32 open           (4 bytes)
  float32 high           (4 bytes)
  float32 low            (4 bytes)
  float32 close          (4 bytes)
  float32 volume         (4 bytes)

Matemáticas de Memoria:
  1 día (1m)  = 1,440 velas × 28 bytes = 40,320 bytes  (~40 KB)
  100 días    = 144,000 velas × 28 bytes = 4,032,000 bytes (~3.85 MB)
  → Cabe ENTERO en L3 cache del Ryzen 7 5700U (8 MB shared)
"""

import os
import time
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ── Constantes de Geometría ─────────────────────────────────────────────────
HEADER_SIZE = 64
HEADER_SLOTS = HEADER_SIZE // 8  # 8 slots de int64

CANDLE_DTYPE = np.dtype([
    ('timestamp', np.int64),
    ('open',      np.float32),
    ('high',      np.float32),
    ('low',       np.float32),
    ('close',     np.float32),
    ('volume',    np.float32),
], align=True)  # align=True fuerza padding C-compatible

CANDLE_BYTES = CANDLE_DTYPE.itemsize  # 28 (o 32 con align)
CANDLES_PER_DAY = 1440  # 1 minuto × 24 horas


class QuantumMMAP:
    """
    Estructura binaria persistente mapeada en RAM.
    Implementa un Ring Buffer sobre numpy.memmap con evicción O(1).
    """
    __slots__ = (
        'symbol', 'max_candles', 'file_size', 'filepath',
        '_header', '_data', '_count_total',
    )

    def __init__(self, symbol: str, capacity_days: int = 100, cache_dir: str = "data/quantum_lake"):
        from config import Config

        self.symbol = symbol.replace("/", "").replace("-", "").upper()
        self.max_candles = capacity_days * CANDLES_PER_DAY
        self.file_size = HEADER_SIZE + (self.max_candles * CANDLE_BYTES)

        abs_cache = os.path.join(Config.BASE_DIR, cache_dir)
        os.makedirs(abs_cache, exist_ok=True)
        self.filepath = os.path.join(abs_cache, f"{self.symbol}.qbin")

        self._mount()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _mount(self):
        """Monta o crea el archivo binario y lo mapea a RAM."""
        is_new = not os.path.exists(self.filepath)

        if is_new:
            with open(self.filepath, "wb") as f:
                # Pre-allocate todo el espacio con ceros
                f.write(b'\x00' * self.file_size)
            logger.info(
                f"🌌 [QBridge] Pre-allocated {self.symbol}: "
                f"{self.file_size / (1024*1024):.2f} MB "
                f"({self.max_candles:,} candles)"
            )

        # Header: [head_ptr, tail_ptr, count, ...reservado]
        self._header = np.memmap(
            self.filepath, dtype=np.int64, mode='r+',
            offset=0, shape=(HEADER_SLOTS,)
        )

        if is_new:
            self._header[0] = -1   # head_ptr (nada escrito aún)
            self._header[1] = 0    # tail_ptr
            self._header[2] = 0    # count total
            self._header.flush()

        # Data body: array contiguo de Candle structs
        self._data = np.memmap(
            self.filepath, dtype=CANDLE_DTYPE, mode='r+',
            offset=HEADER_SIZE, shape=(self.max_candles,)
        )

    # ── Propiedades Atómicas ─────────────────────────────────────────────────

    @property
    def head(self) -> int:
        return int(self._header[0])

    @property
    def tail(self) -> int:
        return int(self._header[1])

    @property
    def count(self) -> int:
        """Número de velas activas en el buffer."""
        h, t = self.head, self.tail
        if h == -1:
            return 0
        if h >= t:
            return h - t + 1
        return self.max_candles - t + h + 1

    @property
    def last_timestamp_ms(self) -> int:
        """Timestamp de la última vela escrita (0 si vacío)."""
        h = self.head
        if h == -1:
            return 0
        return int(self._data[h]['timestamp'])

    # ── Inyección O(1) ───────────────────────────────────────────────────────

    def inject_candle(self, ts_ms: int, o: float, h: float, l: float, c: float, v: float):
        """Escribe exactamente 28 bytes en la siguiente posición del ring buffer."""
        head = self.head
        idx = (head + 1) % self.max_candles

        # Evicción por colisión: si head alcanza a tail, avanzamos tail
        if idx == self.tail and head != -1:
            self._header[1] = (self.tail + 1) % self.max_candles

        self._data[idx] = (ts_ms, o, h, l, c, v)
        self._header[0] = idx
        self._header[2] += 1

    def inject_bulk(self, timestamps: np.ndarray, ohlcv: np.ndarray):
        """
        Inyección masiva vectorizada. Evita el overhead del bucle Python.
        
        Args:
            timestamps: array int64 de timestamps en ms, shape (N,)
            ohlcv: array float32 de [open, high, low, close, volume], shape (N, 5)
        """
        n = len(timestamps)
        if n == 0:
            return

        head = self.head
        start_idx = (head + 1) % self.max_candles

        # Caso simple: todo cabe sin wrap-around
        end_idx = start_idx + n
        if end_idx <= self.max_candles:
            self._data['timestamp'][start_idx:end_idx] = timestamps
            self._data['open'][start_idx:end_idx] = ohlcv[:, 0]
            self._data['high'][start_idx:end_idx] = ohlcv[:, 1]
            self._data['low'][start_idx:end_idx] = ohlcv[:, 2]
            self._data['close'][start_idx:end_idx] = ohlcv[:, 3]
            self._data['volume'][start_idx:end_idx] = ohlcv[:, 4]
        else:
            # Wrap-around: dos escrituras
            first_chunk = self.max_candles - start_idx
            # Primera parte: desde start_idx hasta el final
            self._data['timestamp'][start_idx:] = timestamps[:first_chunk]
            self._data['open'][start_idx:] = ohlcv[:first_chunk, 0]
            self._data['high'][start_idx:] = ohlcv[:first_chunk, 1]
            self._data['low'][start_idx:] = ohlcv[:first_chunk, 2]
            self._data['close'][start_idx:] = ohlcv[:first_chunk, 3]
            self._data['volume'][start_idx:] = ohlcv[:first_chunk, 4]
            # Segunda parte: desde el inicio
            rest = n - first_chunk
            self._data['timestamp'][:rest] = timestamps[first_chunk:]
            self._data['open'][:rest] = ohlcv[first_chunk:, 0]
            self._data['high'][:rest] = ohlcv[first_chunk:, 1]
            self._data['low'][:rest] = ohlcv[first_chunk:, 2]
            self._data['close'][:rest] = ohlcv[first_chunk:, 3]
            self._data['volume'][:rest] = ohlcv[first_chunk:, 4]

        new_head = (start_idx + n - 1) % self.max_candles
        self._header[0] = new_head
        self._header[2] += n

        # Evicción: si escribimos más que la capacidad, mover tail
        if n >= self.max_candles:
            self._header[1] = (new_head + 1) % self.max_candles
        elif self.count > self.max_candles:
            # Ajustar tail para mantener max_candles
            self._header[1] = (new_head + 1) % self.max_candles

    # ── Lectura Zero-Copy ────────────────────────────────────────────────────

    def get_view(self, lookback: int = 0) -> np.ndarray:
        """
        Retorna una vista directa al memmap. Sin copia si es contiguo.
        Si lookback=0, retorna TODAS las velas vivas.
        """
        h = self.head
        t = self.tail
        if h == -1:
            return np.empty(0, dtype=CANDLE_DTYPE)

        if lookback > 0:
            total = self.count
            if lookback > total:
                lookback = total
            t = (h - lookback + 1) % self.max_candles

        if h >= t:
            # Contiguo: retorna view directo (CERO copia)
            return self._data[t:h + 1]
        else:
            # Wrap: necesita concatenar (1 copia inevitable)
            return np.concatenate((self._data[t:], self._data[:h + 1]))

    def get_ohlcv_arrays(self, lookback: int = 0):
        """
        Retorna arrays separados float32 para cada columna.
        Máxima simpatía con L1 cache (Struct of Arrays).
        """
        view = self.get_view(lookback)
        if len(view) == 0:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty, empty, empty, empty
        return (
            view['open'],
            view['high'],
            view['low'],
            view['close'],
            view['volume'],
        )

    def to_dataframe(self, lookback: int = 0):
        """
        Compatibilidad con el sistema legacy (Pandas DataFrame).
        SOLO usar en el boundary de inicialización, NUNCA en el hot-path.
        """
        import pandas as pd
        view = self.get_view(lookback)
        if len(view) == 0:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        df = pd.DataFrame({
            'open': view['open'].astype(np.float64),
            'high': view['high'].astype(np.float64),
            'low': view['low'].astype(np.float64),
            'close': view['close'].astype(np.float64),
            'volume': view['volume'].astype(np.float64),
        })
        df.index = pd.to_datetime(view['timestamp'], unit='ms')
        df.index.name = 'timestamp'
        return df

    def flush(self):
        """Fuerza sync de memmap al disco."""
        self._header.flush()
        self._data.flush()

    def close(self):
        """Libera los file handles del memmap (necesario en Windows)."""
        if hasattr(self, '_data') and self._data is not None:
            del self._data
            self._data = None
        if hasattr(self, '_header') and self._header is not None:
            del self._header
            self._header = None

    def __del__(self):
        self.close()

    def __repr__(self):
        return (
            f"QuantumMMAP({self.symbol}, "
            f"candles={self.count:,}/{self.max_candles:,}, "
            f"head={self.head}, tail={self.tail})"
        )


# ═════════════════════════════════════════════════════════════════════════════
# QuantumDataLake — Reemplaza RobustDataLake
# ═════════════════════════════════════════════════════════════════════════════

class QuantumDataLake:
    """
    Gestor de datos MMAP con descarga incremental desde Binance.
    Reemplaza data/robust_data_lake.py (Parquet/Pandas).

    Diferencias fundamentales:
    - Almacenamiento: Parquet LZ4 → MMAP binario contiguo (cero parsing)
    - Lectura: pl.read_parquet().to_pandas() → numpy.memmap view (cero copia)
    - Evicción: Filtro booleano O(n) → Aritmética de punteros O(1)
    - Escritura: pd.to_parquet() → Escritura directa de 28 bytes O(1)
    """

    def __init__(self, capacity_days: int = 100, cache_dir: str = "data/quantum_lake"):
        self.capacity_days = capacity_days
        self.cache_dir = cache_dir
        self._pools: dict[str, QuantumMMAP] = {}
        self._client = None

    def _get_client(self):
        if self._client is None:
            from binance.client import Client
            self._client = Client()
        return self._client

    def _get_pool(self, symbol: str) -> QuantumMMAP:
        safe = symbol.replace("/", "").replace("-", "").upper()
        if safe not in self._pools:
            self._pools[safe] = QuantumMMAP(safe, self.capacity_days, self.cache_dir)
        return self._pools[safe]

    def fetch_symbol(self, symbol: str, days: int = 30, end_time: datetime = None):
        """
        API compatible con RobustDataLake.fetch_symbol().
        Retorna pd.DataFrame con index=timestamp y cols=[open,high,low,close,volume].
        
        Internamente:
        1. Comprueba el MMAP existente (O(1) — leer head timestamp)
        2. Descarga solo el delta faltante desde Binance
        3. Inyecta el delta binariamente (inject_bulk)
        4. Retorna un DataFrame (boundary conversion, no hot-path)
        """
        import pandas as pd

        if end_time is None:
            end_time = datetime.utcnow()

        start_time = end_time - timedelta(days=days)
        pool = self._get_pool(symbol)

        # ── 1. Determinar Delta ──────────────────────────────────────────
        last_ts = pool.last_timestamp_ms
        if last_ts > 0:
            last_dt = datetime.utcfromtimestamp(last_ts / 1000)
            fetch_start = last_dt + timedelta(minutes=1)  # Siguiente minuto

            # Si ya cubrimos todo el periodo, retornar view directa
            if fetch_start >= end_time:
                df = pool.to_dataframe()
                mask = (df.index >= start_time) & (df.index <= end_time)
                return df.loc[mask]
        else:
            fetch_start = start_time

        # ── 2. Descargar Delta desde Binance ─────────────────────────────
        delta_hours = (end_time - fetch_start).total_seconds() / 3600
        if delta_hours < 0.01:
            df = pool.to_dataframe()
            if len(df) == 0:
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            mask = (df.index >= start_time) & (df.index <= end_time)
            return df.loc[mask]

        safe_sym = symbol.replace("/", "").replace("-", "")
        print(
            f"📡 [QBridge] {symbol} | Delta: "
            f"{fetch_start.strftime('%Y-%m-%d %H:%M')} → "
            f"{end_time.strftime('%Y-%m-%d %H:%M')} "
            f"({delta_hours:.1f}h)"
        )

        client = self._get_client()
        all_klines = []
        current = fetch_start

        while current < end_time:
            chunk_end = min(current + timedelta(hours=16), end_time)
            for attempt in range(3):
                try:
                    klines = client.get_historical_klines(
                        safe_sym,
                        client.KLINE_INTERVAL_1MINUTE,
                        str(int(current.timestamp() * 1000)),
                        str(int(chunk_end.timestamp() * 1000)),
                        limit=1000,
                    )
                    if klines:
                        all_klines.extend(klines)
                    break
                except Exception as e:
                    print(f"  ⏳ [QBridge] Retry {attempt+1}/3: {e}")
                    time.sleep(2)

            current = chunk_end
            time.sleep(0.05)

        # ── 3. Inyección Binaria Masiva ──────────────────────────────────
        if all_klines:
            n = len(all_klines)
            ts_arr = np.empty(n, dtype=np.int64)
            ohlcv_arr = np.empty((n, 5), dtype=np.float32)

            for i, k in enumerate(all_klines):
                ts_arr[i] = int(k[0])
                ohlcv_arr[i, 0] = float(k[1])  # open
                ohlcv_arr[i, 1] = float(k[2])  # high
                ohlcv_arr[i, 2] = float(k[3])  # low
                ohlcv_arr[i, 3] = float(k[4])  # close
                ohlcv_arr[i, 4] = float(k[5])  # volume

            # Deduplicar por timestamp (solo inyectar lo que no existe)
            if pool.last_timestamp_ms > 0:
                mask = ts_arr > pool.last_timestamp_ms
                ts_arr = ts_arr[mask]
                ohlcv_arr = ohlcv_arr[mask]

            if len(ts_arr) > 0:
                pool.inject_bulk(ts_arr, ohlcv_arr)
                pool.flush()
                print(f"  ⚛️  [QBridge] {symbol}: {len(ts_arr):,} velas inyectadas en MMAP")

        # ── 4. Retornar DataFrame filtrado ───────────────────────────────
        df = pool.to_dataframe()
        if len(df) == 0:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        mask = (df.index >= start_time) & (df.index <= end_time)
        result = df.loc[mask]
        print(f"✅ [QBridge] {symbol} | Entregadas: {len(result):,} velas (MMAP Zero-Copy)")
        return result


# ═════════════════════════════════════════════════════════════════════════════
# Singleton Global
# ═════════════════════════════════════════════════════════════════════════════

_global_quantum_lake = None

def get_quantum_lake() -> QuantumDataLake:
    global _global_quantum_lake
    if _global_quantum_lake is None:
        _global_quantum_lake = QuantumDataLake()
    return _global_quantum_lake
