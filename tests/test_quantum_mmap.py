"""
Test de validación del Quantum Memory Bridge (MMAP Storage).
Verifica: creación, inyección, lectura zero-copy, ring buffer, y DataFrame export.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import tempfile

def test_quantum_mmap():
    from core.quantum.mmap_storage import QuantumMMAP, CANDLE_DTYPE

    # Usar directorio temporal para no contaminar el proyecto
    with tempfile.TemporaryDirectory() as tmpdir:
        # Monkey-patch Config.BASE_DIR
        import config
        original_base = config.Config.BASE_DIR
        config.Config.BASE_DIR = tmpdir

        try:
            # ── Test 1: Creación ────────────────────────────────────────
            print("=" * 60)
            print("TEST 1: Creación de QuantumMMAP")
            pool = QuantumMMAP("BTCUSDT", capacity_days=1)  # 1 día = 1440 velas
            assert pool.head == -1, f"Head debería ser -1, es {pool.head}"
            assert pool.count == 0, f"Count debería ser 0, es {pool.count}"
            assert os.path.exists(pool.filepath), "Archivo .qbin no creado"
            file_size = os.path.getsize(pool.filepath)
            expected = 64 + 1440 * CANDLE_DTYPE.itemsize
            print(f"  ✅ Archivo: {file_size:,} bytes (esperado: {expected:,})")
            assert file_size == expected, f"Tamaño incorrecto: {file_size} vs {expected}"
            print("  ✅ PASSED")

            # ── Test 2: Inyección Individual O(1) ───────────────────────
            print("\nTEST 2: Inyección Individual O(1)")
            t0 = time.perf_counter_ns()
            pool.inject_candle(1718000000000, 67000.0, 67100.0, 66900.0, 67050.0, 123.45)
            t1 = time.perf_counter_ns()
            inject_ns = t1 - t0
            print(f"  ⚡ Latencia inject_candle: {inject_ns:,} ns")
            assert pool.head == 0, f"Head debería ser 0, es {pool.head}"
            assert pool.count == 1
            # Verificar datos
            view = pool.get_view()
            assert len(view) == 1
            assert view[0]['timestamp'] == 1718000000000
            assert abs(view[0]['close'] - 67050.0) < 0.1
            print("  ✅ PASSED")

            # ── Test 3: Inyección Masiva Vectorizada ────────────────────
            print("\nTEST 3: Inyección Masiva (1000 velas)")
            n = 1000
            ts = np.arange(1718000060000, 1718000060000 + n * 60000, 60000, dtype=np.int64)
            ohlcv = np.random.uniform(66000, 68000, (n, 5)).astype(np.float32)
            
            t0 = time.perf_counter_ns()
            pool.inject_bulk(ts, ohlcv)
            t1 = time.perf_counter_ns()
            bulk_ns = t1 - t0
            print(f"  ⚡ Latencia inject_bulk(1000): {bulk_ns:,} ns ({bulk_ns/n:.0f} ns/vela)")
            assert pool.count == 1001, f"Count: {pool.count}"
            print("  ✅ PASSED")

            # ── Test 4: Lectura Zero-Copy ───────────────────────────────
            print("\nTEST 4: Lectura Zero-Copy (get_view)")
            t0 = time.perf_counter_ns()
            view = pool.get_view()
            t1 = time.perf_counter_ns()
            read_ns = t1 - t0
            print(f"  ⚡ Latencia get_view(): {read_ns:,} ns ({len(view)} velas)")
            assert len(view) == 1001
            
            # Lookback
            t0 = time.perf_counter_ns()
            view100 = pool.get_view(lookback=100)
            t1 = time.perf_counter_ns()
            print(f"  ⚡ Latencia get_view(100): {t1-t0:,} ns ({len(view100)} velas)")
            assert len(view100) == 100
            print("  ✅ PASSED")

            # ── Test 5: DataFrame Export (Backward Compat) ──────────────
            print("\nTEST 5: DataFrame Export (Legacy Compat)")
            t0 = time.perf_counter_ns()
            df = pool.to_dataframe()
            t1 = time.perf_counter_ns()
            df_ns = t1 - t0
            print(f"  ⚡ Latencia to_dataframe(): {df_ns:,} ns ({df_ns/1e6:.2f} ms)")
            assert len(df) == 1001
            assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
            assert df.index.name == 'timestamp'
            print("  ✅ PASSED")

            # ── Test 6: Ring Buffer Wrap-Around ─────────────────────────
            print("\nTEST 6: Ring Buffer Wrap-Around (Evicción)")
            pool2 = QuantumMMAP("ETHUSDT", capacity_days=1)  # max 1440
            n_overflow = 1500  # 60 más que la capacidad
            ts2 = np.arange(1718000000000, 1718000000000 + n_overflow * 60000, 60000, dtype=np.int64)
            ohlcv2 = np.random.uniform(3000, 3500, (n_overflow, 5)).astype(np.float32)
            pool2.inject_bulk(ts2, ohlcv2)
            
            count = pool2.count
            print(f"  Inyectadas: {n_overflow}, Vivas: {count}, Max: {pool2.max_candles}")
            assert count <= pool2.max_candles, f"Count {count} > max {pool2.max_candles}"
            
            view_wrap = pool2.get_view()
            print(f"  View length: {len(view_wrap)}")
            # Los timestamps deben estar ordenados
            ts_view = view_wrap['timestamp']
            assert np.all(ts_view[1:] >= ts_view[:-1]), "Timestamps desordenados después del wrap!"
            print("  ✅ PASSED")

            # ── Test 7: Persistencia (Re-mount) ────────────────────────
            print("\nTEST 7: Persistencia (Re-mount)")
            old_count = pool.count
            old_head = pool.head
            pool.flush()
            
            pool_reloaded = QuantumMMAP("BTCUSDT", capacity_days=1)
            assert pool_reloaded.count == old_count, f"Count mismatch: {pool_reloaded.count} vs {old_count}"
            assert pool_reloaded.head == old_head, f"Head mismatch: {pool_reloaded.head} vs {old_head}"
            print(f"  Count persistido: {pool_reloaded.count}, Head: {pool_reloaded.head}")
            print("  ✅ PASSED")

            # ── Resumen ─────────────────────────────────────────────────
            print("\n" + "=" * 60)
            print("📊 RESUMEN DE LATENCIAS:")
            print(f"  inject_candle:     {inject_ns:>12,} ns")
            print(f"  inject_bulk(1000): {bulk_ns:>12,} ns ({bulk_ns/n:.0f} ns/vela)")
            print(f"  get_view(all):     {read_ns:>12,} ns")
            print(f"  to_dataframe:      {df_ns:>12,} ns ({df_ns/1e6:.2f} ms)")
            print("=" * 60)
            print("✅ TODOS LOS TESTS PASARON")

        finally:
            config.Config.BASE_DIR = original_base

if __name__ == "__main__":
    test_quantum_mmap()
