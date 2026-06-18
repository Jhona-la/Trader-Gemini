"""
AUTOPSIA DE LATENCIA - BISTURIMETRO NANOSEGUNDOS
Mide en nanosegundos cada frontera del pipeline de ingesta WebSocket.
F1: RED PURA (ws.recv) | F2: PARSING (orjson) | F3: DEDUP+ROUTE | F4: HANDLER
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import time
import statistics
import websockets
import orjson
import collections

TOTAL_TICKS = 100
BINANCE_WSS = "wss://fstream.binance.com/stream?streams="

class NanoAccumulator:
    __slots__ = ['name', 'samples', 'count']
    def __init__(self, name):
        self.name = name
        self.samples = []
        self.count = 0
    def record(self, ns):
        self.samples.append(ns)
        self.count += 1
    def report(self):
        if not self.samples:
            return {'name': self.name, 'count': 0}
        s = sorted(self.samples)
        n = len(s)
        return {
            'name': self.name, 'count': n,
            'mean_ns': statistics.mean(s), 'median_ns': statistics.median(s),
            'p95_ns': s[int(n*0.95)], 'p99_ns': s[int(n*0.99)],
            'min_ns': s[0], 'max_ns': s[-1],
            'stddev_ns': statistics.stdev(s) if n > 1 else 0,
        }

def format_ns(ns):
    if ns >= 1_000_000_000: return f"{ns/1_000_000_000:.3f}s"
    elif ns >= 1_000_000: return f"{ns/1_000_000:.3f}ms"
    elif ns >= 1_000: return f"{ns/1_000:.3f}us"
    else: return f"{ns:.0f}ns"

async def autopsy_stream(stream_type, symbol="btcusdt"):
    if stream_type == 'depth':
        stream_name = f"{symbol}@depth10@100ms"
    elif stream_type == 'aggTrade':
        stream_name = f"{symbol}@aggTrade"
    elif stream_type == 'kline':
        stream_name = f"{symbol}@kline_1m"
    else:
        raise ValueError(f"Unknown: {stream_type}")

    url = BINANCE_WSS + stream_name
    f1 = NanoAccumulator(f"F1_RED_{stream_type}")
    f2 = NanoAccumulator(f"F2_PARSE_{stream_type}")
    f3 = NanoAccumulator(f"F3_DEDUP_{stream_type}")
    f4 = NanoAccumulator(f"F4_HANDLER_{stream_type}")
    ft = NanoAccumulator(f"TOTAL_{stream_type}")
    drift_samples = []
    seen = collections.OrderedDict()
    count = 0

    print(f"\n[{stream_type.upper()}] Conectando a {url}...")
    try:
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            print(f"[{stream_type.upper()}] Conectado. Capturando {TOTAL_TICKS} ticks...")
            while count < TOTAL_TICKS:
                # F1: RED PURA
                t0 = time.perf_counter_ns()
                raw_msg = await ws.recv()
                t1 = time.perf_counter_ns()
                f1.record(t1 - t0)

                # F2: PARSING ORJSON
                t2 = time.perf_counter_ns()
                if isinstance(raw_msg, str):
                    raw_bytes = raw_msg.encode('utf-8')
                else:
                    raw_bytes = raw_msg
                msg = orjson.loads(raw_bytes)
                t3 = time.perf_counter_ns()
                f2.record(t3 - t2)

                # F3: DEDUP + ROUTING
                t4 = time.perf_counter_ns()
                data = msg.get('data', msg)
                uid = None
                if stream_type == 'depth':
                    uid = data.get('lastUpdateId')
                elif stream_type == 'aggTrade':
                    uid = data.get('a')
                elif stream_type == 'kline':
                    uid = data.get('k', {}).get('t')
                is_dup = False
                if uid is not None:
                    dk = f"{stream_type}_{uid}"
                    if dk in seen:
                        is_dup = True
                    else:
                        seen[dk] = True
                        if len(seen) > 5000:
                            seen.popitem(last=False)
                t5 = time.perf_counter_ns()
                f3.record(t5 - t4)
                if is_dup:
                    continue

                # F4: HANDLER SIMULADO
                t6 = time.perf_counter_ns()
                if stream_type == 'depth':
                    bids = data.get('bids', [])
                    asks = data.get('asks', [])
                    tb = sum(float(b[0])*float(b[1]) for b in bids[:10])
                    ta = sum(float(a[0])*float(a[1]) for a in asks[:10])
                    obi = (tb - ta) / (tb + ta + 1e-8)
                    if bids and asks:
                        bp, bv = float(bids[0][0]), float(bids[0][1])
                        ap, av = float(asks[0][0]), float(asks[0][1])
                        micro = (bp*av + ap*bv) / (bv + av + 1e-12)
                elif stream_type == 'aggTrade':
                    price = float(data.get('p', 0))
                    qty = float(data.get('q', 0))
                    is_buyer = not data.get('m', False)
                    usd_vol = price * qty
                elif stream_type == 'kline':
                    k = data.get('k', {})
                    o, h, l, c, v = float(k.get('o',0)), float(k.get('h',0)), float(k.get('l',0)), float(k.get('c',0)), float(k.get('v',0))
                t7 = time.perf_counter_ns()
                f4.record(t7 - t6)
                ft.record(t7 - t0)

                # F5: DRIFT
                evt_ms = data.get('E', 0)
                if evt_ms > 0:
                    local_ms = int(time.time() * 1000)
                    drift = local_ms - evt_ms
                    if -5000 < drift < 5000:
                        drift_samples.append(drift)
                count += 1
                if count % 50 == 0:
                    print(f"  [{stream_type}] {count}/{TOTAL_TICKS}")
    except Exception as e:
        print(f"[{stream_type}] Error: {e}")

    return {'f1': f1.report(), 'f2': f2.report(), 'f3': f3.report(), 'f4': f4.report(), 'total': ft.report(), 'drift_samples': drift_samples}

async def measure_binance_drift(n=10):
    import aiohttp
    url = "https://fapi.binance.com/fapi/v1/time"
    drifts, rtts = [], []
    print(f"\nMidiendo Drift contra Binance Server Time ({n} muestras)...")
    async with aiohttp.ClientSession() as session:
        for i in range(n):
            ts = time.time() * 1000
            async with session.get(url) as resp:
                tr = time.time() * 1000
                data = await resp.json()
                st = data['serverTime']
                rtt = tr - ts
                drift = (ts + rtt/2) - st
                drifts.append(drift)
                rtts.append(rtt)
                if i < 3:
                    print(f"  Muestra {i+1}: Drift={drift:.1f}ms, RTT={rtt:.1f}ms")
            await asyncio.sleep(0.2)
    return {
        'drift_median_ms': statistics.median(drifts), 'drift_mean_ms': statistics.mean(drifts),
        'drift_min_ms': min(drifts), 'drift_max_ms': max(drifts),
        'rtt_median_ms': statistics.median(rtts), 'rtt_mean_ms': statistics.mean(rtts),
        'rtt_min_ms': min(rtts), 'rtt_max_ms': max(rtts),
    }

def benchmark_retina_bridge():
    print("\nBenchmarking Retina Bridge (Cython)...")
    try:
        from core.metal.ingester_bridge import retina_bridge
        if not retina_bridge.available:
            print("  Retina Bridge NO disponible (Cython no compilado)")
            return {'available': False}
        payload = b'{"stream":"btcusdt@depth10@100ms","data":{"lastUpdateId":123456789,"bids":[["50000.00","1.234"]],"asks":[["50001.00","3.456"]],"E":1234567890123}}'
        times = []
        for _ in range(1000):
            t0 = time.perf_counter_ns()
            retina_bridge.ingest(0, payload, 0)
            t1 = time.perf_counter_ns()
            times.append(t1 - t0)
        s = sorted(times)
        n = len(s)
        print(f"  Retina Median: {format_ns(statistics.median(s))}")
        print(f"  Retina P99:    {format_ns(s[int(n*0.99)])}")
        return {'available': True, 'median_ns': statistics.median(s), 'p99_ns': s[int(n*0.99)]}
    except Exception as e:
        print(f"  Error: {e}")
        return {'available': False, 'error': str(e)}

def benchmark_parsers():
    import json as stdlib_json
    print("\nBenchmarking JSON Parsers...")
    payload = b'{"stream":"btcusdt@depth10@100ms","data":{"lastUpdateId":123456789,"bids":[["50000.00","1.234"],["49999.00","2.345"],["49998.00","3.456"],["49997.00","4.567"],["49996.00","5.678"],["49995.00","6.789"],["49994.00","7.890"],["49993.00","8.901"],["49992.00","9.012"],["49991.00","0.123"]],"asks":[["50001.00","1.234"],["50002.00","2.345"],["50003.00","3.456"],["50004.00","4.567"],["50005.00","5.678"],["50006.00","6.789"],["50007.00","7.890"],["50008.00","8.901"],["50009.00","9.012"],["50010.00","0.123"]],"E":1234567890123}}'
    for _ in range(100):
        orjson.loads(payload); stdlib_json.loads(payload)
    ot, jt = [], []
    for _ in range(10000):
        t0 = time.perf_counter_ns(); orjson.loads(payload); ot.append(time.perf_counter_ns() - t0)
    for _ in range(10000):
        t0 = time.perf_counter_ns(); stdlib_json.loads(payload); jt.append(time.perf_counter_ns() - t0)
    om, jm = statistics.median(ot), statistics.median(jt)
    print(f"  orjson: {format_ns(om)}  |  json: {format_ns(jm)}  |  Speedup: {jm/om:.1f}x")
    return {'orjson_median_ns': om, 'json_median_ns': jm, 'speedup': jm/om}

async def main():
    print("=" * 70)
    print("AUTOPSIA DE LATENCIA - BISTURIMETRO NANOSEGUNDOS")
    print("=" * 70)
    print(f"Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    benchmark_parsers()
    benchmark_retina_bridge()

    streams = ['depth', 'aggTrade', 'kline']
    results = {}
    for s in streams:
        results[s] = await autopsy_stream(s, "btcusdt")

    try:
        drift = await measure_binance_drift(10)
    except Exception as e:
        print(f"Drift error: {e}")
        drift = None

    # REPORTE
    print("\n" + "=" * 70)
    print("REPORTE FINAL DE AUTOPSIA")
    print("=" * 70)
    labels = {'f1': 'F1:RED_PURA', 'f2': 'F2:ORJSON', 'f3': 'F3:DEDUP', 'f4': 'F4:HANDLER', 'total': 'TOTAL'}
    for st, res in results.items():
        print(f"\n--- {st.upper()} ---")
        tot = res['total']
        for key in ['f1','f2','f3','f4','total']:
            r = res[key]
            if r['count'] == 0: continue
            pct = (r['mean_ns']/tot['mean_ns']*100) if key != 'total' and tot['count'] > 0 else 100
            print(f"  {labels[key]:>12s}: Mean={format_ns(r['mean_ns']):>10s} Median={format_ns(r['median_ns']):>10s} P99={format_ns(r['p99_ns']):>10s} ({pct:.1f}%)")
        ds = res.get('drift_samples', [])
        if ds:
            print(f"  {'DRIFT_WS':>12s}: Median={statistics.median(ds):.1f}ms Mean={statistics.mean(ds):.1f}ms Range=[{min(ds):.0f},{max(ds):.0f}]ms")

    if drift:
        print(f"\n--- DRIFT vs BINANCE SERVER ---")
        print(f"  Drift Median: {drift['drift_median_ms']:.1f}ms | RTT Median: {drift['rtt_median_ms']:.1f}ms")

    print("\n" + "=" * 70)
    print("VEREDICTO")
    print("=" * 70)
    for st, res in results.items():
        tot = res['total']
        if tot['count'] == 0: continue
        frontiers = {k: res[k]['mean_ns'] for k in ['f1','f2','f3','f4'] if res[k]['count'] > 0}
        if frontiers:
            bn = max(frontiers, key=frontiers.get)
            pct = frontiers[bn]/tot['mean_ns']*100
            print(f"  [{st.upper()}] Cuello: {labels[bn]} ({pct:.1f}%) Total/tick: {format_ns(tot['mean_ns'])}")

    print("\nAUTOPSIA COMPLETADA.")

if __name__ == "__main__":
    asyncio.run(main())
