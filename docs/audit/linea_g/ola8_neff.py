"""LG-ola8 — tamaño efectivo de la muestra entre monedas (cuánta información
independiente aportan N monedas por vela). Tapes TGMTICK1 reales de agosto de
2026; precio del último trade en una rejilla de 60 s y 300 s.

- Equicorrelación: N_eff = N/(1+(N-1)ρ̄), techo 1/ρ̄.
- Cambio COMÚN a todas las monedas (lo que importa para detectar un cambio de
  régimen agregando entre monedas): información de Fisher de un desplazamiento
  común = 1ᵀΣ⁻¹1 (GLS), con Σ la correlación de signos condicionada a que las
  dos monedas se muevan (Cs = SSᵀ/(|S||S|ᵀ), robusta a los retornos cero).

Uso: python -I ola8_neff.py <directorio de tapes>
"""
import numpy as np, glob, os, sys, json
D = sys.argv[1]
files = sorted(glob.glob(os.path.join(D, '*_2026-08_REAL.bin')))
names = {os.path.basename(f).split('_')[0]: f for f in files}
for s, alt in (('BTCUSDT','BTCUSDT_AUG_REAL.bin'), ('ETHUSDT','ETHUSDT_AUG_REAL.bin')):
    p = os.path.join(D, alt)
    if os.path.exists(p): names[s] = p
dt = np.dtype([('ts','<u8'),('bid','<f8'),('ask','<f8'),('q1','<f8'),('q2','<f8')])
series = {}
for s, f in sorted(names.items()):
    raw = np.memmap(f, dtype=np.uint8, mode='r')
    if bytes(raw[:8]) != b'TGMTICK1' or (len(raw)-8) % 40: continue
    rec = np.frombuffer(raw[8:], dtype=dt)
    ts = rec['ts'].astype(np.int64); mid = (rec['bid'] + rec['ask']) * 0.5
    ok = np.isfinite(mid) & (mid > 0)
    series[s] = (ts[ok], mid[ok])
t0 = max(v[0][0] for v in series.values()); t1 = min(v[0][-1] for v in series.values())
out = {'monedas': len(series), 'desde_ms': int(t0), 'hasta_ms': int(t1)}
for paso in (60_000, 300_000):
    grid = np.arange(t0 - t0 % paso + paso, t1, paso, dtype=np.int64)
    R = []
    keys = sorted(series)
    for s in keys:
        ts, mid = series[s]
        idx = np.searchsorted(ts, grid, side='right') - 1
        px = mid[idx]
        R.append(np.diff(np.log(px)))
    R = np.array(R)
    S = np.sign(R)
    A = np.abs(S)
    C = np.corrcoef(R)
    Cs = (S @ S.T) / np.maximum(A @ A.T, 1.0)  # correlación de signo condicionada a que ambos se muevan
    np.fill_diagonal(Cs, 1.0)
    n = len(keys); off = ~np.eye(n, dtype=bool)
    rbar, sbar = C[off].mean(), Cs[off].mean()
    ev = np.linalg.eigvalsh(Cs); pr = ev.sum()**2 / (ev**2).sum()
    uno = np.ones(n); gls = float(uno @ np.linalg.solve(Cs, uno))
    five = [keys.index(k) for k in ('BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT') if k in keys]
    s5 = Cs[np.ix_(five, five)][~np.eye(len(five), dtype=bool)].mean()
    zero = float((R == 0).mean())
    TRECE = ('BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','DOGEUSDT','ADAUSDT',
             'DOTUSDT','LINKUSDT','LTCUSDT','BCHUSDT','AVAXUSDT','TRXUSDT')
    trece = [keys.index(k) for k in TRECE if k in keys]
    assert len(trece) == 13, [k for k in TRECE if k not in keys]
    def neff(N, r): return N / (1 + (N - 1) * r)
    out[f'paso_{paso//1000}s'] = {
        'barras': int(R.shape[1]), 'fraccion_retornos_cero': round(zero, 3),
        'rho_pearson_medio': round(float(rbar), 3), 'rho_signo_medio': round(float(sbar), 3),
        'rho_signo_5_del_evolver': round(float(s5), 3),
        'N_eff_5_evolver': round(neff(len(five), s5), 2),
        'N_eff_13_con_rho_de_las_38': round(neff(13, sbar), 2), f'N_eff_{n}': round(neff(n, sbar), 2),
        'participacion_autovalores_signo': round(float(pr), 2),
        f'N_eff_gls_{n}_cambio_comun': round(gls, 2),
        'N_eff_gls_13_cambio_comun': round(float(np.ones(len(trece)) @ np.linalg.solve(Cs[np.ix_(trece,trece)], np.ones(len(trece)))), 2),
        'N_eff_13_pearson': round(neff(13, float(C[np.ix_(trece,trece)][~np.eye(len(trece),dtype=bool)].mean())), 2),
        'N_eff_gls_5_evolver': round(float(np.ones(len(five)) @ np.linalg.solve(Cs[np.ix_(five,five)], np.ones(len(five)))), 2),
        'cota_1_sobre_rho': round(1 / float(sbar), 2),
    }
print(json.dumps(out, indent=1, ensure_ascii=False))
