"""Medidor del minimo de klines M y de la tolerancia (MEDIDASV4-P4), offline y determinista.

Consumidor de las klines del warmup: StatefulEngine::try_process_kline
(crates/god-engine-core/src/stateful_engine.rs:1417-1493), llamado por el host en src/bin/god_engine.rs
(bucle de inyeccion del warmup, `process_kline` por cada barra de historical_klines). Este modulo:
- inventaria los rasgos con su lookback y fuente archivo:linea (`inventario`), y de ahi L_max;
- replica en Python los rasgos de lookback finito (DFA de Hurst, espectro legacy) y el rasgo
  geometrico vt_rasgo_ml. Es una REPLICA: la paridad con el consumidor Rust no se acredita aqui
  (`paridad_consumidor = N/D`) y sin ella ni tol ni M quedan acreditados;
- mide el error relativo de cada rasgo con M barras frente a la referencia M = 5000 sobre la misma
  barra final (`errores_rejilla`), con la convencion de denominador nulo declarada (`error_relativo`);
- informa la primera transicion observada del consumidor en una rejilla de perturbaciones
  r' = r*(1 + delta) (`sensibilidad`), que NUNCA es una tol segura por si sola;
- elige M_candidato / M_acreditado (`m_minimo`) solo con tol acreditada, M >= L_max, >= 59 arranques
  sin excesos y una cohorte de validacion aparte; si falta algo, N/D.

El piso 0,01 del clamp de lift_eff_long/short (crates/god-engine-core/src/lib.rs, `.clamp(0.01, 0.30)`)
se informa como piso de la compuerta, no como tolerancia.
"""

import argparse
import hashlib
import io
import json
import math
import os
import sys
import zipfile

M_REFERENCIA = 5000
N_ARRANQUES = 59
PISO_COMPUERTA = 0.01

DFA_ESCALAS = (4, 8, 16, 32, 64, 128, 256)  # hurst_dfa.rs:59
DFA_MAX_HISTORIA = 1024  # hurst_dfa.rs:65
DFA_CADA = 32  # hurst_dfa.rs:68
DFA_MIN = 512  # hurst_dfa.rs:72
DFA_R2_MIN = 0.85  # math_kernels.rs:471 (hurst_or_neutral(0.85))
FFT = 64  # spectral.rs:6


def inventario():
    rasgos = [
        {"rasgo": "hurst_dfa", "fuente": "crates/feature-engine/src/hurst_dfa.rs:65",
         "lookback_cierres": DFA_MAX_HISTORIA + 1, "memoria": "finita",
         "nota": "1024 retornos; minimo 512 (:72); recalculo cada 32 retornos (:68): el estado depende de la fase",
         "replicado": True},
        {"rasgo": "spectral_legacy", "fuente": "crates/feature-engine/src/spectral.rs:6",
         "lookback_cierres": FFT + 1, "memoria": "finita",
         "nota": "64 retornos simples empujados en stateful_engine.rs:1461-1464", "replicado": True},
        {"rasgo": "multifractal_macro", "fuente": "crates/feature-engine/src/multifractal.rs:497",
         "lookback_cierres": 51, "memoria": "finita", "nota": "ventana 50 (micro 10, meso 25)",
         "replicado": False},
        {"rasgo": "sigma60", "fuente": "crates/god-engine-core/src/sigma_minuto.rs:44",
         "lookback_cierres": 61, "memoria": "finita", "nota": "SEGUNDOS = 60", "replicado": False},
        {"rasgo": "vt_rasgo_ml", "fuente": "crates/god-engine-core/src/stateful_engine.rs:1487",
         "lookback_cierres": None, "memoria": "infinita",
         "nota": "media exponencial 0,8/0,2 del rango verdadero: la influencia decae como 0,8^n", "replicado": True},
        {"rasgo": "omni_bb_stats", "fuente": "crates/feature-engine/src/welford.rs:34",
         "lookback_cierres": None, "memoria": "infinita",
         "nota": "Welford acumulado hasta 2000 muestras y luego con olvido (omni_strategies.rs:84)",
         "replicado": False},
        {"rasgo": "omni_local_max_min", "fuente": "crates/feature-engine/src/omni_strategies.rs:97",
         "lookback_cierres": None, "memoria": "infinita", "nota": "extremos acumulados de toda la historia",
         "replicado": False},
    ]
    finitos = [r["lookback_cierres"] for r in rasgos if r["lookback_cierres"] is not None]
    return {
        "consumidor": "crates/god-engine-core/src/stateful_engine.rs:1417 (try_process_kline)",
        "rasgos": rasgos,
        "L_max_finito": max(finitos),
        "memoria_infinita": [r["rasgo"] for r in rasgos if r["memoria"] == "infinita"],
        "no_replicados": [r["rasgo"] for r in rasgos if not r["replicado"]],
        "limit_warmup_vivo": 1000,
        "nota": "con limit = 1000 (bootloader.rs:255) el DFA nunca llena sus 1024 retornos (999 retornos)",
    }


# ------------------------------------------------------------------------------ replicas de rasgos

def _retorno_log(p, prev):
    ratio = p / prev
    if 0.5 <= ratio <= 2.0:
        return math.log1p((p - prev) / prev)
    if math.isfinite(ratio) and ratio > 0.0:
        return math.log(ratio)
    return math.log(p) - math.log(prev)


def ultimo_recalculo(n_retornos):
    """Retornos acumulados en el ultimo recalculo del DFA: 512, 544, ... (None si aun no hubo)."""
    if n_retornos < DFA_MIN:
        return None
    return DFA_MIN + DFA_CADA * ((n_retornos - DFA_MIN) // DFA_CADA)


def _dfa(r):
    n = len(r)
    amp = max(abs(x) for x in r) if r else 0.0
    if n < DFA_MIN or amp == 0.0 or not all(math.isfinite(x) for x in r):
        return None, 0.0
    mean = sum(x / amp for x in r) / n
    walk, acc = [], 0.0
    for x in r:
        acc += x / amp - mean
        walk.append(acc)
    ln_s, ln_f = [], []
    for s in DFA_ESCALAS:
        if n // s < 4:
            continue
        ventanas = n // s
        m = float(s)
        sum_x = (s * (s - 1)) / 2.0
        sum_xx = ((s - 1) * s * (2 * s - 1)) / 6.0
        denom = m * sum_xx - sum_x * sum_x
        suma = 0.0
        for w in range(ventanas):
            seg = walk[w * s:(w + 1) * s]
            sy = sum(seg)
            sxy = sum(i * y for i, y in enumerate(seg))
            if abs(denom) > 1e-12:
                b = (m * sxy - sum_x * sy) / denom
                a = (sy - b * sum_x) / m
            else:
                a, b = sy / m, 0.0
            suma += sum((y - (a + b * i)) ** 2 for i, y in enumerate(seg)) / m
        f = math.sqrt(suma / ventanas)
        if f > 0.0 and math.isfinite(f):
            ln_s.append(math.log(s))
            ln_f.append(math.log(f))
    if len(ln_s) < 3:
        return None, 0.0
    k = len(ln_s)
    mx, my = sum(ln_s) / k, sum(ln_f) / k
    sxy = sum((ln_s[i] - mx) * (ln_f[i] - my) for i in range(k))
    sxx = sum((ln_s[i] - mx) ** 2 for i in range(k))
    syy = sum((ln_f[i] - my) ** 2 for i in range(k))
    if sxx <= 0.0 or syy <= 0.0:
        return None, 0.0
    slope = sxy / sxx
    r2 = min(1.0, max(0.0, (sxy / math.sqrt(sxx) / math.sqrt(syy)) ** 2))
    return slope, r2


def hurst_final(cierres):
    """Estado final del DFA tras alimentar los cierres en orden (misma cadencia que el motor)."""
    r = [_retorno_log(c, p) for p, c in zip(cierres, cierres[1:])]
    j = ultimo_recalculo(len(r))
    if j is None:
        return 0.5, None
    ventana = r[max(0, j - DFA_MAX_HISTORIA):j]
    raw, r2 = _dfa(ventana)
    valido = raw is not None and 0.0 < raw < 1.0
    neutral = min(0.95, max(0.05, raw)) if valido and r2 >= DFA_R2_MIN else 0.5
    return neutral, raw


def espectro_final(cierres):
    r = [(c - p) / p for p, c in zip(cierres, cierres[1:])]
    if len(r) < FFT:
        return 0, 0.0, 0.0
    buf = r[-FFT:]
    taper = FFT // 4
    x = [buf[i] * (0.5 * (1.0 - math.cos(math.pi * i / taper)) if i < taper else 1.0) for i in range(FFT)]
    maxp, dom, total, pond = 0.0, 0, 0.0, 0.0
    for k in range(1, FFT // 2 + 1):
        re = sum(x[n] * math.cos(2 * math.pi * k * n / FFT) for n in range(FFT))
        im = -sum(x[n] * math.sin(2 * math.pi * k * n / FFT) for n in range(FFT))
        p = (re * re + im * im) / FFT
        if p > maxp:
            maxp, dom = p, k
        total += p
        pond += k * p
    cen = pond / total if total > 1e-12 else 0.0
    return dom, maxp, cen


def vt_final(barras):
    vt = 0.0
    for o, h, lo, c, v in barras:
        tr = max(h - lo, c * 0.0005)
        vt = tr if (vt == 0.0 or not math.isfinite(vt)) else max(vt * 0.8 + tr * 0.2, c * 0.0005)
    return vt


def rasgos(barras):
    cierres = [b[3] for b in barras]
    h, raw = hurst_final(cierres)
    dom, maxp, cen = espectro_final(cierres)
    return {"hurst_dfa": h, "hurst_dfa_raw": raw, "spectral_dominant_bin": dom, "spectral_max_power": maxp,
            "spectral_centroid": cen, "vt_rasgo_ml": vt_final(barras)}


DISCRETOS = {"spectral_dominant_bin"}


def error_relativo(x, ref, discreto=False):
    """|x - ref| / |ref|. Discreto: 0 si igual, 1 si distinto. ref = 0: 0 si x = 0, si no None.
    Ambos sin valor (None): 0 (mismo estado 'sin ajuste'); uno solo sin valor: None (no evaluable)."""
    if x is None and ref is None:
        return 0.0
    if x is None or ref is None:
        return None
    if discreto:
        return 0.0 if x == ref else 1.0
    if ref == 0:
        return 0.0 if x == 0 else None
    return abs(x - ref) / abs(ref)


def errores_rejilla(barras, finales, rejilla, m_ref=M_REFERENCIA):
    por_final = []
    for t in finales:
        if t < m_ref or t > len(barras):
            raise ValueError(f"final {t} fuera de [{m_ref}, {len(barras)}]")
        ref = rasgos(barras[t - m_ref:t])
        errs = {}
        for m in rejilla:
            val = rasgos(barras[t - m:t])
            errs[m] = {k: error_relativo(val[k], ref[k], k in DISCRETOS) for k in ref}
        por_final.append({"final": t, "referencia": ref, "errores": errs})
    return {"m_ref": m_ref, "rejilla": list(rejilla), "por_final": por_final, "paridad_consumidor": "N/D",
            "motivo_paridad": "replica Python de los rasgos; la paridad con el consumidor Rust no se acredito"}


def peor_error(errores_m, incluir=None):
    vals = [v for k, v in errores_m.items() if incluir is None or k in incluir]
    if any(v is None for v in vals):
        return None
    return max(vals) if vals else None


# ----------------------------------------------------------------------------- sensibilidad y M

def sensibilidad(consumidor, base, deltas):
    """Primera transicion de la decision del consumidor entre los puntos ensayados r' = r*(1 + delta)."""
    cobertura = sorted(deltas)
    d0 = consumidor(base)
    trans = [d for d in sorted(cobertura, key=lambda d: (abs(d), d)) if consumidor(base * (1.0 + d)) != d0]
    sin = []
    if not any(d < 0 for d in trans):
        sin.append("negativa")
    if not any(d > 0 for d in trans):
        sin.append("positiva")
    return {
        "delta_primera_transicion_observada": trans[0] if trans else None,
        "transiciones": trans,
        "cobertura": cobertura,
        "direcciones_sin_transicion": sin,
        "frontera_entre_puntos_posible": True,
        "tol_segura": "N/D",
        "motivo": "una rejilla finita no acredita una cota para errores menores, otras direcciones o estados",
        "piso_compuerta": PISO_COMPUERTA,
    }


def tol_desde_sensibilidad(r):
    return {"valor": None, "acreditada": False, "fuente": "sensibilidad: " + r["motivo"],
            "delta_primera_transicion_observada": r["delta_primera_transicion_observada"]}


def _sin_excesos(errs, tol, n):
    return len(errs) >= n and all(e is not None for e in errs) and all(e <= tol for e in errs)


def m_minimo(errores, tol, l_max, paridad, validacion=None, n_requerido=N_ARRANQUES):
    """errores: {M: [peor error por arranque]}. M_candidato exige tol con valor y acreditada."""
    ms = sorted(errores)
    r = {"L_max": l_max, "condicion_M_ge_L_max": True, "n_requerido": n_requerido, "tol": tol,
         "descartados_por_L_max": [m for m in ms if m < l_max], "excesos": {}, "M_candidato": None,
         "M_acreditado": "N/D", "paridad": paridad,
         "alcance": "un M y simbolo predeclarados; la seleccion del minimo exige una cohorte de validacion aparte"}
    valor = tol.get("valor") if tol else None
    if valor is None or not tol.get("acreditada"):
        r["motivo"] = "tol no acreditada: sin M candidato ni acreditado"
        return r
    for m in ms:
        r["excesos"][m] = sum(1 for e in errores[m] if e is None or e > valor)
    for m in ms:
        if m >= l_max and _sin_excesos(errores[m], valor, n_requerido):
            r["M_candidato"] = m
            break
    if r["M_candidato"] is None:
        r["motivo"] = "ningun M >= L_max con >= n arranques evaluables y 0 excesos"
        return r
    if not paridad:
        r["motivo"] = "sin paridad con el consumidor: M acreditado N/D"
        return r
    val = (validacion or {}).get(r["M_candidato"])
    if val is None or not _sin_excesos(val, valor, n_requerido):
        r["motivo"] = "la cohorte de validacion no confirma el candidato: M acreditado N/D"
        return r
    r["M_acreditado"] = r["M_candidato"]
    r["motivo"] = "tol acreditada, paridad, M >= L_max y validacion sin excesos"
    return r


# ------------------------------------------------------------------------- Vision (recibo offline)

def _leer_zip(ruta):
    import sonda  # paridad de barra con parse_warmup_kline

    with open(ruta, "rb") as fh:
        crudo = fh.read()
    filas = []
    with zipfile.ZipFile(io.BytesIO(crudo)) as z:
        for nombre in z.namelist():
            for linea in io.TextIOWrapper(z.open(nombre), encoding="utf-8"):
                c = linea.strip().split(",")
                if not c or not c[0].isdigit():
                    continue
                b = sonda.barra_valida([int(c[0])] + c[1:6])
                filas.append((int(c[0]), b))
    return hashlib.sha256(crudo).hexdigest(), filas


def cargar_simbolo(raiz, sym):
    d = os.path.join(raiz, sym, "1m")
    zips = sorted(f for f in os.listdir(d) if f.endswith(".zip"))
    hashes, filas = {}, []
    for z in zips:
        h, f = _leer_zip(os.path.join(d, z))
        hashes[z] = h
        filas.extend(f)
    filas.sort(key=lambda x: x[0])
    tiempos = [t for t, _ in filas]
    barras = [b for _, b in filas]
    return {"simbolo": sym, "zips": hashes, "tiempos": tiempos, "barras": barras,
            "rechazadas": sum(1 for b in barras if b is None)}


def finales_sin_hueco(tiempos, barras, m_ref):
    """Indices finales t cuya ventana [t - m_ref, t) es continua (60 000 ms) y sin barras rechazadas."""
    ok, malos = [], 0
    malo = [0] * (len(tiempos) + 1)
    for i in range(len(tiempos)):
        roto = barras[i] is None or (i > 0 and tiempos[i] - tiempos[i - 1] != 60000)
        malo[i + 1] = malo[i] + (1 if roto else 0)
    for t in range(m_ref, len(tiempos) + 1):
        # la primera barra de la ventana puede seguir a un hueco: se mira desde la segunda
        if malo[t] - malo[t - m_ref + 1] == 0 and barras[t - m_ref] is not None:
            ok.append(t)
        else:
            malos += 1
    return ok


def elegir(finales, n, semilla):
    """Seleccion determinista con LCG registrado (sin reemplazo)."""
    x, pool, out = semilla, list(finales), []
    while pool and len(out) < n:
        x = (6364136223846793005 * x + 1442695040888963407) % (1 << 64)
        out.append(pool.pop((x >> 33) % len(pool)))
    return sorted(out)


def medir_vision(raiz, simbolos, rejilla, n=N_ARRANQUES, semilla=20261117, m_ref=M_REFERENCIA):
    inv = inventario()
    out = {"inventario": inv, "rejilla": rejilla, "m_ref": m_ref, "n_arranques": n, "semilla": semilla,
           "lcg": "x = (6364136223846793005*x + 1442695040888963407) mod 2^64; indice = (x >> 33) mod len",
           "piso_compuerta": PISO_COMPUERTA, "tol_segura": "N/D", "paridad_consumidor": "N/D",
           "M_acreditado": "N/D", "simbolos": {}}
    for sym in simbolos:
        if not os.path.isdir(os.path.join(raiz, sym, "1m")):
            out["simbolos"][sym] = {"estado": "N/D", "motivo": "sin cinta Vision local"}
            continue
        c = cargar_simbolo(raiz, sym)
        fin = finales_sin_hueco(c["tiempos"], c["barras"], m_ref)
        sel = elegir(fin, n, semilla)
        e = errores_rejilla(c["barras"], sel, rejilla, m_ref)
        resumen = {}
        for m in rejilla:
            por_rasgo = {}
            for k in e["por_final"][0]["errores"][m] if e["por_final"] else []:
                vals = [f["errores"][m][k] for f in e["por_final"]]
                ev = [v for v in vals if v is not None]
                por_rasgo[k] = {"max": max(ev) if ev else None,
                                "mediana": sorted(ev)[len(ev) // 2] if ev else None,
                                "no_evaluables": len(vals) - len(ev)}
            resumen[m] = por_rasgo
        out["simbolos"][sym] = {
            "estado": "completo" if len(sel) >= n else "insuficiente",
            "barras": len(c["barras"]), "rechazadas": c["rechazadas"], "finales_continuos": len(fin),
            "arranques": len(sel), "solapados": True,
            "primer_ms": c["tiempos"][0] if c["tiempos"] else None,
            "ultimo_ms": c["tiempos"][-1] if c["tiempos"] else None,
            "zips": c["zips"], "errores_por_M": resumen,
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Medidor de M (offline, Vision local)")
    ap.add_argument("--vision", required=True, help=r"...\vision\data\futures\um\daily\klines")
    ap.add_argument("--simbolos", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--rejilla", default="65,500,513,1000,1025,1500,2000,3000,5000")
    ap.add_argument("--arranques", type=int, default=N_ARRANQUES)
    ap.add_argument("--semilla", type=int, default=20261117)
    a = ap.parse_args(argv)
    rejilla = [int(x) for x in a.rejilla.split(",")]
    sims = [s.strip().upper() for s in a.simbolos.split(",") if s.strip()]
    r = medir_vision(a.vision, sims, rejilla, a.arranques, a.semilla)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "medida_m.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in r.items() if k != "simbolos"}, ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
