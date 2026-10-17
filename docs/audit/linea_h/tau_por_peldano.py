"""τ y tamaño por serie medida del T-3, leídos de las trazas de apertura del
núcleo (`[OPEN TRACE]`: sólo las 100 primeras aperturas de cada núcleo, tope
literal de `god-engine-core/src/lib.rs` — pedido a Qoder/AGY de hacerlo
configurable; el contador `#n` vuelve a 1 con cada núcleo nuevo).

Ola 4 de la Línea H: la verificación barata de LH-06 (`docs/DISENO_LH06_TAU.md`
§5, puntos 1 y 3) y la huella del dimensionado LH-04 (nocional, margen y
apalancamiento por apertura). Una serie medida = las trazas anteriores a su
línea `[T3] …` (la escalera, el valle y la adaptación imprimen una por serie;
en la adaptación las compuestas E2–E4' llevan la etiqueta de su primer tramo
y n = 2·velas, por eso la columna `orden` las distingue).

Uso: python -I tau_por_peldano.py [--tramo-velas N] <log> [...]
Con `--tramo-velas N` (velas de 60 s), las aperturas con ts ≥ N·60 000 ms se
cuentan aparte como segundo tramo (`trazas_t2`, `tau_mediana_t2_s`): sirve para
las series compuestas de `t3_adaptacion_en_el_tiempo`; en una serie simple el
segundo tramo queda vacío (NaN).

Salida: una fila TSV por serie con la mediana y los cuartiles de τ (s), el nodo
de la malla 4^k µs más frecuente, la fracción de aperturas por ranura, los
pares de aperturas consecutivas en ranuras distintas y la fracción de esos
pares con |Δ ln τ| ≥ 0,80 (umbral del co-despacho D-431), el nocional, el
margen y el apalancamiento medianos, y el reparto por rama.
"""
import io
import math
import re
import sys

TRAZA = re.compile(
    r"OPEN TRACE\] #(\d+) dir=(LONG|SHORT) slot=(\d+) tau=(\d+)ms rama=(\d+) .*?"
    r"lev=([0-9.]+)x margin=\$([0-9.]+) notional=\$([0-9.]+) .* ts=(\d+)"
)
PELDANO = re.compile(
    r"\[T3\] a=([0-9.]+) d=([0-9.]+)% σ=([0-9.]+)% n=(\d+)(?: C0=(\d+) USD)?(?: gen=(v\d))?"
)
CABECERA = (
    "orden\ta\td_pct\tn\tC0\tgen\ttrazas\ttau_mediana_s\ttau_q25_s\ttau_q75_s"
    "\tnodo_moda_s\tfrac_slot0\tfrac_slot1\tfrac_slot2\tpares_ranuras_distintas"
    "\tfrac_dlntau_ge_0_80\tnocional_med_usd\tnocional_max_usd\tmargen_med_usd"
    "\tlev_med\tramas\ttrazas_t2\ttau_mediana_t2_s\tnocional_med_t2_usd"
)
BAR_MS = 60_000


def cuantil(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    k = (len(s) - 1) * q
    i = int(math.floor(k))
    j = min(i + 1, len(s) - 1)
    return s[i] + (s[j] - s[i]) * (k - i)


def nodo_moda_s(taus_ms):
    nodos = {}
    for t in taus_ms:
        k = round(math.log(t * 1000.0) / math.log(4.0))
        nodos[k] = nodos.get(k, 0) + 1
    k_moda = max(nodos, key=nodos.get)
    return (4.0 ** k_moda) / 1e6


def resumir(opens, corte_ms):
    n = len(opens)
    if n == 0:
        return "0\t" + "\t".join(["nan"] * 13) + "\t{}\t0\tnan\tnan"
    taus = [o["tau"] for o in opens]
    slots = [0, 0, 0]
    for o in opens:
        if 0 <= o["slot"] < 3:
            slots[o["slot"]] += 1
    pares = 0
    separados = 0
    for prev, cur in zip(opens, opens[1:]):
        if prev["slot"] != cur["slot"]:
            pares += 1
            if abs(math.log(cur["tau"]) - math.log(prev["tau"])) >= 0.80:
                separados += 1
    ramas = {}
    for o in opens:
        ramas[o["rama"]] = ramas.get(o["rama"], 0) + 1
    t2 = [o for o in opens if corte_ms is not None and o["ts"] >= corte_ms]
    return "\t".join(
        [
            str(n),
            f"{cuantil(taus, 0.5) / 1000:.0f}",
            f"{cuantil(taus, 0.25) / 1000:.0f}",
            f"{cuantil(taus, 0.75) / 1000:.0f}",
            f"{nodo_moda_s(taus):.0f}",
            f"{slots[0] / n:.2f}",
            f"{slots[1] / n:.2f}",
            f"{slots[2] / n:.2f}",
            str(pares),
            f"{(separados / pares) if pares else float('nan'):.2f}",
            f"{cuantil([o['notional'] for o in opens], 0.5):.2f}",
            f"{max(o['notional'] for o in opens):.2f}",
            f"{cuantil([o['margin'] for o in opens], 0.5):.2f}",
            f"{cuantil([o['lev'] for o in opens], 0.5):.1f}",
            str(dict(sorted(ramas.items()))),
            str(len(t2)),
            f"{cuantil([o['tau'] for o in t2], 0.5) / 1000:.0f}" if t2 else "nan",
            f"{cuantil([o['notional'] for o in t2], 0.5):.2f}" if t2 else "nan",
        ]
    )


def analizar(path, tramo_velas):
    corte_ms = tramo_velas * BAR_MS if tramo_velas else None
    opens = []
    filas = []
    orden = 0
    for line in io.open(path, encoding="utf-8", errors="replace"):
        m = TRAZA.search(line)
        if m:
            opens.append(
                {
                    "num": int(m.group(1)),
                    "dir": m.group(2),
                    "slot": int(m.group(3)),
                    "tau": int(m.group(4)),
                    "rama": int(m.group(5)),
                    "lev": float(m.group(6)),
                    "margin": float(m.group(7)),
                    "notional": float(m.group(8)),
                    "ts": int(m.group(9)),
                }
            )
            continue
        m = PELDANO.match(line)
        if m:
            orden += 1
            cap = m.group(5) or "13"
            gen = m.group(6) or "v1"
            filas.append(
                f"{orden}\t{m.group(1)}\t{m.group(2)}\t{m.group(4)}\t{cap}\t{gen}"
                f"\t{resumir(opens, corte_ms)}"
            )
            opens = []
    return filas


if __name__ == "__main__":
    args = sys.argv[1:]
    tramo = None
    if args and args[0] == "--tramo-velas":
        tramo = int(args[1])
        args = args[2:]
    print(CABECERA)
    for ruta in args:
        for fila in analizar(ruta, tramo):
            print(fila)
