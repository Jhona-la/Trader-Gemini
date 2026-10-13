"""Fracción de aperturas del motor del lado del estado oculto, por peldaño.

Regenera el estado oculto del T-3 (misma LCG de 64 bits que el test) y cruza
las trazas OPEN TRACE (sólo las 100 primeras por corrida) con la vela en la
que se abrió cada posición.
"""
import io, re, sys

M64 = (1 << 64) - 1


def estado_oculto(a, d, n):
    sal = (int(a * 1e6) ^ (int(d * 1e8) << 20) ^ (n << 44)) & M64
    seed = (0x2545F4914F6CDD1D ^ ((sal * 0x9E3779B97F4A7C15) & M64)) & M64
    estados = []
    s = 1

    def uniforme():
        nonlocal seed
        seed = (seed * 6364136223846793005 + 1442695040888963407) & M64
        return (seed >> 11) / float(1 << 53)

    for _ in range(n):
        if uniforme() > a:
            s = -s
        uniforme()  # el ruido
        estados.append(s)
    return estados


def analizar(path):
    txt = io.open(path, encoding="utf-8", errors="replace").read().split("\n")
    opens = []
    filas = []
    for line in txt:
        m = re.search(r"OPEN TRACE\] #\d+ dir=(LONG|SHORT) .* tau=(\d+)ms rama=(\d+) .* ts=(\d+)", line)
        if m:
            opens.append((m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))))
            continue
        m = re.match(r"\[T3\] a=([0-9.]+) d=([0-9.]+)% σ=([0-9.]+)% n=(\d+)(?: C0=(\d+) USD)?", line)
        if m:
            a, d, n = float(m.group(1)), float(m.group(2)) / 100.0, int(m.group(4))
            cap = m.group(5) or "13"
            est = estado_oculto(a, d, n)
            lado_ok = 0
            ramas = {}
            taus = []
            for dir_, tau, rama, ts in opens:
                bar = min(ts // 60000, n - 1)
                s = est[bar]
                if (dir_ == "LONG" and s > 0) or (dir_ == "SHORT" and s < 0):
                    lado_ok += 1
                ramas[rama] = ramas.get(rama, 0) + 1
                taus.append(tau)
            total = len(opens)
            taus.sort()
            med = taus[len(taus) // 2] if taus else 0
            frac = lado_ok / total if total else float("nan")
            # fracción teórica de velas cuyo signo observado coincide con el estado no aplica aquí
            filas.append((a, d, cap, total, frac, med, dict(sorted(ramas.items()))))
            opens = []
    return filas


for path in sys.argv[1:]:
    print("==", path.split("/")[-1])
    for a, d, cap, total, frac, med, ramas in analizar(path):
        print(f"a={a:.2f} d={d*100:.2f}% C0={cap:>4} | aperturas trazadas={total:3d} lado_del_estado={frac:.3f} tau_mediana={med/1000:.0f}s ramas={ramas}")
