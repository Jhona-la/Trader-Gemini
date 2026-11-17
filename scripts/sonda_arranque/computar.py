"""Computador de la sonda de arranque C3 (MEDIDASV4-P1): epsilon Bonferroni y estimadores con sus n.

Solo biblioteca estandar. No hace red ni E/S salvo `resumir_csv` (lectura de un CSV de la sonda).

Politica (decision del coordinador, especificacion v4 §1): epsilon = alpha / S (cota de union), caso
aprobado S = 26 (ROSTER, crates/god-engine-core/src/bootloader.rs:126-131). La formula independiente
1 - (1 - alpha)^(1/S) se informa solo como comparacion y no alimenta ninguna decision.

Ningun estimador convierte N/D en exito: cada salida lleva n requerido/observado, formula y estado.
"""

import csv
import math
from fractions import Fraction

S_APROBADO = 26
ALFA_POR_DEFECTO = 0.05
N_PRIMEROS_INTENTOS = 299  # ceil(ln 0.05 / ln 0.99)
N_PARES = 8  # ((1.645*0.5 + 0.842*0.3)/0.4)^2 = 7.224 -> 8
N_RACHAS = 59  # ceil(ln 0.05 / ln 0.95)
N_LATENCIAS = 2995  # ceil(ln 0.05 / ln 0.999)


def epsilon(s, alpha=ALFA_POR_DEFECTO, s_acreditado=False, s_aprobado=S_APROBADO):
    """epsilon por simbolo derivado de S y alpha. Nunca es una constante fija."""
    r = {
        "S": s,
        "alpha": alpha,
        "politica": "bonferroni",
        "formula": "epsilon = alpha / S",
        "formula_informativa": "1 - (1 - alpha)^(1/S) (solo bajo independencia; no decide)",
        "independiente_alimenta_decision": False,
        "S_aprobado": s_aprobado,
        "S_estado": "acreditado" if s_acreditado else "N/D",
        "epsilon": None,
        "epsilon_independiente_informativo": None,
        "recibo_cerrado": False,
        "motivo": "",
    }
    if not isinstance(s, int) or s <= 0 or not (0.0 < alpha < 1.0):
        r["motivo"] = "S o alpha invalidos: no se inventa epsilon"
        return r
    r["epsilon"] = alpha / s
    r["epsilon_independiente_informativo"] = 1.0 - (1.0 - alpha) ** (1.0 / s)
    if not s_acreditado:
        r["motivo"] = "S no acreditado (universo efectivo sin recibo de validacion/backfill): N/D"
    elif s != s_aprobado:
        r["motivo"] = f"S={s} distinto del caso aprobado S={s_aprobado}: se informa y no se cierra"
    else:
        r["recibo_cerrado"] = True
        r["motivo"] = f"S={s} acreditado = caso aprobado; epsilon = {alpha}/{s}"
    return r


# ---------------------------------------------------------------- binomial y Clopper-Pearson exactos

def binom_cdf(k, n, p):
    """P(X <= k) con X ~ Bin(n, p), suma directa en log para estabilidad."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    total = 0.0
    lp, lq = math.log(p), math.log1p(-p)
    for i in range(0, k + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    return min(1.0, total)


def _biseccion(f, objetivo, creciente):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        v = f(mid)
        if (v < objetivo) == creciente:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def cp_superior(k, n, a):
    """Limite superior exacto: p tal que P(X <= k | p) = a."""
    if k >= n:
        return 1.0
    if k == 0:
        return 1.0 - a ** (1.0 / n)
    return _biseccion(lambda p: binom_cdf(k, n, p), a, creciente=False)


def cp_inferior(k, n, a):
    """Limite inferior exacto: p tal que P(X >= k | p) = a."""
    if k <= 0:
        return 0.0
    return _biseccion(lambda p: 1.0 - binom_cdf(k - 1, n, p), a, creciente=True)


def proporcion(k, n, nivel=0.95):
    """Proporcion de fallos k/n con IC Clopper-Pearson; la puntual y los limites van por separado."""
    a = 1.0 - nivel
    if n <= 0 or k < 0 or k > n:
        return {"estado": "N/D", "puntual": None, "n_observado": n, "k": k, "motivo": "n o k invalidos"}
    return {
        "estado": "estimacion",
        "puntual": k / n,
        "k": k,
        "n_observado": n,
        "nivel": nivel,
        "inferior_bilateral": cp_inferior(k, n, a / 2.0),
        "superior_bilateral": cp_superior(k, n, a / 2.0),
        "superior_unilateral": cp_superior(k, n, a),
        "formula": "Clopper-Pearson exacto; unilateral superior con alfa = 1 - nivel",
    }


def p_c(pares, politica_d, n_requerido=N_PARES, nivel=0.95):
    """p_c = P(reintento falla | fallo previo), solo con pares de la MISMA politica de espera D."""
    validos = [p for p in pares if p.get("politica_d") == politica_d]
    descartados = len(pares) - len(validos)
    base = {"politica_d": politica_d, "n_requerido": n_requerido, "n_observado": len(validos),
            "pares_de_otra_politica": descartados}
    if len(validos) < n_requerido:
        base.update(estado="N/D", puntual=None,
                    motivo=f"{len(validos)} pares con {politica_d} < {n_requerido} requeridos; no se transfiere de otra D")
        return base
    k = sum(1 for p in validos if p.get("reintento_fallo"))
    r = proporcion(k, len(validos), nivel)
    base.update(r)
    base["n_requerido"] = n_requerido
    return base


def cuantil_empirico(valores, q):
    """Regla declarada: rango mas cercano, el valor de orden ceil(q*n) (1-based) de la muestra ordenada."""
    n = len(valores)
    if n == 0:
        return {"valor": None, "regla": "rango mas cercano ceil(q*n)", "n": 0}
    rango = max(1, math.ceil(Fraction(str(q)) * n))
    return {"valor": sorted(valores)[rango - 1], "regla": "rango mas cercano ceil(q*n)", "rango": rango, "n": n}


def rachas_y_d(rachas_ms, retry_after_ms=0, n_requerido=N_RACHAS):
    """Duracion de rachas de fallo: q95 empirico y, aparte, el maximo como limite unilateral del q95."""
    n = len(rachas_ms)
    r = {
        "n_requerido": n_requerido,
        "n_observado": n,
        "q95_empirico": cuantil_empirico(rachas_ms, 0.95),
        "limite_q95": None,
        "confianza_limite": None,
        "d_medido_ms": None,
        "d_minimo_ms": retry_after_ms,
        "formula": "limite = max de n rachas; confianza 1 - 0.95^n; D = max(Retry-After, limite)",
    }
    if n < n_requerido:
        r.update(estado="N/D", motivo=f"{n} rachas < {n_requerido}: limite y D medido N/D; manda Retry-After")
        return r
    r["limite_q95"] = max(rachas_ms)
    r["confianza_limite"] = 1.0 - 0.95 ** n
    r["d_medido_ms"] = max(retry_after_ms, r["limite_q95"])
    r["estado"] = "completo"
    return r


def latencias(lat_ms, censuradas=0, n_requerido=N_LATENCIAS):
    """P99,9 empirico y, aparte, el maximo como limite unilateral. Un timeout es censura, no latencia."""
    n = len(lat_ms)
    r = {
        "n_requerido": n_requerido,
        "n_observado": n,
        "censuradas": censuradas,
        "p999_empirico": cuantil_empirico(lat_ms, 0.999),
        "limite_p999": None,
        "confianza_limite": None,
        "formula": "limite = max de n latencias completas; confianza 1 - 0.999^n",
    }
    if censuradas > 0:
        r.update(estado="N/D", motivo=f"{censuradas} observaciones censuradas (timeout) sin resolver: no se certifica la cola")
        return r
    if n < n_requerido:
        r.update(estado="N/D", motivo=f"{n} latencias < {n_requerido}")
        return r
    r["limite_p999"] = max(lat_ms)
    r["confianza_limite"] = 1.0 - 0.999 ** n
    r["estado"] = "completo"
    return r


def k_reintentos(p1, pc, epsilon, k_max=100):
    """K = min k con p1 * p_c^(k-1) <= epsilon. K_estimado con puntuales; K_acreditado queda N/D.

    K_acreditado exigiria cotas conjuntas validas y la justificacion del modelo de reintentos; los IC
    marginales de p1 y p_c no la dan. Este computador nunca lo cierra por su cuenta.
    """
    r = {"epsilon": epsilon, "modelo": "p1 * p_c^(k-1) <= epsilon", "K_estimado": None,
         "K_estimado_rotulo": "estimacion puntual", "K_acreditado": "N/D",
         "motivo_acreditado": "IC marginales separados no acreditan una garantia conjunta"}
    if p1.get("estado") != "estimacion" or pc.get("estado") != "estimacion":
        r["motivo_estimado"] = "p1 o p_c N/D"
        return r
    a, b = p1["puntual"], pc["puntual"]
    for k in range(1, k_max + 1):
        if a * b ** (k - 1) <= epsilon:
            r["K_estimado"] = k
            return r
    r["motivo_estimado"] = f"sin K <= {k_max}"
    return r


def resumir_csv(ruta, politica_d, s, s_acreditado):
    """Resumen de un intentos.csv de la sonda: p1 de primeros intentos, p_c de pares, latencias."""
    with open(ruta, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    primeros = [x for x in filas if x.get("reintento") == "0"]
    fallos = sum(1 for x in primeros if x.get("exito") != "1")
    pares = []
    por_cohorte = {}
    for x in filas:
        por_cohorte.setdefault(x.get("cohorte"), []).append(x)
    for grupo in por_cohorte.values():
        grupo.sort(key=lambda y: int(y.get("reintento") or 0))
        for prev, sig in zip(grupo, grupo[1:]):
            if prev.get("exito") != "1":
                pares.append({"politica_d": sig.get("d_previsto_ms") and f"D={sig['d_previsto_ms']}",
                              "reintento_fallo": sig.get("exito") != "1"})
    lat = [float(x["latencia_ms"]) for x in filas if x.get("latencia_ms") not in ("", "N/D", None)]
    cens = sum(1 for x in filas if x.get("error") == "timeout")
    eps = epsilon(s, s_acreditado=s_acreditado)
    p1 = proporcion(fallos, len(primeros)) if primeros else {"estado": "N/D", "puntual": None}
    pc = p_c(pares, politica_d)
    return {"epsilon": eps, "p1": p1, "p_c": pc, "latencias": latencias(lat, cens),
            "K": k_reintentos(p1, pc, eps["epsilon"] or 0.0), "filas": len(filas)}
