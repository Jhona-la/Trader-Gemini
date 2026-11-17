"""Sonda de arranque C3 (MEDIDASV4-P3): GET publico de klines contra Testnet, SOLO LECTURA.

Unico endpoint: GET https://testnet.binancefuture.com/fapi/v1/klines?symbol=<SYM>&interval=1m&limit=1000,
el mismo que warmup_klines_url(true, ...) (crates/god-engine-core/src/bootloader.rs:28-34). Sin llaves,
sin cabeceras de autenticacion, sin firmas, sin redirecciones (otro host o endpoint = fallo registrado).
No existe en este modulo ningun camino que envie ordenes ni otro metodo que GET. No inicia god_engine.

Cada intento deja una fila en intentos.csv (COLUMNAS). Transporte, esquema y suficiencia van separados:
- exito_http: 200 y cuerpo leido; esquema_valido: JSON lista de listas; barras_validas: paridad con
  parse_warmup_kline (bootloader.rs:5-19): >= 6 campos, OHLCV en texto, finitos, l > 0, v >= 0, l<=o<=h, l<=c<=h.
- exito = exito_http y esquema_valido (definicion de fallo del estimador p1/p_c).
- warmup_suficiente = barras_validas >= M_acreditado; sin M acreditado es N/D, nunca exito.
peso_delta es la diferencia del contador IP X-MBX-USED-WEIGHT-1M; solo es atribuible con concurrencia
acotada, lectura previa, misma ventana y sin retroceso. En otro caso N/D con motivo, sin inventar ceros.
Frenos (docs/BINANCE_API_MAP.md:117-123): 429 -> cero envios hasta Retry-After (60 s si falta);
418 -> pausa Retry-After (300 s si falta) y TERMINA la tanda marcada.
"""

import argparse
import csv
import datetime
import json
import math
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HOST_TESTNET = "testnet.binancefuture.com"
RUTA_KLINES = "/fapi/v1/klines"
CABECERA_PESO = "X-MBX-USED-WEIGHT-1M"
LIMIT_WARMUP = 1000  # bootloader.rs:255

COLUMNAS = [
    "ts_envio_iso", "ts_respuesta_iso", "simbolo", "codigo_http", "exito", "latencia_ms", "retry_after_s",
    "peso_delta", "bytes", "error",
    "cohorte", "reintento", "limit", "mono_envio_ms", "mono_respuesta_ms", "d_previsto_ms", "espera_real_ms",
    "exito_http", "esquema_valido", "barras_recibidas", "barras_validas", "barras_rechazadas",
    "warmup_suficiente", "header_peso_anterior", "header_peso_posterior", "instante_contador_anterior_ms",
    "instante_contador_posterior_ms", "ventana_contador", "peso_delta_estado", "reserva_w",
]

_NUM_RUST = re.compile(r"^[+-]?(inf|infinity|nan|\d+\.?\d*(e[+-]?\d+)?|\.\d+(e[+-]?\d+)?)$", re.IGNORECASE)


def barra_valida(k):
    """Paridad con parse_warmup_kline (bootloader.rs:5-19). Devuelve [o,h,l,c,v] o None."""
    if not isinstance(k, list) or len(k) < 6:
        return None
    vals = []
    for i in range(1, 6):
        x = k[i]
        if not isinstance(x, str) or not _NUM_RUST.match(x):
            return None
        vals.append(float(x))
    o, h, lo, c, v = vals
    if all(math.isfinite(x) for x in vals) and lo > 0.0 and v >= 0.0 and lo <= o <= h and lo <= c <= h:
        return vals
    return None


def validar_base(base, permitir_local=False):
    u = urllib.parse.urlsplit(base)
    if u.scheme == "https" and u.hostname == HOST_TESTNET and u.port is None and u.path in ("", "/") \
            and not u.query and not u.username:
        return f"https://{HOST_TESTNET}"
    if permitir_local and u.scheme == "http" and u.hostname == "127.0.0.1" and u.path in ("", "/"):
        return f"http://127.0.0.1:{u.port}"
    raise ValueError(f"base no admitida (solo https://{HOST_TESTNET}): {base}")


class _SinRedireccion(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # una redireccion queda como HTTPError registrado, nunca se sigue


def _iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def _entero(x):
    try:
        return int(str(x).strip())
    except (TypeError, ValueError):
        return None


class Sonda:
    def __init__(self, base, salida_dir, presupuesto, reloj_ms=None, dormir_ms=None, timeout_s=10.0,
                 permitir_local=False, concurrencia_acotada=False, m_acreditado=None, limit=LIMIT_WARMUP,
                 nombre_csv="intentos.csv"):
        self.base = validar_base(base, permitir_local)
        self.presupuesto = presupuesto
        self.reloj_ms = reloj_ms or (lambda: time.monotonic_ns() // 1_000_000)
        self.dormir_ms = dormir_ms or (lambda ms: time.sleep(ms / 1000.0))
        self.timeout_s = timeout_s
        self.concurrencia_acotada = concurrencia_acotada
        self.m_acreditado = m_acreditado
        self.limit = int(limit)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SinRedireccion())
        os.makedirs(salida_dir, exist_ok=True)
        self.ruta_csv = os.path.join(salida_dir, nombre_csv)
        self._f = open(self.ruta_csv, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=COLUMNAS)
        self._w.writeheader()
        self._lectura = None  # (valor, instante_ms, ventana)
        self._w_atribuible = None
        self.intentos = 0

    def cerrar(self):
        if not self._f.closed:
            self._f.close()

    def url(self, simbolo):
        q = urllib.parse.urlencode({"symbol": simbolo.upper(), "interval": "1m", "limit": self.limit})
        return f"{self.base}{RUTA_KLINES}?{q}"

    def _peso(self, crudo, t_ms):
        """Delta del contador IP respecto a la lectura previa, o N/D con motivo."""
        valor = _entero(crudo)
        ventana = t_ms // 60000
        previa = self._lectura
        self._lectura = None if valor is None else (valor, t_ms, ventana)
        if valor is None:
            return "N/D", "N/D:header_ausente", previa
        if not self.concurrencia_acotada:
            return "N/D", "N/D:concurrencia_no_acotada", previa
        if previa is None:
            return "N/D", "N/D:sin_lectura_previa", previa
        if valor < previa[0]:
            return "N/D", "N/D:retroceso_o_reinicio", previa
        if ventana != previa[2]:
            return "N/D", "N/D:cambio_de_ventana", previa
        if valor - previa[0] > self.presupuesto.cota_reserva:
            # mas que la cota documentada del endpoint (BINANCE_API_MAP.md:27): hubo trafico no contabilizado
            return "N/D", "N/D:excede_cota_endpoint", previa
        return str(valor - previa[0]), "atribuible", previa

    def intento(self, simbolo, cohorte, reintento, d_previsto_ms, espera_previa_ms=0):
        espera = self.presupuesto.espera_ms(self.reloj_ms(), self._w_atribuible)
        if espera:
            self.dormir_ms(espera)
        fila = {c: "" for c in COLUMNAS}
        fila.update(simbolo=simbolo.upper(), cohorte=cohorte, reintento=str(reintento), limit=str(self.limit),
                    d_previsto_ms=str(d_previsto_ms), espera_real_ms=str(int(espera_previa_ms) + int(espera)),
                    warmup_suficiente="N/D", exito="0", exito_http="0", esquema_valido="0",
                    barras_recibidas="0", barras_validas="0", barras_rechazadas="0",
                    peso_delta="N/D", peso_delta_estado="N/D:sin_respuesta", codigo_http="N/D", latencia_ms="N/D")
        t_envio = self.reloj_ms()
        idx = self.presupuesto.reservar(t_envio, self._w_atribuible, "envio")
        fila["reserva_w"] = str(self.presupuesto.reservas[idx][1])
        fila["ts_envio_iso"] = _iso()
        fila["mono_envio_ms"] = str(t_envio)
        req = urllib.request.Request(self.url(simbolo), method="GET", headers={"User-Agent": "sonda-arranque-c3"})
        p0 = time.perf_counter()
        cuerpo, cabeceras, codigo = b"", {}, None
        try:
            with self.opener.open(req, timeout=self.timeout_s) as r:
                codigo = r.status
                cabeceras = dict(r.headers)
                cuerpo = r.read()
        except urllib.error.HTTPError as e:
            codigo = e.code
            cabeceras = dict(e.headers or {})
            try:
                cuerpo = e.read()
            except (OSError, ValueError):
                cuerpo = b""
            finally:
                e.close()
        except (TimeoutError, socket.timeout):
            fila["error"] = "timeout"
        except urllib.error.URLError as e:
            fila["error"] = "timeout" if isinstance(e.reason, (TimeoutError, socket.timeout)) else f"red:{e.reason}"
        except (OSError, ValueError) as e:
            fila["error"] = f"red:{type(e).__name__}"
        p1 = time.perf_counter()
        t_resp = self.reloj_ms()
        fila["ts_respuesta_iso"] = _iso()
        fila["mono_respuesta_ms"] = str(t_resp)
        self.intentos += 1
        if codigo is None:  # timeout o error de transporte: observacion censurada, reserva intacta
            self.presupuesto.resolver(idx, None, fila["error"])
            self._lectura = None
            self._escribir(fila)
            return fila
        fila["codigo_http"] = str(codigo)
        fila["latencia_ms"] = f"{(p1 - p0) * 1000.0:.3f}"
        fila["bytes"] = str(len(cuerpo))
        ra = _entero(cabeceras.get("Retry-After"))
        fila["retry_after_s"] = "" if ra is None else str(ra)
        crudo = cabeceras.get(CABECERA_PESO)
        delta, estado, previa = self._peso(crudo, t_resp)
        fila["header_peso_anterior"] = "" if previa is None else str(previa[0])
        fila["instante_contador_anterior_ms"] = "" if previa is None else str(previa[1])
        fila["header_peso_posterior"] = "" if crudo is None else str(crudo)
        fila["instante_contador_posterior_ms"] = str(t_resp)
        fila["ventana_contador"] = "1m"
        fila["peso_delta"], fila["peso_delta_estado"] = delta, estado
        w_medido = _entero(delta) if estado == "atribuible" else None
        self.presupuesto.resolver(idx, w_medido, estado)
        self._w_atribuible = w_medido if w_medido and w_medido > 0 else None
        if codigo != 200:
            fila["error"] = f"http_{codigo}"
            self._escribir(fila)
            return fila
        fila["exito_http"] = "1"
        try:
            datos = json.loads(cuerpo.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            datos = None
        if isinstance(datos, list) and all(isinstance(k, list) for k in datos):
            fila["esquema_valido"] = "1"
            validas = sum(1 for k in datos if barra_valida(k) is not None)
            fila["barras_recibidas"] = str(len(datos))
            fila["barras_validas"] = str(validas)
            fila["barras_rechazadas"] = str(len(datos) - validas)
            fila["exito"] = "1"
            if self.m_acreditado is not None:
                fila["warmup_suficiente"] = "1" if validas >= self.m_acreditado else "0"
        else:
            fila["error"] = "esquema"
        self._escribir(fila)
        return fila

    def _escribir(self, fila):
        self._w.writerow(fila)
        self._f.flush()

    def correr(self, simbolos, max_reintentos, max_intentos, max_duracion_ms, d_reintento_ms=250):
        """Tanda con limites finitos. Termina al acabar, al tocar un limite o con cualquier 418."""
        t0 = self.reloj_ms()
        resumen = {"simbolos": len(simbolos), "intentos": 0, "terminacion": "completa", "fallidos": []}
        for n, sym in enumerate(simbolos):
            cohorte = f"{sym.upper()}-{n}"
            reintento, d_prev, espera = 0, 0, 0
            while True:
                if self.intentos >= max_intentos:
                    resumen["terminacion"] = "incompleta_limite_intentos"
                    resumen["intentos"] = self.intentos
                    return resumen
                if self.reloj_ms() - t0 >= max_duracion_ms:
                    resumen["terminacion"] = "incompleta_limite_duracion"
                    resumen["intentos"] = self.intentos
                    return resumen
                fila = self.intento(sym, cohorte, reintento, d_prev, espera)
                if fila["exito"] == "1":
                    break
                ra = _entero(fila["retry_after_s"])
                if fila["codigo_http"] == "418":
                    self.dormir_ms(1000 * (ra if ra is not None else 300))
                    resumen["terminacion"] = "terminada_418"
                    resumen["intentos"] = self.intentos
                    return resumen
                if reintento >= max_reintentos:
                    resumen["fallidos"].append(sym)
                    break
                if fila["codigo_http"] == "429":
                    d_prev = 1000 * (ra if ra is not None else 60)
                else:
                    d_prev = d_reintento_ms
                self.dormir_ms(d_prev)
                espera = d_prev
                reintento += 1
        resumen["intentos"] = self.intentos
        return resumen


def main(argv=None):
    import computar
    import presupuesto as pres

    ap = argparse.ArgumentParser(description="Sonda de arranque C3, solo lectura contra Testnet")
    ap.add_argument("--simbolos", required=True, help="lista separada por comas (universo efectivo)")
    ap.add_argument("--s-acreditado", action="store_true", help="el universo sale de un recibo de validacion")
    ap.add_argument("--salida", required=True, help="carpeta salidas/sonda_<ts>")
    ap.add_argument("--sin-red", action="store_true", help="solo plan y epsilon, ningun GET")
    ap.add_argument("--confirmo-solo-lectura", action="store_true")
    ap.add_argument("--max-intentos", type=int, default=computar.N_PRIMEROS_INTENTOS + computar.N_LATENCIAS)
    ap.add_argument("--max-reintentos", type=int, default=3)
    ap.add_argument("--max-duracion-min", type=int, default=120)
    a = ap.parse_args(argv)
    simbolos = [s.strip().upper() for s in a.simbolos.split(",") if s.strip()]
    p = pres.Presupuesto()
    plan = {"epsilon": computar.epsilon(len(simbolos), s_acreditado=a.s_acreditado),
            "presupuesto": p.plan(a.max_intentos), "limites": {"max_intentos": a.max_intentos,
            "max_reintentos": a.max_reintentos, "max_duracion_min": a.max_duracion_min},
            "endpoint": f"GET https://{HOST_TESTNET}{RUTA_KLINES}?symbol=<SYM>&interval=1m&limit={LIMIT_WARMUP}"}
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if a.sin_red:
        return 0
    if not a.confirmo_solo_lectura:
        print("falta --confirmo-solo-lectura: no se envia nada", file=sys.stderr)
        return 2
    s = Sonda(f"https://{HOST_TESTNET}", a.salida, p)
    try:
        estado = s.correr(simbolos * max(1, a.max_intentos // max(1, len(simbolos))), a.max_reintentos,
                          a.max_intentos, a.max_duracion_min * 60000)
    finally:
        s.cerrar()
    with open(os.path.join(a.salida, "estado.json"), "w", encoding="utf-8") as f:
        json.dump(estado, f, indent=2, ensure_ascii=False)
    print(json.dumps(estado, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
