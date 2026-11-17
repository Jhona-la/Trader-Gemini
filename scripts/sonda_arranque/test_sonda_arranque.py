"""Pruebas de la sonda de arranque C3 (Carril E, Lote 16, adenda nominal C3-SONDA-ARRANQUE-PYTHON).

Comando: python -B -m unittest discover -s scripts/sonda_arranque -p "test_*.py"

Puras: computar, presupuesto y medidor con fixtures sinteticos en memoria.
Integracion identificada: servidor HTTP propio en 127.0.0.1 y CSV en un directorio temporal.
Ninguna prueba sale a la red externa ni lee .env, llaves, data/ o Vision real.
"""

import importlib
import math
import unittest


def cargar(nombre):
    """Importa el modulo de la sonda en cada prueba: si falta, la prueba sale roja (no se salta)."""
    return importlib.import_module(nombre)


class P1Epsilon(unittest.TestCase):
    def test_epsilon_se_deriva_de_S_y_no_se_harda(self):
        computar = cargar("computar")
        for s in (26, 30, 7):
            r = computar.epsilon(s, alpha=0.05, s_acreditado=True)
            self.assertEqual(r["S"], s)
            self.assertEqual(r["alpha"], 0.05)
            self.assertEqual(r["politica"], "bonferroni")
            self.assertAlmostEqual(r["epsilon"], 0.05 / s, places=15)
            self.assertAlmostEqual(r["epsilon_independiente_informativo"], 1 - 0.95 ** (1 / s), places=15)
            self.assertFalse(r["independiente_alimenta_decision"])
        # caso aprobado: S = 26 acreditado cierra el recibo
        ok = computar.epsilon(26, alpha=0.05, s_acreditado=True)
        self.assertTrue(ok["recibo_cerrado"])
        self.assertAlmostEqual(ok["epsilon"], 0.001923076923, places=12)
        self.assertAlmostEqual(ok["epsilon_independiente_informativo"], 0.001971, places=6)  # v4 §1, tabla S=26
        # S distinto del aprobado: se informa, no cierra
        otro = computar.epsilon(30, alpha=0.05, s_acreditado=True)
        self.assertFalse(otro["recibo_cerrado"])
        self.assertIn("S=30", otro["motivo"])
        # S no acreditado: N/D
        nd = computar.epsilon(26, alpha=0.05, s_acreditado=False)
        self.assertFalse(nd["recibo_cerrado"])
        self.assertEqual(nd["S_estado"], "N/D")
        # S invalido no inventa epsilon
        malo = computar.epsilon(0, alpha=0.05, s_acreditado=True)
        self.assertIsNone(malo["epsilon"])
        self.assertFalse(malo["recibo_cerrado"])


class P4Estimadores(unittest.TestCase):
    def test_estimadores_cerrados_con_csv_sintetico(self):
        computar = cargar("computar")
        # p1: 10 fallos en 299 primeros intentos -> estimacion puntual y limites aparte
        p1 = computar.proporcion(10, 299, nivel=0.95)
        self.assertEqual(p1["estado"], "estimacion")
        self.assertAlmostEqual(p1["puntual"], 10 / 299, places=15)
        self.assertLess(p1["inferior_bilateral"], p1["puntual"])
        self.assertGreater(p1["superior_bilateral"], p1["puntual"])
        self.assertGreater(p1["superior_unilateral"], p1["puntual"])
        self.assertLess(p1["superior_unilateral"], p1["superior_bilateral"])
        # Clopper-Pearson: la binomial en el limite superior unilateral da alfa
        self.assertAlmostEqual(computar.binom_cdf(10, 299, p1["superior_unilateral"]), 0.05, places=9)
        # p_c: 2 de 8 pares bajo la MISMA politica de D
        pares = [{"politica_d": "D=250", "reintento_fallo": i < 2} for i in range(8)]
        pc = computar.p_c(pares, politica_d="D=250", n_requerido=8)
        self.assertEqual(pc["estado"], "estimacion")
        self.assertAlmostEqual(pc["puntual"], 0.25, places=15)
        self.assertEqual(pc["n_observado"], 8)
        # pares de otra politica no se transfieren
        otra = computar.p_c(pares, politica_d="D=500", n_requerido=8)
        self.assertEqual(otra["estado"], "N/D")
        self.assertIsNone(otra["puntual"])
        # menos de 8 pares: N/D
        pocos = computar.p_c(pares[:5], politica_d="D=250", n_requerido=8)
        self.assertEqual(pocos["estado"], "N/D")
        # cuantil empirico con regla declarada (rango mas cercano, ceil(q*n))
        self.assertEqual(computar.cuantil_empirico(list(range(1, 101)), 0.95)["valor"], 95)
        self.assertEqual(computar.cuantil_empirico([5, 1, 3], 0.95)["valor"], 5)
        # 3 rachas: limite q95 y D medido N/D, Retry-After sigue mandando
        rachas = computar.rachas_y_d([1200, 800, 950], retry_after_ms=30000, n_requerido=59)
        self.assertEqual(rachas["q95_empirico"]["valor"], 1200)
        self.assertIsNone(rachas["limite_q95"])
        self.assertEqual(rachas["estado"], "N/D")
        self.assertIsNone(rachas["d_medido_ms"])
        self.assertEqual(rachas["d_minimo_ms"], 30000)
        # 59 rachas: el maximo es limite unilateral con confianza 1-0.95^59, D = max(Retry-After, limite)
        r59 = computar.rachas_y_d(list(range(100, 159)), retry_after_ms=0, n_requerido=59)
        self.assertEqual(r59["estado"], "completo")
        self.assertEqual(r59["limite_q95"], 158)
        self.assertAlmostEqual(r59["confianza_limite"], 1 - 0.95 ** 59, places=12)
        self.assertEqual(r59["d_medido_ms"], 158)
        # latencias: menos de 2995 o con censura no resuelta no certifican P99,9
        lat = computar.latencias([10.0] * 2000, censuradas=0, n_requerido=2995)
        self.assertIsNone(lat["limite_p999"])
        self.assertEqual(lat["estado"], "N/D")
        cens = computar.latencias([10.0] * 3000, censuradas=4, n_requerido=2995)
        self.assertIsNone(cens["limite_p999"])
        self.assertIn("censur", cens["motivo"])
        bien = computar.latencias([float(i) for i in range(3000)], censuradas=0, n_requerido=2995)
        self.assertEqual(bien["limite_p999"], 2999.0)
        self.assertAlmostEqual(bien["confianza_limite"], 1 - 0.999 ** 3000, places=12)
        self.assertEqual(bien["p999_empirico"]["valor"], 2996.0)  # rango ceil(0,999*3000)=2997 -> valor 2996
        # 299 intentos sin fallos: p1_superior <= 0.01, p_c N/D, K_acreditado N/D
        cero = computar.proporcion(0, 299, nivel=0.95)
        self.assertAlmostEqual(cero["superior_unilateral"], 1 - 0.05 ** (1 / 299), places=12)
        self.assertLessEqual(cero["superior_unilateral"], 0.01)
        sin_pares = computar.p_c([], politica_d="D=250", n_requerido=8)
        self.assertEqual(sin_pares["estado"], "N/D")
        k = computar.k_reintentos(cero, sin_pares, epsilon=0.05 / 26)
        self.assertEqual(k["K_acreditado"], "N/D")
        self.assertIsNone(k["K_estimado"])
        # con puntuales: K_estimado rotulado como estimacion, K_acreditado sigue N/D (IC marginales)
        k2 = computar.k_reintentos(p1, pc, epsilon=0.05 / 26)
        p1v, pcv = 10 / 299, 0.25
        esperado = next(j for j in range(1, 100) if p1v * pcv ** (j - 1) <= 0.05 / 26)
        self.assertEqual(k2["K_estimado"], esperado)
        self.assertEqual(k2["K_estimado_rotulo"], "estimacion puntual")
        self.assertEqual(k2["K_acreditado"], "N/D")
        self.assertTrue(math.isfinite(k2["epsilon"]))


class P2Presupuesto(unittest.TestCase):
    def test_presupuesto_nunca_pasa_de_la_mitad(self):
        presupuesto = cargar("presupuesto")
        p = presupuesto.Presupuesto(b_por_minuto=2400, fraccion=0.5, ventana_ms=60000, cota_reserva=10)
        self.assertEqual(p.tope, 1200)
        reloj = 0
        envios = []
        # secuencia heredada: 500 respuestas de peso 5, latencia 0; la primera sin medida reserva la cota 10
        for i in range(500):
            reserva = None if i == 0 else 5
            espera = p.espera_ms(reloj, reserva)
            reloj += espera
            usado = p.usado(reloj)
            w = p.reserva_para(reserva)
            self.assertLessEqual(usado + w, 1200, f"envio {i}: {usado} + {w} > 1200")
            p.reservar(reloj, reserva)
            envios.append((reloj, w))
        # reconstruccion de la serie: en ninguna ventana de 60 s la suma de reservas pasa de 1200
        for t, _ in envios:
            en_ventana = sum(w for (u, w) in envios if t - 60000 < u <= t)
            self.assertLessEqual(en_ventana, 1200)
        self.assertEqual(envios[0][1], 10)  # primera peticion: cota documentada, no w medido
        # con w = 5 el ritmo estable es una peticion cada 250 ms (60000*5/1200)
        self.assertEqual(envios[-1][0] - envios[-2][0], 250)
        # sin burst: con cupo libre, el siguiente envio igual espera su espaciado (60000*w/1200)
        e = presupuesto.Presupuesto(b_por_minuto=2400, fraccion=0.5, ventana_ms=60000, cota_reserva=10)
        e.reservar(0, 5)
        self.assertEqual(e.espera_ms(0, 5), 250)
        # frontera del cupo (solo la ventana, sin espaciado): exactamente 1200 entra, 1201 no
        q = presupuesto.Presupuesto(b_por_minuto=2400, fraccion=0.5, ventana_ms=60000, cota_reserva=10,
                                    espaciar=False)
        for _ in range(119):
            q.reservar(0, 10)
        self.assertEqual(q.usado(0), 1190)
        self.assertEqual(q.espera_ms(0, 10), 0)
        self.assertGreater(q.espera_ms(0, 11), 0)
        # timeout y contador incierto: la reserva no se libera antes de su ventana
        t = presupuesto.Presupuesto(b_por_minuto=2400, fraccion=0.5, ventana_ms=60000, cota_reserva=10)
        idx = t.reservar(0, None)
        t.resolver(idx, w_medido=None, motivo="timeout")
        self.assertEqual(t.usado(59999), 10)
        self.assertEqual(t.usado(60000), 0)
        idx2 = t.reservar(100000, None)
        t.resolver(idx2, w_medido=3, motivo="contador incierto")
        self.assertEqual(t.usado(100001), 10)  # nunca baja de la reserva por un delta incierto
        idx3 = t.reservar(200000, 5)
        t.resolver(idx3, w_medido=8, motivo="delta mayor")
        self.assertEqual(t.usado(200001), 8)  # si el delta medido supera la reserva, cuenta el mayor
        # un siguiente envio que nunca cabe no espera para siempre
        with self.assertRaises(ValueError):
            t.espera_ms(0, 1201)


# ------------------------------------------------------------------ integracion: HTTP en 127.0.0.1

import csv as _csv
import http.server
import json as _json
import os
import shutil
import tempfile
import threading
import time


def _barra(t, o, h, l, c, v):
    return [t, str(o), str(h), str(l), str(c), str(v), t + 59999, "0", 1, "0", "0", "0"]


class _Guion(http.server.BaseHTTPRequestHandler):
    """Responde segun una lista de pasos; cada paso = (codigo, cabeceras, cuerpo, demora_s)."""

    def do_GET(self):  # noqa: N802
        srv = self.server
        srv.peticiones.append((self.path, dict(self.headers)))
        codigo, cabeceras, cuerpo, demora = srv.pasos.pop(0) if srv.pasos else (500, {}, b"", 0)
        if demora:
            time.sleep(demora)
        try:
            self.send_response(codigo)
            for k, v in cabeceras.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, *a):
        pass


class _ConServidor(unittest.TestCase):
    def setUp(self):
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Guion)
        self.srv.daemon_threads = True
        self.srv.pasos = []
        self.srv.peticiones = []
        self.hilo = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.hilo.start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.dir = tempfile.mkdtemp(prefix="sonda_c3_prueba_")
        self.reloj = [1_000_000]

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.hilo.join(5)
        shutil.rmtree(self.dir, ignore_errors=True)

    def ahora(self):
        return self.reloj[0]

    def dormir(self, ms):
        self.reloj[0] += int(ms)

    def filas(self, sonda):
        with open(sonda.ruta_csv, newline="", encoding="utf-8") as f:
            return list(_csv.DictReader(f))


class P1Csv(_ConServidor):
    def test_csv_registra_cada_intento(self):
        sonda_m = cargar("sonda")
        presupuesto = cargar("presupuesto")
        ok = _json.dumps([_barra(0, 10, 11, 9, 10.5, 3), _barra(60000, 10.5, 12, 10, 11, 0),
                          _barra(120000, 11, 10, 12, 11, 1)]).encode()  # la tercera: low > high
        self.srv.pasos = [
            (200, {"X-MBX-USED-WEIGHT-1M": "7"}, ok, 0),
            (429, {"Retry-After": "30", "X-MBX-USED-WEIGHT-1M": "1250"}, b"{}", 0),
            (200, {}, ok, 1.5),  # timeout del cliente (0,3 s)
            (200, {"X-MBX-USED-WEIGHT-1M": "20"}, b'{"code": -1121, "msg": "Invalid symbol."}', 0),
            (200, {}, ok, 0),  # contador ausente
            (200, {"X-MBX-USED-WEIGHT-1M": "3"}, ok, 0),  # retroceso: reinicio de ventana
            (200, {"X-MBX-USED-WEIGHT-1M": "8"}, ok, 0),  # 3 -> 8 atribuible (+5)
        ]
        s = sonda_m.Sonda(self.base, self.dir, presupuesto.Presupuesto(espaciar=False), reloj_ms=self.ahora,
                          dormir_ms=self.dormir, timeout_s=0.3, permitir_local=True, concurrencia_acotada=True,
                          m_acreditado=None)
        for i in range(7):
            s.intento("BTCUSDT", cohorte=f"c{i}", reintento=0, d_previsto_ms=0)
        s.cerrar()
        f = self.filas(s)
        self.assertEqual(len(f), 7)
        for fila in f:
            for col in sonda_m.COLUMNAS:
                self.assertIn(col, fila)
            self.assertNotEqual(fila["ts_envio_iso"], "")
        a, b, c, d, e, g, h = f
        self.assertEqual((a["codigo_http"], a["exito_http"], a["esquema_valido"]), ("200", "1", "1"))
        self.assertEqual((a["barras_recibidas"], a["barras_validas"], a["barras_rechazadas"]), ("3", "2", "1"))
        self.assertEqual(a["warmup_suficiente"], "N/D")  # sin M acreditado no hay exito de suficiencia
        self.assertTrue(a["peso_delta_estado"].startswith("N/D"))  # primera lectura: sin previa
        self.assertEqual(a["limit"], "1000")
        self.assertEqual((b["codigo_http"], b["exito"], b["retry_after_s"], b["error"]), ("429", "0", "30", "http_429"))
        # 7 -> 1250 no es el peso de UNA peticion (cota documentada 10): trafico no contabilizado
        self.assertEqual((b["peso_delta"], b["peso_delta_estado"]), ("N/D", "N/D:excede_cota_endpoint"))
        self.assertEqual((c["error"], c["exito"], c["latencia_ms"], c["codigo_http"]), ("timeout", "0", "N/D", "N/D"))
        self.assertEqual((d["codigo_http"], d["exito_http"], d["esquema_valido"], d["exito"]), ("200", "1", "0", "0"))
        self.assertEqual(d["barras_validas"], "0")
        self.assertEqual(e["peso_delta_estado"], "N/D:header_ausente")
        self.assertEqual(e["header_peso_posterior"], "")
        self.assertEqual(g["peso_delta_estado"], "N/D:sin_lectura_previa")  # e no dejo lectura
        self.assertEqual((h["peso_delta"], h["peso_delta_estado"]), ("5", "atribuible"))
        self.assertEqual((h["header_peso_anterior"], h["header_peso_posterior"]), ("3", "8"))
        # retroceso: un tramo 8 -> 3 no se recorta a cero ni se inventa
        s2 = sonda_m.Sonda(self.base, self.dir, presupuesto.Presupuesto(espaciar=False), reloj_ms=self.ahora,
                           dormir_ms=self.dormir, timeout_s=0.3, permitir_local=True, concurrencia_acotada=True,
                           nombre_csv="b.csv")
        self.srv.pasos = [(200, {"X-MBX-USED-WEIGHT-1M": "8"}, ok, 0), (200, {"X-MBX-USED-WEIGHT-1M": "3"}, ok, 0),
                          (200, {"X-MBX-USED-WEIGHT-1M": "9"}, ok, 0)]
        s2.intento("ETHUSDT", "x", 0, 0)
        s2.intento("ETHUSDT", "y", 0, 0)
        self.dormir(60000)  # cambia la ventana del contador
        s2.intento("ETHUSDT", "z", 0, 0)
        s2.cerrar()
        f2 = self.filas(s2)
        self.assertEqual(f2[1]["peso_delta_estado"], "N/D:retroceso_o_reinicio")
        self.assertEqual(f2[1]["peso_delta"], "N/D")
        self.assertEqual(f2[2]["peso_delta_estado"], "N/D:cambio_de_ventana")
        # concurrencia no acotada: nunca hay peso exclusivo atribuible
        s3 = sonda_m.Sonda(self.base, self.dir, presupuesto.Presupuesto(espaciar=False), reloj_ms=self.ahora,
                           dormir_ms=self.dormir, timeout_s=0.3, permitir_local=True, concurrencia_acotada=False,
                           nombre_csv="c.csv")
        self.srv.pasos = [(200, {"X-MBX-USED-WEIGHT-1M": "1"}, ok, 0), (200, {"X-MBX-USED-WEIGHT-1M": "6"}, ok, 0)]
        s3.intento("ETHUSDT", "x", 0, 0)
        s3.intento("ETHUSDT", "y", 0, 0)
        s3.cerrar()
        self.assertEqual(self.filas(s3)[1]["peso_delta_estado"], "N/D:concurrencia_no_acotada")
        # la peticion es el GET publico del warmup: ruta, parametros y sin llaves
        ruta, cab = self.srv.peticiones[0]
        self.assertEqual(ruta, "/fapi/v1/klines?symbol=BTCUSDT&interval=1m&limit=1000")
        self.assertFalse(any(k.lower() in ("x-mbx-apikey", "authorization") for k in cab))

    def test_solo_testnet_y_solo_lectura(self):
        sonda_m = cargar("sonda")
        presupuesto = cargar("presupuesto")
        for base in ("https://fapi.binance.com", "http://testnet.binancefuture.com", "https://example.org",
                     "https://testnet.binancefuture.com.evil.org", self.base):
            with self.assertRaises(ValueError, msg=base):
                sonda_m.Sonda(base, self.dir, presupuesto.Presupuesto(), permitir_local=False)
        sonda_m.Sonda("https://testnet.binancefuture.com", self.dir, presupuesto.Presupuesto()).cerrar()
        # sin redirecciones a otro host o endpoint
        self.srv.pasos = [(302, {"Location": "https://fapi.binance.com/fapi/v1/order"}, b"", 0)]
        s = sonda_m.Sonda(self.base, self.dir, presupuesto.Presupuesto(), reloj_ms=self.ahora, dormir_ms=self.dormir,
                          timeout_s=0.5, permitir_local=True, nombre_csv="r.csv")
        s.intento("BTCUSDT", "r", 0, 0)
        s.cerrar()
        fila = self.filas(s)[0]
        self.assertEqual((fila["codigo_http"], fila["exito"]), ("302", "0"))
        self.assertEqual(len(self.srv.peticiones), 1)
        # la sonda no tiene ningun camino que envie ordenes: solo GET a /fapi/v1/klines
        with open(sonda_m.__file__, encoding="utf-8") as fh:
            fuente = fh.read()
        for prohibido in ("/fapi/v1/order", "POST", "X-MBX-APIKEY", "signature=", ".env"):
            self.assertNotIn(prohibido, fuente)


class P3RetryAfter(_ConServidor):
    def test_retry_after_se_respeta(self):
        sonda_m = cargar("sonda")
        presupuesto = cargar("presupuesto")
        ok = _json.dumps([_barra(0, 10, 11, 9, 10.5, 3)]).encode()
        self.srv.pasos = [(429, {"Retry-After": "30"}, b"{}", 0), (200, {}, ok, 0)]
        s = sonda_m.Sonda(self.base, self.dir, presupuesto.Presupuesto(), reloj_ms=self.ahora, dormir_ms=self.dormir,
                          timeout_s=0.5, permitir_local=True)
        estado = s.correr(["BTCUSDT"], max_reintentos=3, max_intentos=10, max_duracion_ms=10 * 60000)
        s.cerrar()
        f = self.filas(s)
        self.assertEqual(len(f), 2)
        self.assertEqual(f[0]["codigo_http"], "429")
        self.assertGreaterEqual(int(f[1]["mono_envio_ms"]) - int(f[0]["mono_respuesta_ms"]), 30000)
        self.assertEqual(f[1]["codigo_http"], "200")
        self.assertEqual((f[1]["cohorte"], f[1]["reintento"]), (f[0]["cohorte"], "1"))
        self.assertGreaterEqual(int(f[1]["espera_real_ms"]), 30000)
        self.assertEqual(estado["terminacion"], "completa")
        # 429 sin Retry-After: 60 s
        self.srv.pasos = [(429, {}, b"{}", 0), (200, {}, ok, 0)]
        s2 = sonda_m.Sonda(self.base, self.dir, presupuesto.Presupuesto(), reloj_ms=self.ahora, dormir_ms=self.dormir,
                           timeout_s=0.5, permitir_local=True, nombre_csv="b.csv")
        s2.correr(["BTCUSDT"], max_reintentos=3, max_intentos=10, max_duracion_ms=10 * 60000)
        s2.cerrar()
        g = self.filas(s2)
        self.assertGreaterEqual(int(g[1]["mono_envio_ms"]) - int(g[0]["mono_respuesta_ms"]), 60000)
        # 418: pausa la ventana (o 300 s) y TERMINA la tanda marcandola, sin insistir
        n0 = len(self.srv.peticiones)
        self.srv.pasos = [(418, {"Retry-After": "120"}, b"{}", 0), (200, {}, ok, 0), (200, {}, ok, 0)]
        s3 = sonda_m.Sonda(self.base, self.dir, presupuesto.Presupuesto(), reloj_ms=self.ahora, dormir_ms=self.dormir,
                           timeout_s=0.5, permitir_local=True, nombre_csv="c.csv")
        t0 = self.ahora()
        e3 = s3.correr(["BTCUSDT", "ETHUSDT"], max_reintentos=3, max_intentos=10, max_duracion_ms=10 * 60000)
        s3.cerrar()
        self.assertEqual(len(self.srv.peticiones) - n0, 1)
        self.assertEqual(e3["terminacion"], "terminada_418")
        self.assertGreaterEqual(self.ahora() - t0, 120000)
        # limite finito de intentos: termina incompleta, no persigue fallos
        self.srv.pasos = [(500, {}, b"", 0)] * 5
        s4 = sonda_m.Sonda(self.base, self.dir, presupuesto.Presupuesto(), reloj_ms=self.ahora, dormir_ms=self.dormir,
                           timeout_s=0.5, permitir_local=True, nombre_csv="d.csv")
        e4 = s4.correr(["BTCUSDT"], max_reintentos=10, max_intentos=3, max_duracion_ms=10 * 60000)
        s4.cerrar()
        self.assertEqual(len(self.filas(s4)), 3)
        self.assertEqual(e4["terminacion"], "incompleta_limite_intentos")


if __name__ == "__main__":
    unittest.main()
