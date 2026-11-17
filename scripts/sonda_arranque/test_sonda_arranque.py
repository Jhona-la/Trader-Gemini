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


if __name__ == "__main__":
    unittest.main()
