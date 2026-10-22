"""Contrato del emisor económico del informe integral (GLM 144, §52 del mandato).

El worker de telemetría del audit-engine vuelca outputs/economia_*.json con
ROI pre/post fees, PnL bruto/neto, WR pre/post costos, Sharpe y EV/trade.
Este contrato verifica que el emisor:
  - lee el recibo más reciente y expone TODOS los campos obligatorios,
  - degrada con honestidad (ausencia explícita) cuando no hay recibo,
  - renderiza la sección del informe con las métricas y sin inventar null→0.
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODULO = REPO / "scripts" / "informe_integral.py"

spec = importlib.util.spec_from_file_location("informe_integral", str(MODULO))
informe = importlib.util.module_from_spec(spec)
sys.modules["informe_integral"] = informe
spec.loader.exec_module(informe)

RECIBO_EJEMPLO = {
    "trades": 30,
    "pnl_bruto": 12.5,
    "fees": 3.0,
    "pnl_neto": 9.5,
    "ganadoras_brutas": 18,
    "ganadoras_netas": 16,
    "wr_pre_costos": 0.6,
    "wr_post_costos": 0.5333333333333333,
    "capital_inicial": 240.5,
    "roi_pre_fees_pct": 5.197505197505198,
    "roi_post_fees_pct": 3.95010395010395,
    "ev_por_trade": 0.31666666666666665,
    "sharpe_post_fees": 1.25,
    "periodo_ns": [100, 200],
    "actualizado_ns": 200,
}


class EmisorEconomicoContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="informe_econ_")
        self.dir_outputs = Path(self.tmp.name).resolve() / "outputs"
        self.dir_outputs.mkdir(parents=True, exist_ok=True)
        self.original_root = informe.ROOT
        informe.ROOT = self.tmp.name

    def tearDown(self):
        informe.ROOT = self.original_root
        self.tmp.cleanup()

    def _recibo(self) -> Path:
        """Ruta del recibo CONFINADA al directorio temporal del test."""
        ruta = (self.dir_outputs / "economia_live.json").resolve()
        if self.dir_outputs.resolve() not in ruta.parents:
            raise ValueError("ruta fuera del sandbox del test")
        return ruta

    def test_lee_el_recibo_mas_reciente_con_todos_los_campos(self):
        self._recibo().write_text(json.dumps(RECIBO_EJEMPLO), encoding="utf-8")
        dato, err = informe.emisor_economico()
        self.assertIsNone(err)
        for campo in ("trades", "pnl_bruto", "fees", "pnl_neto", "wr_pre_costos",
                      "wr_post_costos", "roi_pre_fees_pct", "roi_post_fees_pct",
                      "ev_por_trade", "sharpe_post_fees", "periodo_ns"):
            self.assertIn(campo, dato, f"§52 exige {campo}")
        self.assertEqual(dato["pnl_neto"], 9.5)

    def test_ausencia_honesta_sin_recibo(self):
        dato, err = informe.emisor_economico()
        self.assertIsNone(dato)
        self.assertIn("sin outputs/economia_*.json", err)

    def test_render_incluye_las_metricas_y_no_inventa_nulos(self):
        texto = informe.generar("test", None)
        self.assertIn("## 4. Económico", texto)
        self.assertIn("sin datos:", texto)  # sin recibo en tmp ⇒ ausencia explícita
        # Con recibo: el render muestra las métricas reales y null → "—"
        recibo = dict(RECIBO_EJEMPLO)
        recibo["sharpe_post_fees"] = None  # ausencia: std≈0 con 1 muestra
        self._recibo().write_text(json.dumps(recibo), encoding="utf-8")
        texto2 = informe.generar("test", None)
        self.assertIn("PnL bruto: **12.5 USD**", texto2)
        self.assertIn("ROI pre-fees: **5.197505197505198%**", texto2)
        self.assertIn("Sharpe post-fees", texto2)
        self.assertIn("**—**", texto2, "null se muestra como ausencia, no como 0")




RECIBO_ORTOGONALIDAD = """=============== ORTOGONALIDAD DE VOCES (GLM 146) ===============
tape: BTCUSDT_TEST.bin | ticks: 2000000 | muestras: 500000 | fuentes: 19 | pares observados: 78
core bien-observado (pares >=100, todos-contra-todos): 11 de 19 fuentes
casos completos del core (todas las fuentes hablaron): 14732 de 500000 muestras
espectro propio casos-completos (descendente, 11):
borde Marchenko-Pastur (N=11, T=14732): 1.055397
★ APUESTAS EFECTIVAS INDEPENDIENTES: 2.828 de 11 fuentes
--- dendrograma co-disparo >=90% del máx (n>=449954): 1 clúster(es) ---
  clúster [...] casos completos: 499920
    espectro: [3.346 1.226 1.000 0.319 0.094 0.014] · borde MP 1.0069
    ★ APUESTAS EFECTIVAS: 1.886 de 6 fuentes
--- max |corr| por fuente (core, por pares) ---
  RenyiTsallisEntropyEngine      max|corr| 0.9748
  StochasticResonanceEngine      max|corr| 0.9748
"""


class EmisorOrtogonalidadContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="informe_orto_")
        self.original_root = informe.ROOT
        informe.ROOT = self.tmp.name
        self.dir_out = Path(self.tmp.name) / "outputs"
        self.dir_out.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        informe.ROOT = self.original_root
        self.tmp.cleanup()

    def _recibo(self) -> Path:
        ruta = (self.dir_out / "ortogonalidad_voces_X.log").resolve()
        if self.dir_out.resolve() not in ruta.parents:
            raise ValueError("fuera del sandbox")
        return ruta

    def test_parsea_apuestas_por_cluster_y_global(self):
        self._recibo().write_text(RECIBO_ORTOGONALIDAD, encoding="utf-8")
        dato, err = informe.emisor_ortogonalidad()
        self.assertIsNone(err)
        self.assertEqual(dato["tape"], "BTCUSDT_TEST.bin")
        self.assertEqual(dato["fuentes"], 19)
        self.assertEqual(dato["core_completos"], 14732)
        self.assertAlmostEqual(dato["bets_global"], 2.828)
        self.assertAlmostEqual(dato["bets_cluster"], 1.886)
        self.assertEqual(dato["bets_cluster_de"], 6)
        self.assertEqual(dato["clones_top"][0][1], 0.9748)
        self.assertEqual(len(dato["espectro_cluster"]), 6)

    def test_ausencia_honesta_sin_recibo(self):
        dato, err = informe.emisor_ortogonalidad()
        self.assertIsNone(dato)
        self.assertIn("correr el veh", err)

if __name__ == "__main__":
    unittest.main()
