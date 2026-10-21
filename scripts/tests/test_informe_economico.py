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


if __name__ == "__main__":
    unittest.main()
