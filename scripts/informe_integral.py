#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Informe integral del sistema (GLM 141 — mandato del operador).

Teje las fuentes de métricas existentes en UN informe periódico con
serie histórica comparable. Diseño: docs/DISENO_META_ANALITICA_INTEGRAL.md.

Emisores:
  1. meta-analítica de código  — docs/audit/meta_analitica/*.tsv (por agente/área)
  2. oráculos con testigo      — outputs/oracle_witness_*.log (gate de conducta)
  3. censo de voces            — recibo del test #[ignore] censo_mudos (si existe output)
  4. económico (GLM 144, §52)  — outputs/economia_*.json volcado por el worker de
     telemetría del audit-engine: ROI pre/post fees, PnL bruto/neto, WR pre/post
     costos, Sharpe (Welford por trade), EV/trade y periodo exacto.

Uso: python scripts/informe_integral.py [--out docs/audit/informe_integral/]
Determinista: sólo lee archivos existentes; no ejecuta suites.
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def leer_tsv(ruta):
    """Lee un TSV con cabecera a lista de dicts."""
    with open(ruta, encoding="utf-8") as f:
        cab = f.readline().rstrip("\n").split("\t")
        return [dict(zip(cab, linea.rstrip("\n").split("\t"))) for linea in f if linea.strip()]


def emisor_meta_analitica():
    """(1) Impacto por cambio: commits/líneas/tests por agente."""
    ruta = os.path.join(ROOT, "docs", "audit", "meta_analitica", "libro_impacto.tsv")
    if not os.path.exists(ruta):
        return None, "libro_impacto.tsv ausente"
    filas = leer_tsv(ruta)
    if not filas:
        return None, "libro_impacto.tsv vacío"
    # Cabecera esperada: commit, fecha, autor, asunto, prod_mas, ... (flexible)
    por_autor = {}
    for f in filas:
        a = f.get("autor", f.get("agente", "?"))
        e = por_autor.setdefault(a, {"commits": 0})
        e["commits"] += 1
    total = len(filas)
    return {"total_commits": total, "por_autor": por_autor,
            "rango": (filas[0].get("fecha", "?"), filas[-1].get("fecha", "?"))}, None


def emisor_oraculos():
    """(2) Gates de conducta: parsea los oracle_witness_*.log."""
    ruta_glob = os.path.join(ROOT, "outputs", "oracle_witness_*.log")
    logs = sorted(glob.glob(ruta_glob))
    if not logs:
        return None, "sin oracle_witness_*.log"
    registros = []
    for ruta in logs:
        info = {"archivo": os.path.basename(ruta)}
        try:
            with open(ruta, encoding="utf-8", errors="replace") as f:
                for linea in f:
                    if linea.startswith("# HEAD:"):
                        info["head"] = linea.split(":", 1)[1].strip()[:8]
                    elif linea.startswith("test result:"):
                        m = re.search(r"(ok|FAILED)\. (\d+) passed; (\d+) failed", linea)
                        if m:
                            info["resultado"] = m.group(1)
                            info["pasados"] = int(m.group(2))
                            info["fallidos"] = int(m.group(3))
                    elif linea.startswith("duracion_s:"):
                        info["duracion_s"] = int(linea.split(":", 1)[1].strip())
                    elif linea.startswith("exit_code:"):
                        info["exit"] = int(linea.split(":", 1)[1].strip())
        except OSError:
            continue
        if "resultado" in info:
            registros.append(info)
    return {"total": len(registros), "registros": registros[-10:]}, None


def emisor_censo(recibo_ruta=None):
    """(3) Salud del ensamble: parsea un recibo del censo #[ignore] si
    el operador dejó el output (el test corre con --nocapture)."""
    candidatos = [recibo_ruta] if recibo_ruta else []
    # Convención futura: outputs/censo_voces_*.log
    candidatos += sorted(glob.glob(os.path.join(ROOT, "outputs", "censo_voces_*.log")))
    for ruta in candidatos:
        if ruta and os.path.exists(ruta):
            try:
                with open(ruta, encoding="utf-8", errors="replace") as f:
                    texto = f.read()
                m = re.search(r"hablantes \(tasa ≥1%\): (\d+) de (\d+)", texto)
                if m:
                    return {"hablantes": int(m.group(1)), "fuentes": int(m.group(2)),
                            "archivo": os.path.basename(ruta)}, None
            except OSError:
                pass
    return None, "sin recibo de censo (correr tests #[ignore] con --nocapture)"


def emisor_economico():
    """(4) Económico (§52 del mandato): lee el recibo outputs/economia_live.json
    que vuelca el worker de telemetría del audit-engine (ROI pre/post fees,
    PnL bruto/neto, WR pre/post costos, Sharpe, EV/trade, periodo exacto)."""
    candidatos = sorted(glob.glob(os.path.join(ROOT, "outputs", "economia_*.json")))
    if not candidatos:
        return None, "sin outputs/economia_*.json (worker de telemetría sin trades aún)"
    try:
        with open(candidatos[-1], encoding="utf-8") as f:
            return json.load(f), None
    except (OSError, json.JSONDecodeError) as e:
        return None, f"recibo ilegible: {e}"


def generar(fecha, previo_ruta):
    lineas = []
    w = lineas.append
    w(f"# Informe integral del sistema — {fecha}")
    w("")
    w("> GLM 141 · Diseño: `docs/DISENO_META_ANALITICA_INTEGRAL.md` · "
      "Serie histórica inmutable: los informes pasados no se regeneran.")
    w("")

    meta, err1 = emisor_meta_analitica()
    w("## 1. Meta-analítica de cambios (código)")
    if meta:
        w(f"- Commits en el libro: **{meta['total_commits']}** "
          f"(rango {meta['rango'][0][:10]} → {meta['rango'][1][:10]}).")
        w("- Por agente: " + ", ".join(
            f"{a}={e['commits']}" for a, e in
            sorted(meta["por_autor"].items(), key=lambda kv: -kv[1]["commits"])))
    else:
        w(f"- (sin datos: {err1})")
    w("")

    orac, err2 = emisor_oraculos()
    w("## 2. Gates de conducta (oráculos T-1 con testigo)")
    if orac:
        w(f"- Oráculos con recibo completo: **{orac['total']}**.")
        w("| fecha | head | resultado | pasados | fallidos | duración (s) |")
        w("|---|---|---|---|---|---|")
        for r in orac["registros"]:
            fecha_log = r["archivo"].replace("oracle_witness_", "").replace(".log", "")
            w(f"| {fecha_log[:15]} | {r.get('head','?')} | {r['resultado']} | "
              f"{r['pasados']} | {r['fallidos']} | {r.get('duracion_s','?')} |")
    else:
        w(f"- (sin datos: {err2})")
    w("")

    censo, err3 = emisor_censo()
    w("## 3. Salud del ensamble (censo de voces)")
    if censo:
        w(f"- Hablantes: **{censo['hablantes']}/{censo['fuentes']}** fuentes "
          f"(recibo: {censo['archivo']}).")
    else:
        w(f"- (sin datos: {err3})")
    w("")

    econ, err4 = emisor_economico()
    w("## 4. Económico (ROI/PnL/WR/Sharpe/EV con fees — §52 del mandato)")
    if econ:
        fmt = lambda v, suf="": "—" if v is None else f"{v}{suf}"
        w(f"- Trades del periodo: **{econ.get('trades', 0)}** · "
          f"periodo_ns: {econ.get('periodo_ns')}.")
        w(f"- PnL bruto: **{fmt(econ.get('pnl_bruto'))} USD** · fees: "
          f"{fmt(econ.get('fees'))} USD · **PnL neto: {fmt(econ.get('pnl_neto'))} USD**.")
        w(f"- ROI pre-fees: **{fmt(econ.get('roi_pre_fees_pct'), '%')}** · "
          f"ROI post-fees: **{fmt(econ.get('roi_post_fees_pct'), '%')}** · "
          f"capital inicial: {fmt(econ.get('capital_inicial'))} USD.")
        w(f"- WR pre-costos: **{fmt(econ.get('wr_pre_costos'))}** · "
          f"WR post-costos: **{fmt(econ.get('wr_post_costos'))}** · "
          f"EV/trade: {fmt(econ.get('ev_por_trade'))} USD.")
        w(f"- Sharpe post-fees (anualizado, Welford por trade): "
          f"**{fmt(econ.get('sharpe_post_fees'))}**.")
        w("- Ausencias serializadas como null (CL-15): lo no medible no se inventa.")
    else:
        w(f"- (sin datos: {err4})")
    w("")

    # 5. Comparativa con el informe anterior (la serie histórica)
    w("## 5. Comparativa con el informe anterior")
    if previo_ruta and os.path.exists(previo_ruta):
        try:
            previo = open(previo_ruta, encoding="utf-8").read()
            pm = re.search(r"Commits en el libro: \*\*(\d+)\*\*", previo)
            if pm and meta:
                delta = meta["total_commits"] - int(pm.group(1))
                w(f"- Δ commits desde el informe anterior: **{delta:+d}**.")
            po = re.search(r"recibo completo: \*\*(\d+)\*\*", previo)
            if po and orac:
                w(f"- Δ oráculos: **{orac['total'] - int(po.group(1)):+d}**.")
            w(f"- Informe anterior: `{os.path.basename(previo_ruta)}`.")
        except OSError:
            w("- (no se pudo leer el informe anterior)")
    else:
        w("- (primer informe de la serie — sin anterior)")
    w("")
    w("---")
    w("*Generado por `scripts/informe_integral.py` — determinista, sólo lectura "
      "de artifacts existentes. Emisores de telemetría-bin y sombras: átomos "
      "siguientes del diseño.*")
    return "\n".join(lineas) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Informe integral del sistema")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "docs", "audit", "informe_integral"), help="directorio de salida")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fecha = datetime.date.today().isoformat()
    salida = os.path.join(args.out, f"informe_{fecha}.md")
    # Informe anterior = el más reciente que NO sea el de hoy
    previos = sorted(glob.glob(os.path.join(args.out, "informe_*.md")))
    previos = [p for p in previos if os.path.basename(p) != f"informe_{fecha}.md"]
    previo = previos[-1] if previos else None
    texto = generar(fecha, previo)
    with open(salida, "w", encoding="utf-8", newline="\n") as f:
        f.write(texto)
    print(f"informe escrito: {salida}")
    if previo:
        print(f"comparado contra: {previo}")


if __name__ == "__main__":
    sys.exit(main())
