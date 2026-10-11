"""Herramientas del plan de Línea C (docs/PLAN_LINEA_C_DESDE_LA_BASE_2026-10-11.md).

`citas`: resuelve cada cita `ruta:línea` de un documento contra los archivos
versionados de una revisión de Git y comprueba que la línea (o el rango)
exista. No juzga si la línea dice lo que el plan afirma: eso es trabajo del
recibo por archivo. Código 1 si alguna cita apunta a un archivo inexistente
o a una línea fuera del archivo. Las citas con nombre ambiguo (p. ej.
`lib.rs:120` sin directorio ni alias) se informan pero no fallan.

`asignacion`: da a cada archivo versionado su única fase dueña (C-F0…C-F11)
según las listas «Archivos» del plan (REGLAS). Gana la regla más específica:
ruta exacta, luego patrón con `*`, luego el prefijo de directorio más largo.
Los archivos que el plan no nombra caen en la fase de su directorio más
cercano (tipo «directorio»), como dice la regla de asignación del plan, y
eso se anota en el recibo. Escribe un TSV (ruta, fase, tipo, regla,
recibo). Código 1 si algún archivo queda sin fase o si una ruta exacta del
plan no existe en la revisión (nombre equivocado en el plan).

Uso:
    python3 scripts/plan_linea_c.py citas DOCUMENTO [--rev REV] [--alias CORTO=RUTA ...]
    python3 scripts/plan_linea_c.py asignacion [--rev REV] [--salida TSV]
"""
import argparse
import fnmatch
import re
import subprocess
import sys

EXTENSIONES = "rs|md|toml|yml|yaml|bat|ps1|py|json|sh|txt"
CITA = re.compile(
    r"(?P<ruta>\.?[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:%s)):(?P<ini>\d+)(?:-(?P<fin>\d+))?" % EXTENSIONES
)
# `:1235` suelto tras una cita en la misma línea se refiere al mismo archivo.
# Un `:9091` entre comillas invertidas es código (un puerto), no una cita.
CONTINUACION = re.compile(r"(?<![\w.\])}`]):(?P<ini>\d+)(?:-(?P<fin>\d+))?")


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                          text=True, check=True).stdout


def archivos_versionados(repo, rev):
    return git(repo, "ls-tree", "-r", "--name-only", rev).splitlines()


def resolver(ruta, versionados, alias=None):
    """Devuelve (estado, ruta_resuelta): exacta, sufijo único, ambigua o inexistente.

    `alias` traduce un nombre corto que el documento declara (p. ej. `lib.rs`
    sin ruta = el del núcleo) antes de buscarlo.
    """
    if ruta.startswith("./"):
        ruta = ruta[2:]
    ruta = (alias or {}).get(ruta, ruta)
    if ruta in versionados:
        return "ok", ruta
    candidatos = [v for v in versionados if v.endswith("/" + ruta)]
    if len(candidatos) == 1:
        return "ok", candidatos[0]
    if len(candidatos) > 1:
        return "ambigua", None
    return "inexistente", None


def extraer_citas(texto):
    """Lista de (n_linea_doc, ruta, ini, fin) con las continuaciones resueltas."""
    citas = []
    for n, linea in enumerate(texto.splitlines(), start=1):
        ocupado = []
        ultima = None
        eventos = []
        for m in CITA.finditer(linea):
            ocupado.append((m.start(), m.end()))
            eventos.append((m.start(), m.group("ruta"), m.group("ini"), m.group("fin")))
        for m in CONTINUACION.finditer(linea):
            if any(a <= m.start() < b for a, b in ocupado):
                continue
            eventos.append((m.start(), None, m.group("ini"), m.group("fin")))
        for _, ruta, ini, fin in sorted(eventos, key=lambda e: e[0]):
            if ruta is not None:
                ultima = ruta
            elif ultima is None:
                continue
            citas.append((n, ruta or ultima, int(ini), int(fin) if fin else int(ini)))
    return citas


def comprobar(repo, rev, texto, alias=None):
    versionados = archivos_versionados(repo, rev)
    largos = {}
    problemas = []
    resumen = {"ok": 0, "ambigua": 0, "inexistente": 0, "fuera_de_rango": 0}
    for n, ruta, ini, fin in extraer_citas(texto):
        estado, real = resolver(ruta, versionados, alias)
        if estado != "ok":
            resumen[estado] += 1
            problemas.append((n, estado, "%s:%d" % (ruta, ini)))
            continue
        if real not in largos:
            largos[real] = len(git(repo, "show", "%s:%s" % (rev, real)).splitlines())
        if ini < 1 or fin < ini or fin > largos[real]:
            resumen["fuera_de_rango"] += 1
            problemas.append((n, "fuera_de_rango", "%s:%d-%d (tiene %d)" % (real, ini, fin, largos[real])))
            continue
        resumen["ok"] += 1
    return resumen, problemas


def expandir(patron):
    """`a/{x.rs, y.rs}` -> [`a/x.rs`, `a/y.rs`] (un solo nivel de llaves)."""
    m = re.search(r"\{([^{}]*)\}", patron)
    if not m:
        return [patron]
    return [patron[:m.start()] + alt.strip() + patron[m.end():] for alt in m.group(1).split(",")]


# Listas «Archivos» del plan, fase por fase. Una entrada que acaba en `/` es
# un directorio; con `*`, un patrón; si no, una ruta exacta.
REGLAS = {
    "C-F0": [".agents/", ".agent/", "AGENTS.md", "CLAUDE.md", "TABLERO.md",
             "COORDINACION_CODEX_2026-09-28.md", "docs/adr/"],
    "C-F1": [
        "crates/quantum-arena/src/{state.rs, config.rs, horizon_policy.rs, symbol_registry.rs, symbols.rs, feed_health.rs, paths.rs, atomic_float.rs, lib.rs}",
        "crates/omniscient-registry/src/lib.rs",
        "crates/strategy-core/src/{lib.rs, types.rs}",
        "crates/execution-engine/src/{binance_api.rs, order_types.rs, lib.rs}",
        "crates/{storage-engine, data-pipeline, signal-engine, feature-engine}/src/lib.rs",
        "src/{config.rs, env_manager.rs, lib.rs, risk/mod.rs}",
        "crates/risk-engine/src/veto_registry.rs",
        "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/config.toml", "template.env",
        "crates/*/Cargo.toml",
    ],
    "C-F2": [
        "crates/risk-engine/src/{cramer_lundberg.rs, random_matrix.rs, hodge.rs}",
        "crates/risk-engine/src/{evidence.rs, selection_stats.rs, ville_e_process.rs}",
        "crates/quantum-arena/src/{evalues.rs, adaptive_quantiles.rs}",
        "crates/god-engine-core/src/{calibration.rs, conformal.rs, diffusion.rs}",
        "crates/feature-engine/src/{welford.rs, ewma.rs, correlation.rs, normalizer.rs}",
        "crates/quantum-arena/src/{temporal_spectrum.rs, spectral_tape.rs, espectral_multiactivo.rs, spectral_regime.rs}",
        "crates/god-engine-core/src/math_kernels.rs",
        "crates/feature-engine/src/{hurst_dfa.rs, multifractal.rs, spectral.rs, navier_stokes.rs, hodge_flow.rs, kalman.rs, tensor_ring.rs, shannon_entropy.rs, path_signatures.rs}",
        "crates/strategy-core/src/yang_mills_gauge.rs",
        "crates/signal-engine/src/hawkes_bessel.rs",
        "crates/feature-engine/src/{hawkes.rs, hawkes_cross.rs, lead_lag.rs, transfer_entropy.rs, copulas.rs}",
        "crates/metacortex-engine/src/prospect_theory.rs",
        "crates/feature-engine/src/microstructure.rs",
        "crates/signal-engine/src/{feynman_propagator.rs, navier_stokes_manifold.rs}",
        "crates/feature-engine/src/{quantum_tensor_store.rs, simd_neural_network.rs}",
    ],
    "C-F3": [
        "crates/god-engine-core/src/lib.rs",
        "crates/god-engine-core/src/{ml_inference.rs, entes_consejo.rs, evidence_publication.rs, contagion_publisher.rs}",
        "crates/metacortex-engine/src/consejo_seniors.rs",
        "crates/signal-engine/src/{orchestrator.rs, voto_espectral.rs, skill_motores.rs, flow_impulse.rs, flow_excitation_confluence.rs, quantum_oscillator.rs, soliton_wave.rs, supersonic_shockwave.rs, stochastic_resonance.rs, game_theoretic_nash.rs, conformal_reversion_filter.rs, trend_runner.rs, renyi_tsallis_entropy.rs, coaxial_breakout.rs, perceptron_gate.rs, proyeccion_espectral.rs, contagion_modulator.rs}",
        "crates/strategy-core/src/{stat_arb.rs, vecm_arbitrage.rs, maker.rs, momentum_booster.rs, strategy_telemetry.rs, multivariate_coint.rs}",
        "crates/dark-alpha-engine/src/lib.rs",
        "crates/risk-engine/src/regime.rs",
        "src/trailing.rs", "src/features/", "src/multi_asset_orchestrator.rs",
    ],
    "C-F4": [
        "crates/risk-engine/src/{lib.rs, tp_sl.rs, correlation_guard.rs, kelly_envelope.rs, leverage_matrix.rs, drawdown.rs, ruin.rs, capital_regime.rs, kelly.rs, orchestrator.rs, copulas_store.rs, guard.rs, capital_compounder.rs, epigenetic_capital_alloc.rs, epigenetic_fitness_landscape.rs}",
        "crates/god-engine-core/src/{entry_reservation.rs, quantum_kelly_risk.rs}",
        "crates/quantum-arena/src/position.rs",
        "crates/audit-engine/src/{drift_auditor.rs, shadow_ledger.rs}",
        "config_dir/copulas_manifest.json",
    ],
    "C-F5": [
        "src/bin/god_engine.rs",
        "crates/execution-engine/src/{executor.rs, reconciliation.rs, user_data_stream.rs, ioc_evidence.rs, order_registry.rs, entry_dispatch.rs, trade_accounting.rs, income_evidence.rs, client.rs, execution_evidence.rs, ntp.rs, ws_executor.rs, simulator.rs, shadow.rs, router.rs, quantum_multiplexer.rs, quantum_socket.rs, hot_swap.rs}",
        "crates/risk-engine/src/envio.rs",
        "crates/god-engine-core/src/{trailing.rs, reality_physics.rs, slippage_predictor.rs, latency_accelerator.rs, order_flow_aggregator.rs}",
        "crates/quantum-arena/src/protection_health.rs",
        "crates/backtest-engine/src/network_jitter.rs",
    ],
    "C-F6": [
        "crates/god-engine-core/src/{stateful_engine.rs, bootloader.rs, liquidation_feed.rs}",
        "crates/quantum-arena/src/{tick_source.rs, active_universe.rs, symbol_ranker_engine.rs}",
        "src/{parsers.rs, symbol_manager.rs, orderbook.rs}",
        "src/bin/{binance_vision_sync.rs, download_history.rs, oi_history_sync.rs, macro_history_sync.rs, parquet_to_bin.rs}",
        "crates/data-ingest/src/",
        "crates/data-pipeline/src/{omni_multiplexer.rs, dynamic_ranker.rs, spot_feed.rs, ws_client.rs, parser.rs, validation.rs, state_db.rs, historical.rs, macro_data.rs, market_context.rs, multiplexer.rs, resilient_stream.rs, api_client.rs, asset_selector.rs, macro_feed.rs, onchain_feed.rs, feature_vm.rs, persistence.rs}",
        "crates/storage-engine/src/{history_store.rs, raw_zone.rs}",
        "crates/feature-engine/src/omni_strategies.rs",
        "crates/execution-engine/src/dynamic_symbols.rs",
        "crates/backtest-engine/src/{label_evidence.rs, tick_replayer.rs}",
        "data/oihist/",
    ],
    "C-F7": [
        "crates/god-engine-core/src/{ensemble.rs, model_reload.rs, outcome_context.rs, recarga_genoma.rs, darwin.rs, fitness_contract.rs}",
        "crates/quantum-arena/src/{genome.rs, genome_store.rs}",
        "src/bin/{train_forest.rs, auto_trainer_daemon.rs, evolver.rs, walkforward_evolver.rs, evolution.rs, backtest_windows.rs, train_dark_alpha.rs}",
        "src/model_manifest.rs",
        "crates/metacortex-engine/src/{online_learning.rs, compiler_sandbox.rs, evolutionary_templates.rs, hot_swap_controller.rs, fases_autonomous.rs, reminiscence_and_adn.rs, cazador_constantes.rs, epigenoma_store.rs}",
        "crates/dark-alpha-engine/src/{online_ppo.rs, neuro_plasticity.rs}",
        "crates/evolution-engine/src/",
        "crates/backtest-engine/src/{lib.rs, booktick_replay.rs, vectorized.rs, oos_context.rs, metrics.rs}",
        "crates/storage-engine/src/evolution_ledger.rs",
        "config_dir/genomes/", "config_dir/genotypes/", "config_dir/models_manifest.json", "models/",
        "scripts/revalidate_roster.ps1",
    ],
    "C-F8": [
        "crates/god-engine-core/src/{orchestrator.rs, direction_diag.rs, ml_coverage.rs}",
        "crates/quantum-arena/src/state_continuity.rs",
        "src/{dashboard.rs, bin/dashboard.rs, dark_alpha_router.rs, dark_alpha_sniffer.rs, quantum_arena.rs}",
        "crates/storage-engine/src/{mmap_bus.rs, lakehouse.rs, ledger.rs, temporal_store.rs}",
        "crates/os-guardian/src/",
        "crates/data-pipeline/src/{teleonomia.rs, telemetry_bus.rs, lakehouse_mmap.rs, storage.rs}",
        "crates/metacortex-engine/src/{shadow_graph_auditor.rs, immune_system.rs}",
        "crates/audit-engine/src/{trajectory_auditor.rs, telemetry.rs, auditor_interno.rs, behavioral_auditor.rs, cybernetic_resilience.rs, resilience.rs, state_validator.rs}",
        "crates/telemetry-server/src/", "crates/telemetry-engine/src/", "crates/flight-recorder/src/lib.rs",
        "deployment/", "docker-compose.yml", "Dockerfile", ".dockerignore",
        "static/", "dashboard/",
        "scripts/monitor_demo.ps1",
    ],
    "C-F9": [
        "crates/god-engine-core/src/{ml_registry.rs, bin/test_ml.rs, bin/nano_benchmark.rs}",
        "src/bin/{audit_forensic_backtest.rs, multi_coin_simulator.rs, feature_exporter.rs, spectral_bench.rs, tape_spread_fix.rs, feature_validator.rs, votes_export.rs, income_report.rs, system_health.rs, latency_flow_audit.rs, ml_path_probe.rs, oco_probe.rs, algo_fetch_probe.rs, test_forest.rs, test_forest2.rs, benchmark.rs, config_compiler.rs, graph_server.rs, copulas_manifest.rs, model_manifest.rs}",
        "crates/quantum-arena/src/{arena_alloc.rs, ring_buffer.rs, zero_alloc_pool.rs, zero_copy_stream.rs, net_multiplexer.rs}",
        "crates/data-pipeline/src/{chaos_monkey.rs, bypass.rs}",
        "crates/metacortex-engine/src/lib.rs",
        "crates/audit-engine/src/{lib.rs, main.rs}",
        "crates/backtest-engine/src/bin/continuous_evolution_backtest.rs",
        "crates/graph-architecture/", "crates/graph-4d/", "crates/phase-runner/",
        "crates/{graph-architecture, graph-4d, phase-runner}/Cargo.toml",
        "scripts/", "launchers/", "LAUNCHER.bat", "LAUNCH_*.bat", "LAUNCH_GOD_MODE.ps1",
        "build.ps1", "build.sh", "build_loop.ps1", "run_god_engine.ps1", "run_sim_5.ps1",
    ],
    "C-F10": [
        "crates/*/tests/*",
        "crates/omniscient-registry/src/tests.rs",
        "crates/backtest-engine/src/booktick_causality_contract.rs",
        "crates/audit-engine/src/forensics.rs",
        ".github/workflows/{replay-contracts.yml, unit-suites.yml}",
        ".githooks/",
        "scripts/{quality_gate.ps1, t1_oraculo.sh}",
    ],
    "C-F11": [
        "docs/", "graphify-out/", "artifacts/",
        "ARCHITECTURE.md", "ATLAS_ANALITICO.md", "CERTIFICADO_SISTEMA_v1.md",
        "FORENSIC_INTELLIGENCE_AUDIT.md", "INFORME_*.md",
        "PLAN_REPARACION_EVOLUCION_SENIOR.md", "README.md", "RESULTADO_CERTIFICACION_R2_30D.md",
        "STRATEGIES.md", "WALKTHROUGH_FORENSE_SISTEMICO.md", "implementation_plan_fee_slayer.md",
        "summary_table.md", "task.md", "walkthrough.md",
        "equity_curve.csv", "graph_3d_lite.html", "graph_3d_topology.html",
        "rustc-ice-2026-09-07T14_42_23-32692.txt",
        ".agignore", ".gitattributes", ".gitignore", ".freebuff/project-id",
    ],
}

# Regla del plan para lo que sus listas no nombran: la fase del directorio
# más cercano. Estas son las fases por defecto de cada directorio.
POR_DIRECTORIO = {
    "crates/god-engine-core/": "C-F3", "crates/signal-engine/": "C-F3",
    "crates/strategy-core/": "C-F3", "crates/quantum-arena/": "C-F1",
    "crates/omniscient-registry/": "C-F1", "crates/feature-engine/": "C-F2",
    "crates/risk-engine/": "C-F4", "crates/execution-engine/": "C-F5",
    "crates/data-ingest/": "C-F6", "crates/data-pipeline/": "C-F6",
    "crates/metacortex-engine/": "C-F7", "crates/dark-alpha-engine/": "C-F7",
    "crates/evolution-engine/": "C-F7", "crates/backtest-engine/": "C-F7",
    "crates/storage-engine/": "C-F8", "crates/audit-engine/": "C-F8",
    "crates/os-guardian/": "C-F8", "crates/telemetry-server/": "C-F8",
    "crates/telemetry-engine/": "C-F8", "crates/flight-recorder/": "C-F8",
    "src/bin/": "C-F9", "src/": "C-F3", "config_dir/": "C-F7", "data/": "C-F6",
    ".github/": "C-F10", ".cargo/": "C-F1", ".freebuff/": "C-F11",
}


def reglas_planas():
    planas = []
    for fase, patrones in REGLAS.items():
        for patron in patrones:
            for regla in expandir(patron):
                planas.append((fase, regla, "explícita"))
    for prefijo, fase in POR_DIRECTORIO.items():
        planas.append((fase, prefijo, "directorio"))
    return planas


def prioridad(regla, tipo):
    """Mayor es más específica: exacta > patrón > prefijo largo > respaldo."""
    if tipo == "directorio":
        return (0, len(regla))
    if regla.endswith("/"):
        return (1, len(regla))
    if "*" in regla:
        return (2, len(regla))
    return (3, len(regla))


def coincide(regla, ruta):
    if regla.endswith("/"):
        return ruta.startswith(regla)
    if "*" in regla:
        return fnmatch.fnmatchcase(ruta, regla)
    return ruta == regla


def asignar(versionados):
    """Devuelve (filas, sin_fase, exactas_inexistentes, empates)."""
    planas = reglas_planas()
    filas, sin_fase, empates = [], [], []
    for ruta in versionados:
        candidatas = [(prioridad(r, t), f, r, t) for f, r, t in planas if coincide(r, ruta)]
        if not candidatas:
            sin_fase.append(ruta)
            continue
        candidatas.sort(key=lambda c: c[0], reverse=True)
        mejor = candidatas[0]
        rivales = {c[1] for c in candidatas if c[0] == mejor[0]}
        if len(rivales) > 1:
            empates.append((ruta, sorted(rivales)))
        filas.append((ruta, mejor[1], mejor[3], mejor[2]))
    conjunto = set(versionados)
    exactas_inexistentes = sorted(
        (f, r) for f, r, t in planas
        if t == "explícita" and not r.endswith("/") and "*" not in r and r not in conjunto)
    return filas, sin_fase, exactas_inexistentes, empates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="orden", required=True)
    citas = sub.add_parser("citas")
    citas.add_argument("documento")
    citas.add_argument("--rev", default="HEAD")
    citas.add_argument("--repo", default=".")
    citas.add_argument("--alias", action="append", default=[], metavar="CORTO=RUTA",
                       help="nombre corto que el documento declara, p. ej. lib.rs=crates/x/src/lib.rs")
    asig = sub.add_parser("asignacion")
    asig.add_argument("--rev", default="HEAD")
    asig.add_argument("--repo", default=".")
    asig.add_argument("--salida", help="ruta del TSV (si falta, sólo el resumen)")
    args = parser.parse_args(argv)
    if args.orden == "asignacion":
        return orden_asignacion(args)
    alias = dict(a.split("=", 1) for a in args.alias)
    with open(args.documento, encoding="utf-8") as f:
        texto = f.read()
    resumen, problemas = comprobar(args.repo, args.rev, texto, alias)
    for n, estado, cita in problemas:
        print("%s:%d: %s %s" % (args.documento, n, estado, cita))
    print("citas: %s" % ", ".join("%s=%d" % kv for kv in resumen.items()))
    return 1 if resumen["inexistente"] or resumen["fuera_de_rango"] else 0


def orden_asignacion(args):
    versionados = archivos_versionados(args.repo, args.rev)
    filas, sin_fase, inexistentes, empates = asignar(versionados)
    if args.salida:
        with open(args.salida, "w", encoding="utf-8") as f:
            f.write("ruta\tfase\ttipo\tregla\trecibo\n")
            for ruta, fase, tipo, regla in filas:
                f.write("%s\t%s\t%s\t%s\tpendiente\n" % (ruta, fase, tipo, regla))
    for ruta in sin_fase:
        print("sin fase: %s" % ruta)
    for fase, regla in inexistentes:
        print("ruta del plan inexistente en %s: %s (%s)" % (args.rev, regla, fase))
    for ruta, fases in empates:
        print("empate: %s %s" % (ruta, fases))
    por_fase = {}
    for _, fase, tipo, _ in filas:
        por_fase.setdefault(fase, [0, 0])[0 if tipo == "explícita" else 1] += 1
    orden = sorted(por_fase, key=lambda f: int(f.split("F")[1]))
    print("archivos: %d; por fase (explícita+directorio): %s" % (
        len(filas), ", ".join("%s=%d+%d" % (f, *por_fase[f]) for f in orden)))
    return 1 if sin_fase or inexistentes or empates else 0


if __name__ == "__main__":
    sys.exit(main())
