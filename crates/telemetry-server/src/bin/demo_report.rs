//! CLI del informe por operación de una corrida Demo (SOL M1, ítem 2).
//!
//! Uso: `demo_report --log <run.log> --income <income.jsonl> [--funnel <gate.json>]`
//! Un log sin marcas `[ts=...]` aborta con código 2 (no se juzga la Demo con
//! tiempos fabricados).

fn main() {
    telemetry_server::report_demo_run::cli_main();
}
