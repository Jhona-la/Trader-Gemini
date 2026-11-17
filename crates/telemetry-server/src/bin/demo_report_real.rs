//! CLI del SEGUNDO lector: informe por operación con lo que el motor escribe hoy.
//!
//! Uso: `demo_report_real --log <run.log>`
//! Sólo LEE el log de la corrida; no toca el motor, no lee `.env`, no llama al
//! exchange, no escribe en `models/`, `config_dir/` ni `data/demo/`.
//! Las horas de líneas sin `[ts=]` se prestan de la línea con hora más cercana
//! y se declaran en la salida; nunca se fabrican tiempos ni latencias.

fn main() {
    telemetry_server::report_demo_real::cli_main_real();
}
