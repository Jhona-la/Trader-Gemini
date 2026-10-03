//! CL-41 — el host y el replay deciden el apalancamiento de envío con la
//! misma función, que nunca supera el validado por el riesgo. Guardia sobre
//! las fuentes: el ajuste D-382 vivía copiado en ambos con el doble conteo
//! de la reserva propia y sin techo.

fn compacta(src: &str) -> String {
    src.split_whitespace().collect()
}

#[test]
fn cl41_host_y_replay_usan_la_funcion_de_envio() {
    let host = compacta(include_str!("../../../src/bin/god_engine.rs"));
    let replay = compacta(include_str!("../../backtest-engine/src/booktick_replay.rs"));
    for (nombre, src) in [("host", &host), ("replay", &replay)] {
        assert!(
            src.contains("risk_engine::envio::apalancamiento_de_envio("),
            "{nombre}: el apalancamiento de envío sale de risk_engine::envio"
        );
        assert!(
            src.contains("risk_engine::envio::margen_libre_sin_la_propia("),
            "{nombre}: el margen libre no descuenta la reserva propia"
        );
        let copia = ["let", "needed_leverage"].concat();
        assert!(!src.contains(&copia), "{nombre}: ajuste D-382 copiado en línea");
    }
}
