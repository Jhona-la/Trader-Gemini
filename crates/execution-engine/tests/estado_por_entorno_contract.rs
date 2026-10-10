//! CL-48: el estado aprendido y los diarios viven por entorno
//! (`data/{demo|prod}/`). Antes la envolvente Kelly, el fee-breaker, el
//! diario de posiciones y el de fills usaban rutas fijas en `data/`, comunes
//! a demo, testnet y producción: una sesión de testnet sembraba el posterior
//! Kelly real, suspendía símbolos reales y prestaba su τ y su precio de
//! entrada a la adopción y a los cierres de bracket reales.

/// Fuente sin comentarios de línea completa.
fn codigo(src: &str) -> String {
    src.lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

const ARCHIVOS: [&str; 4] = [
    "kelly_envelope.json",
    "fee_breaker.json",
    "position_journal.jsonl",
    "trade_fills.jsonl",
];

#[test]
fn cl48_ningun_estado_aprendido_usa_una_ruta_comun_a_los_entornos() {
    for (nombre, src) in [
        ("god_engine.rs", codigo(include_str!("../../../src/bin/god_engine.rs"))),
        ("trade_accounting.rs", codigo(include_str!("../src/trade_accounting.rs"))),
    ] {
        for archivo in ARCHIVOS {
            assert!(
                !src.contains(&format!("\"data/{archivo}")),
                "{nombre}: {archivo} con ruta fija en data/"
            );
        }
    }
}

#[test]
fn cl48_host_y_ejecutor_leen_el_mismo_diario_de_posiciones() {
    let host = codigo(include_str!("../../../src/bin/god_engine.rs"));
    let ejecutor = codigo(include_str!("../src/trade_accounting.rs"));
    let ruta = "quantum_arena::paths::env_data_path(\"position_journal.jsonl\")";
    assert!(host.matches(ruta).count() >= 3, "host: lectura, compactación y escritura");
    assert!(ejecutor.contains(ruta), "ejecutor: respaldo del precio de entrada");
    for archivo in ["kelly_envelope.json", "fee_breaker.json"] {
        assert!(
            host.contains(&format!("quantum_arena::paths::env_data_path(\"{archivo}\")")),
            "{archivo} por entorno"
        );
    }
    assert!(ejecutor.contains("quantum_arena::paths::env_data_path(\"trade_fills.jsonl\")"));
}
