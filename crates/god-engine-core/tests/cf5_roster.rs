//! C-F5 · CL-123: el roster nombra los símbolos como el exchange, y el
//! manifiesto de cópulas usa los mismos nombres.
//!
//! El bootloader suscribía `shibusdt`, que no existe en futuros USDⓈ-M (el
//! contrato es `1000SHIBUSDT`, mil unidades por contrato): esa moneda no
//! recibía datos. El generador de cópulas tenía su propia copia del roster,
//! con el mismo nombre.

use god_engine_core::bootloader::{SystemDiagnostics, ROSTER};

/// Esperado: SHIB figura con su símbolo de futuros, y la fase 1-2 del
/// arranque devuelve exactamente el roster.
#[test]
fn cl123_el_roster_nombra_shib_como_el_exchange() {
    assert!(ROSTER.contains(&"1000SHIBUSDT"));
    assert!(!ROSTER.contains(&"SHIBUSDT"));
    assert!(ROSTER.iter().all(|s| s.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())));
    let rt = tokio::runtime::Builder::new_current_thread().build().expect("runtime");
    let (_, syms) = rt
        .block_on(SystemDiagnostics::execute_phase_1_and_2(true))
        .expect("roster");
    assert_eq!(syms, ROSTER.iter().map(|s| s.to_string()).collect::<Vec<_>>());
}

/// Esperado: toda moneda del manifiesto de cópulas vivo es del roster (si no,
/// su par nunca se aplica).
#[test]
fn cl123_las_monedas_del_manifiesto_de_copulas_son_del_roster() {
    let manifiesto = include_str!("../../../config_dir/copulas_manifest.json");
    let mut fuera = Vec::new();
    for linea in manifiesto.lines().filter(|l| l.contains("\"a\":")) {
        for clave in ["\"a\": \"", "\"b\": \""] {
            let Some(i) = linea.find(clave) else { continue };
            let resto = &linea[i + clave.len()..];
            let simbolo = &resto[..resto.find('"').unwrap_or(0)];
            if !ROSTER.contains(&simbolo) {
                fuera.push(simbolo.to_string());
            }
        }
    }
    fuera.sort();
    fuera.dedup();
    assert!(fuera.is_empty(), "monedas del manifiesto fuera del roster: {fuera:?}");
}
