//! C-F5c · R3, R5 y B2 de la revisión posterior del PR #62 (CL-132, CL-133):
//! el pico del capital se registra y se mide en un solo plano, es de un solo
//! venue, y un archivo ilegible no es un archivo ausente.
//!
//! R3. El riesgo registraba el pico compartido con el capital del arena y el
//! host medía la caída con el mínimo entre el arena y el exchange (X-013). En
//! papel sobre mainnet el exchange es el saldo real, que no se mueve: si el
//! papel gana de 13 a 14,04 USD, el pico sube a 14,04 y el host ve una caída
//! del 7,4 % del saldo real, que se guarda y vuelve tras reiniciar. Ahora los
//! dos registran y miden con `drawdown::capital_del_freno`: el más
//! conservador de los dos planos, publicado por el host en el arena.
//!
//! R5. Testnet y papel de mainnet guardan en `data/demo` (CL-61). Una sesión
//! de testnet con 5 000 USDT dejaba el pico en 5 000; la siguiente sesión de
//! papel con 13 USD reales lo restauraba, veía una caída del 99,7 % ≥ d* y se
//! bloqueaba. Ahora cada venue tiene su archivo (`drawdown::archivo_pico`).
//!
//! B2. Un `pico_capital.txt` ilegible (coma decimal, BOM, error de E/S)
//! contaba como ausente: perdonaba la caída sin aviso y el host lo
//! sobrescribía a los 5 s. Ahora `leer_pico` distingue ausente de ilegible;
//! con el pico ilegible el riesgo no admite entradas y el host no lo
//! sobrescribe, como con el estado aprendido (CL-108f).

use quantum_arena::GlobalArena;
use risk_engine::drawdown::{self, CargaPico};
use risk_engine::{RiskEngine, REJECT_NAMES};
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Mutex};

static ENTORNO: Mutex<()> = Mutex::new(());

fn sin_codigo(s: &str) -> String {
    s.split_whitespace().collect()
}

/// Esperado (R3): el host y el riesgo registran el pico y miden la caída con
/// el mismo capital.
#[test]
fn cl132_host_y_riesgo_registran_el_pico_en_el_mismo_plano() {
    let host = sin_codigo(include_str!("../../../src/bin/god_engine.rs"));
    let riesgo = sin_codigo(include_str!("../src/lib.rs"));
    assert!(host.contains("letcap=risk_engine::drawdown::capital_del_freno(&arena_imm);"));
    assert!(host.contains("registrar_pico(&arena_imm.pico_capital,cap)"));
    assert!(riesgo.contains("letcapital_freno=crate::drawdown::capital_del_freno(arena);"));
    assert!(riesgo.contains("registrar_pico(&arena.pico_capital,capital_freno.max(self.peak_capital),)"));
}

/// Esperado (R3): en papel sobre mainnet, ganar en papel no sube el pico del
/// saldo real ni fabrica una caída; perder en cualquiera de los dos planos sí
/// cuenta.
#[test]
fn cl132_ganar_en_papel_no_fabrica_una_caida() {
    // La sesión arrancó con 13 USD reales y el papel ganó hasta 14,04.
    let a = GlobalArena::build_in_own_stack(13.0);
    a.unified_capital.store(14.04, Relaxed);
    a.capital_exchange.store(13.0, Relaxed);
    assert_eq!(drawdown::capital_del_freno(&a), 13.0, "el plano más conservador");
    let larga = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        win_probability: 0.9,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    };
    let _ = RiskEngine::new(13.0).evaluate_quantum_order(0, &larga, &a);
    assert_eq!(a.pico_capital.load(Relaxed), 13.0, "el papel no sube el pico del saldo real");
    a.capital_exchange.store(0.0, Relaxed);
    assert_eq!(drawdown::capital_del_freno(&a), 14.04, "sin exchange (backtest), el arena");
    a.capital_exchange.store(f64::NAN, Relaxed);
    assert_eq!(drawdown::capital_del_freno(&a), 14.04, "un saldo inválido no cuenta");
}

/// Esperado (R5): cada venue guarda su pico en su archivo.
#[test]
fn cl133_cada_venue_guarda_su_pico() {
    let _g = ENTORNO.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("USE_TESTNET", "true");
    let prueba = drawdown::archivo_pico();
    std::env::remove_var("USE_TESTNET");
    let mainnet = drawdown::archivo_pico();
    println!("observado: testnet {prueba}, mainnet {mainnet}");
    assert_ne!(prueba, mainnet);
    let host = sin_codigo(include_str!("../../../src/bin/god_engine.rs"));
    assert!(host.contains("env_data_path(&risk_engine::drawdown::archivo_pico()"));
    assert!(!host.contains("env_data_path(\"pico_capital.txt\""), "el archivo sin venue no se escribe");
}

/// Esperado (B2): ausente, ilegible y legible son tres casos distintos.
#[test]
fn cl133_un_pico_ilegible_no_es_un_pico_ausente() {
    let ruta = std::env::temp_dir().join(format!("cl133_{}_pico.txt", std::process::id()));
    let _ = std::fs::remove_file(&ruta);
    assert!(matches!(drawdown::leer_pico(&ruta), CargaPico::Ausente));
    drawdown::guardar_pico(&ruta, 13.25).unwrap();
    assert!(matches!(drawdown::leer_pico(&ruta), CargaPico::Leido(p) if p == 13.25));
    for ilegible in ["", "abc", "13,25", "\u{feff}13.25", "NaN", "inf", "-5", "0"] {
        std::fs::write(&ruta, ilegible).unwrap();
        assert!(
            matches!(drawdown::leer_pico(&ruta), CargaPico::Ilegible(_)),
            "contenido {ilegible:?}"
        );
        assert_eq!(drawdown::cargar_pico(&ruta), None);
    }
    let _ = std::fs::remove_file(&ruta);
}

/// Esperado (B2): con el pico ilegible no entran posiciones nuevas.
#[test]
fn cl133_con_el_pico_ilegible_el_riesgo_no_admite_entradas() {
    let a = GlobalArena::build_in_own_stack(13.0);
    a.pico_ilegible.store(true, Relaxed);
    let larga = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        win_probability: 0.9,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    };
    let orden = RiskEngine::new(13.0).evaluate_quantum_order(0, &larga, &a);
    assert_eq!(orden.signal, SignalType::Flat);
    assert_eq!(risk_engine::ultimo_rechazo().map(|i| REJECT_NAMES[i]), Some("drawdown"));
    let host = sin_codigo(include_str!("../../../src/bin/god_engine.rs"));
    assert!(host.contains("CargaPico::Ilegible("), "el host distingue el pico ilegible");
}
