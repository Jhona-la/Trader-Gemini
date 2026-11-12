//! CONTRATO RB-V1a/RB-V1b (Codex RB, 2026-11-10): los vetos posteriores al
//! riesgo y los rechazos del riesgo posteriores a la geometría se juzgan en
//! sombra, y la sombra NO decide nada.
//!
//! QUÉ: `GodEngineCore` registra en `sombra_vetos` cada intención que el riesgo
//! rechaza después de calcular su geometría (RB-V1b) o que valida y vetan el
//! consejo o el ML-GATE (RB-V1a), con la geometría que el riesgo publicó
//! (`geometria_evaluada`), y avanza el libro con el mid de cada evento de
//! `process_tick_dual`.
//! POR QUÉ: el veto que estrangula la Demo (el ML-GATE) tiene que recibir su
//! prueba contrafactual; y un observador que influyera en la decisión dejaría
//! de medir el sistema que observa.
//! CÓMO:
//! 1. Cableado (fuente): la observación precede a la evaluación del riesgo en
//!    `process_tick_dual`; el registro de rechazos sigue a la única evaluación
//!    del riesgo y precede a la deliberación; el de vetos vive en el bloque
//!    `!(consejo && ML)`, después de calcular `ml_gate_ok` y antes de la rama
//!    que abre.
//! 2. Pureza (fuente): el núcleo sólo escribe en la sombra (`observar`,
//!    `registrar`) y el módulo no importa ni toca estado de decisión.
//! 3. Conducta: el `process_tick_dual` real resuelve por primer toque (TP) y por
//!    vencimiento, cada intención con los mids de SU moneda.

use god_engine_core::sombra_vetos::{Veto, FUENTE_CONSEJO, FUENTE_ML_SIN_VENTAJA};
use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use risk_engine::medida_sombra::GeometriaSombra;
use std::sync::Mutex;

const NUCLEO: &str = include_str!("../src/lib.rs");
const SOMBRA: &str = include_str!("../src/sombra_vetos.rs");

/// Las pruebas de conducta cambian el universo y el registro de símbolos.
static ENTORNO_GLOBAL: Mutex<()> = Mutex::new(());

fn posicion(texto: &str, desde: usize, aguja: &str) -> usize {
    match texto.get(desde..).and_then(|t| t.find(aguja)) {
        Some(p) => desde + p,
        None => panic!("no aparece `{aguja}` a partir del byte {desde}"),
    }
}

/// Código sin comentarios de línea, sólo la parte de producción del módulo.
fn codigo_de_produccion(texto: &str) -> String {
    let fin = texto.find("#[cfg(test)]").unwrap_or(texto.len());
    texto[..fin]
        .lines()
        .map(str::trim)
        .filter(|l| !l.starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn rb_v1a_la_observacion_precede_al_riesgo_y_el_registro_sigue_a_la_decision() {
    let ptd = posicion(NUCLEO, 0, "pub fn process_tick_dual(");
    let observa = posicion(NUCLEO, ptd, "self.sombra_vetos.observar(");
    let riesgo = posicion(NUCLEO, ptd, ".evaluate_quantum_order(");
    assert!(
        observa < riesgo,
        "la sombra avanza antes de evaluar intenciones nuevas"
    );

    let ml = posicion(NUCLEO, riesgo, "let ml_gate_ok =");
    let veto = posicion(NUCLEO, ml, "if !(aprobado_por_consejo && ml_gate_ok) {");
    let registra = posicion(
        NUCLEO,
        veto,
        "self.sombra_vetos.registrar(sombra_vetos::Veto {",
    );
    let abre = posicion(NUCLEO, veto, "if aprobado_por_consejo && ml_gate_ok {");
    assert!(
        registra < abre,
        "el registro está dentro del bloque del veto"
    );
    let bloque = &NUCLEO[registra..abre];
    for campo in [
        "sombra_vetos::fuente_del_veto(",
        "has_roster_model",
        "geometria: self.risk_engine.geometria_evaluada(coin_id)",
        "promesa_bruta: self.risk_engine.promesa_evaluada(coin_id)",
        "t0_ms: event_time_ms",
    ] {
        assert!(bloque.contains(campo), "el registro usa `{campo}`");
    }
}

#[test]
fn rb_v1b_los_rechazos_del_riesgo_se_registran_tras_evaluar_y_antes_de_deliberar() {
    // Una sola evaluación del riesgo por tick: la geometría que publica es la
    // de ESTA intención en los dos registros (rechazo y veto).
    assert_eq!(NUCLEO.matches(".evaluate_quantum_order(").count(), 1);
    let ptd = posicion(NUCLEO, 0, "pub fn process_tick_dual(");
    let riesgo = posicion(NUCLEO, ptd, ".evaluate_quantum_order(");
    let rechazo = posicion(NUCLEO, riesgo, "risk_engine::ultimo_rechazo()");
    let registra = posicion(
        NUCLEO,
        rechazo,
        "self.sombra_vetos.registrar(sombra_vetos::Veto {",
    );
    let delibera = posicion(NUCLEO, riesgo, "if order.signal != SignalType::Flat {");
    assert!(
        rechazo < registra && registra < delibera,
        "el registro de rechazos sigue a la evaluación y precede a la deliberación"
    );
    let bloque = &NUCLEO[rechazo..delibera];
    for campo in [
        ".filter(|_| order.signal == SignalType::Flat)",
        ".and_then(sombra_vetos::fuente_juzgable)",
        "es_largo: calibrated_intent.signal == SignalType::Long",
        "geometria: self.risk_engine.geometria_evaluada(coin_id)",
        "promesa_bruta: self.risk_engine.promesa_evaluada(coin_id)",
        "t0_ms: event_time_ms",
    ] {
        assert!(
            bloque.contains(campo),
            "el registro de rechazos usa `{campo}`"
        );
    }
}

#[test]
fn rb_v1a_el_nucleo_solo_escribe_en_la_sombra() {
    // (uso permitido, veces): sólo se escribe en la sombra.
    let permitidos = [
        ("pub mod sombra_vetos;", 1),
        ("pub sombra_vetos: sombra_vetos::SombraVetos,", 1),
        ("sombra_vetos: sombra_vetos::SombraVetos::new(n_coins),", 1),
        ("self.sombra_vetos.observar(", 1),
        ("self.sombra_vetos.registrar(sombra_vetos::Veto {", 2),
        ("fuente: sombra_vetos::fuente_del_veto(", 1),
        (".and_then(sombra_vetos::fuente_juzgable)", 1),
    ];
    let usos: Vec<&str> = NUCLEO
        .lines()
        .map(str::trim)
        .filter(|l| l.contains("sombra_vetos") && !l.starts_with("//"))
        .collect();
    for uso in &usos {
        assert!(
            permitidos.iter().any(|(p, _)| uso.starts_with(p)),
            "uso de la sombra no permitido en el núcleo: `{uso}`"
        );
    }
    for (p, veces) in permitidos {
        assert_eq!(
            usos.iter().filter(|u| u.starts_with(p)).count(),
            veces,
            "`{p}` {veces} vez/veces"
        );
    }

    let modulo = codigo_de_produccion(SOMBRA);
    // Cada sentencia `use` completa, sin espacios ni la coma final que rustfmt
    // añade al partirla en varias líneas.
    let importaciones: Vec<String> = modulo
        .split(';')
        .map(|s| s.split_whitespace().collect::<String>().replace(",}", "}"))
        .filter(|s| s.starts_with("use"))
        .collect();
    assert_eq!(
        importaciones,
        [
            "useaudit_engine::shadow_ledger::{EstadisticaFuente,IntencionVetada,LibroSombra,Promesa,Veredicto}",
            "userisk_engine::medida_sombra::GeometriaSombra",
        ],
        "el módulo sólo depende del libro y de la geometría"
    );
    for prohibido in [
        "GlobalArena",
        "quantum_arena",
        "Ordering",
        ".store(",
        "fetch_",
    ] {
        assert!(
            !modulo.contains(prohibido),
            "la sombra no toca estado compartido: `{prohibido}`"
        );
    }
}

fn nucleo_nuevo() -> GodEngineCore {
    symbols::update_dynamic_universe(vec!["BTCUSDT".into(), "ETHUSDT".into()]);
    symbol_registry::update_registry(vec![
        symbol_registry::get_official_binance_spec("BTCUSDT"),
        symbol_registry::get_official_binance_spec("ETHUSDT"),
    ]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    GodEngineCore::new(arena)
}

/// Un evento de libro con mid `mid`, sin permitir entradas.
fn evento(core: &mut GodEngineCore, moneda: usize, mid: f64, t_ms: u64) {
    let (_cierre, _salida, _cotizacion) = core.process_tick_dual(
        moneda,
        mid - 0.005,
        mid + 0.005,
        5.0,
        5.0,
        t_ms,
        &[0.0; 54],
        true,
        false,
    );
}

#[test]
fn rb_v1a_el_nucleo_real_avanza_la_sombra_con_los_mids_de_cada_moneda() {
    let _entorno = ENTORNO_GLOBAL.lock().unwrap_or_else(|p| p.into_inner());
    let mut core = nucleo_nuevo();
    const T0: u64 = 1_700_000_000_000;
    let g = GeometriaSombra {
        tp_pct: 0.010,
        sl_pct: 0.005,
        friccion_pct: 0.001,
        tau_ms: 60_000.0,
    };
    // Largo vetado por el ML-GATE en la moneda 0 y corto vetado por el consejo
    // en la moneda 1, ambos con τ de un minuto.
    assert!(core.sombra_vetos.registrar(Veto {
        fuente: FUENTE_ML_SIN_VENTAJA,
        moneda: 0,
        es_largo: true,
        entrada: 100.0,
        geometria: Some(g),
        promesa_bruta: None,
        t0_ms: T0,
    }));
    assert!(core.sombra_vetos.registrar(Veto {
        fuente: FUENTE_CONSEJO,
        moneda: 1,
        es_largo: false,
        entrada: 50.0,
        geometria: Some(g),
        promesa_bruta: None,
        t0_ms: T0,
    }));

    evento(&mut core, 1, 50.1, T0 + 1_000);
    evento(&mut core, 0, 100.4, T0 + 2_000);
    assert_eq!(
        core.sombra_vetos.libro().abiertas(),
        2,
        "nada tocó barreras"
    );

    // Un mid de 101,5 en la moneda 1 resuelve SU corto (SL, x = −1,03) y no
    // toca el largo de la moneda 0: a cada intención sólo la mueve su moneda.
    evento(&mut core, 1, 101.5, T0 + 3_000);
    assert!(core
        .sombra_vetos
        .estadistica(FUENTE_ML_SIN_VENTAJA)
        .is_some_and(|e| e.n == 0));
    evento(&mut core, 0, 101.5, T0 + 4_000);
    let Some(ml) = core.sombra_vetos.estadistica(FUENTE_ML_SIN_VENTAJA) else {
        panic!("la fuente del ML-GATE existe");
    };
    assert_eq!((ml.n, ml.tp, ml.sl), (1, 1, 0), "el largo tocó el TP");

    let Some(consejo) = core.sombra_vetos.estadistica(FUENTE_CONSEJO) else {
        panic!("la fuente del consejo existe");
    };
    assert_eq!(
        (consejo.n, consejo.sl),
        (1, 1),
        "el corto tocó el SL con su mid"
    );
    assert_eq!(core.sombra_vetos.libro().abiertas(), 0);
}

#[test]
fn rb_v1a_el_nucleo_real_resuelve_por_vencimiento() {
    let _entorno = ENTORNO_GLOBAL.lock().unwrap_or_else(|p| p.into_inner());
    let mut core = nucleo_nuevo();
    const T0: u64 = 1_700_000_000_000;
    assert!(core.sombra_vetos.registrar(Veto {
        fuente: FUENTE_CONSEJO,
        moneda: 1,
        es_largo: false,
        entrada: 50.0,
        geometria: Some(GeometriaSombra {
            tp_pct: 0.010,
            sl_pct: 0.005,
            friccion_pct: 0.001,
            tau_ms: 60_000.0,
        }),
        promesa_bruta: None,
        t0_ms: T0,
    }));
    evento(&mut core, 1, 49.95, T0 + 30_000);
    assert_eq!(
        core.sombra_vetos.libro().abiertas(),
        1,
        "antes de τ sigue abierta"
    );
    evento(&mut core, 1, 49.9, T0 + 60_000);
    let Some(e) = core.sombra_vetos.estadistica(FUENTE_CONSEJO) else {
        panic!("la fuente del consejo existe");
    };
    assert_eq!((e.n, e.vencidas), (1, 1));
    // Corto a 50 que vence a 49,9: +0,002 a favor, (0,002 − 0,001)/0,005 = 0,2 R.
    assert!(
        e.media_r().is_some_and(|m| (m - 0.2).abs() < 1e-9),
        "{:?}",
        e.media_r()
    );
}
