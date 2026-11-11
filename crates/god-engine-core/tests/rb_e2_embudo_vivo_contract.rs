//! CONTRATO RB-E2 (Codex RB, 2026-11-10): el embudo vivo cuenta por qué no se
//! abre cada intención, y contar no cambia ninguna decisión.
//!
//! QUÉ: el núcleo atribuye cada tick bloqueado a su primera causa
//! (`embudo_vivo::BloqueosEntrada`), cuenta las intenciones sin ranura y los
//! rechazos del riesgo por motivo, y el host publica una línea por minuto
//! (`EmbudoVivo`) con las órdenes que la fase no deja salir.
//! POR QUÉ: la Demo del 2026-11-10 pasó 6,5 h sin operar y el embudo del núcleo
//! (`direction_diag`) sólo lo imprimía el backtest forense; los bloqueos previos
//! a la evaluación (llamador, comisiones, kill switch, deriva, feed, latencia,
//! liquidaciones) no los contaba nadie.
//! CÓMO: (1) conducta con el núcleo real; (2) cableado en la fuente del núcleo
//! y del host; (3) pureza: los contadores no los lee ninguna decisión.

use god_engine_core::embudo_vivo::{BLOQUEO_LLAMADOR, CAUSAS_BLOQUEO};
use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use std::sync::atomic::Ordering;
use std::sync::Mutex;

const NUCLEO: &str = include_str!("../src/lib.rs");
const EMBUDO: &str = include_str!("../src/embudo_vivo.rs");
const HOST: &str = include_str!("../../../src/bin/god_engine.rs");

/// Las pruebas de conducta cambian el universo y el registro de símbolos.
static ENTORNO_GLOBAL: Mutex<()> = Mutex::new(());

/// Índice de la causa «comisiones sin observar».
const BLOQUEO_COMISIONES: usize = 1;

fn posicion(texto: &str, desde: usize, aguja: &str) -> usize {
    match texto.get(desde..).and_then(|t| t.find(aguja)) {
        Some(p) => desde + p,
        None => panic!("no aparece `{aguja}` a partir del byte {desde}"),
    }
}

/// Núcleo con BTC y ETH registrados.
fn nucleo_nuevo() -> GodEngineCore {
    symbols::update_dynamic_universe(vec!["BTCUSDT".into(), "ETHUSDT".into()]);
    symbol_registry::update_registry(vec![
        symbol_registry::get_official_binance_spec("BTCUSDT"),
        symbol_registry::get_official_binance_spec("ETHUSDT"),
    ]);
    GodEngineCore::new(GlobalArena::build_in_own_stack(100.0))
}

fn tick(core: &mut GodEngineCore, moneda: usize, mid: f64, t_ms: u64, permite: bool) {
    let (_cierre, _salida, _cotizacion) = core.process_tick_dual(
        moneda,
        mid - 0.005,
        mid + 0.005,
        5.0,
        5.0,
        t_ms,
        &[0.0; 54],
        true,
        permite,
    );
}

#[test]
fn rb_e2_el_nucleo_real_cuenta_cada_tick_por_su_primera_causa() {
    let _entorno = ENTORNO_GLOBAL.lock().unwrap_or_else(|p| p.into_inner());
    let mut core = nucleo_nuevo();
    const T0: u64 = 1_700_000_000_000;

    // El llamador no permite entradas: diez ticks, los diez al llamador.
    for k in 0..10 {
        tick(&mut core, 0, 100.0, T0 + k, false);
    }
    let b = core.diag_embudo.bloqueos;
    assert_eq!(b.ticks, 10);
    assert_eq!(b.por_causa[BLOQUEO_LLAMADOR], 10);
    assert_eq!(b.bloqueados(), 10);
    assert_eq!(core.diag_dir.evaluations, 0, "bloqueado: no se evalúa");

    // Sesión viva sin calendario de comisiones publicado (lo que hace el OMS
    // al empezar cada refresco, `reset_fee_requirement`): el llamador permite,
    // las comisiones no.
    core.arena
        .config
        .observed_fees_required
        .store(true, Ordering::Release);
    tick(&mut core, 0, 100.0, T0 + 20, true);
    let b = core.diag_embudo.bloqueos;
    assert_eq!(b.ticks, 11);
    assert_eq!(b.por_causa[BLOQUEO_COMISIONES], 1, "{:?}", b.por_causa);
    core.arena
        .config
        .observed_fees_required
        .store(false, Ordering::Release);

    // Con permiso y comisiones observadas, el tick cruza la frontera.
    tick(&mut core, 0, 100.0, T0 + 30, true);
    let b = core.diag_embudo.bloqueos;
    assert_eq!(b.ticks, 12);
    assert_eq!(
        b.bloqueados(),
        11,
        "este tick no se bloquea: {:?}",
        b.por_causa
    );
    assert!(
        core.diag_dir.evaluations <= b.ticks - b.bloqueados(),
        "sólo se evalúa lo que cruza la frontera"
    );
}

#[test]
fn rb_e2_process_event_distingue_el_evento_de_libro_del_panico_del_host() {
    let _entorno = ENTORNO_GLOBAL.lock().unwrap_or_else(|p| p.into_inner());
    let mut core = nucleo_nuevo();
    const T0: u64 = 1_700_000_000_000;
    let evento = |core: &mut GodEngineCore, t_ms: u64, es_trade: bool, panico: bool| {
        let (_orden, _cierre) = core.process_event(
            0, es_trade, false, !es_trade, 100.0, 0.01, 99.995, 100.005, 5.0, 5.0, 0.0, 0.0, t_ms,
            panico, &[0.0; 54], false,
        );
    };
    evento(&mut core, T0, false, false); // sólo libro
    evento(&mut core, T0 + 1, false, true); // libro en pánico: cuenta el pánico
    evento(&mut core, T0 + 2, true, true); // trade en pánico
    evento(&mut core, T0 + 3, true, false); // trade sin pánico: oportunidad
    let n = core.diag_embudo;
    assert_eq!((n.solo_libro, n.panico_host), (1, 2), "{n:?}");
    assert!(
        n.bloqueos.por_causa[BLOQUEO_LLAMADOR] >= 3,
        "los tres eventos sin permiso llegan al núcleo como «el llamador no las permite»: {:?}",
        n.bloqueos
    );
}

#[test]
fn rb_e2_la_disyuncion_del_nucleo_es_la_de_siempre_con_sus_siete_causas_en_orden() {
    let ptd = posicion(NUCLEO, 0, "pub fn process_tick_dual(");
    let registro = posicion(
        NUCLEO,
        ptd,
        "let entries_blocked = self.diag_embudo.bloqueos.registrar([",
    );
    let mut desde = registro;
    for causa in [
        "!allow_entries,",
        "!observed_fees_valid,",
        "kill_latched,",
        "self.drift_entry_veto,",
        "quantum_arena::feed_health::is_stalled(),",
        "is_latency_panic,",
        "liquidation_as_of_invalid,",
    ] {
        desde = posicion(NUCLEO, desde, causa) + causa.len();
    }
    assert_eq!(
        NUCLEO.matches("let entries_blocked").count(),
        1,
        "una sola definición de la frontera"
    );
    assert!(
        !NUCLEO.contains("let allow_entries = allow_entries && observed_fees_valid;"),
        "las comisiones vetan como causa propia, no escondidas en el permiso"
    );
    assert_eq!(CAUSAS_BLOQUEO, 7);
    let frontera = posicion(NUCLEO, registro, "if entries_blocked {");
    assert!(
        registro < frontera,
        "se cuenta antes de cortar la evaluación"
    );

    let pe = posicion(NUCLEO, 0, "pub fn process_event(");
    let permiso = posicion(
        NUCLEO,
        pe,
        "let allow_entries = !latency_panic && (!is_depth || is_trade);",
    );
    let llamada = posicion(
        NUCLEO,
        permiso,
        ".registrar_llamada(is_depth && !is_trade, latency_panic);",
    );
    assert!(llamada < posicion(NUCLEO, permiso, "self.process_tick_dual("));

    let ranura = posicion(
        NUCLEO,
        ptd,
        "let slot_available = raw_slot_available && !same_dir_unsecured;",
    );
    let cuenta = posicion(
        NUCLEO,
        ranura,
        ".registrar_sin_ranura(!coin.positions.is_any_open(), same_dir_unsecured);",
    );
    let riesgo = posicion(NUCLEO, ranura, ".evaluate_quantum_order(");
    assert!(cuenta < riesgo, "sin ranura se cuenta antes del riesgo");
    let motivo = posicion(NUCLEO, riesgo, "self.diag_embudo.registrar_riesgo(");
    assert!(
        motivo < posicion(NUCLEO, riesgo, "if order.signal != SignalType::Flat {"),
        "el motivo se lee justo tras la evaluación"
    );
}

#[test]
fn rb_e2_el_host_publica_el_embudo_y_cuenta_las_ordenes_sin_permiso_de_fase() {
    let creacion = posicion(
        HOST,
        0,
        "let mut embudo_vivo = god_engine_core::embudo_vivo::EmbudoVivo::new(60_000);",
    );
    let bucle = posicion(
        HOST,
        creacion,
        "while let Ok(mensaje_ws) = rx_events.recv() {",
    );
    let fase = posicion(HOST, bucle, "current_phase = orch.on_tick();");
    let cuenta = posicion(
        HOST,
        fase,
        "if !is_trading_allowed && new_order.is_some() {",
    );
    let descarte = posicion(HOST, fase, "if !is_trading_allowed {");
    assert!(
        cuenta < descarte,
        "se cuenta la orden ANTES de que la fase la descarte"
    );
    let toca = posicion(HOST, fase, "if embudo_vivo.toca(ahora_embudo_ms) {");
    let bloque = &HOST[toca..posicion(HOST, toca, "// M6-H02")];
    for campo in [
        "god_engine_core::embudo_vivo::Instantanea::de(",
        "ordenes_sin_permiso_de_fase,",
        "embudo_vivo.cerrar(instantanea, ahora_embudo_ms)",
        "telemetry_server::telemetry_log!(",
        "engine_real.diag_dir.report()",
    ] {
        assert!(bloque.contains(campo), "el bloque del embudo usa `{campo}`");
    }
}

#[test]
fn rb_e2_ninguna_decision_lee_los_contadores() {
    // En el núcleo, `diag_embudo` sólo aparece en su declaración, su
    // inicialización y en llamadas que REGISTRAN.
    let usos: Vec<&str> = NUCLEO
        .lines()
        .map(str::trim)
        .filter(|l| l.contains("diag_embudo") && !l.starts_with("//"))
        .collect();
    for uso in &usos {
        assert!(
            uso.starts_with("pub diag_embudo: embudo_vivo::ContadoresNucleo,")
                || uso.starts_with("diag_embudo: embudo_vivo::ContadoresNucleo::default(),")
                || uso.starts_with("self.diag_embudo.registrar_")
                || uso.starts_with("self.diag_embudo.bloqueos.registrar(")
                || uso.starts_with("let entries_blocked = self.diag_embudo.bloqueos.registrar(")
                || *uso == "self.diag_embudo",
            "uso de `diag_embudo` que no registra: `{uso}`"
        );
    }
    // El módulo no toca estado compartido ni decide.
    let produccion = EMBUDO.split("#[cfg(test)]").next().unwrap_or(EMBUDO);
    for prohibido in ["Ordering", ".store(", "fetch_", "GlobalArena", "registry"] {
        assert!(
            !produccion.contains(prohibido),
            "el embudo no toca estado compartido: `{prohibido}`"
        );
    }
}
