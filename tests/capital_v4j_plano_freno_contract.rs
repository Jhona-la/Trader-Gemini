//! CAPITAL-UNICO V4J, punto 5 (A6 de `qra_v4j_criterios.md`, ocho casos de
//! `qra_plano_del_freno.md`): el freno duro y el pico miden la equity conservadora del puente; el
//! libro del dimensionable gobierna solo la capacidad de entrada. No cambia ningun umbral.
//!
//! Piezas reales: el puente de produccion (`PuenteCuenta`), el libro bajo la puerta del arena,
//! `drawdown::capital_del_freno`, `registrar_pico`, `freno_duro`, el escritor del pico de Risk
//! (`RiskEngine::evaluate_quantum_order`), el arranque del pico (`decidir_arranque_pico`,
//! `guardar_pico`, `guardar_marca_definicion`) y `FlujosExternos`. Lo que vive en linea en el lazo
//! del inmune del host (cuatro lineas) esta en el ADAPTADOR, atado a sus lineas con anclas.
use execution_engine::puente_cuenta::{DatoPosicion, LecturaCuenta, PosicionRest, PuenteCuenta};
use quantum_arena::dimensionable::ClaveReserva;
use quantum_arena::GlobalArena;
use risk_engine::drawdown::{self, CargaPico, FlujosExternos, OrigenDeLaCaida, RegistroDeFlujo};
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const HOST: &str = include_str!("../src/bin/god_engine.rs");
const RIESGO: &str = include_str!("../crates/risk-engine/src/lib.rs");
const SESION: u64 = 1;
static SERIE: Mutex<()> = Mutex::new(());

fn texto(f: &str) -> String {
    f.replace("\r\n", "\n")
}

fn orden(fuente: &str, pasos: &[&str]) {
    let fuente = texto(fuente);
    let mut resto = fuente.as_str();
    for p in pasos {
        resto = resto.split_once(p).unwrap_or_else(|| panic!("ancla ausente o fuera de orden: {p}")).1;
    }
}

// ============================ ADAPTADOR (lazo del inmune del host) ============================

fn anclas_del_inmune() {
    orden(HOST, &[
        "let cap_exchange = puente_imm.equity_del_freno().unwrap_or(f64::NAN);",
        "arena_imm.capital_exchange.store(",
        "let cap = risk_engine::drawdown::capital_del_freno(&arena_imm);",
        "risk_engine::drawdown::registrar_pico(&arena_imm.pico_capital, cap);",
        "let freno = risk_engine::drawdown::freno_duro(&arena_imm, peak_capital, cap);",
        "let insolvent = cap <= 0.0;",
    ]);
    // Cambio de espacio: nada del ambito anterior mide el pico nuevo.
    orden(HOST, &[
        "if let Some(rutas) = espacio_nuevo {",
        "arena_imm.capital_exchange.store(f64::NAN, Ordering::Relaxed);",
        "aplicar_arranque_pico(&arena_imm, &rutas, true);",
    ]);
    // Relevo: la era nueva es de la cuenta anunciada; con otra cuenta, la equity pasa a ausencia.
    orden(HOST, &[
        "puente_cuenta.anunciar_cuenta(cuenta_del_relevo);",
        "if libro_relevo.cuenta().is_some_and(|c| c != cuenta_del_relevo) {",
        "arena_real.capital_exchange.store(f64::NAN, Ordering::Relaxed);",
        "libro_relevo.fijar_cuenta(&arena_real, cuenta_del_relevo);",
    ]);
    orden(HOST, &[".fijar_cuenta(&arena_real, cuenta_de_arranque);", "puente_cuenta.anunciar_cuenta(cuenta_de_arranque);"]);
    orden(HOST, &["let mut marca_pendiente = !risk_engine::drawdown::marca_vigente(&rutas.marca);"]);
    orden(HOST, &[
        "fn aplicar_arranque_pico(",
        "if !cambio_de_espacio && arena.config.dimensionable_derivado.load(Ordering::Acquire) {",
        "arena.pico_capital.store(0.0, Ordering::Release);",
        "risk_engine::drawdown::CargaPico::Leido(guardado) => {",
        "risk_engine::drawdown::registrar_pico(&arena.pico_capital, guardado);",
    ]);
    orden(HOST, &["risk_engine::drawdown::OrigenDeLaCaida::FlujoIncierto => format!("]);
    // El escritor de Risk.
    orden(RIESGO, &[
        "if arena.config.dimensionable_derivado.load(Ordering::Acquire) {",
        "self.peak_capital = arena.pico_capital.load(Ordering::Acquire);",
        "let capital_freno = crate::drawdown::capital_del_freno(arena);",
        "if derivado {",
        "self.peak_capital = arena.pico_capital.load(Ordering::Acquire);",
        "let capital_freno = if derivado && !(self.peak_capital > 0.0) {",
        "self.peak_capital = crate::drawdown::registrar_pico(",
    ]);
}

/// Un ciclo del inmune: (capital del freno, pico, freno).
fn ciclo_del_inmune(m: &Montaje) -> (f64, f64, drawdown::FrenoDuro) {
    let cap_exchange = m.puente.equity_del_freno().unwrap_or(f64::NAN);
    m.arena.capital_exchange.store(if cap_exchange.is_finite() { cap_exchange } else { f64::NAN }, Ordering::Relaxed);
    let cap = drawdown::capital_del_freno(&m.arena);
    let pico = drawdown::registrar_pico(&m.arena.pico_capital, cap);
    (cap, pico, drawdown::freno_duro(&m.arena, pico, cap))
}

/// El relevo del host (`paso_transicion` dentro de `relevar_sesion`): cuenta anunciada, libro y,
/// con otra cuenta, la equity del freno en ausencia.
fn relevo(m: &Montaje, cuenta: u64) -> u64 {
    let sesion = m.oms.load(Ordering::SeqCst) + 1;
    let (_, era) = m.puente.relevar_sesion(|| {
        m.oms.store(sesion, Ordering::SeqCst);
        m.puente.anunciar_cuenta(cuenta);
        let mut libro = m.arena.config.entry_publication_gate.lock().unwrap();
        if libro.cuenta().is_some_and(|c| c != cuenta) {
            m.arena.capital_exchange.store(f64::NAN, Ordering::Relaxed);
        }
        libro.fijar_cuenta(&m.arena, cuenta);
        ((), Some(sesion))
    });
    let era = era.unwrap();
    m.puente.abrir_socket(era).unwrap();
    era
}
// ========================== FIN DEL ADAPTADOR ==========================

struct Montaje {
    puente: Arc<PuenteCuenta>,
    arena: Arc<GlobalArena>,
    oms: Arc<AtomicU64>,
    riesgo: RiskEngine,
}

fn montaje(cuenta: u64) -> Montaje {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let oms = Arc::new(AtomicU64::new(SESION));
    let puente = Arc::new(PuenteCuenta::con_sesion(SESION));
    {
        let oms = Arc::clone(&oms);
        puente.vincular_testigo_de_sesion(move || oms.load(Ordering::SeqCst));
    }
    {
        let mut libro = arena.config.entry_publication_gate.lock().unwrap();
        libro.activar(&arena);
        libro.fijar_cuenta(&arena, cuenta);
    }
    // ADAPTADOR (`aplicar_arranque_pico` del host con el libro activo): el pico nace vacio.
    arena.pico_capital.store(0.0, Ordering::Release);
    puente.anunciar_cuenta(cuenta);
    {
        let (arena, oms) = (Arc::clone(&arena), Arc::clone(&oms));
        puente.vincular_publicador(move |p| {
            if p.veta {
                arena.config.entry_publication_gate.lock().unwrap().veto_del_puente();
                arena.capital_incompleto.store(true, Ordering::Release);
                return false;
            }
            let Some((saldo, marca)) = p.vigente else {
                arena.capital_incompleto.store(true, Ordering::Release);
                return false;
            };
            if oms.load(Ordering::SeqCst) != p.sesion {
                return false;
            }
            let mut l = arena.config.entry_publication_gate.lock().unwrap();
            l.fijar_alcance(p.era, p.sesion);
            l.saldo_aceptado(&arena, saldo, marca);
            drop(l);
            arena.capital_incompleto.store(false, Ordering::Release);
            true
        });
    }
    puente.abrir_socket(puente.era()).unwrap();
    Montaje { puente, arena, oms, riesgo: RiskEngine::new(100.0) }
}

fn lectura(saldo: f64, upnl: Option<f64>, t: u64, completa: bool) -> LecturaCuenta {
    let mut posiciones = HashMap::new();
    if let Some(u) = upnl {
        posiciones.insert(("BTCUSDT".to_string(), "BOTH".to_string()), PosicionRest { dato: DatoPosicion::Abierta { cantidad: 0.01, upnl: u }, tiempo_ms: Some(t) });
    }
    LecturaCuenta {
        saldo,
        tiempo_saldo_ms: Some(t),
        posiciones: completa.then_some(posiciones),
        rechazo_arranque: None,
        upnl_del_activo: Some(upnl.unwrap_or(0.0)),
    }
}

impl Montaje {
    fn rest(&self, saldo: f64, upnl: Option<f64>, t: u64) {
        let _ = self.puente.observar_rest_de_sesion(self.puente.ticket(), self.puente.sesion(), &lectura(saldo, upnl, t, true));
    }
    fn rest_incompleta(&self, saldo: f64, upnl: f64, t: u64) {
        let _ = self.puente.observar_rest_de_sesion(self.puente.ticket(), self.puente.sesion(), &lectura(saldo, Some(upnl), t, false));
    }
    fn dimensionable(&self) -> f64 {
        self.arena.unified_capital.load(Ordering::Relaxed)
    }
    fn pico(&self) -> f64 {
        self.arena.pico_capital.load(Ordering::Relaxed)
    }
    /// El escritor de Risk por su entrada real (decision de una entrada larga).
    fn riesgo(&mut self) -> (f64, f64) {
        let intento = SignalIntent { signal: SignalType::Long, confidence: 0.9, win_probability: 0.9, expected_duration_ms: 60_000, ..Default::default() };
        let _ = self.riesgo.evaluate_quantum_order(0, &intento, &self.arena);
        (drawdown::capital_del_freno(&self.arena), self.riesgo.peak_capital)
    }
}

/// F1: equity y pico 100. Reservar, cargar un debito provisional, cancelar y reconciliar con la
/// equity constante: el dimensionable varia, los dos consumidores ven 100, el pico queda en 100 y
/// el freno no dispara.
#[test]
fn f1_reserva_y_debito_provisional_no_disparan_ni_bajan_pico() {
    let _g = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_inmune();
    let mut m = montaje(1);
    m.rest(100.0, None, 100);
    assert_eq!(ciclo_del_inmune(&m).1, 100.0);
    let clave = ClaveReserva { coin_id: 0, slot: 0, generation: 1 };
    let mut dims = Vec::new();
    for paso in ["reserva", "debito provisional", "cancelacion", "reconciliacion"] {
        {
            let mut l = m.arena.config.entry_publication_gate.lock().unwrap();
            match paso {
                "reserva" => l.reservar(&m.arena, clave, 0.75),
                "debito provisional" => l.debito_cierre_local(&m.arena, ClaveReserva { slot: 1, ..clave }, 3.0),
                "cancelacion" => l.cancelar(&m.arena, clave, 0.75),
                _ => {}
            }
        }
        if paso == "reconciliacion" {
            m.rest(100.0, None, 200);
        }
        dims.push(m.dimensionable());
        let (cap, pico, freno) = ciclo_del_inmune(&m);
        let (cap_riesgo, pico_riesgo) = m.riesgo();
        assert_eq!((cap, pico), (100.0, 100.0), "{paso}: el inmune midio la capacidad");
        assert_eq!((cap_riesgo, pico_riesgo), (100.0, 100.0), "{paso}: Risk midio la capacidad");
        assert!(!freno.echado(), "{paso}: el freno disparo sin perdida de patrimonio: {freno}");
    }
    assert_eq!(dims, [99.25, 96.25, 97.0, 97.0], "fixture: el dimensionable si varia");
    assert_eq!(m.pico(), 100.0);
}

/// F2: dimensionable 95 y equity 100, 103, 100, 87 con cartera abierta e intercalado de los dos
/// escritores: los dos ven la misma magnitud, el pico 103 y una caida coherente con la serie, con
/// la misma decision del freno con sus umbrales vigentes.
#[test]
fn f2_risk_y_host_comparten_pico_y_drawdown_de_equity() {
    let _g = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    let mut m = montaje(1);
    m.rest(100.0, Some(0.0), 100);
    m.arena.config.entry_publication_gate.lock().unwrap().debito_cierre_local(&m.arena, ClaveReserva { coin_id: 0, slot: 1, generation: 1 }, 5.0);
    assert_eq!(m.dimensionable(), 95.0, "fixture");
    let _ = ciclo_del_inmune(&m);
    for (i, (upnl, t)) in [(3.0, 200), (0.0, 300), (-13.0, 400)].into_iter().enumerate() {
        m.rest(100.0, Some(upnl), t);
        // Intercalado: en los pasos pares escribe primero Risk, en los impares el inmune.
        let ((cap, pico, freno), (cap_r, pico_r)) = if i % 2 == 0 {
            // El inmune publica la equity del puente; Risk escribe el pico antes que el inmune.
            m.arena.capital_exchange.store(m.puente.equity_del_freno().unwrap_or(f64::NAN), Ordering::Relaxed);
            let r = m.riesgo();
            (ciclo_del_inmune(&m), r)
        } else {
            let h = ciclo_del_inmune(&m);
            (h, m.riesgo())
        };
        assert_eq!(cap, 100.0 + upnl, "la magnitud del freno no es la equity");
        assert_eq!(cap_r, cap, "Risk y el host miden distinto");
        assert_eq!(m.dimensionable(), 95.0, "la capacidad cambio con el PnL abierto");
        let pico_r = drawdown::registrar_pico(&m.arena.pico_capital, f64::NAN).max(pico_r);
        assert_eq!((pico, pico_r), (103.0, 103.0), "pico distinto entre los dos escritores");
        let freno_r = drawdown::freno_duro(&m.arena, pico_r, cap_r);
        assert_eq!(freno.echado(), freno_r.echado(), "decision distinta entre los dos");
        assert_eq!(freno.caida_capital, (103.0 - cap) / 103.0, "caida incoherente con la serie");
    }
}

/// F3: equity 100 a 91 con el cargo aun retenido en el libro. El freno y el pico reciben una sola
/// perdida, tambien al soltar el debito y ante su duplicado; comision y funding con identidad
/// independiente cuentan en el libro una vez cada uno, y nunca en el freno.
#[test]
fn f3_funding_y_comision_acreditados_afectan_equity_una_vez() {
    let _g = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    let mut m = montaje(1);
    m.rest(100.0, None, 100);
    assert_eq!(ciclo_del_inmune(&m).1, 100.0);
    {
        let mut l = m.arena.config.entry_publication_gate.lock().unwrap();
        l.debito_funding(&m.arena, 7, 8.5, 150);
        l.fill_de_entrada(&m.arena, 0, "suelto", 8, Some(0.5), Some(150));
    }
    // La cuenta ya refleja 9 (8,5 de funding y 0,5 de comision), con una marca anterior al cargo.
    m.rest(91.0, None, 140);
    let (cap, pico, freno) = ciclo_del_inmune(&m);
    assert_eq!((cap, pico), (91.0, 100.0));
    assert_eq!(freno.caida_capital, 0.09, "la perdida no llego una vez");
    assert_eq!(m.dimensionable(), 82.0, "fixture: el libro todavia retiene el cargo");
    // Duplicados (misma identidad): nada cambia.
    {
        let mut l = m.arena.config.entry_publication_gate.lock().unwrap();
        l.debito_funding(&m.arena, 7, 8.5, 150);
        l.fill_de_entrada(&m.arena, 0, "suelto", 8, Some(0.5), Some(150));
    }
    assert_eq!((ciclo_del_inmune(&m).0, m.dimensionable()), (91.0, 82.0), "duplicado");
    // Se suelta el debito (foto posterior al cargo): el freno sigue en 91, sin una segunda perdida.
    m.rest(91.0, None, 200);
    let (cap, pico, freno) = ciclo_del_inmune(&m);
    let (cap_r, pico_r) = m.riesgo();
    assert_eq!((cap, pico, cap_r, pico_r), (91.0, 100.0, 91.0, 100.0));
    assert_eq!(freno.caida_capital, 0.09);
    assert_eq!(m.dimensionable(), 91.0, "fixture: el cargo salio del libro");
}

/// F4: en frio, dato ausente con capacidad local positiva no escribe pico ni permite abrir. Cero y
/// deuda observados conservan la insolvencia. En las tres variantes, el cierre defensivo y la
/// cancelacion siguen disponibles (fuera de una entrada guardada la frontera cruza siempre).
#[test]
fn f4_frio_ausente_cero_y_deuda_conservan_protecciones() {
    let _g = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    for variante in ["ausente", "cero", "deuda"] {
        let mut m = montaje(1);
        assert!(m.dimensionable() > 0.0, "fixture: capacidad local positiva");
        match variante {
            "cero" => m.rest(0.0, None, 100),
            "deuda" => m.rest(10.0, Some(-15.0), 100),
            _ => {}
        }
        let (cap, pico, _) = ciclo_del_inmune(&m);
        let (_, pico_r) = m.riesgo();
        match variante {
            "ausente" => {
                assert!(!cap.is_finite(), "ausente: el freno recibio la capacidad {cap}");
                assert_eq!((pico, pico_r, m.pico()), (0.0, 0.0, 0.0), "ausente: se escribio un pico sin foto");
                assert!(m.arena.capital_incompleto.load(Ordering::Acquire), "ausente: entradas abiertas");
            }
            "cero" => assert!(cap == 0.0 && cap <= 0.0, "cero: no es insolvencia ({cap})"),
            // El puente acota la equity de la deuda en cero: sigue siendo insolvencia.
            _ => assert!(cap <= 0.0 && cap.is_finite(), "deuda: no es insolvencia ({cap})"),
        }
        let clave = ClaveReserva { coin_id: 0, slot: 0, generation: 1 };
        assert!(!m.arena.config.entry_publication_gate.lock().unwrap().admite(&m.arena, Some(m.dimensionable())) || variante != "ausente", "{variante}: admitida");
        assert!(quantum_arena::dimensionable::cruzar_frontera_de_envio().is_ok(), "{variante}: cierre o cancelacion bloqueados");
        let mut l = m.arena.config.entry_publication_gate.lock().unwrap();
        l.cancelar(&m.arena, clave, 0.0);
    }
}

/// F5: en la misma cuenta se pierde la evidencia, llega una cota menor y despues una ganancia no
/// conciliada: el pico se conserva, la cota menor actua y la ganancia no reabre. Con la foto valida
/// posterior, los dos consumidores adoptan la misma vista.
#[test]
fn f5_relevo_retiene_cota_conservadora_y_no_aprueba_ganancia() {
    let _g = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    let mut m = montaje(1);
    m.rest(100.0, None, 100);
    let _ = ciclo_del_inmune(&m);
    m.puente.evidencia_perdida(m.puente.era()).unwrap();
    assert!(m.arena.capital_incompleto.load(Ordering::Acquire), "fixture: evidencia perdida veta");
    m.rest_incompleta(100.0, -10.0, 150);
    let (cap, pico, freno) = ciclo_del_inmune(&m);
    assert_eq!((cap, pico), (90.0, 100.0), "la cota menor no actuo o el pico cambio");
    assert_eq!(freno.caida_capital, 0.1);
    m.rest_incompleta(100.0, 20.0, 160);
    let (cap, pico, _) = ciclo_del_inmune(&m);
    assert_eq!((cap, pico), (90.0, 100.0), "una ganancia no conciliada reabrio");
    assert!(m.arena.capital_incompleto.load(Ordering::Acquire), "la ganancia solto el veto");
    // Relevo de la MISMA cuenta: la cota conservada sigue siendo la evidencia del freno.
    let _ = relevo(&m, 1);
    let (cap, pico, _) = ciclo_del_inmune(&m);
    assert_eq!((cap, pico), (90.0, 100.0), "el relevo de la misma cuenta perdio la cota conservada");
    // Foto valida: los dos consumidores adoptan la misma vista.
    m.rest(104.0, None, 300);
    let (cap, pico, _) = ciclo_del_inmune(&m);
    let (cap_r, pico_r) = m.riesgo();
    assert_eq!((cap, pico, cap_r, pico_r), (104.0, 104.0, 104.0, 104.0));
}

/// F6: relevo a otra cuenta con el callback y la cache anteriores retenidos: ninguno escribe el
/// pico sucesor. Un reinicio restaura el pico compatible. Un pico legado (otra marca o sin ella)
/// requiere procedencia o veto y no se adopta aunque coincida con la cifra actual.
#[test]
fn f6_pico_persistido_y_cache_aislados_por_ambito() {
    let _g = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    let mut m = montaje(1);
    m.rest(100.0, None, 100);
    let _ = ciclo_del_inmune(&m);
    let _ = m.riesgo();
    assert_eq!((m.pico(), m.riesgo.peak_capital), (100.0, 100.0), "fixture: cache de Risk en 100");
    let ticket_viejo = m.puente.ticket();
    let sesion_vieja = m.puente.sesion();
    // Relevo a OTRA cuenta; el inmune toma el espacio nuevo (pico nace vacio, sin archivo).
    let _ = relevo(&m, 2);
    m.arena.capital_exchange.store(f64::NAN, Ordering::Relaxed);
    m.arena.pico_capital.store(0.0, Ordering::Release);
    // Cache de Risk y callback viejos retenidos: la REST de la sesion anterior y Risk con 100.
    let _ = m.puente.observar_rest_de_sesion(ticket_viejo, sesion_vieja, &lectura(100.0, None, 120, true));
    let (cap, pico, _) = ciclo_del_inmune(&m);
    let (_, pico_r) = m.riesgo();
    assert!(!cap.is_finite(), "la equity de la cuenta anterior quedo como evidencia del freno sucesor: {cap}");
    assert_eq!((pico, pico_r, m.pico()), (0.0, 0.0, 0.0), "un escritor puso el pico de la cuenta anterior en el sucesor");
    // Primer saldo de la cuenta nueva: su pico.
    m.rest(50.0, None, 10);
    let (cap, pico, _) = ciclo_del_inmune(&m);
    let (_, pico_r) = m.riesgo();
    assert_eq!((cap, pico, pico_r), (50.0, 50.0, 50.0));

    // Reinicio: pico compatible (marca vigente) restaurado; legado o sin marca, ilegible (veto).
    let dir = std::env::temp_dir().join(format!("tg_v4j_f6_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let rutas = drawdown::RutasPico::del_espacio("prueba_abc", |n| dir.join(n));
    drawdown::guardar_pico(&rutas.pico, 50.0).unwrap();
    drawdown::guardar_marca_definicion(&rutas.marca).unwrap();
    assert_eq!(drawdown::decidir_arranque_pico(&rutas).carga, CargaPico::Leido(50.0), "el pico compatible no se restauro");
    for legado in [Some("capital = walletBalance de USDT (CAPITAL-UNICO V2)\n"), None] {
        match legado {
            Some(t) => std::fs::write(&rutas.marca, t).unwrap(),
            None => std::fs::remove_file(&rutas.marca).unwrap(),
        }
        let a = drawdown::decidir_arranque_pico(&rutas);
        assert!(matches!(a.carga, CargaPico::Ilegible(_)), "legado {legado:?}: adoptado por su cifra ({:?})", a.carga);
        assert!(!a.rearmar_stops, "legado {legado:?}: rearmo los stops");
        assert!(!drawdown::marca_vigente(&rutas.marca));
    }
    // Con el pico ilegible, Risk no admite entradas (veto) aunque el capital sea el mismo.
    m.arena.pico_ilegible.store(true, Ordering::Relaxed);
    let intento = SignalIntent { signal: SignalType::Long, confidence: 0.9, win_probability: 0.9, expected_duration_ms: 60_000, ..Default::default() };
    let r = m.riesgo.evaluate_quantum_order(0, &intento, &m.arena);
    assert_eq!(r.signal, SignalType::Flat, "pico de procedencia incierta: entrada admitida");
    let _ = std::fs::remove_dir_all(&dir);
}

/// F7: sin el libro (acumulador: backtest, replay, simulacion) el freno conserva su contrato: la
/// equity propia si falta el exchange y el minimo si los dos planos de ese modo son validos.
#[test]
fn f7_acumulador_preserva_su_contrato() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    assert!(!arena.config.dimensionable_derivado.load(Ordering::Acquire));
    for (capital, exchange, esperado) in [(100.0, f64::NAN, 100.0), (100.0, 90.0, 90.0), (90.0, 100.0, 90.0), (90.0, 0.0, 0.0)] {
        arena.unified_capital.store(capital, Ordering::Relaxed);
        arena.capital_exchange.store(exchange, Ordering::Relaxed);
        assert_eq!(drawdown::capital_del_freno(&arena), esperado, "acumulador {capital} / {exchange}");
    }
}

/// F8: un flujo externo identificado (tranId y revision) con replay. Sin politica de ajuste: el
/// pico se conserva (no se rearma), la proteccion sigue midiendo la caida del patrimonio y el
/// flujo queda como incierto en la atribucion, nunca como perdida de mercado probada.
#[test]
fn f8_flujo_identificado_y_retiro_incierto_no_rearman() {
    let _g = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    let mut m = montaje(1);
    m.rest(100.0, None, 100);
    let _ = ciclo_del_inmune(&m);
    let mut flujos = FlujosExternos::default();
    assert_eq!(flujos.origen_de_la_caida(), OrigenDeLaCaida::Patrimonio);
    assert!(drawdown::es_flujo_externo("TRANSFER"));
    assert!(!drawdown::es_flujo_externo("FUNDING_FEE") && !drawdown::es_flujo_externo("COMMISSION") && !drawdown::es_flujo_externo("REALIZED_PNL"));
    assert_eq!(flujos.registrar(9_001, -30.0, m.puente.era()), RegistroDeFlujo::Nuevo);
    assert_eq!(flujos.registrar(9_001, -30.0, m.puente.era()), RegistroDeFlujo::Repetido, "replay");
    assert_eq!(flujos.registrar(9_002, f64::NAN, m.puente.era()), RegistroDeFlujo::Invalido);
    assert_eq!((flujos.cantidad(), flujos.neto_incierto()), (1, -30.0), "el replay conto dos veces");
    // La cuenta refleja el retiro: la caida se mide (proteccion), el pico no se rearma.
    m.rest(70.0, None, 200);
    let (cap, pico, freno) = ciclo_del_inmune(&m);
    let (_, pico_r) = m.riesgo();
    assert_eq!((cap, pico, pico_r), (70.0, 100.0, 100.0), "el retiro rearmo el pico o no se midio");
    assert_eq!(freno.caida_capital, 0.3);
    assert_eq!(flujos.origen_de_la_caida(), OrigenDeLaCaida::FlujoIncierto, "el retiro se etiqueto como perdida de mercado");
    assert!(m.arena.capital_incompleto.load(Ordering::Acquire) == false, "fixture: veto segun el estado del puente");
    assert!(texto(HOST).contains("sin politica de ajuste (no se etiqueta como perdida de mercado; pico y freno se conservan)"));
}
