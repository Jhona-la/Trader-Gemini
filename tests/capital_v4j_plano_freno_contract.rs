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
const OMS: &str = include_str!("../src/host_oms.rs");
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
    // Publicacion sincrona: el observador del puente publica la magnitud y su pico.
    orden(HOST, &[
        "puente_cuenta.vincular_freno(move |equity| {",
        "risk_engine::drawdown::publicar_equity_del_freno(&arena_freno, equity);",
    ]);
    // El lazo del inmune: lectura, magnitud y pico en UN paso bajo el mutex del puente.
    orden(HOST, &[
        "puente_imm.con_equity_del_freno(|_| {",
        "let espacio_nuevo = buzon_imm.lock().unwrap_or_else(|p| p.into_inner()).take();",
        "aplicar_arranque_pico(&arena_imm, &rutas, true);",
        // V4N (F2/F5): anclas adaptadas. El paso (buzon, magnitud, pico, revision y persistencia)
        // es `host_oms::paso_del_freno_del_inmune`; el bin le pasa su buzon y su estado del pico.
        "let paso_del_freno = quantum_engine::host_oms::paso_del_freno_del_inmune(",
        "&buzon_imm,",
        "&flujos_imm,",
        "&mut pico_imm,",
        "aplicar_arranque_pico,",
        "let freno = risk_engine::drawdown::freno_duro(&arena_imm, peak_capital, cap);",
        "let insolvent = cap <= 0.0;",
    ]);
    // V4N (F2/F5): el orden del paso, en la funcion productiva.
    orden(OMS, &[
        "pub fn paso_del_freno_del_inmune(",
        "puente.con_equity_del_freno(|equity| {",
        "let plano = risk_engine::drawdown::EscrituraDelPlano::abrir(arena);",
        "let espacio_nuevo = buzon.lock().unwrap_or_else(|p| p.into_inner()).take();",
        "let cap_exchange = equity.unwrap_or(f64::NAN);",
        "arena.capital_exchange.store(",
        "let cap = risk_engine::drawdown::capital_del_freno(arena);",
        "let pico = risk_engine::drawdown::registrar_pico(&arena.pico_capital, cap);",
        "revision = plano.revision_al_cerrar();",
        "risk_engine::drawdown::guardar_pico(&ruta_del_pico, pico)",
        "risk_engine::drawdown::guardar_marca_definicion(&estado.ruta_marca)",
    ]);
    // Relevo: dentro del cierre de `relevar_sesion` (mutex del puente), antes de la era nueva.
    orden(HOST, &[
        "let (paso, era_del_relevo) = puente_cuenta.relevar_sesion(|| {",
        "puente_cuenta.anunciar_cuenta(cuenta_del_relevo);",
        "if libro_relevo.cuenta().is_some_and(|c| c != cuenta_del_relevo) {",
        "arena_real.capital_exchange.store(f64::NAN, Ordering::Release);",
        "arena_real.pico_capital.store(0.0, Ordering::Release);",
        "libro_relevo.fijar_cuenta(&arena_real, cuenta_del_relevo);",
        "*buzon_espacio_pico.lock().unwrap_or_else(|p| p.into_inner()) =",
        "(paso, sesion_publicada)",
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
    orden(HOST, &[
        "let _ = puente_brk.con_era_vigente(era_brk, |_| {",
        "quantum_engine::host_oms::registrar_flujos_externos_de_era(",  // V4K (F8): ancla adaptada, el registro vive en host_oms
    ]);
    orden(HOST, &["risk_engine::drawdown::OrigenDeLaCaida::FlujoIncierto => format!("]);
    // Risk: solo lee el pico con el libro derivado.
    orden(RIESGO, &[
        "if arena.config.dimensionable_derivado.load(Ordering::Acquire) {",
        "let Some(vista) = crate::drawdown::vista_del_freno(arena) else {",  // V4L (L-F2): ancla adaptada, UNA vista por decision antes de validar el pico
        "self.peak_capital = vista.pico;",  // V4L (L-F2): ancla adaptada, el pico de la validacion es el de la vista (antes: carga suelta del arena)
        "let capital_freno = crate::drawdown::capital_del_freno(arena);",
        "if derivado {",
        "let Some(vista) = vista_de_la_decision else {",  // V4L (L-F2): ancla adaptada, la misma vista (antes: segunda lectura del plano)
        "self.peak_capital = if capital_freno.is_finite() { vista.pico.max(capital_freno) } else { vista.pico };",
        "} else {",
        "self.peak_capital = crate::drawdown::registrar_pico(",
    ]);
}

/// Un ciclo del inmune: (capital del freno, pico, freno), bajo el mutex del puente.
/// V4N (F2/F5): el paso es el PRODUCTIVO (`host_oms::paso_del_freno_del_inmune`), no una copia.
fn ciclo_del_inmune(m: &Montaje) -> (f64, f64, drawdown::FrenoDuro) {
    let p = m.paso.correr(&m.puente, &m.arena);
    (p.cap, p.pico, drawdown::freno_duro(&m.arena, p.pico, p.cap))
}

/// El relevo del host (`paso_transicion` dentro de `relevar_sesion`): cuenta anunciada, libro y,
/// con otra cuenta, magnitud en ausencia y pico del espacio nuevo vacio, en el mismo paso.
fn relevo(m: &Montaje, cuenta: u64) -> u64 {
    let sesion = m.oms.load(Ordering::SeqCst) + 1;
    let (_, era) = m.puente.relevar_sesion(|| {
        m.oms.store(sesion, Ordering::SeqCst);
        m.puente.anunciar_cuenta(cuenta);
        let mut libro = m.arena.config.entry_publication_gate.lock().unwrap();
        if libro.cuenta().is_some_and(|c| c != cuenta) {
            m.arena.capital_exchange.store(f64::NAN, Ordering::Release);
            m.arena.pico_capital.store(0.0, Ordering::Release);
        }
        libro.fijar_cuenta(&m.arena, cuenta);
        ((), Some(sesion))
    });
    let era = era.unwrap();
    m.puente.abrir_socket(era).unwrap();
    era
}// ========================== FIN DEL ADAPTADOR ==========================

struct Montaje {
    puente: Arc<PuenteCuenta>,
    arena: Arc<GlobalArena>,
    oms: Arc<AtomicU64>,
    riesgo: RiskEngine,
    /// V4N (F2/F5): el paso REAL del host con su estado.
    paso: PasoReal,
}

// ============================ PASO REAL DEL HOST (V4N, F2/F5) ============================
// CAPITAL-UNICO V4N (F2/F5, dictamen V4L :83 y :118): el paso del host en el ciclo del inmune ya no
// se copia: es `host_oms::paso_del_freno_del_inmune`, la MISMA funcion que llama el binario, con su
// estado (buzon del espacio, flujos y pico persistido en un directorio temporal propio).

static DIRECTORIOS_DEL_PASO: AtomicU64 = AtomicU64::new(0);

struct PasoReal {
    buzon: Mutex<Option<drawdown::RutasPico>>,
    flujos: Mutex<FlujosExternos>,
    pico: Mutex<quantum_engine::host_oms::PicoDelInmune>,
    dir: std::path::PathBuf,
}

fn paso_real() -> PasoReal {
    let n = DIRECTORIOS_DEL_PASO.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("tg_v4n_paso_{}_{}_{n}", env!("CARGO_CRATE_NAME"), std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let rutas = drawdown::RutasPico::del_espacio("v4n_paso", |n| dir.join(n));
    let pico = quantum_engine::host_oms::PicoDelInmune {
        pico_ilegible: false,
        marca_pendiente: !drawdown::marca_vigente(&rutas.marca),
        ruta_pico: rutas.pico.clone(),
        ruta_marca: rutas.marca.clone(),
        pico_guardado: 0.0,
    };
    PasoReal { buzon: Mutex::new(None), flujos: Mutex::new(FlujosExternos::default()), pico: Mutex::new(pico), dir }
}

impl PasoReal {
    fn correr(&self, puente: &PuenteCuenta, arena: &GlobalArena) -> quantum_engine::host_oms::PasoDelFreno {
        let mut pico = self.pico.lock().unwrap();
        quantum_engine::host_oms::paso_del_freno_del_inmune(puente, arena, &self.buzon, &self.flujos, &mut pico, |_, _, _| {
            panic!("montaje: el buzon del espacio del pico esta vacio en estas pruebas")
        })
    }
    /// El pico que el paso persistio en disco (None: no escribio).
    #[allow(dead_code)]
    fn pico_en_disco(&self) -> Option<f64> {
        let ruta = self.pico.lock().unwrap().ruta_pico.clone();
        std::fs::read_to_string(ruta).ok().and_then(|t| t.trim().parse().ok())
    }
}

impl Drop for PasoReal {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
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
    // ADAPTADOR (`vincular_freno` del host): publicacion sincrona de la magnitud y su pico.
    {
        let arena = Arc::clone(&arena);
        puente.vincular_freno(move |equity| drawdown::publicar_equity_del_freno(&arena, equity));
    }
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
    Montaje { puente, arena, oms, riesgo: RiskEngine::new(100.0), paso: paso_real() }
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
    /// Risk por su entrada real (decision de una entrada larga): la magnitud y el pico que usa.
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
            // Risk decide ANTES del ciclo del inmune: la foto aceptada ya publico su magnitud.
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
    // La decision de Risk inmediata (antes del ciclo del inmune) ya ve la cota menor.
    assert_eq!(m.riesgo(), (90.0, 100.0), "Risk decidio con la vista anterior a la cota");
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
    let era_vieja = m.puente.era();
    // Relevo a OTRA cuenta: magnitud en ausencia y pico del espacio nuevo vacio en el mismo paso.
    let _ = relevo(&m, 2);
    assert_eq!(m.pico(), 0.0, "el relevo dejo el pico de la cuenta anterior");
    // Un escritor retenido de la era anterior (su lectura de 100) no compromete nada: el testigo
    // de era se valida junto al commit, bajo el mutex del puente.
    let escrito = m.puente.con_era_vigente(era_vieja, |_| drawdown::registrar_pico(&m.arena.pico_capital, 100.0));
    assert!(escrito.is_none(), "el escritor retenido de la cuenta anterior comprometio su pico");
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
    let era = m.puente.era();
    // Un deposito no explica una caida.
    assert_eq!(flujos.registrar(1, 9_000, "USDT", 5.0, era), RegistroDeFlujo::Nuevo);
    assert_eq!(flujos.origen_de_la_caida(), OrigenDeLaCaida::Patrimonio, "un deposito se tomo por la causa de la caida");
    assert_eq!(flujos.registrar(1, 9_001, "USDT", -35.0, era), RegistroDeFlujo::Nuevo);
    assert_eq!(flujos.registrar(1, 9_001, "USDT", -35.0, era), RegistroDeFlujo::Repetido, "replay");
    assert_eq!(flujos.registrar(1, 9_002, "USDT", f64::NAN, era), RegistroDeFlujo::Invalido);
    // Una respuesta retenida de OTRA cuenta no se mezcla; otro activo no se suma como USDT.
    assert_eq!(flujos.registrar(2, 9_003, "USDT", -50.0, era), RegistroDeFlujo::OtroAmbito);
    assert_eq!(flujos.registrar(1, 9_004, "BNB", -1.0, era), RegistroDeFlujo::OtroActivo);
    assert_eq!((flujos.cantidad(), flujos.neto_incierto()), (3, -30.0), "el replay, la otra cuenta o el otro activo contaron en USDT");
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

// ================================ V4K: F2, F5, F6 y F8 ================================

/// V4K: el ciclo del inmune con su revision y su ambito.
/// V4N (F2/F5): el paso es el PRODUCTIVO (`host_oms::paso_del_freno_del_inmune`, antes un
/// adaptador copiado); la revision es la que el paso devuelve como testigo.
fn ciclo_del_inmune_k(m: &Montaje) -> (f64, f64, drawdown::FrenoDuro, u64, u64) {
    let p = m.paso.correr(&m.puente, &m.arena);
    assert_eq!(p.revision, m.arena.plano_freno_rev.load(Ordering::Acquire), "el testigo del paso no es la revision que dejo");
    (p.cap, p.pico, drawdown::freno_duro(&m.arena, p.pico, p.cap), p.revision, p.ambito)
}

fn anclas_v4k() {
    orden(HOST, &[
        // V4N (F2/F5): anclas adaptadas. El paso ya no esta en linea: el bin llama a la funcion que
        // estas pruebas ejecutan (`PasoReal::correr`); se ata la llamada y el uso del resultado.
        "let paso_del_freno = quantum_engine::host_oms::paso_del_freno_del_inmune(",
        "&mut pico_imm,",
        "(paso_del_freno.cap, paso_del_freno.pico, paso_del_freno.revision, paso_del_freno.ambito);",
        "let freno = risk_engine::drawdown::freno_duro(&arena_imm, peak_capital, cap);",
        "let patrimonial = !operator_lock && !lat_breach;",
        // V4L (L-F6): anclas adaptadas. La comprobacion del ambito, la toma del ejecutor y los
        // efectos locales (kill-switch de arena y ejecutor) pasan a `comprometer_disparo`, en el
        // mismo paso bajo el mutex del puente (antes: comprobacion bajo el mutex y efectos fuera).
        "let Some((executor, autoria)) = puente_imm.con_equity_del_freno(|_| {",
        "execution_engine::aplanado_inmune::comprometer_disparo(",
        "|| exec_imm.load_full(),",
        "patrimonial.then_some(ambito_de_la_evidencia),",
        "[SISTEMA INMUNE] ACTIVADO",
        "autoria_del_latch = Some(autoria);",
    ]);
    orden(HOST, &["risk_engine::drawdown::preparar_ambito_del_pico(&arena_real);", "puente_cuenta.vincular_freno(move |equity| {"]);
    orden(HOST, &[
        "if libro_relevo.cuenta().is_some_and(|c| c != cuenta_del_relevo) {",
        "risk_engine::drawdown::preparar_ambito_del_pico(&arena_real);",
        "libro_relevo.fijar_cuenta(&arena_real, cuenta_del_relevo);",
        "if espacio == espacio_pico && pico_pendiente_del_relevo {",
    ]);
    orden(HOST, &["fn aplicar_arranque_pico(", "risk_engine::drawdown::registrar_pico(&arena.pico_capital, guardado);", "risk_engine::drawdown::pico_restaurado(arena);", "(ilegible, marca_pendiente)"]);
    orden(HOST, &["let _ = puente_brk.con_era_vigente(era_brk, |_| {", "quantum_engine::host_oms::registrar_flujos_externos_de_era("]);
}

/// La decision real de Risk: (vetada por drawdown, vista con la que decidio).
fn decision_de_risk(m: &mut Montaje) -> (bool, Option<drawdown::VistaDelFreno>) {
    let intento = SignalIntent { signal: SignalType::Long, confidence: 0.9, win_probability: 0.9, expected_duration_ms: 60_000, ..Default::default() };
    let r = m.riesgo.evaluate_quantum_order(0, &intento, &m.arena);
    let vetada = r.signal == SignalType::Flat && risk_engine::ultimo_rechazo() == Some(risk_engine::REJ_DRAWDOWN);
    (vetada, m.riesgo.vista_del_freno())
}

/// V4K F2 (dictamen V4J :40 y :56): secuencia de Criterios :24 (dimensionable 95, equity 100, 103,
/// 100, 87) con los escritores intercalados. Se captura la decision REAL de Risk y la revision con
/// que decidio, y se compara con la decision del host (adaptador del lazo) bajo sus umbrales: misma
/// revision, misma equity, mismo pico 103 y misma decision del freno. Ademas, con un escritor a
/// mitad (pico de una observacion nueva sin su equity), Risk no decide con la mezcla: veta.
#[test]
fn v4k_f2_risk_y_host_deciden_con_la_misma_revision() {
    let _g = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    anclas_v4k();
    let mut m = montaje(1);
    m.rest(100.0, Some(0.0), 100);
    m.arena.config.entry_publication_gate.lock().unwrap().debito_cierre_local(&m.arena, ClaveReserva { coin_id: 0, slot: 1, generation: 1 }, 5.0);
    assert_eq!(m.dimensionable(), 95.0, "fixture");
    let _ = ciclo_del_inmune_k(&m);
    for (i, (upnl, t)) in [(3.0, 200), (0.0, 300), (-13.0, 400)].into_iter().enumerate() {
        m.rest(100.0, Some(upnl), t);
        if i % 2 == 0 {
            // Risk decide ANTES del ciclo del inmune: la observacion ya publico equity y pico.
            let (_, vista) = decision_de_risk(&mut m);
            let vista = vista.expect("Risk no leyo una revision estable");
            assert_eq!((vista.equity, vista.pico), (100.0 + upnl, 103.0), "Risk antes del inmune: mezcla de observaciones");
        }
        let (cap, pico, freno, revision, _) = ciclo_del_inmune_k(&m);
        let (vetada, vista) = decision_de_risk(&mut m);
        let vista = vista.expect("Risk no leyo una revision estable");
        assert_eq!(vista.revision, revision, "Risk y el host decidieron con revisiones distintas");
        assert_eq!((vista.equity, vista.pico), (cap, pico), "Risk y el host vieron otra equity o pico");
        assert_eq!((cap, pico), (100.0 + upnl, 103.0), "la serie");
        assert_eq!(vetada, freno.echado() || cap <= 0.0, "decision distinta entre Risk y el host con sus umbrales ({freno})");
        assert_eq!(freno.caida_capital, (103.0 - cap) / 103.0, "caida incoherente con la serie");
        assert_eq!(m.dimensionable(), 95.0, "la capacidad cambio con el PnL abierto");
    }
    // Escritor a mitad en OTRO hilo: pico de una observacion nueva (120) escrito, su equity no.
    let a = Arc::clone(&m.arena);
    let (dentro, seguir) = (Arc::new(std::sync::Barrier::new(2)), Arc::new(std::sync::Barrier::new(2)));
    let escritor = {
        let (dentro, seguir) = (Arc::clone(&dentro), Arc::clone(&seguir));
        std::thread::spawn(move || {
            let _p = drawdown::EscrituraDelPlano::abrir(&a);
            drawdown::registrar_pico(&a.pico_capital, 120.0);
            dentro.wait();
            seguir.wait();
            a.capital_exchange.store(120.0, Ordering::Release);
        })
    };
    dentro.wait();
    let (vetada, vista) = decision_de_risk(&mut m);
    assert!(vetada && vista.is_none(), "Risk decidio con la equity de una observacion y el pico de otra: {vista:?}");
    seguir.wait();
    escritor.join().unwrap();
    let (_, vista) = decision_de_risk(&mut m);
    let vista = vista.expect("revision estable tras el escritor");
    assert_eq!((vista.equity, vista.pico, vista.revision % 2), (120.0, 120.0, 0), "la vista posterior no es la del escritor");
}

/// V4K F5 (dictamen V4J :43): la secuencia conservadora de F5 con la decision REAL de Risk y su
/// revision comun con el host, al recibir la cota y al recuperar la foto valida.
#[test]
fn v4k_f5_cota_y_recuperacion_con_la_misma_revision() {
    let _g = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    let mut m = montaje(1);
    m.rest(100.0, None, 100);
    let _ = ciclo_del_inmune_k(&m);
    m.puente.evidencia_perdida(m.puente.era()).unwrap();
    m.rest_incompleta(100.0, -10.0, 150);
    let (cap, pico, freno, revision, _) = ciclo_del_inmune_k(&m);
    let (vetada, vista) = decision_de_risk(&mut m);
    let vista = vista.expect("vista de la cota");
    assert_eq!((cap, pico), (90.0, 100.0), "la cota menor no actuo o el pico cambio");
    assert_eq!((vista.equity, vista.pico, vista.revision), (cap, pico, revision), "Risk y el host no comparten la cota");
    assert!(vetada, "con la evidencia perdida Risk admitio una entrada");
    assert_eq!(freno.caida_capital, 0.1);
    m.rest_incompleta(100.0, 20.0, 160);
    let (cap, pico, _, _, _) = ciclo_del_inmune_k(&m);
    assert_eq!((cap, pico), (90.0, 100.0), "una ganancia no conciliada reabrio");
    let _ = relevo(&m, 1);
    let (cap, pico, _, _, _) = ciclo_del_inmune_k(&m);
    assert_eq!((cap, pico), (90.0, 100.0), "el relevo de la misma cuenta perdio la cota");
    m.rest(104.0, None, 300);
    let (cap, pico, freno, revision, _) = ciclo_del_inmune_k(&m);
    let (vetada, vista) = decision_de_risk(&mut m);
    let vista = vista.expect("vista de la foto valida");
    assert_eq!((cap, pico, vista.equity, vista.pico, vista.revision), (104.0, 104.0, 104.0, 104.0, revision));
    assert_eq!(vetada, freno.echado() || cap <= 0.0, "decision distinta con la foto valida");
}

/// ADAPTADOR V4K de `aplicar_arranque_pico` (libro derivado, anclado arriba): pico vacio, el
/// persistido compatible, y el ambito pasa a listo.
fn aplicar_arranque_k(m: &Montaje, rutas: &drawdown::RutasPico) {
    m.puente.con_equity_del_freno(|_| {
        let _plano = drawdown::EscrituraDelPlano::abrir(&m.arena);
        m.arena.pico_capital.store(0.0, Ordering::Release);
        if let CargaPico::Leido(g) = drawdown::decidir_arranque_pico(rutas).carga {
            drawdown::registrar_pico(&m.arena.pico_capital, g);
        }
        drawdown::pico_restaurado(&m.arena);
    });
}

/// V4K F6 (dictamen V4J :44, :58 y :67): arranque, persistencia del ambito, callback, cache y
/// relevo juntos. El ambito se prepara antes de vincular el observador: la primera equity (100)
/// llega antes de restaurar el pico persistido (120) y no es evidencia lista (ni pico ni admision;
/// Risk veta). Restaurado, host y Risk miden desde 120 con la misma revision. Un relevo a otra
/// cuenta intercalado entre la captura de la evidencia y la toma del ejecutor descarta la decision
/// del ambito anterior; sin relevo (control), el efecto se aplica.
#[test]
fn v4k_f6_arranque_restaura_antes_de_la_primera_evidencia_y_el_relevo_descarta_efectos() {
    let _g = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    anclas_v4k();
    let dir = std::env::temp_dir().join(format!("tg_v4k_f6_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let rutas = drawdown::RutasPico::del_espacio("prueba_v4k", |n| dir.join(n));
    drawdown::guardar_pico(&rutas.pico, 120.0).unwrap();
    drawdown::guardar_marca_definicion(&rutas.marca).unwrap();
    // Arranque: el host prepara el ambito antes de vincular el freno (anclas).
    let arena = GlobalArena::build_in_own_stack(100.0);
    drawdown::preparar_ambito_del_pico(&arena);
    let mut m = montaje_con_arena(1, arena);
    m.rest(100.0, None, 100);
    assert!(!m.arena.capital_exchange.load(Ordering::Acquire).is_finite(), "la primera equity se publico antes de restaurar el pico");
    assert_eq!(m.pico(), 0.0, "la primera equity escribio el pico antes de restaurar el persistido");
    // Veto efectivo de la entrada (pico vacio: `REJ_INVALID_INPUT`; pico pendiente con vista:
    // `REJ_DRAWDOWN`), nunca una decision con una vista lista.
    let intento = SignalIntent { signal: SignalType::Long, confidence: 0.9, win_probability: 0.9, expected_duration_ms: 60_000, ..Default::default() };
    let r = m.riesgo.evaluate_quantum_order(0, &intento, &m.arena);
    let vista = m.riesgo.vista_del_freno();
    assert!(
        r.signal == SignalType::Flat && !vista.is_some_and(|v| v.lista),
        "Risk decidio con el pico sin restaurar: {:?} vista={vista:?} rechazo={:?}",
        r.signal,
        risk_engine::ultimo_rechazo()
    );
    assert!(
        matches!(risk_engine::ultimo_rechazo(), Some(risk_engine::REJ_INVALID_INPUT) | Some(risk_engine::REJ_DRAWDOWN)),
        "veto por otra compuerta: {:?}",
        risk_engine::ultimo_rechazo()
    );
    let (cap, pico, _, _, _) = ciclo_del_inmune_k(&m);
    assert!(!cap.is_finite() && pico == 0.0, "el inmune uso la equity antes de restaurar ({cap}, {pico})");
    aplicar_arranque_k(&m, &rutas);
    let (cap, pico, freno, revision, ambito) = ciclo_del_inmune_k(&m);
    let (vetada, vista) = decision_de_risk(&mut m);
    let vista = vista.expect("vista restaurada");
    assert_eq!((cap, pico), (100.0, 120.0), "el pico persistido compatible no gobierna");
    assert_eq!((vista.equity, vista.pico, vista.revision, vista.ambito), (cap, pico, revision, ambito));
    assert_eq!(vetada, freno.echado() || cap <= 0.0);
    // Relevo intercalado entre la captura (ciclo) y la toma del ejecutor del disparo.
    let tomar = |m: &Montaje, capturado: u64| m.puente.con_equity_del_freno(|_| (drawdown::ambito_del_plano(&m.arena) == capturado).then_some(()));
    assert!(tomar(&m, ambito).is_some(), "control: sin relevo el efecto se aplica");
    let _ = relevo_k(&m, 2);
    assert!(tomar(&m, ambito).is_none(), "la decision del ambito anterior aplico sus efectos al sucesor");
    assert_eq!(m.pico(), 0.0, "el relevo dejo el pico de la cuenta anterior");
    m.rest(50.0, None, 10);
    let (cap, _, _, _, _) = ciclo_del_inmune_k(&m);
    assert!(!cap.is_finite(), "el sucesor uso evidencia antes de restaurar su pico");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Relevo del host con la preparacion del ambito (V4K), como `relevo` mas `preparar_ambito_del_pico`.
fn relevo_k(m: &Montaje, cuenta: u64) -> u64 {
    let sesion = m.oms.load(Ordering::SeqCst) + 1;
    let (_, era) = m.puente.relevar_sesion(|| {
        m.oms.store(sesion, Ordering::SeqCst);
        m.puente.anunciar_cuenta(cuenta);
        let mut libro = m.arena.config.entry_publication_gate.lock().unwrap();
        if libro.cuenta().is_some_and(|c| c != cuenta) {
            m.arena.capital_exchange.store(f64::NAN, Ordering::Release);
            m.arena.pico_capital.store(0.0, Ordering::Release);
            drawdown::preparar_ambito_del_pico(&m.arena);
        }
        libro.fijar_cuenta(&m.arena, cuenta);
        ((), Some(sesion))
    });
    let era = era.unwrap();
    m.puente.abrir_socket(era).unwrap();
    era
}

fn montaje_con_arena(cuenta: u64, arena: Arc<GlobalArena>) -> Montaje {
    // Mismo cableado que `montaje`, sobre el arena dado (el del arranque del host).
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
    puente.anunciar_cuenta(cuenta);
    {
        let arena = Arc::clone(&arena);
        puente.vincular_freno(move |equity| drawdown::publicar_equity_del_freno(&arena, equity));
    }
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
    Montaje { puente, arena, oms, riesgo: RiskEngine::new(100.0), paso: paso_real() }
}

/// V4K F8 (dictamen V4J :46 y :67): una respuesta de income retenida de la era anterior atraviesa
/// el registro PRODUCTIVO (`host_oms::registrar_flujos_externos_de_era`, el que llama el lazo del
/// host) despues del relevo: no registra nada en el sucesor. Control: la misma respuesta con la era
/// vigente registra; replay, otra cuenta y otro activo conservan su identidad; el pico y la
/// proteccion siguen la evidencia y la caida queda atribuida a un flujo incierto.
#[test]
fn v4k_f8_income_retenido_no_registra_en_el_sucesor() {
    let _g = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    anclas_v4k();
    let m = montaje(1);
    m.rest(100.0, None, 100);
    let _ = ciclo_del_inmune_k(&m);
    let flujos = Mutex::new(FlujosExternos::default());
    let entradas: Vec<execution_engine::order_types::IncomeEntry> = serde_json::from_str(
        r#"[{"symbol":"","incomeType":"TRANSFER","income":"-35","asset":"USDT","time":150,"tranId":9001,"tradeId":""},
            {"symbol":"","incomeType":"TRANSFER","income":"-1","asset":"BNB","time":151,"tranId":9002,"tradeId":""},
            {"symbol":"BTCUSDT","incomeType":"FUNDING_FEE","income":"-0.1","asset":"USDT","time":152,"tranId":9003,"tradeId":""}]"#,
    )
    .unwrap();
    let era_vieja = m.puente.era();
    // Relevo de la misma cuenta con la respuesta retenida: no registra en el sucesor.
    let _ = relevo_k(&m, 1);
    let r = quantum_engine::host_oms::registrar_flujos_externos_de_era(&m.puente, era_vieja, &m.arena, &flujos, &entradas);
    assert!(r.is_none(), "la respuesta retenida de la era anterior registro en el sucesor: {r:?}");
    assert_eq!(flujos.lock().unwrap().cantidad(), 0, "movimientos registrados en el sucesor");
    // Control: con la era vigente, registra los dos flujos externos (no el funding) una vez.
    let era = m.puente.era();
    let r = quantum_engine::host_oms::registrar_flujos_externos_de_era(&m.puente, era, &m.arena, &flujos, &entradas).expect("control: era vigente");
    assert_eq!(r, vec![(9001, RegistroDeFlujo::Nuevo), (9002, RegistroDeFlujo::OtroActivo)]);
    let r = quantum_engine::host_oms::registrar_flujos_externos_de_era(&m.puente, era, &m.arena, &flujos, &entradas).unwrap();
    assert_eq!(r, vec![(9001, RegistroDeFlujo::Repetido), (9002, RegistroDeFlujo::Repetido)], "replay");
    assert_eq!(flujos.lock().unwrap().neto_incierto(), -35.0);
    // Relevo a OTRA cuenta: la respuesta de la era anterior no registra; la del sucesor, con la
    // cuenta nueva, se rechaza por ambito (el registro sigue siendo de la cuenta 1).
    let era_cuenta_1 = m.puente.era();
    let _ = relevo_k(&m, 2);
    assert!(quantum_engine::host_oms::registrar_flujos_externos_de_era(&m.puente, era_cuenta_1, &m.arena, &flujos, &entradas).is_none());
    let r = quantum_engine::host_oms::registrar_flujos_externos_de_era(&m.puente, m.puente.era(), &m.arena, &flujos, &entradas).unwrap();
    assert!(r.iter().all(|(_, x)| *x == RegistroDeFlujo::OtroAmbito), "otra cuenta mezclo flujos: {r:?}");
    assert_eq!(flujos.lock().unwrap().origen_de_la_caida(), OrigenDeLaCaida::FlujoIncierto, "el retiro se etiqueto como perdida de mercado");
}
