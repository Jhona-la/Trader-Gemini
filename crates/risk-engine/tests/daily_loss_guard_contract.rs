//! DEMO-B (especificacion Demo §B.4, casos B1-B3; revision N1 de NVIDIA, cambio 7): nucleo puro
//! de la perdida diaria. Reloj, fotos de cuenta y persistencia inyectados (carpeta temporal).
//! Los valores esperados del ejemplo (gen 5 en frio) son ORACULOS de la especificacion; la ruta
//! de decision los calcula con las constantes y funciones del motor.

use risk_engine::drawdown::{drawdown_de_falsacion, ALFA_FALSACION, FRACCION_KELLY_MAXIMA};
use risk_engine::perdida_diaria::{
    cargar_estado, dia_utc, guardar_estado, presupuesto_diario, q_del_roster, tope_capital_del_freno, CausaNd,
    DecisionDiaria, EstadoDiario, GuardaDiaria, Transicion, MS_POR_DIA_UTC,
};
use risk_engine::ruin::{clamp_ruin, expected_loss_streak, CONSERVATIVE_Q};

/// `global_max_drawdown` del genoma Demo gen 5 (active.bin, SHA-256 39cbb631..., segun la
/// especificacion §B.2): dato de entrada del caso, no una constante del motor.
const GMDD_GEN5: f64 = 0.50;
const TOL: f64 = 1e-9;

fn dir_temporal(nombre: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "tg_perdida_diaria_{nombre}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&d).expect("carpeta temporal");
    d
}

// ------------------------------------------------------------------ B1 ----

#[test]
fn b1_frio_gen5_en_el_pico_da_6_4644_por_ciento() {
    let q = q_del_roster(&[(f64::NAN, 0.0), (0.0, 0.0)]).expect("roster frio");
    assert_eq!(q, CONSERVATIVE_Q, "frio: el q conservador del motor");
    let p = presupuesto_diario(100.0, 100.0, q, GMDD_GEN5).expect("presupuesto");
    assert!((p.d_d - 0.0646441763).abs() < TOL, "d_D = {}", p.d_d);
    assert!((p.b_d - 6.46441763).abs() < 1e-7, "B_D = {}", p.b_d);
    // Cada termino sale de su funcion del motor.
    assert_eq!(p.h, expected_loss_streak(q));
    assert_eq!(p.f_sizing, clamp_ruin(1.0, q));
    assert_eq!(p.f_sizing, 0.25);
    assert!((p.h - 10.37206655).abs() < 1e-7);
    assert_eq!(p.d_duro, GMDD_GEN5.min(drawdown_de_falsacion(FRACCION_KELLY_MAXIMA, ALFA_FALSACION)));
    assert!(((1.0 - p.d_unidad).powf(p.h) - (1.0 - p.d_duro)).abs() < 1e-12, "(1-d_unidad)^h = 1-d_duro");
    // E0 = 100, P0 = 190: el techo patrimonial manda (5 %); P0 = 200: el dia empieza en el freno.
    let p190 = presupuesto_diario(100.0, 190.0, q, GMDD_GEN5).expect("190");
    assert!((p190.d_d - 0.05).abs() < TOL, "{}", p190.d_d);
    let p200 = presupuesto_diario(100.0, 200.0, q, GMDD_GEN5).expect("200");
    assert_eq!(p200.d_d, 0.0);
    assert_eq!(p200.b_d, 0.0);
}

#[test]
fn b1_los_tres_techos_y_el_freno_vigente() {
    // d_duro es el tope del freno duro: el mismo valor que freno_duro con cualquier gen.
    let arena = quantum_arena::GlobalArena::build_in_own_stack(100.0);
    for gmdd in [0.5, 0.2, 0.9, 0.0, 1.0, f64::NAN, -0.3] {
        arena.config.global_max_drawdown.store(gmdd, std::sync::atomic::Ordering::Relaxed);
        let freno = risk_engine::drawdown::freno_duro(&arena, 100.0, 100.0);
        assert_eq!(freno.tope_capital.to_bits(), tope_capital_del_freno(gmdd).to_bits(), "gmdd {gmdd}");
    }
    let q = CONSERVATIVE_Q;
    // Con un gen permisivo manda la cota d* del freno; con uno estrecho, el gen.
    let ancho = presupuesto_diario(100.0, 100.0, q, 0.9).expect("ancho");
    assert_eq!(ancho.d_duro, drawdown_de_falsacion(FRACCION_KELLY_MAXIMA, ALFA_FALSACION));
    let estrecho = presupuesto_diario(100.0, 100.0, q, 0.05).expect("estrecho");
    assert!(estrecho.d_d < ancho.d_d);
    // Variar q cambia h y el presupuesto; con evidencia buena manda f_sizing o d_unidad, nunca mas.
    let q_bueno = q_del_roster(&[(0.9, 500.0)]).expect("evidencia");
    let p = presupuesto_diario(100.0, 100.0, q_bueno, GMDD_GEN5).expect("q bueno");
    assert!(p.h < expected_loss_streak(q));
    assert!(p.d_d <= p.f_sizing && p.d_d <= p.d_unidad && p.d_d <= p.d_restante);
    // El roster toma el peor simbolo (racha mas larga, presupuesto menor).
    let mezcla = q_del_roster(&[(0.9, 500.0), (0.0, 0.0)]).expect("mezcla");
    assert_eq!(mezcla, CONSERVATIVE_Q.max(q_bueno));
    // N/D con causa exacta.
    assert_eq!(q_del_roster(&[]), Err(CausaNd::SinRoster));
    assert_eq!(q_del_roster(&[(0.5, f64::NAN)]), Err(CausaNd::QInvalido));
    assert_eq!(presupuesto_diario(0.0, 100.0, q, GMDD_GEN5), Err(CausaNd::EquityInvalida));
    assert_eq!(presupuesto_diario(f64::NAN, 100.0, q, GMDD_GEN5), Err(CausaNd::EquityInvalida));
    assert_eq!(presupuesto_diario(100.0, f64::INFINITY, q, GMDD_GEN5), Err(CausaNd::PicoInvalido));
    assert_eq!(presupuesto_diario(100.0, 99.0, q, GMDD_GEN5), Err(CausaNd::PicoMenorQueEquity));
    assert_eq!(presupuesto_diario(100.0, 100.0, 1.0, GMDD_GEN5), Err(CausaNd::QInvalido));
}

// ------------------------------------------------------------------ B2 ----

fn dia_frio(e0: f64, p0: f64) -> EstadoDiario {
    EstadoDiario::abrir(7, 20_000, e0, p0, CONSERVATIVE_Q, GMDD_GEN5).expect("dia").0
}

#[test]
fn b2_frontera_inclusive_y_latch() {
    let mut d = dia_frio(100.0, 100.0);
    let suelo = d.e0 - d.b_d;
    assert_eq!(d.evaluar(suelo + 1e-6), DecisionDiaria::Permitida, "por debajo del limite pasa");
    assert!(!d.bloqueado);
    assert_eq!(d.evaluar(suelo), DecisionDiaria::DayLossLocked, "en el limite, inclusive");
    assert_eq!(d.evaluar(1_000.0), DecisionDiaria::DayLossLocked, "un rebote no rearma");
    let mut d2 = dia_frio(100.0, 100.0);
    assert_eq!(d2.evaluar(suelo - 1.0), DecisionDiaria::DayLossLocked, "por arriba se enclava");
    // B_D = 0 bloquea desde que se publica el dia.
    let d0 = dia_frio(100.0, 200.0);
    assert!(d0.bloqueado);
    assert_eq!(d0.clone().evaluar(150.0), DecisionDiaria::DayLossLocked);
}

#[test]
fn b2_flujos_externos_una_vez_por_identidad() {
    let mut d = dia_frio(100.0, 100.0);
    d.registrar_flujo(11, "USDT", -5.0).expect("retiro");
    assert_eq!(d.perdida(95.0), Some(0.0), "E=95 con retiro -5: perdida cero");
    let mut d = dia_frio(100.0, 100.0);
    d.registrar_flujo(12, "USDT", 20.0).expect("deposito");
    assert_eq!(d.perdida(113.0), Some(7.0), "E=113 con deposito +20: perdida 7");
    d.registrar_flujo(12, "USDT", 20.0).expect("misma identidad: idempotente");
    assert_eq!(d.perdida(113.0), Some(7.0));
    assert_eq!(d.registrar_flujo(12, "USDT", 21.0), Err(CausaNd::FlujoContradictorio));
    assert_eq!(d.nd, Some(CausaNd::FlujoContradictorio));
    assert_eq!(d.evaluar(200.0), DecisionDiaria::DayEquityNd(CausaNd::FlujoContradictorio), "ID contradictorio: N/D");
    let mut d = dia_frio(100.0, 100.0);
    assert_eq!(d.registrar_flujo(0, "USDT", 1.0), Err(CausaNd::FlujoSinIdentidad));
    let mut d = dia_frio(100.0, 100.0);
    assert_eq!(d.registrar_flujo(5, "BNB", 1.0), Err(CausaNd::FlujoOtroActivo));
    // Fee y funding ya estan en E: entran una vez por la equity, no como flujo.
    let d = dia_frio(100.0, 100.0);
    let equity_tras_fee_y_funding = 100.0 - 0.04 - 0.01;
    assert!((d.perdida(equity_tras_fee_y_funding).unwrap_or(f64::NAN) - 0.05).abs() < 1e-12);
    // Un deposito no rearma un dia enclavado.
    let mut d = dia_frio(100.0, 100.0);
    assert_eq!(d.evaluar(d.e0 - d.b_d), DecisionDiaria::DayLossLocked);
    d.registrar_flujo(99, "USDT", 50.0).expect("deposito");
    assert_eq!(d.evaluar(150.0), DecisionDiaria::DayLossLocked);
}

// ------------------------------------------------------------------ B3 ----

#[test]
fn b3_reloj_dos_dias_reinicio_y_cuenta() {
    let dir = dir_temporal("b3");
    let ruta = dir.join("perdida_diaria_7.txt");
    let dia = 20_400u64;
    let corte = dia * MS_POR_DIA_UTC;
    let mut g = GuardaDiaria::recuperar(7, &ruta, corte - 10_000);
    g.publicar_evidencia(vec![(0.0, 0.0)]);
    // Antes del corte: sin dia vigente, N/D.
    assert_eq!(g.observar(corte - 5_000, 100.0, 100.0, GMDD_GEN5), Transicion::SinAncla(CausaNd::SinAncla));
    assert_eq!(g.permite_apertura(corte - 4_000, 100.0), DecisionDiaria::DayEquityNd(CausaNd::SinAncla));
    // Dos observaciones consecutivas que bracketean el corte abren el dia (E0 = el mayor lado).
    match g.observar(corte + 5_000, 99.0, 100.0, GMDD_GEN5) {
        Transicion::DiaAbierto(p) => assert!((p.d_d - 0.0646441763).abs() < TOL),
        otra => panic!("se esperaba dia abierto: {otra:?}"),
    }
    let estado = g.estado.clone().expect("estado");
    assert_eq!((estado.dia, estado.e0, estado.p0), (dia, 100.0, 100.0));
    guardar_estado(&ruta, &estado).expect("persistir");
    let bytes1 = std::fs::read(&ruta).expect("leer");
    guardar_estado(&ruta, &estado).expect("idempotente");
    assert_eq!(std::fs::read(&ruta).expect("leer"), bytes1, "mismo estado, mismos bytes");
    // Cruce del suelo y latch, persistido.
    let suelo = estado.e0 - estado.b_d;
    assert_eq!(g.observar(corte + 3_600_000, suelo, 100.0, GMDD_GEN5), Transicion::Bloqueado);
    guardar_estado(&ruta, g.estado.as_ref().expect("estado")).expect("persistir latch");
    // Reinicio a las 12:00: misma base y latch, aunque la equity rebote.
    let doce = corte + 12 * 3_600_000;
    let mut r = GuardaDiaria::recuperar(7, &ruta, doce);
    assert_eq!(r.estado.as_ref().map(|e| (e.e0, e.bloqueado)), Some((100.0, true)));
    assert_eq!(r.permite_apertura(doce, 500.0), DecisionDiaria::DayLossLocked);
    // Reloj regresivo: se ignora y el estado se conserva.
    assert_eq!(r.observar(doce, 500.0, 500.0, GMDD_GEN5), Transicion::Nada);
    assert_eq!(r.observar(doce - 1_000, 500.0, 500.0, GMDD_GEN5), Transicion::RelojRegresivo);
    assert_eq!(r.permite_apertura(doce, 500.0), DecisionDiaria::DayLossLocked);
    // Otra cuenta (rekey no atestado) nunca toma base, pico ni latch de esta.
    let mut otra = GuardaDiaria::recuperar(8, &ruta, doce);
    assert!(otra.estado.is_none());
    assert_eq!(otra.permite_apertura(doce, 100.0), DecisionDiaria::DayEquityNd(CausaNd::SinAncla));
    assert_eq!(cargar_estado(&ruta, 8, dia), Err(CausaNd::CuentaDistinta));
    // Rollover valido (observaciones consecutivas a ambos lados del corte siguiente) rearma solo
    // el ledger diario; un salto de dia sin ancla sigue bloqueado.
    let corte2 = corte + MS_POR_DIA_UTC;
    r.publicar_evidencia(vec![(0.0, 0.0)]);
    assert_eq!(r.observar(corte2 - 2_000, 120.0, 130.0, GMDD_GEN5), Transicion::Nada);
    assert!(matches!(r.observar(corte2 + 2_000, 121.0, 130.0, GMDD_GEN5), Transicion::DiaAbierto(_)));
    assert_eq!(r.permite_apertura(corte2 + 3_000, 121.0), DecisionDiaria::Permitida);
    let mut salto = GuardaDiaria::recuperar(7, &ruta, corte2 + 3 * MS_POR_DIA_UTC);
    salto.publicar_evidencia(vec![(0.0, 0.0)]);
    assert_eq!(salto.observar(corte2 + 3 * MS_POR_DIA_UTC, 100.0, 100.0, GMDD_GEN5), Transicion::SinAncla(CausaNd::SinAncla));
    // Sin evidencia publicada, NaN o foto invalida: el dia no abre.
    let mut sin = GuardaDiaria::recuperar(9, &dir.join("otra.txt"), corte - 1);
    sin.observar(corte - 1, 100.0, 100.0, GMDD_GEN5);
    assert_eq!(sin.observar(corte + 1, 100.0, 100.0, GMDD_GEN5), Transicion::SinAncla(CausaNd::SinEvidencia));
    let mut nan = GuardaDiaria::recuperar(9, &dir.join("otra2.txt"), corte - 1);
    nan.publicar_evidencia(vec![(0.0, 0.0)]);
    nan.observar(corte - 1, 100.0, 100.0, GMDD_GEN5);
    assert_eq!(nan.observar(corte + 1, f64::NAN, 100.0, GMDD_GEN5), Transicion::SinAncla(CausaNd::EquityInvalida));
    let mut pico_bajo = GuardaDiaria::recuperar(9, &dir.join("otra3.txt"), corte - 1);
    pico_bajo.publicar_evidencia(vec![(0.0, 0.0)]);
    pico_bajo.observar(corte - 1, 100.0, 100.0, GMDD_GEN5);
    assert_eq!(pico_bajo.observar(corte + 1, 100.0, 50.0, GMDD_GEN5), Transicion::SinAncla(CausaNd::PicoMenorQueEquity));
    // Estado ilegible: el dia queda N/D, no se adopta.
    std::fs::write(dir.join("roto.txt"), "perdida_diaria v1\ncuenta=zz\n").expect("escribir");
    let mut roto = GuardaDiaria::recuperar(7, &dir.join("roto.txt"), doce);
    assert_eq!(roto.permite_apertura(doce, 100.0), DecisionDiaria::DayEquityNd(CausaNd::EstadoIlegible));
    assert_eq!(dia_utc(doce), dia);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn b3_flujos_del_corte_pertenecen_al_ancla() {
    let mut g = GuardaDiaria::recuperar(7, &std::env::temp_dir().join("tg_no_existe_b3f.txt"), 0);
    let corte = 20_500 * MS_POR_DIA_UTC;
    g.publicar_evidencia(vec![(0.0, 0.0)]);
    g.observar(corte - 1, 100.0, 100.0, GMDD_GEN5);
    assert!(matches!(g.observar(corte + 1, 100.0, 100.0, GMDD_GEN5), Transicion::DiaAbierto(_)));
    g.registrar_flujo(corte, 1, "USDT", 10.0).expect("del corte");
    g.registrar_flujo(corte - 50, 2, "USDT", 10.0).expect("de ayer");
    g.registrar_flujo(corte + 50, 3, "USDT", -4.0).expect("de hoy");
    let e = g.estado.as_ref().expect("estado");
    assert_eq!(e.flujos.len(), 1);
    assert_eq!(e.flujo_neto(), -4.0);
    // Ida y vuelta exacta del texto persistido.
    assert_eq!(EstadoDiario::deserializar(&e.serializar()).as_ref(), Ok(e));
}
