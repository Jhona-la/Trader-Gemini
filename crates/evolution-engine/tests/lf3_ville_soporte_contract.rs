//! LF-3 y LF-4 (Línea F) — el vigilante de rollback post-promoción.
//!
//! LF-3. El demonio revertía un genoma cuando un e-proceso que apuesta A FAVOR
//! del genoma perdía el 50 % desde su pico, con cada retorno recortado a
//! [−1, +1] R. Ese criterio no es una prueba: con varianza, la riqueza de
//! cualquier e-proceso cae tarde o temprano un 50 % desde su máximo, también
//! con ventaja positiva; y el recorte a +1 R hacía que vigilara la tasa de
//! acierto contra 0,5 en vez de la esperanza. La vigilancia nueva
//! (`VigilanciaDegradacion`) apuesta EN CONTRA (H0: E[X] ≥ 0 sobre el soporte
//! [−1, RR máx]) y revierte sólo si la riqueza supera 1/α: por Ville, revertir
//! un genoma con esperanza ≥ 0 tiene probabilidad ≤ α.
//!
//! LF-4. La vigilancia se armaba detrás del `return` del gate de evolución en
//! vivo: en prod (desarmado por defecto) nunca se armaba.
use evolution_engine::return_evidence::{
    SequentialVilleEvidence, VigilanciaDegradacion, soporte_superior_en_r,
};

const SEMILLAS: u64 = 400;

/// Generador congruencial (Knuth MMIX): trayectorias i.i.d. reproducibles.
struct Lcg(u64);
impl Lcg {
    fn uniforme(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `n` operaciones i.i.d.: con probabilidad `p` pagan `+rr` R, si no −1 R.
fn trayectoria(semilla: u64, p: f64, rr: f64, n: usize) -> Vec<f64> {
    let mut g = Lcg(semilla + 1);
    (0..n).map(|_| if g.uniforme() < p { rr } else { -1.0 }).collect()
}

/// Fracción de trayectorias que la vigilancia nueva revierte.
fn fraccion_revertida(p: f64, n: usize) -> f64 {
    let mut revertidas = 0u64;
    for s in 0..SEMILLAS {
        let mut v = VigilanciaDegradacion::new(0.05, 0.05, 0.50, 2.25).unwrap();
        for x in trayectoria(s, p, 2.25, n) {
            v.observe_r(x);
        }
        revertidas += v.degradada() as u64;
    }
    revertidas as f64 / SEMILLAS as f64
}

#[test]
fn lf3_testigo_el_criterio_anterior_revierte_genomas_rentables() {
    // 40 % de acierto con RR 2,25: esperanza +0,30 R por operación.
    let mut revertidas = 0u64;
    for s in 0..SEMILLAS {
        let mut v = SequentialVilleEvidence::with_bounds(0.05, 0.05, 0.50).unwrap();
        let mut degradado = false;
        for x in trayectoria(s, 0.40, 2.25, 200) {
            v.observe(x.clamp(-1.0, 1.0));
            degradado |= v.is_exhausted() || v.is_evidence_decayed(0.50);
        }
        revertidas += degradado as u64;
    }
    assert!(
        revertidas as f64 / SEMILLAS as f64 > 0.9,
        "testigo del defecto: el criterio de caída del 50 % desde el pico revierte \
         casi todos los genomas de +0,30 R ({revertidas}/{SEMILLAS})"
    );
}

#[test]
fn lf3_un_genoma_con_esperanza_no_negativa_se_revierte_con_probabilidad_a_lo_sumo_alfa() {
    // p* = 1/(1+2,25) ≈ 0,3077 es el punto de equilibrio con RR 2,25.
    for (p, nombre) in [(0.40, "+0,30 R"), (0.35, "+0,14 R"), (1.0 / 3.25, "0 R")] {
        let f = fraccion_revertida(p, 1000);
        // α = 0,05 más el error de Monte Carlo con 400 trayectorias (~2,2 σ).
        assert!(f <= 0.075, "esperanza {nombre}: revertida en {f} de las trayectorias (> α)");
    }
}

#[test]
fn lf3_un_genoma_perdedor_se_revierte() {
    // 10 % de acierto: −0,675 R por operación; 20 %: −0,35 R.
    assert_eq!(fraccion_revertida(0.10, 200), 1.0);
    assert!(fraccion_revertida(0.20, 1000) > 0.95);
}

#[test]
fn lf3_el_retorno_se_acota_al_soporte_y_los_no_finitos_se_ignoran() {
    let mut a = VigilanciaDegradacion::new(0.05, 0.05, 0.50, 2.25).unwrap();
    let mut b = VigilanciaDegradacion::new(0.05, 0.05, 0.50, 2.25).unwrap();
    for x in [1000.0, -50.0, f64::NAN, f64::NEG_INFINITY] {
        a.observe_r(x);
    }
    b.observe_r(2.25);
    b.observe_r(-1.0);
    assert_eq!(a.e_valor(), b.e_valor());
    assert_eq!(a.observaciones(), 2);
}

#[test]
fn lf3_soporte_invalido_se_rechaza() {
    for x in [0.5, f64::NAN, f64::INFINITY, -1.0] {
        assert!(VigilanciaDegradacion::new(0.05, 0.05, 0.50, x).is_err(), "soporte = {x}");
    }
}

#[test]
fn lf3_soporte_superior_es_el_rr_maximo_y_nunca_menor_que_uno() {
    assert_eq!(soporte_superior_en_r(&[(0.009, 0.004), (0.0225, 0.01)]), 2.25);
    assert_eq!(soporte_superior_en_r(&[(0.002, 0.004)]), 1.0);
    assert_eq!(soporte_superior_en_r(&[(f64::NAN, 0.01), (0.01, 0.0), (-1.0, 0.01)]), 1.0);
    assert_eq!(soporte_superior_en_r(&[]), 1.0);
}

/// LF-4 — guarda de fuente: el armado de la vigilancia precede al gate de la
/// evolución en vivo. No prueba la conducta del demonio completo (exige un
/// almacén de genomas en disco); fija el orden que causó el defecto.
#[test]
fn lf4_la_vigilancia_se_arma_antes_del_gate_de_evolucion_en_vivo() {
    let src = include_str!("../src/online_daemon.rs");
    let armado = src
        .find("[WATCHDOG-EXTERN] Promoción/restauración externa detectada")
        .expect("el armado por promoción externa existe");
    let gate = src
        .find("if !live_evolution_armed_for_env() {\n            return;\n        }")
        .expect("el gate de evolución en vivo existe");
    assert!(
        armado < gate,
        "el armado de la vigilancia de rollback debe quedar antes del gate de evolución en vivo"
    );
}
