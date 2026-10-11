//! C-F3 · F3.1 (Línea C): el consenso espectral de 13 motores frente a un
//! nulo martingala. Eventos sintéticos de profundidad, sin red.
//!
//! Teoría que se prueba:
//! - En una caminata aleatoria log-normal (martingala) ninguna escala tiene
//!   información direccional: el IC verdadero de cualquier voto es 0. Un
//!   veredicto que decide (Long/Short) en más del α = 5 % de las
//!   evaluaciones con dominante activo decide sobre ruido (F3.1-01).
//! - Una escala τ que todavía no ha visto un período completo no sabe nada
//!   de su escala (D-742): su z es el retorno desde el arranque. El
//!   dominante no puede salir de ella: τ_dom ≤ tiempo transcurrido
//!   (F3.1-02).
//! - El motor vivo del solitón es par en el signo de la amplitud
//!   (`compute_soliton_amplitude` usa |amplitud|); su sombra espectral debe
//!   serlo también (F3.1-05 / F3.6-02).
//! - La concentración espectral promete «0 = masa difusa (ruido blanco)»
//!   (proyeccion_espectral.rs): en una caminata debe ser baja (F3.1-06).
//!
//! Convención (fase C-F3, docs/audit/HALLAZGOS_C-F3_2026-10-11.md): un test
//! que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto; se
//! activa con el arreglo. Para correrlos: `cargo test ... -- --include-ignored`.

use god_engine_core::GodEngineCore;
use quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS;
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use signal_engine::SignalType;
use std::sync::{Arc, Mutex};

const SIM: &str = "CF3ESPUSDT";
const T0: u64 = 1_700_000_000_000;

/// El universo y el registro de símbolos son globales del proceso.
static ENTORNO: Mutex<()> = Mutex::new(());

/// LCG de 64 bits (constantes de Knuth/MMIX) con salida de 53 bits en (0, 1).
struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    /// Normal estándar por Box-Muller.
    fn n(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn nucleo() -> (Arc<GlobalArena>, GodEngineCore) {
    symbols::update_dynamic_universe(vec![SIM.into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(SIM)]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
    (arena, core)
}

/// Evento de profundidad puro con spread de 1 pb: actualiza el espectro y
/// compone el consenso de 13 motores (sombras y veredicto espectral).
fn profundidad(core: &mut GodEngineCore, mid: f64, t_ms: u64) {
    let medio = mid * 0.5e-4;
    let _ = core.process_event(
        0,
        false,
        false,
        true,
        mid,
        0.0,
        mid - medio,
        mid + medio,
        5.0,
        5.0,
        0.0,
        0.0,
        t_ms,
        false,
        &[0.0; 54],
        false,
    );
}

/// Nulo martingala: σ = 2 pb por evento, 1 evento/s, 7 200 eventos (2 h).
/// Verdad: ninguna dirección tiene ventaja, así que la fracción de
/// evaluaciones con dominante activo (v ≠ 0, τ ≥ 30 s) que acaban en
/// Long/Short debe ser ≤ α = 0,05. Con ≥ 50 evaluaciones activas el error
/// binomial de la fracción bajo un α verdadero de 0,05 es ≤ √(0,05·0,95/50)
/// ≈ 0,031; la prueba exige el nominal, no una banda: decidir más a menudo
/// que α es exactamente el defecto. Se lee también la fracción que hoy
/// supera el corte, para el informe.
#[test]
#[ignore = "OPEN F3.1-01 (Qoder, espectro y consenso): el veredicto espectral es una sola fuente (momentum_z) votada 13 veces y un argmax sin nulo; el corte sale del gen min_confidence, no del nulo de |v_dom| en ruido, así que decide en una martingala"]
fn cf3_en_ruido_el_veredicto_espectral_no_decide() {
    let _g = ENTORNO.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, mut core) = nucleo();
    let mut rng = Lcg(0xF31_0001);
    let mut mid = 100.0f64;
    let (mut activas, mut decididas) = (0usize, 0usize);
    let mut suma_v = 0.0f64;
    for i in 0..7_200u64 {
        // Martingala: ninguna señal tiene IC distinto de 0.
        mid *= 1.0 + 2e-4 * rng.n();
        profundidad(&mut core, mid, T0 + i * 1_000);
        if i < 600 {
            continue;
        }
        let v = arena
            .registry
            .get_for_coin_or(0, "consenso_espectral_dominante", 0.0);
        let tau = arena
            .registry
            .get_for_coin_or(0, "consenso_espectral_tau", 0.0);
        if v.abs() > 1e-9 && tau >= TAU_ANCHOR_FAST_MS {
            activas += 1;
            suma_v += v.abs();
            let d = core
                .tensor_orchestrator
                .evaluate_continuous_consensus_for_coin(0, SIM);
            if d.signal != SignalType::Flat {
                decididas += 1;
            }
        }
    }
    let frac = decididas as f64 / activas.max(1) as f64;
    println!(
        "observado: {decididas}/{activas} evaluaciones activas deciden ({:.1} %), |v_dom| medio {:.3}",
        100.0 * frac,
        suma_v / activas.max(1) as f64
    );
    assert!(
        activas >= 50,
        "el veredicto espectral apenas se activó ({activas}); prueba no concluyente"
    );
    assert!(
        frac <= 0.05,
        "en ruido puro el veredicto espectral decide en {:.1} % de las evaluaciones activas ({decididas}/{activas})",
        100.0 * frac
    );
}

/// Misma caminata, 7 200 eventos; tras 30 min de calentamiento, en todo
/// evento con dominante no nulo, τ_dom ≤ tiempo transcurrido desde el
/// primer evento (D-742: una escala que no ha visto su τ no sabe nada de
/// ella). Tolerancia: 0 violaciones.
#[test]
#[ignore = "OPEN F3.1-02 (Qoder, espectro y consenso): el consenso de 13 motores no aplica la masa D-742 en la selección del dominante; las escalas no observadas comparten z ≈ retorno desde el arranque y ganan el argmax"]
fn cf3_el_dominante_no_sale_de_una_escala_no_observada() {
    let _g = ENTORNO.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, mut core) = nucleo();
    let mut rng = Lcg(0xF31_0002);
    let mut mid = 100.0f64;
    let (mut con_voto, mut violaciones) = (0usize, 0usize);
    let mut peor_razon = 0.0f64;
    for i in 0..7_200u64 {
        mid *= 1.0 + 2e-4 * rng.n();
        let t = T0 + i * 1_000;
        profundidad(&mut core, mid, t);
        if i < 1_800 {
            continue; // 30 min de calentamiento
        }
        let v = arena
            .registry
            .get_for_coin_or(0, "consenso_espectral_dominante", 0.0);
        let tau = arena
            .registry
            .get_for_coin_or(0, "consenso_espectral_tau", 0.0);
        let transcurrido = (t - T0) as f64;
        if v.abs() > 1e-9 {
            con_voto += 1;
            if tau > transcurrido {
                violaciones += 1;
                peor_razon = peor_razon.max(tau / transcurrido);
            }
        }
    }
    println!(
        "observado: {violaciones}/{con_voto} dominantes con τ > transcurrido (peor τ/transcurrido = {peor_razon:.1})"
    );
    assert!(con_voto > 0, "el consenso no produjo ningún dominante");
    assert_eq!(
        violaciones, 0,
        "{violaciones}/{con_voto} dominantes con τ mayor que el tiempo observado (peor τ/transcurrido = {peor_razon:.1})"
    );
}

/// Corre la caminata con el OFI per-coin fijado a `ofi` antes de cada
/// evento: la sombra del solitón lee `order_flow_imbalance` antes de que
/// process_tick_dual lo reescriba en el mismo evento.
fn soliton_con_ofi(ofi: f64) -> f64 {
    let (arena, mut core) = nucleo();
    let mut rng = Lcg(0xF31_0005);
    let mut mid = 100.0f64;
    for i in 0..1_800u64 {
        mid *= 1.0 + 2e-4 * rng.n();
        arena.registry.set_for_coin(0, "order_flow_imbalance", ofi);
        profundidad(&mut core, mid, T0 + i * 1_000);
    }
    arena
        .registry
        .get_for_coin_or(0, "sombra_soliton_amplitud_max", f64::NAN)
}

/// Verdad: el solitón vivo es par en el signo de la amplitud
/// (`compute_soliton_amplitude` toma |amplitud|), así que su sombra
/// espectral debe dar los mismos votos por escala con OFI +0,5 y −0,5
/// (|Δ| < 1e-12, error de redondeo de tanh). Hoy `voto_espectral` sustituye
/// amp ≤ 0 por 1,0: con +0,5 vota tanh(0,5·x) y con −0,5 vota tanh(x).
#[test]
#[ignore = "OPEN F3.1-05 / F3.6-02 (Qoder, sombra espectral del consenso): voto_espectral sanea amp ≤ 0 a 1,0 y usa amp > 0 tal cual; la sombra pasa el OFI firmado, así que su pendiente depende del lado del flujo (el vivo usa |OFI|)"]
fn cf3_el_soliton_espectral_es_par_en_el_signo_del_ofi() {
    use signal_engine::soliton_wave::SolitonWaveEngine as S;
    let _g = ENTORNO.lock().unwrap_or_else(|p| p.into_inner());
    // Referencia: el motor vivo es par en el signo de la amplitud.
    assert_eq!(
        S::compute_soliton_amplitude(0.5, 0.0, 0.3, 0.05),
        S::compute_soliton_amplitude(-0.5, 0.0, 0.3, 0.05)
    );
    let mut x = [0.0f64; 32];
    for (k, xi) in x.iter_mut().enumerate() {
        *xi = (k as f64 - 15.5) / 8.0;
    }
    let pos = S::voto_espectral(&x, 0.5);
    let neg = S::voto_espectral(&x, -0.5);
    for k in [0usize, 15, 31] {
        println!(
            "observado: escala {k}, x = {:+.4}: OFI +0,5 → {:+.6}, OFI −0,5 → {:+.6}",
            x[k],
            pos.en_escala(k),
            neg.en_escala(k)
        );
    }
    for k in 0..32 {
        assert!(
            (pos.en_escala(k) - neg.en_escala(k)).abs() < 1e-12,
            "escala {k}: {} con OFI +0,5 frente a {} con OFI -0,5",
            pos.en_escala(k),
            neg.en_escala(k)
        );
    }
    let (a, b) = (soliton_con_ofi(0.5), soliton_con_ofi(-0.5));
    assert!(
        a.is_finite() && b.is_finite(),
        "la sombra no publicó su amplitud"
    );
    assert!(
        (a - b).abs() < 1e-12,
        "amplitud dominante {a} (OFI +0,5) frente a {b} (OFI -0,5)"
    );
}

/// Verdad: en una caminata aleatoria (ruido blanco en retornos) la
/// concentración debe quedar en el extremo «difuso», ≤ 0,2, como promete el
/// doc de proyeccion_espectral.rs («0 = masa difusa (ruido blanco)»). 0,2
/// equivale a una razón de participación ≥ 25,6 de 32 escalas: holgura de
/// sobra frente al ideal 32 de pesos iguales.
#[test]
#[ignore = "OPEN F3.1-06 (Qoder, espectro): la concentración es 1 − PR/32 con N = 32 fijo; mide cuántas escalas no tienen evidencia (piso de fusion_weight × masa), no estructura, y en ruido blanco queda ≈ 0,9"]
fn cf3_la_concentracion_espectral_de_un_ruido_es_baja() {
    let _g = ENTORNO.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, mut core) = nucleo();
    let mut rng = Lcg(0xF31_0006);
    let mut mid = 100.0f64;
    for i in 0..3_600u64 {
        mid *= 1.0 + 2e-4 * rng.n();
        profundidad(&mut core, mid, T0 + i * 1_000);
    }
    let c = arena
        .registry
        .get_for_coin_or(0, "espectral_concentracion", f64::NAN);
    let masa = arena
        .registry
        .get_for_coin_or(0, "espectral_masa_resuelta", f64::NAN);
    println!("observado: concentración {c:.4}, masa resuelta {masa:.4}");
    assert!(c.is_finite(), "concentración no publicada");
    assert!(
        c <= 0.2,
        "ruido blanco con concentración {c:.3} (masa resuelta {masa:.3}); el doc promete 0 = ruido"
    );
}
