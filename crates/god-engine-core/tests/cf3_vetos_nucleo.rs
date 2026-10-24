//! C-F3.4 (Línea C): vetos del núcleo contra soluciones conocidas.
//!
//! Teoría que se prueba:
//! - Escudo de momento adverso (F3.4-03). La regla viva (lib.rs, D-758 +
//!   Ola XLI·D5) bloquea si algún eje tipificado entre micro, macro (EMA
//!   9-21) y higher (p − EMA120) supera Z95, o si el Stouffer
//!   S = Σz/√3 lo supera, siempre que el eje secular (p − EMA720) no se
//!   oponga. Bajo una caminata aleatoria la tasa nominal por lado es 0,025.
//!   Pero los ejes de vela salen de la misma caminata (ρ = 0,796 / 0,700 /
//!   0,391), así que Var(S) = 1,53 y P(S > Z95) ≈ 0,057, y la regla es una
//!   unión de pruebas sin corrección de multiplicidad. Con un z_micro
//!   independiente, el escudo bloquea ≈ 0,085 por lado.
//! - Muro del libro (F3.4-04). `muro_en_contra(b, a, r, largo)` equivale a
//!   OBI = (b − a)/(b + a) ≤ (1 − r)/(1 + r): es el OBI de nivel 0 con otro
//!   umbral (documenta la no independencia; pasa).
//! - Veto de drawdown del consejo (F3.4-06). SeniorRiesgo veta con
//!   dd > 0,95 − 0,10·s ∈ [0,85; 0,95]; el risk-engine rechaza antes, con el
//!   mismo capital y pico, en dd ≥ umbral ≤ d* = drawdown_de_falsacion(½, 0,05)
//!   ≈ 0,632. Ningún DD que llegue al consejo alcanza su veto.
//! - Test de integración K-26 (F3.4-08). El fixture escribe
//!   ml_model_base = 0,25 por moneda, pero el núcleo sin bosque la reescribe
//!   con `unwrap_or(0.5)` antes del bloque conformal del mismo tick.
//! - Sonda de arranque frío (F3.4-05; la parte de D4, cerrada por CL-98). D4 (LEDGER:212):
//!   una sola sonda abierta en toda la cartera. El núcleo lo decide con
//!   `risk_engine::sonda::sonda_sin_modelo`, la misma función del veto del
//!   riesgo.
//!
//! La prueba del escudo replica literalmente una regla que hoy vive en línea
//! dentro de `process_event` (privado). Para que no pruebe una copia, empieza
//! comprobando que la réplica coincide con el código de producción de lib.rs
//! (sin espacios, antes del primer `#[cfg(test)]`). Si el código cambia, esa
//! premisa falla y el test dice qué extraer. La de la sonda ya llama a la
//! función de producción y su premisa comprueba que el núcleo la usa.
//!
//! Convención: un test que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto. Para
//! correrlos: `cargo test ... -- --include-ignored`.

use god_engine_core::calibration::muro_en_contra;
use god_engine_core::diffusion::{
    ema_distance_z, ema_spread_z, EMA_FAST_BARS, EMA_MACRO_BARS, EMA_SLOW_BARS, EMA_TREND_BARS,
    PARKINSON_RANGE_FACTOR, Z95,
};
use god_engine_core::GodEngineCore;
use metacortex_engine::{ConsejoDeliberacion, MarketSnapshotPayload, SeniorRole};
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use risk_engine::drawdown::{drawdown_de_falsacion, ALFA_FALSACION, FRACCION_KELLY_MAXIMA};
use std::sync::{Arc, Mutex};

static CORE_ENV: Mutex<()> = Mutex::new(());

/// Código de producción de lib.rs, sin espacios ni saltos de línea.
fn codigo_produccion() -> String {
    include_str!("../src/lib.rs")
        .split("#[cfg(test)]")
        .next()
        .expect("lib.rs vacío")
        .split_whitespace()
        .collect()
}

/// xorshift64* con salida de 53 bits en (0, 1) y normal por Box-Muller.
struct Xs(u64);
impl Xs {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn n(&mut self) -> f64 {
        let (a, b) = (self.u(), self.u());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
}

// ─────────────────────────────── Escudo ───────────────────────────────

/// Réplica literal de la regla viva (lib.rs, bloque «INVARIANTE DE
/// MOMENTUM ADVERSO MULTIESCALA»). Propuesta: extraer
/// `pub fn escudo_momento_adverso(z_micro, z_macro, z_higher, z_secular) -> (bool, bool)`
/// en diffusion.rs para que el núcleo y este test usen la misma función.
/// Devuelve (bloquea cortos, bloquea largos).
fn escudo(zm: f64, zma: f64, zh: f64, zs: f64) -> (bool, bool) {
    let ejes = [zm, zma, zh];
    let s = ejes.iter().sum::<f64>() / (ejes.len() as f64).sqrt();
    let corto = (ejes.iter().any(|z| *z > Z95) || s > Z95) && zs > -Z95;
    let largo = (ejes.iter().any(|z| *z < -Z95) || s < -Z95) && zs < Z95;
    (corto, largo)
}

/// Premisa: la réplica `escudo` es la regla de producción.
fn escudo_es_el_vivo() {
    let src = codigo_produccion();
    for trozo in [
        "letz_higher_dir=crate::diffusion::ema_distance_z(higher_trend,atr_pct,crate::diffusion::EMA_TREND_BARS",
        "letz_macro=crate::diffusion::ema_spread_z(macro_trend,atr_pct,crate::diffusion::EMA_FAST_BARS,crate::diffusion::EMA_SLOW_BARS",
        "letz_secular=crate::diffusion::ema_distance_z(secular_trend,atr_pct,crate::diffusion::EMA_MACRO_BARS",
        "letz95_adv=crate::diffusion::Z95;",
        "letejes_adv=[z_micro,z_macro,z_higher_dir];",
        "letstouffer_adv=ejes_adv.iter().sum::<f64>()/(ejes_adv.len()asf64).sqrt();",
        "is_adverse_momentum_short=(ejes_adv.iter().any(|z|*z>z95_adv)||stouffer_adv>z95_adv)&&z_secular>-z95_adv;",
        "is_adverse_momentum_long=(ejes_adv.iter().any(|z|*z<-z95_adv)||stouffer_adv<-z95_adv)&&z_secular<z95_adv;",
    ] {
        assert!(
            src.contains(trozo),
            "premisa: la regla del escudo cambió en lib.rs (falta «{trozo}»); actualizar la réplica o extraer escudo_momento_adverso"
        );
    }
}

/// Caminata log-normal con σ = 1e-3 por barra de 1 min, ATR = σ·√(8/π)
/// (diffusion.rs: σ = ATR/PARKINSON), EMAs 9/21/120/720 sobre el precio y
/// las z de diffusion.rs. z_micro es una N(0, 1) independiente: el caso más
/// favorable al nominal.
///
/// Tolerancia: tasa ≤ 0,05 por lado, el doble de la nominal 0,025. El error
/// de Monte Carlo no la decide: con 2·10⁶ barras medidas, la memoria más
/// larga (EMA720) deja del orden de 2·10⁶/720 ≈ 2 800 bloques independientes
/// y un SE de la tasa ≈ √(0,085·0,915/2 800) ≈ 0,005 en el peor caso, así
/// que 0,085 queda a ≈ 7 SE por encima de 0,05 incluso con esa cota
/// pesimista; la parte rápida (micro, macro, higher) da un SE mucho menor.
/// Corre en menos de un segundo en debug.
///
/// El escéptico de completitud separa dos causas: (a) Stouffer mal
/// especificado sobre ejes correlacionados (denominador √(1ᵀΣ1)); (b) la
/// unión de cuatro pruebas sin control de multiplicidad, que es una decisión
/// de diseño (FWER por lado). Arreglar sólo (a) deja la tasa ≈ 0,080: el
/// test exige el contrato conjunto. Imprime ambas partes por separado.
#[test]
#[ignore = "OPEN F3.4-03 (Línea C): el escudo une tres pruebas z y un Stouffer Σz/√3 sobre ejes correlacionados (Var(S) = 1,53): bloquea ≈ 0,085 por lado en caminata aleatoria, no 0,025"]
fn cf3_stouffer_tasa_en_ruido() {
    escudo_es_el_vivo();
    let sigma = 1e-3;
    let atr = sigma * PARKINSON_RANGE_FACTOR; // σ = ATR/√(8/π), diffusion.rs
    let al = |n: f64| 2.0 / (n + 1.0);
    let (a9, a21, a120, a720) = (
        al(EMA_FAST_BARS),
        al(EMA_SLOW_BARS),
        al(EMA_TREND_BARS),
        al(EMA_MACRO_BARS),
    );
    let mut g = Xs(0x9E37_79B9_7F4A_7C15);
    let mut p = 100.0_f64;
    let (mut e9, mut e21, mut e120, mut e720) = (p, p, p, p);
    let (calent, n) = (8_000usize, 2_000_000usize);
    let (mut bl, mut bc) = (0usize, 0usize);
    let (mut s_l, mut s_c) = (0usize, 0usize); // sólo el término de Stouffer
    for t in 0..calent + n {
        p *= (sigma * g.n()).exp();
        e9 += a9 * (p - e9);
        e21 += a21 * (p - e21);
        e120 += a120 * (p - e120);
        e720 += a720 * (p - e720);
        if t < calent {
            continue;
        }
        let z_micro = g.n(); // eje de tick independiente
        let z_macro = ema_spread_z((e9 - e21) / e21, atr, EMA_FAST_BARS, EMA_SLOW_BARS);
        let z_higher = ema_distance_z((p - e120) / e120, atr, EMA_TREND_BARS);
        let z_secular = ema_distance_z((p - e720) / e720, atr, EMA_MACRO_BARS);
        let (c, l) = escudo(z_micro, z_macro, z_higher, z_secular);
        bc += c as usize;
        bl += l as usize;
        let s = (z_micro + z_macro + z_higher) / 3f64.sqrt();
        s_c += (s > Z95) as usize;
        s_l += (s < -Z95) as usize;
    }
    let (fl, fc) = (bl as f64 / n as f64, bc as f64 / n as f64);
    println!(
        "observado: escudo bloquea largos {fl:.4} y cortos {fc:.4}; sólo Stouffer (sin el veto secular): \
         {:.4} / {:.4} (nominal 0,025 por lado)",
        s_l as f64 / n as f64,
        s_c as f64 / n as f64
    );
    assert!(
        fl <= 0.05 && fc <= 0.05,
        "escudo en caminata aleatoria: bloquea largos {fl:.4} y cortos {fc:.4} (nominal 0,025 por lado)"
    );
}

// ──────────────────────────────── Muro ────────────────────────────────

/// F3.4-04 — el muro es el OBI de nivel 0 con otro umbral. Rejilla
/// determinista de 149 × 149 cantidades positivas y r ∈ {2; 5; 5,324 (gen
/// activo); 15,7; 50}; se excluyen los puntos a menos de 1e-9 de la frontera
/// (allí el ≥ del código y el < del OBI pueden diferir sólo por redondeo).
/// Es álgebra: OBI es monótona decreciente en a/b. Igualdad exacta.
#[test]
fn cf3_muro_es_obi_nivel0() {
    let mut casos = 0usize;
    for &r in &[2.0_f64, 5.0, 5.324_041_785_584_2, 15.7, 50.0] {
        let u = (1.0 - r) / (1.0 + r);
        for i in 1..150 {
            for j in 1..150 {
                let (b, a) = (i as f64 * 0.37, j as f64 * 0.53);
                if ((a / b) - r).abs() < 1e-9 || ((b / a) - r).abs() < 1e-9 {
                    continue;
                }
                let obi = (b - a) / (b + a);
                assert_eq!(
                    muro_en_contra(b, a, r, true),
                    obi < u,
                    "largo b={b} a={a} r={r}"
                );
                assert_eq!(
                    muro_en_contra(b, a, r, false),
                    obi > -u,
                    "corto b={b} a={a} r={r}"
                );
                casos += 1;
            }
        }
    }
    println!("observado: {casos} casos, el veto de muro coincide con el umbral de OBI en todos");
}

// ──────────────────────────── Veto DD consejo ────────────────────────────

/// F3.4-06 — el veto «absoluto» de DD de SeniorRiesgo debe ser alcanzable por
/// debajo del corte del risk-engine. dd = 0,999·d* y τ ∈ {30 s; 10 min; 1 h;
/// 12 h}.
///
/// Correcciones del escéptico aplicadas:
/// - `integrity_failure` devuelve `vetoed_by = Some(Riesgo)` para cualquier
///   payload fuera de dominio, así que mirar `vetoed_by` puede pasar en
///   falso. Se mira la opinión del asiento: `deliberar_traced(..).opinions`
///   con `role == Riesgo && is_veto`, y se exige que la deliberación sea
///   válida (hay opiniones).
/// - La rejilla del consejo incluye n ∈ {0; 500} resultados en el tracker y
///   pesos de Metacognitivo en {1; 2} (regla de C-F2, FICHAS :872).
#[test]
fn cf3_veto_dd_consejo_alcanzable() {
    let d_star = drawdown_de_falsacion(FRACCION_KELLY_MAXIMA, ALFA_FALSACION);
    assert!(
        d_star.is_finite() && d_star > 0.0 && d_star < 1.0,
        "d* = {d_star}"
    );
    // El risk-engine rechaza con dd ≥ umbral ≤ d* ANTES del consejo, con el
    // mismo capital y el mismo pico: el mayor DD que el consejo puede ver es
    // menor que d*.
    let dd = d_star * 0.999;

    let vacio = ConsejoDeliberacion::new();
    let sembrado = ConsejoDeliberacion::new();
    let mut g = Xs(0x5EED_0F_C0DE);
    for _ in 0..500 {
        let mut s = [0.0f64; 11];
        for x in s.iter_mut() {
            *x = 2.0 * g.u() - 1.0;
        }
        sembrado.record_outcome(&s, 0.01 * (2.0 * g.u() - 1.0));
    }
    let mut meta2 = [1.0f64; 11];
    meta2[SeniorRole::Metacognitivo as usize] = 2.0;

    let mut alcanzable = false;
    // n esperada en el tracker; la ventana es 500 (consejo_seniors.rs, Default).
    for (consejo, n_esperada) in [(&vacio, 0usize), (&sembrado, 500)] {
        for pesos in [None, Some(&meta2)] {
            for &tau in &[30_000.0_f64, 600_000.0, 3_600_000.0, 43_200_000.0] {
                let p = MarketSnapshotPayload {
                    intended_direction: 1.0,
                    current_drawdown_pct: dd,
                    dominant_tau_ms: tau,
                    ..Default::default()
                };
                let umbral = 0.95 - 0.10 * p.spectral_s();
                let traza = consejo.deliberar_traced(&p, 0.5, pesos);
                assert!(
                    !traza.opinions.is_empty(),
                    "premisa: payload válido (sin integrity_failure) con τ = {tau}"
                );
                assert_eq!(
                    traza.aggregate_outcome_count, n_esperada,
                    "premisa: resultados en el tracker"
                );
                let veto = traza
                    .opinions
                    .iter()
                    .any(|o| o.role == SeniorRole::Riesgo && o.is_veto);
                println!(
                    "observado: n = {}, pesos meta = {}, τ = {tau} ms: dd = {dd:.4}, umbral del consejo = {umbral:.4}, veto de Riesgo = {veto}",
                    traza.aggregate_outcome_count,
                    if pesos.is_some() { 2 } else { 1 }
                );
                alcanzable |= veto;
            }
        }
    }
    assert!(
        alcanzable,
        "el veto de DD del consejo (0,95 − 0,10·s) no se alcanza bajo d* = {d_star:.3}"
    );
}

// ──────────────────────────────── K-26 ────────────────────────────────

fn fixture() -> (Arc<GlobalArena>, GodEngineCore) {
    symbols::update_dynamic_universe(vec!["BTCSYM".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec("BTCUSDT")]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    (arena, core)
}

/// F3.4-08 — el mismo fixture y el mismo tick que
/// `k26_engine_publishes_symmetric_conformal_at_honest_neutrality`
/// (conformal_direction_symmetry_contract.rs). Ese test cree correr con base
/// 0,25; tras el tick el registro debe seguir en 0,25 (tolerancia 1e-12: es
/// el mismo f64 escrito). Hoy el núcleo sin bosque publica `unwrap_or(0.5)`
/// por moneda antes del bloque conformal.
///
/// El arreglo es del test de #714 (inyectar un bosque con base 0,25 o probar
/// el bloque conformal como función pura), no del núcleo.
#[test]
#[ignore = "CERRADO F3.4-08 (AGY, Ola Ω88 / #726): #714 adopta bosque_constante(0.25) y calibrador caliente en conformal_direction_symmetry_contract.rs"]
fn cf3_k26_corre_con_base_025() {
    let _g = CORE_ENV.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, mut core) = fixture();
    arena.registry.set_for_coin(0, "ml_model_base", 0.25);
    let _ = core.process_tick_dual(0, 100.0, 100.01, 5.0, 5.0, 10_000, &[0.0; 54], true, false);
    let base = arena.registry.get_for_coin_or(0, "ml_model_base", -1.0);
    println!("observado: ml_model_base de la moneda 0 tras el tick = {base}");
    assert!(
        (base - 0.25).abs() < 1e-12,
        "el test K-26 cree correr con base 0,25 y el núcleo la reescribe a {base}"
    );
}

// ──────────────────────────────── Sonda ───────────────────────────────

/// Premisa: el núcleo decide la exención del ML-GATE con la función
/// `risk_engine::sonda::sonda_sin_modelo` (la que usa `admite_sonda`, la misma
/// del veto del riesgo; revisión del PR #61, F3), no con un predicado en
/// línea.
fn sonda_es_la_viva() {
    let src = codigo_produccion();
    for trozo in [
        "letarranque_frio_sin_roster=risk_engine::sonda::sonda_sin_modelo(&self.arena,coin_id,has_roster_model,);",
        "letml_gate_ok=arranque_frio_sin_roster||(has_roster_model&&ml_edge_now>=required_edge);",
    ] {
        assert!(
            src.contains(trozo),
            "premisa: el predicado de la sonda cambió en lib.rs (falta «{trozo}»)"
        );
    }
}

/// Arena de dos monedas sin historial, sin posiciones.
fn arena_sonda() -> Arc<GlobalArena> {
    GlobalArena::build_in_own_stack(1_000.0)
}

/// Posición pequeña abierta en la moneda dada.
fn abrir_en(arena: &GlobalArena, coin: usize) {
    arena.coins[coin].positions.slots()[0].open(true, 100.0, 0.01, 0.05, 1_000, 102.0, 99.0);
}

/// F3.4-05, la parte de D4 (cerrada por CL-98; que la sonda sin modelo no
/// produce la evidencia que exige el gate sigue abierto) — la sonda de
/// arranque frío respeta D4:
/// una sonda abierta en toda la cartera. Casos deterministas sobre la función
/// que llama el núcleo.
#[test]
fn cf3_sonda_respeta_d4() {
    use risk_engine::sonda::sonda_sin_modelo;
    sonda_es_la_viva();
    let a = arena_sonda();
    assert!(sonda_sin_modelo(&a, 0, false), "sin sondas abiertas debe poder sondear");
    assert!(!sonda_sin_modelo(&a, 0, true), "con modelo del roster decide el lift, no la sonda");

    abrir_en(&a, 1);
    assert!(
        !sonda_sin_modelo(&a, 0, false),
        "D4: con una sonda abierta en otra moneda no se abre otra"
    );

    // Una posición de una moneda con historial (fuera de la fase de sonda) no
    // ocupa la sonda.
    a.coins[1].metrics.trade_count.store(5, std::sync::atomic::Ordering::Relaxed);
    assert!(sonda_sin_modelo(&a, 0, false), "una posición con historial no es una sonda");

    // Sólo la primera orden de la moneda está exenta del ML-GATE.
    a.coins[0].metrics.trade_count.store(1, std::sync::atomic::Ordering::Relaxed);
    assert!(!sonda_sin_modelo(&a, 0, false), "con un cierre ya no es la primera orden");
}
