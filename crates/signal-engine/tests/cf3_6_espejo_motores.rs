//! C-F3 / F3.6 (Línea C): espejo largo/corto de los votos de los motores del
//! orquestador, y tres propiedades de motores concretos.
//!
//! Teoría que se prueba (puerta de espejo del plan): reflejar el mercado,
//! P → 2·P0 − P, invierte el lado de toda evidencia con signo y deja igual
//! toda magnitud. Un votante honesto es una función IMPAR del mercado
//! reflejado: v(−) = −v(+). Las entradas con signo (OBI, OFI, desviación de
//! posición, tendencia, lado del libro) cambian de signo; las magnitudes
//! (spread, ATR, VPIN, Hurst, λ/μ̂) no cambian. Desde CL-73 la velocidad
//! también tiene signo, como en el escritor vivo: `dir_velocity` es la EWMA
//! del Δmid CON signo (stateful_engine.rs) y el núcleo la publica como
//! `price_velocity` y `order_flow_velocity` (god-engine-core/src/lib.rs,
//! bloque `set_reg`). Antes de CL-73 era la EWMA de |Δmid|, igual en los
//! dos lados.
//!
//! El registro espejo reproduce a mano lo que publica el núcleo en
//! 8f572f7c: desde CL-66 ya no publica `atr_1s`, `atr_5s` ni `atr_1m`, así
//! que no se escriben aquí y el coaxial se abstiene (0 = 0); desde CL-69/71
//! `atr_pct` = v_t/p y `spread_speed_of_sound` = máx(spread, v_t) en precio,
//! que es lo que dan mid 100, v_t 0,2 y ATR 20 pb. Limitación (faltante del
//! escéptico de F3.6): si el escritor cambia, estas pruebas siguen su propia
//! copia; la variante integrada vive en god-engine-core.
//!
//! Tolerancia de espejo: 1e-9. Un votante impar da la misma operación en
//! coma flotante sobre entradas de signo opuesto, con error ≈ 1e-16; los
//! sesgos medidos son ≥ 0,39. Convención: un test que hoy falla por la causa
//! de un hallazgo lleva `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el
//! assert intacto. Para correrlos: `cargo test ... -- --include-ignored`.

use omniscient_registry::OmniscientRegistry;
use signal_engine::{
    coaxial_breakout::CoaxialBreakoutEngine, flow_impulse::FlowImpulseEngine,
    game_theoretic_nash::GameTheoreticNashEngine, perceptron_gate::PerceptronGateEngine,
    quantum_oscillator::QuantumOscillatorEngine, renyi_tsallis_entropy::RenyiTsallisEntropyEngine,
    soliton_wave::SolitonWaveEngine, stochastic_resonance::StochasticResonanceEngine,
    supersonic_shockwave::SupersonicShockwaveEngine, trend_runner::HighPayoffTrendRunner,
};
use std::sync::Arc;
use strategy_core::QuantumStrategy;

const COIN: usize = 3;
const SYM: &str = "ESPEJOUSDT";

/// Publica como el `set_reg` del núcleo: global, por moneda y por símbolo.
fn publicar(r: &OmniscientRegistry, k: &str, v: f64) {
    r.set(k, v);
    r.set_for_coin(COIN, k, v);
    r.set_scoped(SYM, k, v);
}

/// Tsallis binario q = 1,5 normalizado: la fórmula del escritor del núcleo
/// (`tsallis_raw / TSALLIS_Q15_BINARY_MAX`).
fn tsallis_de_obi(obi: f64) -> f64 {
    let p_bid = ((1.0 + obi) / 2.0).clamp(0.0001, 0.9999);
    let p_ask = ((1.0 - obi) / 2.0).clamp(0.0001, 0.9999);
    let raw = (1.0 - (p_bid.powf(1.5) + p_ask.powf(1.5))) / 0.5;
    (raw / 0.585_786_437_626_905).clamp(0.0, 1.0)
}

/// s = +1 mercado original, s = −1 reflejado.
fn registro(s: f64) -> Arc<OmniscientRegistry> {
    let r = Arc::new(OmniscientRegistry::new());
    let (obi, ofi) = (0.70 * s, 0.40 * s);
    publicar(&r, "order_book_imbalance", obi);
    publicar(&r, "orderbook_imbalance", obi);
    publicar(&r, "order_flow_imbalance", ofi);
    publicar(
        &r,
        "weak_alpha_signal",
        (0.5 * ofi + 0.5 * obi).clamp(-1.0, 1.0),
    );
    publicar(&r, "tsallis_q_entropy", tsallis_de_obi(obi));
    publicar(&r, "order_flow_direction", s);
    publicar(&r, "quantum_position_deviation", 0.5 * s);
    publicar(&r, "trend_direction", s);
    publicar(&r, "mid_price", 100.0);
    // Δmid por evento CON signo (CL-73): un salto de 0,5 (M = 0,005/0,002 = 2,5).
    publicar(&r, "price_velocity", 0.5 * s);
    publicar(&r, "order_flow_velocity", 0.5 * s);
    publicar(&r, "spread_speed_of_sound", 0.2);
    publicar(&r, "hurst_exponent", 0.60);
    publicar(&r, "cvpin", 0.20);
    publicar(&r, "atr_pct", 0.002);
    // CL-66: ya no hay atr_1s / atr_5s / atr_1m en el registro vivo.
    publicar(&r, "microstructure_noise_variance", 0.0002);
    publicar(&r, "hawkes_intensity", 3.0);
    r
}

fn voto<E: QuantumStrategy>(mut e: E, r: Arc<OmniscientRegistry>) -> f64 {
    e.init(r).expect("init");
    e.evaluate_for_coin(COIN, SYM)
}

macro_rules! caso_espejo {
    ($(#[$meta:meta])* $nombre:ident, $ctor:expr) => {
        $(#[$meta])*
        #[test]
        fn $nombre() {
            let largo = voto($ctor, registro(1.0));
            let corto = voto($ctor, registro(-1.0));
            println!(
                "observado: {}: v(+) = {:.6}, v(-) = {:.6}",
                stringify!($nombre),
                largo,
                corto
            );
            assert!(
                (largo + corto).abs() < 1e-9,
                "{}: reflejar el mercado debe invertir el voto: v(+) = {}, v(-) = {}",
                stringify!($nombre),
                largo,
                corto
            );
        }
    };
}

// ─────────────── Espejo escalar: los dos motores que lo rompían ───────────────

caso_espejo!(
    /// Antes de CL-73: v(+) = +0,3922 y v(−) = +0,3921, porque la
    /// «velocidad» era |Δmid| y la amplitud |OFI|. Con la velocidad firmada,
    /// tanh(v) es impar y la fase del sech cambia de signo con el mercado.
    cf3_6_espejo_soliton_escalar,
    SolitonWaveEngine::new()
);
caso_espejo!(
    /// Antes de CL-73: +0,5255 en los dos lados (M = 2,5, salto 0,619,
    /// tanh(1,25)). Con la velocidad firmada, M usa |v| y la dirección el
    /// signo de v.
    cf3_6_espejo_choque_escalar,
    SupersonicShockwaveEngine::new()
);

// ─────────────── Guardia: los ocho motores restantes son impares ───────────────
// El coaxial pasa trivialmente desde CL-66 (se abstiene sin sus tres ATR: 0 = 0).

caso_espejo!(cf3_6_espejo_oscilador, QuantumOscillatorEngine::new());
caso_espejo!(cf3_6_espejo_resonancia, StochasticResonanceEngine::new());
caso_espejo!(cf3_6_espejo_nash, GameTheoreticNashEngine::new());
caso_espejo!(cf3_6_espejo_trend_runner, HighPayoffTrendRunner::new());
caso_espejo!(
    cf3_6_espejo_renyi_tsallis,
    RenyiTsallisEntropyEngine::default()
);
caso_espejo!(cf3_6_espejo_coaxial, CoaxialBreakoutEngine::new());
caso_espejo!(cf3_6_espejo_perceptron, PerceptronGateEngine::new());
caso_espejo!(cf3_6_espejo_flow_impulse, FlowImpulseEngine::default());

// ─────────────── Espejo espectral del solitón ───────────────

/// La sombra viva pasa el OFI FIRMADO de la moneda como amplitud. Con
/// OFI = +0,2 la pendiente es 0,2 (tanh(0,2) = 0,1974); con OFI = −0,2 la
/// amplitud ≤ 0 se sustituye por 1,0 (tanh(−1) = −0,7616). Suma −0,564 en
/// cada escala.
#[test]
#[ignore = "OPEN F3.6-02 (Línea C, voto_espectral del solitón y sombra en lib.rs; test H2-10 de GLM): la amplitud ≤ 0 se sustituye por 1,0 y la > 0 se usa tal cual; la pendiente depende del signo del OFI (0,1974 frente a −0,7616 por escala)"]
fn cf3_6_espejo_espectral_soliton_con_el_ofi_del_vivo() {
    let ofi = 0.2;
    let v = SolitonWaveEngine::voto_espectral(&[1.0; 32], ofi);
    let vm = SolitonWaveEngine::voto_espectral(&[-1.0; 32], -ofi);
    println!(
        "observado: escala 0: v(+) = {:.4}, v(-) = {:.4}",
        v.en_escala(0),
        vm.en_escala(0)
    );
    for k in 0..32 {
        assert!(
            (v.en_escala(k) + vm.en_escala(k)).abs() < 1e-9,
            "escala {}: v(+) = {}, v(-) = {}",
            k,
            v.en_escala(k),
            vm.en_escala(k)
        );
    }
}

// ─────────────── Coaxial: invariante a la unidad de precio ───────────────

/// Lo que publica hoy el núcleo para el coaxial (CL-66/CL-69): `atr_pct` =
/// v_t/p y el lado del libro; ya no hay ATR por escala.
fn registro_coaxial_vivo(p0: f64, v_t: f64) -> Arc<OmniscientRegistry> {
    let r = Arc::new(OmniscientRegistry::new());
    publicar(&r, "atr_pct", v_t / p0);
    publicar(&r, "order_flow_direction", 1.0);
    r
}

/// Control del lector: ATR medidas por escala, en precio, proporcionales a
/// p0 (las mismas fracciones a cualquier precio), con compresión real
/// 1 s ≪ 5 s ≪ 1 min.
fn registro_coaxial_medido(p0: f64) -> Arc<OmniscientRegistry> {
    let r = Arc::new(OmniscientRegistry::new());
    publicar(&r, "atr_1s", 5e-6 * p0);
    publicar(&r, "atr_5s", 5e-5 * p0);
    publicar(&r, "atr_1m", 5e-4 * p0);
    publicar(&r, "order_flow_direction", 1.0);
    r
}

/// Verdad: el mismo mercado en fracción (ATR de 1 min = 20 pb) vota igual a
/// 60 000 USD y a 0,06 USD. Antes de CL-66 los suelos en USD del escritor
/// daban 0 arriba y 0,1853 abajo. Hoy el escritor no publica las tres ATR y
/// el coaxial se abstiene en los dos (0 = 0). La segunda parte comprueba que
/// la invariancia no es sólo la abstención: con ATR medidas proporcionales al
/// precio el lector da el mismo voto, no nulo, a los dos precios.
#[test]
fn cf3_6_coaxial_invariante_a_la_unidad_de_precio() {
    let caro = voto(
        CoaxialBreakoutEngine::new(),
        registro_coaxial_vivo(60_000.0, 120.0),
    );
    let barato = voto(
        CoaxialBreakoutEngine::new(),
        registro_coaxial_vivo(0.06, 0.000_12),
    );
    let caro_m = voto(
        CoaxialBreakoutEngine::new(),
        registro_coaxial_medido(60_000.0),
    );
    let barato_m = voto(CoaxialBreakoutEngine::new(), registro_coaxial_medido(0.06));
    println!(
        "observado: escritor vivo: {caro} frente a {barato}; ATR medidas: {caro_m:.9} frente a {barato_m:.9}"
    );
    assert!(
        (caro - barato).abs() < 1e-9,
        "el voto coaxial depende del precio nominal: {} frente a {}",
        caro,
        barato
    );
    assert!(
        caro_m > 0.0,
        "control: con compresión real el lector debe votar ({caro_m})"
    );
    assert!(
        (caro_m - barato_m).abs() < 1e-9,
        "con ATR proporcionales al precio el voto depende del precio nominal: {} frente a {}",
        caro_m,
        barato_m
    );
}

// ─────────────── TrendRunner: la volatilidad no es convicción ───────────────

/// Mismo Hurst (0,60), mismo VPIN (0,20) y misma tendencia: pasar de un ATR
/// del 0,1 % a uno del 1 % no debe subir el voto (D-746 por analogía).
/// Antes de CL-77: vol_boost = atr·expansión·4 subía tp de 0,03723 a
/// 0,04692 y el voto de 0,2675 a 0,3288. Desde CL-77 la convicción es el
/// TP de expansión sin el término de volatilidad.
#[test]
fn cf3_6_trend_runner_la_volatilidad_no_es_conviccion() {
    let con_atr = |atr: f64| {
        let r = Arc::new(OmniscientRegistry::new());
        publicar(&r, "hurst_exponent", 0.60);
        publicar(&r, "cvpin", 0.20);
        publicar(&r, "trend_direction", 1.0);
        publicar(&r, "atr_pct", atr);
        voto(HighPayoffTrendRunner::new(), r)
    };
    let (calma, tormenta) = (con_atr(0.001), con_atr(0.01));
    println!("observado: ATR 0,1 % -> {calma:.4}; ATR 1 % -> {tormenta:.4}");
    assert!(
        tormenta <= calma + 1e-12,
        "D-746: más ATR no es más convicción: {} -> {}",
        calma,
        tormenta
    );
}

// ─────────────── Perceptrón: la banda espectral es la operable ───────────────

/// La «banda operativa 8..23» del perceptrón trata k = 8 (65,5 µs) como
/// operable con peso 1,0, igual que k = 20 (18,3 min). La banda operable del
/// sistema es [30 s, 12 h] (temporal_spectrum.rs). Con x = 1 en todas las
/// escalas, el voto sub-milisegundo debe ser estrictamente menor.
#[test]
#[ignore = "OPEN F3.6-09 (Línea C, perceptron_gate): el peso espectral vale 1,0 en k ∈ 8..=23; la escala de 65 µs vota infer(1, 1) igual que la de 18 min"]
fn cf3_6_perceptron_banda_espectral_es_la_operable() {
    use quantum_arena::temporal_spectrum::{SPECTRUM_SCALES_MS, TAU_ANCHOR_FAST_MS};
    let v = PerceptronGateEngine::voto_espectral(&[1.0; 32]);
    let (k_sub, k_op) = (8usize, 20usize);
    assert!(
        SPECTRUM_SCALES_MS[k_sub] < 1.0,
        "k = 8 es una escala de microsegundos"
    );
    assert!(SPECTRUM_SCALES_MS[k_op] >= TAU_ANCHOR_FAST_MS);
    println!(
        "observado: k = 8 -> {:.6}; k = 20 -> {:.6}",
        v.en_escala(k_sub),
        v.en_escala(k_op)
    );
    assert!(
        v.en_escala(k_sub) < v.en_escala(k_op),
        "la 'banda operativa 8..23' trata 65 µs como operable: {} frente a {}",
        v.en_escala(k_sub),
        v.en_escala(k_op)
    );
}
