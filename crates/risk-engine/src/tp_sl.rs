//! FUENTE ÚNICA DE TP Y SL (D-637 / D-638 / D-639 / D-640 — DÉCIMA OLA).
//!
//! # El problema que resuelve
//!
//! La auditoría localizó **cinco derivaciones independientes** de la misma
//! magnitud, con fórmulas y acotaciones mutuamente incompatibles:
//!
//! | # | Dónde | Acotación |
//! |---|---|---|
//! | 1 | `risk-engine` `match` #1 | `[0,0005 … 0,0100]` — y su SL se DESCARTABA (`_sl_base`) |
//! | 2 | `risk-engine` `match` #2 | `[0,0040 … 0,0058]` |
//! | 3 | `risk-engine` gate de EV | sin techo |
//! | 4 | `conformal::compute_dynamic_tp_sl` (sin llamadores; eliminado) | `[0,001 … 0,50]` |
//! | 5 | `conformal::compute_continuous_tp_sl` (sin llamadores; eliminado) | `[0,001 … 0,50]` |
//!
//! De ahí nacían tres defectos encadenados:
//!
//! * **D-637** — el gate de EV validaba un trade **distinto** al que se
//!   ejecutaba. `expected_win` crecía linealmente con el ATR sin techo,
//!   mientras el TP ejecutado quedaba acotado a 115 bps: con ATR del 2 % el
//!   EV se sobreestimaba 2,6× y con ATR del 5 %, 6,5×. Como la barrera de
//!   comisiones sí era real, la condición efectivamente aplicada al trade
//!   real era `EV_real > hurdle / k` — es decir, la fricción se desactivaba
//!   justo cuando más pesa.
//! * **D-639** — el piso anti-difusivo se calculaba y se destruía en la misma
//!   expresión: `(atr·1,5).max(min_safe).clamp(min_safe, max_safe)`. Con ATR
//!   del 2 % el piso salía a 300 bps y el `clamp` lo devolvía a 60 bps.
//! * **D-640** — componiendo los límites, el SL vivía SIEMPRE entre 40 y 60
//!   bps, con independencia de la volatilidad, del horizonte y de los genes.
//!
//! # El principio de la corrección
//!
//! El stop no es un número elegido: es **la distancia a la que el ruido
//! difusivo del activo deja de explicar el movimiento**. Bajo difusión, la
//! dispersión a horizonte τ escala como `σ(τ) = σ₁ · τ^H`. El stop debe
//! cubrir esa dispersión; el objetivo debe cubrir el stop más la fricción,
//! con el RR que la propia fricción exige (ver `SuperGenotype::min_rr_for`).
//!
//! El único techo legítimo sobre el SL **no es de precio sino de capital**:
//! `SL · apalancamiento ≤ riesgo máximo por operación`. Eso lo aplica quien
//! dimensiona la posición, no quien calcula el precio del stop.

use quantum_arena::genome::SuperGenotype;

/// Entradas observables del cálculo. Todas son magnitudes medidas o genes:
/// ninguna es un literal de conveniencia.
#[derive(Debug, Clone, Copy)]
pub struct TpSlInputs {
    /// Horizonte operativo de la posición, en milisegundos. Procede del
    /// espectro temporal (`dominant_tau_ms`), no de una etiqueta discreta.
    pub tau_ms: f64,
    /// Volatilidad instantánea como fracción del precio (ATR / precio).
    pub atr_ratio: f64,
    /// Exponente de Hurst medido. 0,5 = difusión browniana pura.
    pub hurst: f64,
    /// Fricción de ida y vuelta como fracción del nocional (fees + slippage).
    pub roundtrip_fee: f64,
    /// Multiplicador genómico del stop sobre la dispersión difusiva.
    pub sl_atr_multiplier: f64,
    /// D-754 — σ PRONOSTICADA para ESTE horizonte (fracción de precio), si el
    /// espectro predictivo ha demostrado habilidad fuera de muestra. `None`
    /// ⇒ se usa la ley de escala sobre la volatilidad medida hacia atrás.
    ///
    /// Por qué importa: el stop debe cubrir la volatilidad que OCURRIRÁ
    /// mientras la posición viva, no la que acaba de ocurrir. `atr · (τ/τ_ref)^H`
    /// es una extrapolación de la volatilidad pasada; el pronóstico espectral
    /// mide, fuera de muestra, entre un 11 % y un 18 % de la varianza del
    /// logaritmo de la varianza realizada futura a horizontes de 1 a 18
    /// minutos. Cuando existe con evidencia, es la magnitud correcta.
    pub sigma_forecast: Option<f64>,
}

/// Resultado. `tp_pct` y `sl_pct` son fracciones del precio, siempre
/// positivas, y satisfacen por construcción `tp_pct ≥ sl_pct · rr_required`
/// y el cap B3.24 `sl_pct ≤ tp_pct · 0.5` (RR ≥ 2) en TODOS los caminos.
#[derive(Debug, Clone, Copy)]
pub struct TpSl {
    pub tp_pct: f64,
    pub sl_pct: f64,
    /// RR efectivamente aplicado (≥ el mínimo exigido por la fricción).
    pub rr_applied: f64,
    /// RR mínimo bajo WORST_TOLERATED_WR y la fricción declarada. Es una
    /// condición de EV bajo ese p supuesto, no una probabilidad calibrada.
    pub rr_required: f64,
    /// `true` si el horizonte solicitado NO es operable: la dispersión
    /// esperada a esa τ no cubre el stop mínimo viable frente a la fricción.
    /// Quien reciba esto debe RECHAZAR la operación, no acotarla.
    pub below_tradeable_floor: bool,
}

/// Horizonte de referencia de la curva de dispersión: la escala a la que se
/// MIDE `atr_ratio`, de modo que `sigma(tau) = atr_ratio · (tau/TAU_REF)^H`.
///
/// D-677 (DÉCIMA OLA) — ERROR DE UNIDADES. Valía 1 segundo, pero `atr_ratio`
/// es `v_t / precio` y `v_t` es la EMA del true range de la vela interna de
/// **1 minuto** (`StatefulEngine::process_tick`, cierre a 60 000 ms), también
/// calentada con velas REST `interval=1m`. Con la referencia en 1 s, a
/// τ = 19 min y H = 0,30 la dispersión salía `(1138)^0,3 ≈ 8,3` veces el ATR
/// en lugar de `(19)^0,3 ≈ 2,4`: stops y objetivos 3,4 veces más anchos de lo
/// que la difusión justifica, y a H = 0,5 hasta 7,7 veces. El objetivo quedaba
/// fuera del alcance del horizonte (0 salidas por TP en el backtest forense)
/// y las posiciones terminaban por stop o por caducidad.
pub const TAU_REFERENCE_MS: f64 = 60_000.0;

/// Rango medio de una vela browniana en desviaciones típicas, `√(8/π)`: el
/// factor de Parkinson que convierte una σ de retorno en la escala del ATR.
/// Mismo valor que `god_engine_core::diffusion::PARKINSON_RANGE_FACTOR`
/// (el risk-engine no puede depender del núcleo: ciclo de dependencias).
pub const RANGO_PARKINSON: f64 = 1.595_769_121_605_730_8;

/// D-747 — DESLIZAMIENTO POR LATENCIA: UNA SOLA LEY, LA DE LA DIFUSIÓN.
///
/// # Qué estaba mal
///
/// La misma magnitud tenía DOS fórmulas incompatibles:
///
/// * la física de ejecución (`god-engine-core::reality_physics`) cobra
///   `σ · √(latencia / τ_ref)` — difusión: el desplazamiento esperado en un
///   tiempo `t` escala con `√t`;
/// * la compuerta de expectativa del risk-engine estimaba
///   `σ · (latencia / umbral_de_pánico)` — **lineal**, y normalizada además
///   contra un gen (`latency_ms_panic_threshold`) que no es una escala de
///   volatilidad sino el umbral a partir del cual el enlace se considera
///   roto.
///
/// La ley lineal subestima el coste de las latencias cortas y sobreestima el
/// de las largas frente a la browniana; peor, el gate y la ejecución cobraban
/// números distintos por el mismo evento, de modo que el gate certificaba como
/// rentables operaciones que la física del propio motor volvía negativas.
///
/// # La derivación
///
/// Entre la decisión y el fill transcurre `t`. Bajo difusión el desplazamiento
/// esperado del precio es `σ(t) = σ(τ_ref) · √(t/τ_ref)`. La dispersión
/// disponible es `atr_ratio` y se MIDE sobre la vela interna de 1 minuto
/// ([`TAU_REFERENCE_MS`]), de modo que `τ_ref = 60 000 ms`. Ni la escala ni el
/// exponente son parámetros: la escala es aquella en la que se estima la
/// volatilidad y el exponente ½ es el de la difusión.
///
/// Esta es la FUENTE ÚNICA del término de latencia. `reality_physics` debe
/// llamarla para que la ejecución y la compuerta cobren el mismo número.
#[inline]
pub fn latency_slippage_pct(atr_ratio: f64, latency_ms: f64) -> f64 {
    if !atr_ratio.is_finite() || atr_ratio <= 0.0 {
        return 0.0;
    }
    if !latency_ms.is_finite() || latency_ms <= 0.0 {
        return 0.0;
    }
    let r = (latency_ms / TAU_REFERENCE_MS).sqrt();
    let s = atr_ratio * r;
    if s.is_finite() {
        s
    } else {
        0.0
    }
}

/// σ del jitter lognormal del RTT a Binance Tokyo/AWS AP-Northeast.
/// Mismo valor por defecto que `NetworkJitterSimulator` (backtest-engine):
/// la dispersión es física de la ruta, no un gen.
pub const LATENCY_SIGMA_JITTER: f64 = 0.35;

/// XLVI·B — MUESTREO DETERMINISTA DE LATENCIA RTT LOGNORMAL (cierre DIV-2).
///
/// La auditoría bt↔vivo (docs/AUDITORIA_BT_VIVO_2026-09-28.md) midió que la
/// física de fills cobra la latencia ESTÁTICA del genoma (`30.68ms`) mientras
/// el RTT real es lognormal: p50 < estática < p99 (≈2×). El bt no conocía la
/// cola — exactamente donde viven los stops que sobreviven por milisegundos.
///
/// El gen `latency_penalty_ms` pasa a calibrar la MEDIA del RTT (la
/// normalización `exp(σz−σ²/2)` conserva la media: mediana ≈ 0.94·base) y
/// la física añade la cola con σ fija ([`LATENCY_SIGMA_JITTER`]). El muestreo
/// es determinista por semilla (xorshift + Box-Muller, réplica bit-exacta de
/// `NetworkJitterSimulator::sample_latency_ms` — el contrato de igualdad vive
/// en backtest-engine/tests/bt_vivo_parity_audit.rs), de modo que el replay
/// conserva su garantía mismo-input ⇒ mismo-output.
///
/// Contornos honestos:
/// - `base_ms` inválido (≤0/NaN) ⇒ se devuelve tal cual: sin calibración no
///   se inventa dispersión (comportamiento previo conservado).
/// - La pérdida de paquetes NO se modela aquí: una orden que no llena cambia
///   la semántica de decisión, no la de contabilidad. Documentado, no callado.
/// - La devolución se acota a [2, 500] ms como en el simulador canónico.
#[inline]
pub fn sample_latency_lognormal_ms(base_ms: f64, seed: u64) -> f64 {
    if !base_ms.is_finite() || base_ms <= 0.0 {
        return base_ms;
    }
    // Réplica exacta de la secuencia de `NetworkJitterSimulator::sample_latency_ms`
    // (sin la bandera de pérdida): mismo seed ⇒ mismo milisegundo.
    let mut rng_state = seed ^ 0x517CC1B727220A95;
    rng_state = rng_state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z1 = rng_state;
    z1 = (z1 ^ (z1 >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z1 = (z1 ^ (z1 >> 27)).wrapping_mul(0x94D049BB133111EB);
    z1 = z1 ^ (z1 >> 31);
    let u1 = ((z1 as f64) / (u64::MAX as f64)).clamp(1e-6, 1.0 - 1e-6);

    rng_state = rng_state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z2 = rng_state;
    z2 = (z2 ^ (z2 >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z2 = (z2 ^ (z2 >> 27)).wrapping_mul(0x94D049BB133111EB);
    z2 = z2 ^ (z2 >> 31);
    let u2 = ((z2 as f64) / (u64::MAX as f64)).clamp(1e-6, 1.0 - 1e-6);

    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    let exponent =
        (LATENCY_SIGMA_JITTER * z - 0.5 * LATENCY_SIGMA_JITTER * LATENCY_SIGMA_JITTER)
            .clamp(-50.0, 50.0);
    (base_ms * exponent.exp()).clamp(2.0, 500.0)
}

/// Semilla determinista por evento y activo para el muestreo de latencia:
/// mezcla el reloj del evento con el nodo para que dos activos no compartan
/// muestra en el mismo milisegundo (y el mismo activo sea reproducible).
#[inline]
pub fn latency_seed(event_time_ms: u64, coin_id: usize) -> u64 {
    event_time_ms
        .rotate_left(8)
        .wrapping_add((coin_id as u64).wrapping_mul(0x9E3779B97F4A7C15))
}

/// XLIV-8 — FRICCIÓN DE IDA Y VUELTA: FUENTE ÚNICA.
///
/// D-747 unificó la ley del deslizamiento por latencia en el gate de
/// expectativa y en la física de ejecución, pero la fricción con la que se
/// construyen los brackets del host (`genome_protection_prices`, fallback de
/// entrada) y la del pre-examen del daemon siguieron cobrando la ley LINEAL
/// `atr · latencia / umbral_de_pánico`. Con el umbral en su cota baja
/// (500 ms) y 50 ms de latencia, la ley lineal cobra 0,10·ATR por lado y la
/// difusiva 0,029·ATR: el gate aprobaba una geometría y el host protegía la
/// posición con otra, y el daemon promovía genomas contra una tercera.
///
/// Modelo D-645: taker en ambas piernas + (piso de deslizamiento +
/// latencia difusiva) por lado, acotado al 5 % por lado.
///
/// Saneamiento (precisado tras la revisión de Codex en el PR #8): hereda el
/// de [`latency_slippage_pct`], de modo que un ATR o una latencia no finitos
/// o no positivos anulan el término de latencia (cobra 0, no lo detecta).
/// `taker_fee` y `slip_floor` NO se sanean: si no son finitos el resultado
/// tampoco lo es y quien lo reciba debe rechazarlo. Unifica la FÓRMULA; que
/// gate, host y daemon lean el mismo ATR y la misma latencia en el mismo
/// instante es otra paridad, todavía abierta.
#[inline]
pub fn roundtrip_friction(taker_fee: f64, slip_floor: f64, atr_ratio: f64, latency_ms: f64) -> f64 {
    let per_side_slip = (slip_floor + latency_slippage_pct(atr_ratio, latency_ms)).clamp(0.0, 0.05);
    2.0 * taker_fee + 2.0 * per_side_slip
}

/// FUNCIÓN PURA ÚNICA. La invocan, con las MISMAS entradas, tanto el gate de
/// expectativa como el constructor de la orden: es imposible por construcción
/// que evalúen trades distintos (D-637).
/// CL-34 — DISPERSIÓN AL HORIZONTE en la unidad del ATR de 1 minuto (un
/// rango): `atr · (τ/τ_ref)^H`, con H acotado a [0,30; 0,75] como en la
/// geometría. Es la ley con la que se fija el stop difusivo (`sl = k·esto`);
/// cualquier distancia de una posición medida en «ATR» —el trailing— debe
/// medirse en esta escala, la de su propio horizonte, o un ATR de 1 minuto
/// aplicado a una posición de horas la corta dentro de su ruido.
#[inline]
pub fn dispersion_al_horizonte(atr: f64, tau_ms: f64, hurst: f64) -> f64 {
    let h = if hurst.is_finite() {
        hurst.clamp(0.30, 0.75)
    } else {
        0.50
    };
    let tau = if tau_ms.is_finite() && tau_ms > 0.0 {
        tau_ms
    } else {
        TAU_REFERENCE_MS
    };
    atr * (tau / TAU_REFERENCE_MS).powf(h)
}

pub fn compute_tp_sl(input: TpSlInputs) -> TpSl {
    let atr = if input.atr_ratio.is_finite() && input.atr_ratio > 0.0 {
        input.atr_ratio
    } else {
        0.005
    };
    let h = if input.hurst.is_finite() {
        input.hurst.clamp(0.30, 0.75)
    } else {
        0.50
    };
    let fee = if input.roundtrip_fee.is_finite() && input.roundtrip_fee > 0.0 {
        input.roundtrip_fee
    } else {
        SuperGenotype::REFERENCE_ROUNDTRIP_FEE
    };
    // El RR mínimo se evalúa en el win rate conservador tolerado (0.40)
    // para garantizar EV >= 0 en el peor escenario de supervivencia extrema (D-636/D-681).
    let w = SuperGenotype::WORST_TOLERATED_WR;
    let k = if input.sl_atr_multiplier.is_finite() && input.sl_atr_multiplier > 0.0 {
        input.sl_atr_multiplier
    } else {
        1.0
    };
    let tau = if input.tau_ms.is_finite() && input.tau_ms > 0.0 {
        input.tau_ms
    } else {
        TAU_REFERENCE_MS
    };

    // 1) DISPERSIÓN ESPERADA AL HORIZONTE. Ley de escala de la difusión
    //    anómala: sigma(tau) = sigma_ref · (tau/tau_ref)^H. Aquí `tau` SÍ es
    //    tiempo — a diferencia de D-604, donde se usaba una fracción de
    //    capital dentro de esta misma ley.
    // D-754: si hay pronóstico CON EVIDENCIA para este horizonte, la
    // dispersión esperada es esa; si no, la ley de escala sobre lo medido.
    //
    // D-754c (auditoría PR #5): las dos ramas deben hablar la MISMA unidad.
    // La ley de escala parte del ATR de 1 minuto, que es un RANGO medio
    // (√(8/π)·σ para una vela browniana, ver `god_engine_core::diffusion`), y
    // el gen `sl_atr_multiplier` está expresado en múltiplos de ese rango. El
    // pronóstico es una σ de retorno. Sin convertirla, el stop y el TP se
    // encogían ~1,6× en el instante en que el pronóstico ganaba habilidad,
    // sin cambio alguno en la volatilidad real.
    let sigma_tau = match input.sigma_forecast {
        Some(s) if s.is_finite() && s > 0.0 => s * RANGO_PARKINSON,
        _ => dispersion_al_horizonte(atr, tau, h),
    };

    // 2) STOP DIFUSIVO. El stop cubre k veces la dispersión del horizonte.
    //    Este es el piso REAL y ya no se destruye con un clamp posterior
    //    (D-639): no existe techo de precio sobre el stop.
    let sl_diffusive = sigma_tau * k;

    // 3) SUELO DE VIABILIDAD. Por debajo del stop mínimo viable la fricción
    //    domina el riesgo y ninguna RR alcanzable produce EV positivo
    //    (D-636b). No se acota hacia arriba: se ELEVA el stop hasta el suelo
    //    y se informa de si el horizonte pedido caía por debajo.
    let sl_floor = SuperGenotype::min_viable_sl(fee);
    let below_floor = sl_diffusive < sl_floor;
    let mut sl_pct = sl_diffusive.max(sl_floor);

    // 4) RR EXIGIDO POR LA FRICCIÓN AL NIVEL DE STOP RESULTANTE (D-636).
    //    Crece cuando el stop se estrecha: la comisión pesa más sobre un
    //    riesgo menor.
    let mut rr_required = SuperGenotype::min_rr_for(w, fee, sl_pct);

    // 5) OBJETIVO. El RR genómico puede ser MÁS ambicioso que el mínimo,
    //    nunca menor: el mínimo es una restricción de rentabilidad, no una
    //    preferencia.
    let mut rr_applied = rr_required.max(1.0);
    let mut tp_pct = sl_pct * rr_applied;

    // 6) CAP B3.24 (friction_floors) — SL ≤ TP/2 EN LA FUNCIÓN PURA.
    //    MOD3/5-019 (INFORME 14): `compute_tp_sl` garantizaba RR ≥ rr_required
    //    pero NO el cap B3.24; solo lo aplicaba el gestor al re-geometrizar a
    //    RR ≥ 2 en el tick siguiente — dos geometrías coherentes por separado,
    //    incoherentes entre sí (gate/orden decían una cosa, gestión/brackets
    //    ejecutaban otra). Aplicándolo AQUÍ, el cap es consistente en TODOS
    //    los caminos por construcción.
    if sl_pct > tp_pct * 0.5 {
        sl_pct = tp_pct * 0.5;
        // El stop estrechado encarece la fricción RELATIVA: re-derivar el RR
        // mínimo al nivel nuevo (min_rr_for crece cuando el stop se estrecha)
        // y re-asegurar tp ≥ sl·rr_min. Si rr_min ≤ 2, el objetivo YA lo
        // cubre (tp = 2·sl tras el cap) y no se toca.
        rr_required = SuperGenotype::min_rr_for(w, fee, sl_pct);
        let rr_min = rr_required.max(1.0);
        if tp_pct < sl_pct * rr_min {
            rr_applied = rr_min;
            tp_pct = sl_pct * rr_min;
        } else {
            rr_applied = tp_pct / sl_pct; // = 2.0: RR efectivo tras el cap
        }
    }

    TpSl {
        tp_pct,
        sl_pct,
        rr_applied,
        rr_required,
        below_tradeable_floor: below_floor,
    }
}

/// El gen sólo puede ampliar el TP base: conserva SL, el piso y la misma
/// probabilidad contractual. Un target no representable deja intacta la base.
/// Aumentar TP no garantiza mantener la probabilidad de primera llegada:
/// esa probabilidad todavía requiere estimación independiente.
pub fn compute_tp_sl_with_target_rr(input: TpSlInputs, target_rr: f64) -> TpSl {
    let mut out = compute_tp_sl(input);
    // FMT-096: recalcular el mínimo con p=0.55 podía reducir el TP que
    // la base había construido con p=0.40 (SL=.002, fee=.001, target=2).
    if target_rr.is_finite() && target_rr > out.rr_applied {
        let target_tp = out.sl_pct * target_rr;
        if target_tp.is_finite() && target_tp >= out.tp_pct {
            out.rr_applied = target_rr;
            out.tp_pct = target_tp;
        }
    }
    out
}

/// Función de distribución acumulada de la normal estándar Φ(z).
/// Aproximación analítica de alta precisión (Abramowitz & Stegun 7.1.26, error absoluto < 7.5e-8).
#[inline]
pub fn normal_cdf(z: f64) -> f64 {
    if !z.is_finite() {
        return 0.0;
    }
    if z == 0.0 {
        return 0.5;
    }
    if z < -10.0 {
        return 0.0;
    }
    if z > 10.0 {
        return 1.0;
    }
    let abs_z = z.abs();
    let t = 1.0 / (1.0 + 0.2316419 * abs_z);
    let poly = t * (0.319381530
        + t * (-0.356563782
            + t * (1.781477937
                + t * (-1.821255978 + t * 1.330274429))));
    let phi = 0.3989422804014327 * (-0.5 * abs_z * abs_z).exp(); // 1/√(2π)
    let cdf = 1.0 - phi * poly;
    if z >= 0.0 {
        cdf.clamp(0.0, 1.0)
    } else {
        (1.0 - cdf).clamp(0.0, 1.0)
    }
}

/// R8-A / CL-34 / Ω3: Probabilidad analítica en forma cerrada de que un Movimiento
/// Browniano con deriva μ y volatilidad σ toque el Stop Loss (-sl) antes
/// que el Take Profit (+tp).
///
/// En martingala pura (μ ≈ 0): P(hit SL first) = tp / (tp + sl) = RR / (1 + RR).
/// Con deriva a favor (μ > 0): decae exponencialmente.
#[inline]
pub fn probabilidad_tocar_sl_antes_de_tp(tp_pct: f64, sl_pct: f64, mu: f64, sigma: f64) -> f64 {
    if !tp_pct.is_finite() || tp_pct <= 0.0 || !sl_pct.is_finite() || sl_pct <= 0.0 {
        return 1.0;
    }
    let sig2 = sigma * sigma;
    if !sig2.is_finite() || sig2 <= 1e-16 {
        return if mu <= 0.0 { 1.0 } else { 0.0 };
    }
    let theta = 2.0 * mu / sig2;
    if theta.abs() < 1e-6 {
        // Límite browniano neutral (regla de la palanca / martingala)
        return (tp_pct / (tp_pct + sl_pct)).clamp(0.0, 1.0);
    }
    // P(hit -sl before +tp) = (e^{θ·tp} - 1) / (e^{θ(tp + sl)} - 1)
    let num = (theta * tp_pct).exp_m1();
    let den = (theta * (tp_pct + sl_pct)).exp_m1();
    if !num.is_finite() || !den.is_finite() || den.abs() <= 1e-16 {
        return if mu <= 0.0 { 1.0 } else { 0.0 };
    }
    (num / den).clamp(0.0, 1.0)
}

/// R8-A / CL-34 / Ω3: Probabilidad analítica de primer toque del Stop Loss antes
/// del horizonte temporal τ (distribución Inversa-Gaussiana en forma cerrada).
///
/// P(τ_sl ≤ τ) = Φ((-sl - μ·t)/(σ·√t)) + e^{-2μ·sl/σ²} · Φ((-sl + μ·t)/(σ·√t))
/// donde t = τ en segundos.
#[inline]
pub fn probabilidad_primer_toque_stop_antes_de_tau(
    sl_pct: f64,
    mu_por_seg: f64,
    sigma_por_seg: f64,
    tau_sec: f64,
) -> f64 {
    if !sl_pct.is_finite() || sl_pct <= 0.0 || !tau_sec.is_finite() || tau_sec <= 0.0 {
        return 0.0;
    }
    let sqrt_t = tau_sec.sqrt();
    let sig = sigma_por_seg.max(1e-8);
    let sig_sqrt_t = sig * sqrt_t;
    let b = sl_pct;
    // Para el stop adverso a distancia b:
    let d1 = (-b - mu_por_seg * tau_sec) / sig_sqrt_t;
    let term1 = normal_cdf(d1);

    let exponent = -2.0 * mu_por_seg * b / (sig * sig);
    if exponent > 700.0 {
        return term1.clamp(0.0, 1.0);
    }
    let d2 = (-b + mu_por_seg * tau_sec) / sig_sqrt_t;
    let term2 = exponent.exp() * normal_cdf(d2);

    let prob = term1 + term2;
    if prob.is_finite() {
        prob.clamp(0.0, 1.0)
    } else {
        term1.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> TpSlInputs {
        TpSlInputs {
            tau_ms: 30_000.0,
            atr_ratio: 0.005,
            hurst: 0.5,
            roundtrip_fee: 0.0010,
            sl_atr_multiplier: 1.0,
            sigma_forecast: None,
        }
    }

    /// D-754: con pronóstico CON EVIDENCIA, la dispersión del horizonte es la
    /// pronosticada y no la extrapolada del pasado. Sin él, nada cambia.
    #[test]
    fn d754_el_pronostico_manda_sobre_la_extrapolacion_del_pasado() {
        let sin = compute_tp_sl(base());
        let mut con = base();
        // El doble de dispersión esperada que la que el pasado extrapola (en
        // σ de retorno: el ATR es un rango, ver D-754c).
        let sigma_pasado =
            (base().atr_ratio / RANGO_PARKINSON) * (base().tau_ms / TAU_REFERENCE_MS).powf(0.5);
        con.sigma_forecast = Some(sigma_pasado * 2.0);
        let salida = compute_tp_sl(con);
        assert!(
            salida.sl_pct > sin.sl_pct,
            "con el doble de σ pronosticada el stop debe ser más ancho: {} vs {}",
            salida.sl_pct,
            sin.sl_pct
        );
        // Y un pronóstico no utilizable no puede cambiar nada.
        let mut basura = base();
        basura.sigma_forecast = Some(f64::NAN);
        assert_eq!(compute_tp_sl(basura).sl_pct, sin.sl_pct);
        let mut cero = base();
        cero.sigma_forecast = Some(0.0);
        assert_eq!(compute_tp_sl(cero).sl_pct, sin.sl_pct);
    }

    /// D-754c — un pronóstico que coincide con la σ que el ATR ya implica NO
    /// puede mover el stop: la geometría sólo cambia si cambia la volatilidad
    /// esperada, no por la unidad en que llega.
    #[test]
    fn d754c_el_pronostico_y_el_atr_hablan_la_misma_unidad() {
        let sin = compute_tp_sl(base());
        let b = base();
        let h = b.hurst.clamp(0.30, 0.75);
        let sigma_implicita =
            (b.atr_ratio / RANGO_PARKINSON) * (b.tau_ms / TAU_REFERENCE_MS).powf(h);
        let mut con = base();
        con.sigma_forecast = Some(sigma_implicita);
        let salida = compute_tp_sl(con);
        assert!(
            (salida.sl_pct - sin.sl_pct).abs() <= 1e-12 * sin.sl_pct.max(1e-12),
            "misma volatilidad, stop distinto: {} vs {}",
            salida.sl_pct,
            sin.sl_pct
        );
    }

    /// D-637: el gate de EV y la orden deben ver EXACTAMENTE lo mismo. Al ser
    /// una función pura, la identidad es estructural — este test la fija como
    /// contrato para que ningún refactor futuro vuelva a bifurcarla.
    #[test]
    fn d637_misma_entrada_produce_mismo_tp_sl() {
        let i = base();
        let a = compute_tp_sl(i);
        let b = compute_tp_sl(i);
        assert_eq!(a.tp_pct, b.tp_pct);
        assert_eq!(a.sl_pct, b.sl_pct);
    }

    /// D-639: el stop DEBE seguir a la volatilidad. Antes, con ATR del 2 %, el
    /// piso difusivo salía a 300 bps y un clamp posterior lo devolvía a 60.
    /// B3.24 (MOD3/5-019): cuando la fricción solo exige RR < 2, el cap
    /// SL ≤ TP/2 recorta parte del crecimiento (el stop queda en
    /// σ·rr_req/2), pero SIGUE escalando con la vol — el factor 4× de vol
    /// entrega ~3.35× de stop, no la compresión a 60 bps del bug original.
    #[test]
    fn d639_el_stop_escala_con_la_volatilidad_sin_techo() {
        let mut lo = base();
        lo.atr_ratio = 0.005;
        let mut hi = base();
        hi.atr_ratio = 0.020; // 4x de volatilidad
        let r_lo = compute_tp_sl(lo);
        let r_hi = compute_tp_sl(hi);
        assert!(
            r_hi.sl_pct > r_lo.sl_pct * 3.0,
            "el stop debe crecer con la vol: {} vs {}",
            r_lo.sl_pct,
            r_hi.sl_pct
        );
        assert!(
            r_hi.sl_pct > 0.010,
            "con ATR del 2 % el stop no puede volver a la banda de 60 bps, dio {}",
            r_hi.sl_pct
        );
    }

    /// D-640: el stop ya no vive confinado entre 40 y 60 bps.
    #[test]
    fn d640_el_stop_no_esta_confinado_a_una_banda_de_20bps() {
        let mut muy_volatil = base();
        muy_volatil.atr_ratio = 0.05;
        let r = compute_tp_sl(muy_volatil);
        assert!(
            r.sl_pct > 0.0060,
            "con ATR del 5 % el stop no puede ser 60 bps, dio {}",
            r.sl_pct
        );
    }

    /// D-636: el resultado es EV-no-negativo por construcción.
    #[test]
    fn d636_el_resultado_tiene_ev_no_negativo() {
        for atr in [0.001, 0.005, 0.02, 0.05] {
            for tau in [1_000.0, 60_000.0, 3_600_000.0] {
                let mut i = base();
                i.atr_ratio = atr;
                i.tau_ms = tau;
                let r = compute_tp_sl(i);
                let w = SuperGenotype::WORST_TOLERATED_WR;
                let f = i.roundtrip_fee;
                let ev = w * (r.tp_pct - f) - (1.0 - w) * (r.sl_pct + f);
                assert!(
                    ev >= -1e-12,
                    "EV negativo con atr={atr} tau={tau}: ev={ev}, tp={}, sl={}",
                    r.tp_pct,
                    r.sl_pct
                );
            }
        }
    }

    /// El horizonte demasiado corto se SEÑALA, no se disfraza acotándolo.
    #[test]
    fn horizonte_no_operable_se_reporta() {
        let mut i = base();
        i.tau_ms = 1.0; // 1 ms
        i.atr_ratio = 0.0005;
        let r = compute_tp_sl(i);
        assert!(
            r.below_tradeable_floor,
            "a 1 ms con vol baja la dispersión no cubre la fricción: debe marcarse"
        );
    }

    /// D-677: a la escala en la que se mide el ATR, la dispersión ES el ATR; y
    /// bajo difusión browniana cuatro veces el horizonte es el doble de
    /// dispersión. Fija las unidades de la ley de escala.
    /// B3.24 (MOD3/5-019): a 4 min la fricción solo exige RR 1,75 < 2, así que
    /// el cap SL ≤ TP/2 recorta el stop a σ·rr_req/2 = 0,00875 — la ley de
    /// escala sigue viva (el stop CRECE con τ), solo que el cap le pone el
    /// techo de geometría que antes aplicaba el gestor un tick después.
    #[test]
    fn d677_la_referencia_temporal_es_la_escala_del_atr() {
        let mut i = base();
        i.atr_ratio = 0.005;
        i.hurst = 0.5;
        i.tau_ms = 60_000.0;
        let a_1m = compute_tp_sl(i);
        assert!(!a_1m.below_tradeable_floor);
        assert!(
            (a_1m.sl_pct - 0.005).abs() < 1e-12,
            "a 1 minuto el stop difusivo debe ser el ATR de 1 minuto, dio {}",
            a_1m.sl_pct
        );
        i.tau_ms = 240_000.0;
        let a_4m = compute_tp_sl(i);
        assert!(
            (a_4m.sl_pct - 0.00875).abs() < 1e-9,
            "a 4 minutos: dispersión 2× (0.010) con RR_req 1.75, cap B3.24 a TP/2 ⇒ 0.00875, dio {}",
            a_4m.sl_pct
        );
        assert!(
            a_4m.sl_pct > a_1m.sl_pct,
            "el stop sigue creciendo con tau pese al cap"
        );
    }

    /// MOD3/5-019 — B3.24 en TODOS los caminos: la función pura garantiza
    /// SL ≤ TP/2 (RR ≥ 2) por sí misma, y el target genómico solo puede
    /// ampliar el recorrido, nunca romper el cap. El gestor ya no
    /// re-geometriza nada al tick siguiente.
    #[test]
    fn b324_el_cap_sl_mitad_de_tp_se_garantiza_en_la_funcion_pura() {
        for atr in [0.002, 0.005, 0.02, 0.05] {
            for tau in [30_000.0, 300_000.0, 3_600_000.0, 43_200_000.0] {
                for target in [0.0, 1.6, 1.9, 2.0, 2.5, 3.0] {
                    let mut i = base();
                    i.atr_ratio = atr;
                    i.tau_ms = tau;
                    let r = compute_tp_sl_with_target_rr(i, target);
                    assert!(
                        r.sl_pct <= r.tp_pct * 0.5 + 1e-12,
                        "cap B3.24 violado: atr={atr} tau={tau} target={target} sl={} tp={}",
                        r.sl_pct,
                        r.tp_pct
                    );
                    assert!(
                        r.tp_pct >= r.sl_pct * r.rr_required.max(1.0) - 1e-12,
                        "RR mínimo roto tras el cap: atr={atr} tau={tau} target={target}",
                    );
                }
            }
        }
    }

    /// D-681: la geometría de la orden no depende del desempeño observado. El
    /// RR aplicado es exactamente el mínimo en el win rate de diseño.
    #[test]
    fn d681_el_objetivo_no_depende_del_desempeno_observado() {
        let i = base();
        let r = compute_tp_sl(i);
        let rr =
            SuperGenotype::min_rr_for(SuperGenotype::WORST_TOLERATED_WR, i.roundtrip_fee, r.sl_pct)
                .max(1.0);
        assert!((r.tp_pct / r.sl_pct - rr).abs() < 1e-12);
        assert!(
            r.rr_applied < 3.0,
            "a la geometría de referencia el RR es acotado: {}",
            r.rr_applied
        );
    }

    /// Un horizonte más largo implica más dispersión y por tanto más recorrido
    /// disponible — la continuidad del espectro, sin buckets.
    #[test]
    fn el_horizonte_escala_de_forma_continua() {
        let mut prev = 0.0;
        for tau in [1_000.0, 10_000.0, 100_000.0, 1_000_000.0, 10_000_000.0] {
            let mut i = base();
            i.tau_ms = tau;
            let r = compute_tp_sl(i);
            assert!(
                r.sl_pct >= prev,
                "el stop debe crecer monótonamente con tau en {tau}"
            );
            prev = r.sl_pct;
        }
    }

    /// D-747: el deslizamiento por latencia obedece a la difusión. Cuadruplicar
    /// la latencia DUPLICA el desplazamiento esperado (√4 = 2); la fórmula
    /// lineal que usaba el gate lo cuadruplicaba.
    #[test]
    fn el_deslizamiento_por_latencia_escala_con_la_raiz_del_tiempo() {
        let atr = 0.005;
        let s1 = latency_slippage_pct(atr, 15.0);
        let s4 = latency_slippage_pct(atr, 60.0);
        assert!(
            (s4 / s1 - 2.0).abs() < 1e-9,
            "×4 latencia ⇒ ×2 desplazamiento, no ×4: {s1} {s4}"
        );
        // A la escala en la que se MIDE la volatilidad, el desplazamiento
        // esperado es exactamente esa volatilidad.
        assert!((latency_slippage_pct(atr, TAU_REFERENCE_MS) - atr).abs() < 1e-12);
        // Entradas degeneradas no inventan fricción.
        assert_eq!(latency_slippage_pct(atr, 0.0), 0.0);
        assert_eq!(latency_slippage_pct(f64::NAN, 10.0), 0.0);
    }

    /// XLIV-8c (guardia portada de XLV-1, PR #9 de otra sesión Claude): el
    /// gate, los brackets del host, el fallback de entrada, el fallback de
    /// gestión del núcleo y el pre-examen del daemon calculan la fricción con
    /// la MISMA función. XLIV-8 olvidó el cuarto; esta guardia sobre las
    /// fuentes impide que una fórmula inline vuelva a divergir.
    #[test]
    fn xliv_todos_los_pisos_de_friccion_usan_la_funcion_unica() {
        let fuentes = [
            ("risk-engine/lib.rs", include_str!("lib.rs"), 1),
            ("god_engine.rs", include_str!("../../../src/bin/god_engine.rs"), 2),
            ("god-engine-core/lib.rs", include_str!("../../god-engine-core/src/lib.rs"), 1),
            (
                "online_daemon.rs",
                include_str!("../../evolution-engine/src/online_daemon.rs"),
                1,
            ),
        ];
        for (nombre, codigo, esperadas) in fuentes {
            let n = codigo.matches("tp_sl::roundtrip_friction(").count();
            assert!(n >= esperadas, "{nombre}: {n} llamadas, se esperaban {esperadas}");
            assert!(
                !codigo.contains("lat_ref") && !codigo.contains("latency_ref_ms"),
                "{nombre} conserva una normalización lineal de la latencia"
            );
        }
    }

    /// XLIV-8: la fricción de ida y vuelta es la MISMA para el gate, los
    /// brackets del host y el daemon, y su latencia es la difusiva. La ley
    /// lineal que conservaban el host y el daemon cobraba, con el umbral de
    /// pánico en su cota baja (500 ms), 3,5× más deslizamiento por lado.
    #[test]
    fn xliv_friccion_de_ida_y_vuelta_usa_la_ley_difusiva() {
        let (taker, floor, atr, lat) = (0.0005, 0.0001, 0.004, 50.0);
        let f = roundtrip_friction(taker, floor, atr, lat);
        // Identidad bit a bit con la expresión que usaba el gate.
        let gate = taker + taker + 2.0 * (floor + latency_slippage_pct(atr, lat)).clamp(0.0, 0.05);
        assert_eq!(f.to_bits(), gate.to_bits());
        // La ley lineal del host (umbral de pánico = 500 ms) no es la misma.
        let lineal = 2.0 * taker + 2.0 * (floor + (atr * lat / 500.0).clamp(0.0, 0.05));
        let lat_difusiva = latency_slippage_pct(atr, lat);
        let lat_lineal = atr * lat / 500.0;
        assert!((lat_lineal / lat_difusiva - 3.464).abs() < 1e-3, "{lat_lineal} {lat_difusiva}");
        assert!(lineal > f);
        // Acotada al 5 % por lado, igual que antes en el gate.
        assert_eq!(roundtrip_friction(taker, 0.2, atr, lat), 2.0 * taker + 0.10);
        // Sin volatilidad ni latencia sólo quedan comisiones y piso.
        assert_eq!(roundtrip_friction(taker, floor, 0.0, lat), 2.0 * taker + 2.0 * floor);
    }

    /// XLVI·B — el muestreo es determinista por semilla: mismo (base, seed)
    /// ⇒ bit-idéntico. Es la garantía que conserva el determinismo del replay.
    #[test]
    fn xlvib_muestreo_latencia_determinista_por_semilla() {
        for base in [5.0_f64, 30.68, 100.0] {
            for seed in [0u64, 1, 42, u64::MAX, 0xDEAD_BEEF] {
                let a = sample_latency_lognormal_ms(base, seed);
                let b = sample_latency_lognormal_ms(base, seed);
                assert_eq!(a.to_bits(), b.to_bits(), "base={base} seed={seed}");
                assert!(a.is_finite() && (2.0..=500.0).contains(&a), "a={a}");
            }
        }
        // Semillas distintas ⇒ muestras distintas (en general): 100 semillas
        // consecutivas no colapsan a un único valor.
        let uniq: std::collections::HashSet<u64> = (0..100u64)
            .map(|s| sample_latency_lognormal_ms(30.0, s).to_bits())
            .collect();
        assert!(uniq.len() > 90, "colapso a {} valores", uniq.len());
    }

    /// XLVI·B — el gen calibra la MEDIA: mean ≈ base (la normalización
    /// conserva la media), mediana ≈ base·exp(−σ²/2) ≈ 0.94·base (firma que
    /// la auditoría midió: p50 < estática), y la cola derecha existe
    /// (p99 ≈ 2.1× base con σ=0.35). Contornos DIV-2 en su forma corregida.
    #[test]
    fn xlvib_media_preservada_y_cola_presente() {
        let base = 30.68_f64;
        let mut s: Vec<f64> = (0..20_000u64)
            .map(|i| sample_latency_lognormal_ms(base, i))
            .collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = s[s.len() / 2];
        let p99 = s[(s.len() as f64 * 0.99) as usize - 1];
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        // Media ≈ base (±3%): el coste esperado en latencia no cambia.
        assert!((mean / base - 1.0).abs() < 0.03, "mean={mean}");
        // Mediana ≈ 0.94·base (±5%): la normalización exp(σz−σ²/2).
        assert!((p50 / base - 0.9401).abs() < 0.05, "p50={p50}");
        // Cola: p99 > 1.7× base (teórico ≈ 2.1×).
        assert!(p99 > base * 1.7, "p99={p99}");
    }

    /// XLVI·B — sin calibración no se inventa dispersión: base inválido se
    /// devuelve tal cual (conserva el comportamiento previo del estático).
    #[test]
    fn xlvib_base_invalida_pasa_sin_muestrear() {
        assert_eq!(sample_latency_lognormal_ms(0.0, 7).to_bits(), 0.0_f64.to_bits());
        assert!(sample_latency_lognormal_ms(-5.0, 7).is_nan() == false);
        assert_eq!(sample_latency_lognormal_ms(-5.0, 7), -5.0);
        assert!(sample_latency_lognormal_ms(f64::NAN, 7).is_nan());
    }

    /// XLVI·B — efecto agregado sobre la ley difusiva: con el gen calibrando
    /// la MEDIANA, la mediana del slippage muestreado ≈ estática, la media
    /// baja ≤2% (Jensen: E[√L] = √base·exp(−σ²/8)) y la cola sube >20% —
    /// el cuerpo no se penaliza, la cola se cobra. Firma numérica de DIV-2.
    #[test]
    fn xlvib_cola_se_cobra_en_la_ley_difusiva() {
        let (atr, base) = (0.004_f64, 30.68);
        let estatica = latency_slippage_pct(atr, base);
        let mut s: Vec<f64> = (0..20_000u64)
            .map(|i| latency_slippage_pct(atr, sample_latency_lognormal_ms(base, i)))
            .filter(|v| *v > 0.0)
            .collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        let p50 = s[s.len() / 2];
        let p95 = s[(s.len() as f64 * 0.95) as usize - 1];
        let p99 = s[(s.len() as f64 * 0.99) as usize - 1];
        // Cuerpo: mediana ≈ estática (el gen sigue calibrando el centro).
        assert!((p50 / estatica - 1.0).abs() < 0.05, "p50={p50} est={estatica}");
        // Media: Jensen permite bajar hasta ~1.5%; nada más.
        assert!(
            mean / estatica > 0.95 && mean / estatica <= 1.005,
            "media {mean} fuera de banda vs estática {estatica}"
        );
        // Cola: se cobra de verdad (p95 ≈ +29%, p99 ≈ +46% teóricos).
        assert!(p95 > estatica * 1.2, "p95 {p95} ≤ 1.2× estática {estatica}");
        assert!(p99 > estatica * 1.35, "p99 {p99} ≤ 1.35× estática {estatica}");
    }

    /// XLVI·B — la semilla mezcla evento y activo: mismo milisegundo en dos
    /// activos NO comparte muestra; mismo activo reproduce su muestra.
    #[test]
    fn xlvib_semilla_por_evento_y_activo() {
        let s0 = latency_seed(1000, 0);
        assert_eq!(s0, latency_seed(1000, 0), "reproducible");
        assert_ne!(latency_seed(1000, 0), latency_seed(1000, 1), "activos distintos");
        assert_ne!(latency_seed(1000, 0), latency_seed(1001, 0), "eventos distintos");
    }

    #[test]
    fn test_normal_cdf_exactitud_y_simetria() {
        assert_eq!(normal_cdf(0.0), 0.5);
        assert!((normal_cdf(1.95996) - 0.975).abs() < 1e-4);
        assert!((normal_cdf(-1.95996) - 0.025).abs() < 1e-4);
        assert_eq!(normal_cdf(f64::NAN), 0.0);
    }

    #[test]
    fn test_probabilidad_tocar_sl_antes_de_tp_neutral_and_drift() {
        // En martingala neutra (mu = 0), P(hit SL) = TP / (TP + SL)
        // Con TP = 0.02 y SL = 0.01 (RR = 2): P(hit SL) = 0.02 / 0.03 = 2/3 ≈ 0.6667
        let p_neutral = probabilidad_tocar_sl_antes_de_tp(0.02, 0.01, 0.0, 0.001);
        assert!((p_neutral - 2.0 / 3.0).abs() < 1e-4, "neutral={p_neutral}");

        // Con deriva a favor (mu > 0), la probabilidad de tocar SL disminuye
        let p_fav = probabilidad_tocar_sl_antes_de_tp(0.02, 0.01, 0.0005, 0.001);
        assert!(p_fav < p_neutral, "favorable={p_fav} < neutral={p_neutral}");

        // Con deriva adversa (mu < 0), la probabilidad de tocar SL aumenta
        let p_adv = probabilidad_tocar_sl_antes_de_tp(0.02, 0.01, -0.0005, 0.001);
        assert!(p_adv > p_neutral, "adverse={p_adv} > neutral={p_neutral}");
    }

    #[test]
    fn test_probabilidad_primer_toque_stop_antes_de_tau() {
        // A tau -> 0, P(touch) -> 0
        let p_zero = probabilidad_primer_toque_stop_antes_de_tau(0.01, 0.0, 0.0001, 1e-6);
        assert!(p_zero < 1e-6, "p_zero={p_zero}");

        // A mayor tiempo, mayor probabilidad de tocar
        let p_short = probabilidad_primer_toque_stop_antes_de_tau(0.01, 0.0, 0.0001, 60.0);
        let p_long = probabilidad_primer_toque_stop_antes_de_tau(0.01, 0.0, 0.0001, 3600.0);
        assert!(p_long > p_short, "monotonía temporal: {p_long} > {p_short}");
        assert!(p_long <= 1.0);
    }
}
