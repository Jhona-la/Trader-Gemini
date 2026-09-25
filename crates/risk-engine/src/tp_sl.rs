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
    /// RR mínimo que la fricción exige al nivel de SL resultante, evaluado en
    /// el win rate de diseño (D-681).
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

/// FUNCIÓN PURA ÚNICA. La invocan, con las MISMAS entradas, tanto el gate de
/// expectativa como el constructor de la orden: es imposible por construcción
/// que evalúen trades distintos (D-637).
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
    // D-681 (DÉCIMA OLA): el RR mínimo se evalúa en el win rate de DISEÑO, no
    // en el observado. Con el observado la geometría era autorreferente: cada
    // pérdida bajaba `w`, subía `RR_req = (1−w)/w + f/(w·SL)`, alejaba el TP y
    // reducía la probabilidad de alcanzarlo, lo que producía más pérdidas. Con
    // `w = 0` tras una primera pérdida (acotado a 0,05) el TP quedaba a 19
    // stops o más: 0 salidas por TP en el backtest forense desde 220b7433, y el
    // breakeven y el trailing —que se arman en fracción del TP— dejaban de
    // activarse. La evidencia observada pertenece al gate y al Kelly, no a la
    // geometría de la orden.
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
        _ => atr * (tau / TAU_REFERENCE_MS).powf(h),
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

/// Variante que respeta un RR genómico más ambicioso que el mínimo exigido.
pub fn compute_tp_sl_with_target_rr(input: TpSlInputs, target_rr: f64) -> TpSl {
    let mut out = compute_tp_sl(input);
    // B3.24 (MOD3/5-019): el objetivo genómico solo puede AMPLIAR el
    // recorrido, nunca romper el cap SL ≤ TP/2 que la función pura garantiza
    // (rr_applied ≥ 2 tras el cap): se aplica únicamente si supera el RR ya
    // aplicado. Antes, un target ∈ (rr_required, 2) volvía a dejar el SL por
    // encima de TP/2 justo después de que la base lo respetara.
    if target_rr.is_finite() && target_rr > out.rr_applied {
        out.rr_applied = target_rr;
        out.tp_pct = out.sl_pct * target_rr;
    }
    out
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
        let rr = SuperGenotype::min_rr_for(SuperGenotype::WORST_TOLERATED_WR, i.roundtrip_fee, r.sl_pct)
            .max(1.0);
        assert!((r.tp_pct / r.sl_pct - rr).abs() < 1e-12);
        assert!(r.rr_applied < 3.0, "a la geometría de referencia el RR es acotado: {}", r.rr_applied);
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
}
