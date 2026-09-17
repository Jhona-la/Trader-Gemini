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
}

/// Resultado. `tp_pct` y `sl_pct` son fracciones del precio, siempre
/// positivas, y satisfacen por construcción `tp_pct ≥ sl_pct · rr_required`.
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
const TAU_REFERENCE_MS: f64 = 60_000.0;

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
    let sigma_tau = atr * (tau / TAU_REFERENCE_MS).powf(h);

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
    let sl_pct = sl_diffusive.max(sl_floor);

    // 4) RR EXIGIDO POR LA FRICCIÓN AL NIVEL DE STOP RESULTANTE (D-636).
    //    Crece cuando el stop se estrecha: la comisión pesa más sobre un
    //    riesgo menor.
    let rr_required = SuperGenotype::min_rr_for(w, fee, sl_pct);

    // 5) OBJETIVO. El RR genómico puede ser MÁS ambicioso que el mínimo,
    //    nunca menor: el mínimo es una restricción de rentabilidad, no una
    //    preferencia.
    let rr_applied = rr_required.max(1.0);
    let tp_pct = sl_pct * rr_applied;

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
    if target_rr.is_finite() && target_rr > out.rr_required {
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
        }
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
    #[test]
    fn d639_el_stop_escala_con_la_volatilidad_sin_techo() {
        let mut lo = base();
        lo.atr_ratio = 0.005;
        let mut hi = base();
        hi.atr_ratio = 0.020; // 4x de volatilidad
        let r_lo = compute_tp_sl(lo);
        let r_hi = compute_tp_sl(hi);
        assert!(
            r_hi.sl_pct > r_lo.sl_pct * 3.5,
            "el stop debe crecer con la vol: {} vs {}",
            r_lo.sl_pct,
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
            (a_4m.sl_pct - 0.010).abs() < 1e-12,
            "a 4 minutos y H = 0,5 la dispersión debe duplicarse, dio {}",
            a_4m.sl_pct
        );
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
}
