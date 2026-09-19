// trailing.rs
// QUANTUM TRAILING ENGINE - Zero-copy, sub-microsecond trailing evaluation

#[repr(C)]
pub struct TrailingResult {
    pub stop_price: f64,
    pub force_close: bool,
    pub new_phase: i32,
    pub max_pnl_pct: f64,
    pub mfe_atr: f64,
}

pub fn evaluate_quantum_trailing(
    pos_side: i32, // 1 for LONG, -1 for SHORT
    entry_price: f64,
    current_price: f64,
    current_atr: f64,
    current_phase: i32,
    mfe_atr: f64,
    max_pnl_pct: f64,
    current_trail_stop: f64,
    // Profile configs
    pullback_tol: f64,
    trail_f1: f64,
    trail_f2: f64,
    trail_f3: f64,
    trail_runner: f64,
) -> TrailingResult {
    evaluate_quantum_trailing_with_fee(
        pos_side,
        entry_price,
        current_price,
        current_atr,
        current_phase,
        mfe_atr,
        max_pnl_pct,
        current_trail_stop,
        pullback_tol,
        trail_f1,
        trail_f2,
        trail_f3,
        trail_runner,
        0.0006, // Fallback fee rate (0.02% maker + 0.04% taker roundtrip VIP0)
        0.012,  // B3.27 — TP fallback nominal 1.2%
        0.0,    // S-2 — persistencia neutral (browniano) para el wrapper legado
    )
}

pub fn evaluate_quantum_trailing_with_fee(
    pos_side: i32, // 1 for LONG, -1 for SHORT
    entry_price: f64,
    current_price: f64,
    current_atr: f64,
    mut current_phase: i32,
    mut mfe_atr: f64,
    mut max_pnl_pct: f64,
    current_trail_stop: f64,
    // Profile configs
    pullback_tol: f64,
    trail_f1: f64,
    trail_f2: f64,
    trail_f3: f64,
    trail_runner: f64,
    fee_rate: f64,
    tp_frac: f64, // B3.27 — distancia al TP como fracción del precio
    // S-2 (ESPECTRALIZACIÓN): persistencia de la escala dominante [-1,+1].
    // +1 (tendencial) ⇒ la escalera se EXTIENDE (be 50%, half 70%, profit
    // 85%, runner 100%: el trade corre hasta el TP — la tendencia sostiene).
    // −1 (mean-revert) ⇒ se COMPRIME (30/45/60/80%: cosecha temprana — la
    // ganancia no se sostiene). 0 (browniano) ⇒ 40/60/80/95% (B3.27 exacto).
    spectral_persistence: f64,
) -> TrailingResult {
    if current_atr <= 0.0
        || !current_atr.is_finite()
        || entry_price <= 0.0
        || !entry_price.is_finite()
        || current_price <= 0.0
        || !current_price.is_finite()
    {
        return TrailingResult {
            stop_price: current_trail_stop,
            force_close: false,
            new_phase: current_phase,
            max_pnl_pct,
            mfe_atr,
        };
    }

    // Sanitizar parámetros de perfil para evitar NaNs en trailing stop
    let pullback_tol = if pullback_tol.is_finite() && pullback_tol > 0.0 {
        pullback_tol
    } else {
        1.5
    };
    let trail_f1 = if trail_f1.is_finite() && trail_f1 > 0.0 {
        trail_f1
    } else {
        1.5
    };
    let trail_f2 = if trail_f2.is_finite() && trail_f2 > 0.0 {
        trail_f2
    } else {
        2.0
    };
    let trail_f3 = if trail_f3.is_finite() && trail_f3 > 0.0 {
        trail_f3
    } else {
        2.5
    };
    let trail_runner = if trail_runner.is_finite() && trail_runner > 0.0 {
        trail_runner
    } else {
        1.5
    };
    let fee_rate = if fee_rate.is_finite() && fee_rate >= 0.0 {
        fee_rate
    } else {
        0.0006
    };

    // FIX #666: Sanitizar mfe_atr y max_pnl_pct para evitar propagación de NaNs en trailing stop
    if !mfe_atr.is_finite() {
        mfe_atr = 0.0;
    }
    if !max_pnl_pct.is_finite() {
        max_pnl_pct = 0.0;
    }

    // 1. Calculate PnL (ATR and Pct)
    let pnl_pct = if pos_side == 1 {
        (current_price - entry_price) / entry_price
    } else {
        (entry_price - current_price) / entry_price
    };

    let pnl_atr = if pos_side == 1 {
        (current_price - entry_price) / current_atr
    } else {
        (entry_price - current_price) / current_atr
    };

    // 2. Update State
    if pnl_atr > mfe_atr {
        mfe_atr = pnl_atr;
    }
    if pnl_pct > max_pnl_pct {
        max_pnl_pct = pnl_pct;
    }

    // Escudo Cuántico — B3.27 + S-2 + D-711: UNA SOLA ESCALERA, relativa al TP.
    //
    // B3.27 midió la transición de fase y la escalera en fracciones del TP (no
    // en múltiplos del fee, que decapitaban los ganadores dentro del rango del
    // propio TP) y S-2 interpoló cada escalón por persistencia espectral. Pero
    // la transición de fase quedó en 0,40·TP fijo mientras el breakeven del
    // escudo vive en lerp(0,30; 0,50)·TP: con persistencia baja (t < 0,5) el
    // escudo pedía proteger a 0,30·TP y la fase seguía en 0 hasta 0,40·TP —el
    // mismo limbo que D-711 cerró cuando las dos cotas salían del fee—. El
    // escudo sólo corre con `current_phase != 0`, así que la transición existe
    // para HABILITARLO: un único `be_trigger`, calculado aquí, sirve a ambos.
    let effective_tp = if tp_frac.is_finite() && tp_frac > 0.001 {
        tp_frac
    } else {
        0.012 // fallback: TP nominal 1.2% cuando no se pasa
    };
    let effective_fee = fee_rate.max(0.0004);
    // S-2 — t∈[0,1]: t=1 tendencial / t=0 mean-revert. Cada nivel es lerp(MR, TEND).
    let t = if spectral_persistence.is_finite() {
        ((spectral_persistence + 1.0) * 0.5).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let lvl = |mr: f64, tend: f64| mr + (tend - mr) * t;
    let be_trigger = effective_tp * lvl(0.30, 0.50);

    // 3. Phase Transitions (Desasfixiadas: permiten que el trade desarrolle su ciclo hasta TP)
    if current_phase == 0 && (pnl_atr >= 1.5 || max_pnl_pct >= be_trigger) {
        current_phase = 1;
    } else if current_phase == 1 && (pnl_atr >= 2.5 || max_pnl_pct >= be_trigger * 1.5) {
        current_phase = 2;
    } else if current_phase == 2 && pnl_atr >= 4.0 {
        current_phase = 3;
    } else if current_phase == 3 && mfe_atr >= 5.0 {
        current_phase = 4;
    }

    // 4. Mechanism Proposals
    let mut best_stop = if current_trail_stop > 1e-9 {
        current_trail_stop
    } else {
        0.0
    };
    let mut proposals = [0.0; 4];
    let mut prop_count = 0;

    // Initial Stop Loss Phase (Phase 0)
    if current_phase == 0 {
        // FIX #1508: Acotación de SL inicial en Fase 0 para evitar rebasar margen de seguridad
        let max_sl_dist = (entry_price * 0.05).max(1e-6); // Máximo 5% de distancia inicial
        let safe_atr_dist = (3.0 * current_atr).min(max_sl_dist);
        let initial_sl = if pos_side == 1 {
            (entry_price - safe_atr_dist).max(1e-8)
        } else {
            entry_price + safe_atr_dist
        };
        proposals[prop_count] = initial_sl;
        prop_count += 1;
    }

    // T1: ATR Step Trailing
    if current_phase != 0 {
        let dist_atr = match current_phase {
            1 => trail_f1,
            2 => trail_f2,
            3 => trail_f3,
            4 => trail_runner,
            _ => 2.0,
        };

        let mut t1_stop = if pos_side == 1 {
            current_price - (dist_atr * current_atr)
        } else {
            current_price + (dist_atr * current_atr)
        };

        // B3.27 — ESCALERA RELATIVA AL TP (no al fee). Hallazgo diario:
        // 73% WR pero RRR 0.30 porque la escalera ×fee disparaba TODA dentro
        // del rango del TP (con VIP0: be a 0.65%, half a 0.94%, profit a
        // 1.05% — y el TP a 0.66-1.2%). Los winners se decapitaban antes de
        // correr. Ahora cada nivel es FRACCIÓN del TP, interpolada por la
        // persistencia espectral (S-2). El ATR-trailing (T1 arriba) sigue
        // dando la distancia de respiración; esta escalera sólo pone SUELOS
        // progresivos — el trade respira hasta su TP. `be_trigger`,
        // `effective_tp`, `effective_fee`, `t` y `lvl` son los de arriba (D-711).
        let be_buffer = (effective_fee * 2.0).clamp(0.0010, 0.0018); // costo neto post-fees — SÍ relativo al fee (es un costo)
        let half_lock_trigger = effective_tp * lvl(0.45, 0.70);
        let half_lock_gain = effective_tp * lvl(0.15, 0.35);
        let profit_lock_trigger = effective_tp * lvl(0.60, 0.85);
        let profit_lock_gain = effective_tp * lvl(0.40, 0.60);
        let runner_lock_trigger = effective_tp * lvl(0.80, 1.00);
        let runner_lock_gain = effective_tp * lvl(0.60, 0.85);

        if max_pnl_pct >= be_trigger {
            if pos_side == 1 {
                // D-711: el largo recibe la MISMA guarda que ya tenía el corto
                // (`&& lock < current_price`). Sin ella, un bloqueo que quedaría
                // POR ENCIMA del mercado se fijaba igual y la acotación final
                // (`best_stop.min(current_price·(1 − 1 pb))`) lo aplastaba a un
                // punto básico bajo el precio: el largo salía al primer tick
                // adverso mientras su corto espejo conservaba un stop a distancia
                // ATR. Misma geometría, dos comportamientos de salida según la
                // dirección. Un bloqueo inalcanzable se OMITE, como en el corto.
                let breakeven_price = entry_price * (1.0 + be_buffer);
                if t1_stop < breakeven_price && breakeven_price < current_price {
                    t1_stop = breakeven_price;
                }
                if max_pnl_pct >= half_lock_trigger {
                    let half_lock = entry_price * (1.0 + half_lock_gain);
                    if t1_stop < half_lock && half_lock < current_price {
                        t1_stop = half_lock;
                    }
                }
                if max_pnl_pct >= profit_lock_trigger {
                    let profit_lock = entry_price * (1.0 + profit_lock_gain);
                    if t1_stop < profit_lock && profit_lock < current_price {
                        t1_stop = profit_lock;
                    }
                }
                if max_pnl_pct >= runner_lock_trigger {
                    let runner_lock = entry_price * (1.0 + runner_lock_gain);
                    if t1_stop < runner_lock && runner_lock < current_price {
                        t1_stop = runner_lock;
                    }
                }
            } else {
                let breakeven_price = entry_price * (1.0 - be_buffer);
                if (t1_stop == 0.0 || t1_stop > breakeven_price) && breakeven_price > current_price
                {
                    t1_stop = breakeven_price;
                }
                if max_pnl_pct >= half_lock_trigger {
                    let half_lock = entry_price * (1.0 - half_lock_gain);
                    if (t1_stop == 0.0 || t1_stop > half_lock) && half_lock > current_price {
                        t1_stop = half_lock;
                    }
                }
                if max_pnl_pct >= profit_lock_trigger {
                    let profit_lock = entry_price * (1.0 - profit_lock_gain);
                    // FIX #600: Garantizar que profit_lock esté por encima del precio actual de mercado para cortos
                    if (t1_stop == 0.0 || t1_stop > profit_lock) && profit_lock > current_price {
                        t1_stop = profit_lock;
                    }
                }
                if max_pnl_pct >= runner_lock_trigger {
                    let runner_lock = entry_price * (1.0 - runner_lock_gain);
                    if (t1_stop == 0.0 || t1_stop > runner_lock) && runner_lock > current_price {
                        t1_stop = runner_lock;
                    }
                }
            }
        }
        proposals[prop_count] = t1_stop;
        prop_count += 1;
    }

    // T3: Parabolic Trailing
    if mfe_atr >= 3.0 {
        let mut factor = 0.02 + (mfe_atr - 3.0) * 0.05;
        if factor > 0.20 {
            factor = 0.20;
        }

        let mut dist_parabolic = trail_f3 - (mfe_atr * factor);
        if dist_parabolic < 0.5 {
            dist_parabolic = 0.5;
        }

        let t3_stop = if pos_side == 1 {
            current_price - (dist_parabolic * current_atr)
        } else {
            current_price + (dist_parabolic * current_atr)
        };
        proposals[prop_count] = t3_stop;
        prop_count += 1;
    }

    // T5: Volatility Contraction (activado en Fase 3+ de aceleración terminal para no asfixiar el runner)
    if current_phase >= 3 {
        let dist_vol = (2.2 * current_atr).max(current_price * 0.0035);
        let t5_stop = if pos_side == 1 {
            current_price - dist_vol
        } else {
            current_price + dist_vol
        };
        proposals[prop_count] = t5_stop;
        prop_count += 1;
    }

    // Evaluate best stop
    for i in 0..prop_count {
        let p = proposals[i];
        if pos_side == 1 {
            if best_stop == 0.0 || p > best_stop {
                best_stop = p;
            }
        } else {
            if best_stop == 0.0 || p < best_stop {
                best_stop = p;
            }
        }
    }

    // FIX #623: El trailing stop nunca debe cruzar el precio actual de mercado en la dirección contraria
    if pos_side == 1 && best_stop > 0.0 {
        best_stop = best_stop.min(current_price * (1.0 - 0.0001));
    } else if pos_side == -1 && best_stop > 0.0 {
        best_stop = best_stop.max(current_price * (1.0 + 0.0001));
    }

    // Force Close Check
    // FIX #711: Garantizar piso de tolerancia a pullback (>= 0.5 ATR) y finitud en best_stop
    let safe_pnl_atr = if pnl_atr.is_finite() { pnl_atr } else { 0.0 };
    let dd_atr = mfe_atr - safe_pnl_atr;
    let mut current_tol = pullback_tol;
    if current_phase == 3 || current_phase == 4 {
        current_tol *= 0.8;
    }
    let safe_tol = if current_tol.is_finite() && current_tol > 0.0 {
        current_tol.max(0.5)
    } else {
        1.5
    };

    let mut force_close = false;
    if mfe_atr > 1.0 && dd_atr.is_finite() && dd_atr > safe_tol {
        force_close = true;
    }

    let final_stop = if best_stop.is_finite() && best_stop > 0.0 {
        best_stop
    } else {
        current_trail_stop
    };

    TrailingResult {
        stop_price: final_stop,
        force_close,
        new_phase: current_phase,
        max_pnl_pct,
        mfe_atr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_god_engine_trailing_long_phases_progression() {
        // FIX #1404 (D-453/454/461): la fase 0→1 YA NO dispara a 0.5 ATR —
        // exige 2.0 ATR de pnl O MFE ≥ be_trigger (1-2%). Un movimiento de
        // solo +0.5 ATR debe DEJAR la posición en fase 0 (desarrollo del
        // ciclo; el breakeven temprano asfixiaba los trades).
        let res_early = evaluate_quantum_trailing(
            1, 60000.0, 60050.0, 100.0, 0, 0.0, 0.0, 0.0, 1.5, 1.5, 2.0, 2.5, 1.5,
        );
        assert_eq!(res_early.new_phase, 0, "+0.5 ATR ya no promueve a fase 1");

        // Entrada a fase 1 por MFE porcentual (60600 = +1.0% ≥ be_trigger):
        let res1 = evaluate_quantum_trailing(
            1,       // LONG
            60000.0, // entry
            60600.0, // current (+1.0% MFE dispara be_trigger)
            100.0,   // ATR
            0,       // phase 0
            0.0, 0.0, 0.0, 1.5, 1.5, 2.0, 2.5, 1.5,
        );
        assert_eq!(res1.new_phase, 1);
        assert!(res1.stop_price > 0.0);
        assert!(!res1.force_close);

        // Price reaches Phase 2: nueva puerta 1→2 exige pnl_atr ≥ 3.5
        // (FIX #1404; +2.0 ATR ya NO promueve) o MFE ≥ 1.5×be_trigger.
        let res2 = evaluate_quantum_trailing(
            1,
            60000.0,
            60350.0, // +3.5 ATR
            100.0,
            1,
            res1.mfe_atr,
            res1.max_pnl_pct,
            res1.stop_price,
            1.5,
            1.5,
            2.0,
            2.5,
            1.5,
        );
        assert_eq!(res2.new_phase, 2);

        // Price advances to Phase 3: puerta 2→3 exige pnl_atr ≥ 5.0
        let res3 = evaluate_quantum_trailing(
            1,
            60000.0,
            60500.0, // +5.0 ATR
            100.0,
            2,
            res2.mfe_atr,
            res2.max_pnl_pct,
            res2.stop_price,
            1.5,
            1.5,
            2.0,
            2.5,
            1.5,
        );
        assert_eq!(res3.new_phase, 3);
        assert!(res3.stop_price > 60000.0); // Stop locked in profit
    }

    #[test]
    fn test_god_engine_trailing_short_pullback_force_close() {
        // Short entry at 60000, drops to 59700 (+3 ATR), then bounces back to 59900 (pullback > 1.5 ATR)
        let res = evaluate_quantum_trailing(
            -1, // SHORT
            60000.0, 59900.0, // current
            100.0,   // ATR
            2,       // phase 2
            3.0,     // mfe_atr was 3.0
            0.005, 59800.0, 1.5, 1.5, 2.0, 2.5, 1.5,
        );
        // mfe_atr = 3.0, current pnl_atr = 1.0 -> dd_atr = 2.0 > safe_tol (1.5) -> force_close
        assert!(res.force_close);
    }

    #[test]
    fn test_god_engine_trailing_nan_and_zero_atr_immunity() {
        let res_nan = evaluate_quantum_trailing(
            1,
            f64::NAN,
            60000.0,
            0.0,
            0,
            0.0,
            0.0,
            59000.0,
            1.5,
            1.5,
            2.0,
            2.5,
            1.5,
        );
        assert_eq!(res_nan.stop_price, 59000.0);
        assert!(!res_nan.force_close);
    }

    #[test]
    fn d711_largo_y_corto_protegen_igual_de_lejos() {
        // Espejo exacto: misma entrada, mismo recorrido a favor, mismo ATR y fase.
        let atr = 100.0;
        let entry = 60_000.0;
        let largo = evaluate_quantum_trailing(
            1, entry, entry + 0.005 * entry, atr, 2, 0.0, 0.02, 0.0, 0.0006, 1.0, 1.5, 2.0, 3.0,
        );
        let corto = evaluate_quantum_trailing(
            -1, entry, entry - 0.005 * entry, atr, 2, 0.0, 0.02, 0.0, 0.0006, 1.0, 1.5, 2.0, 3.0,
        );
        let d_largo = (entry + 0.005 * entry - largo.stop_price).abs();
        let d_corto = (corto.stop_price - (entry - 0.005 * entry)).abs();
        assert!(
            (d_largo - d_corto).abs() < entry * 1e-4,
            "el stop del largo queda a {d_largo} y el del corto a {d_corto}: la protección debe ser simétrica"
        );
    }

}
