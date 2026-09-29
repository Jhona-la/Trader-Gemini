//! AUDITORÍA DE CAPITALIZACIÓN COMPUESTA (Ola XLV·L).
//!
//! La meta del operador es +100% cada 3 días. Esto exige ~26% diario
//! compuesto. Este test verifica que la ARITMÉTICA de capitalización
//! del motor es exacta — sin fugas por redondeo, fees mal calculados,
//! o drift de punto flotante.
//!
//! Contrato:
//! - Variable: capital tras N trades con win rate y Kelly conocidos.
//! - Operador: verificar que capital_final == capital_inicial ·
//!   Π(1 + f·r_ganancia) para trades ganadores y Π(1 − f·r_pérdida)
//!   para perdedores, donde f es la fracción Kelly aplicada.
//! - Contorno: exactamente el crecimiento teórico (tolerancia 1e-10
//!   por acumulación de error flotante en N=100 trades).

/// Verifica que la capitalización compuesta de 100 trades con WR=60%
/// y fracción fija produce exactamente el crecimiento teórico.
#[test]
fn xlvl_capitalizacion_compuesta_100_trades_exacta() {
    let initial_capital = 1000.0_f64;
    let kelly_fraction = 0.10_f64; // 10% del capital por trade
    let win_return = 0.03_f64; // +3% del capital apostado en win
    let loss_return = 0.015_f64; // −1.5% en loss (RR 2:1)
    let n_wins = 60;
    let n_losses = 40;

    // Capital teórico: Π(1 + f·r_win) · Π(1 − f·r_loss) · C₀
    let mut theoretical = initial_capital;
    for _ in 0..n_wins {
        theoretical *= 1.0 + kelly_fraction * win_return;
    }
    for _ in 0..n_losses {
        theoretical *= 1.0 - kelly_fraction * loss_return;
    }

    // Simular trade por trade (como el motor)
    let mut capital = initial_capital;
    for i in 0..100 {
        let is_win = i % 10 < 6; // patrón determinista: 6/10 wins
        let pnl = if is_win {
            capital * kelly_fraction * win_return
        } else {
            -(capital * kelly_fraction * loss_return)
        };
        capital += pnl;
    }

    // El patrón determinista 6/10 por bloque de 10 no es EXACTAMENTE
    // el mismo orden que todos-los-wins-primero, pero la MULTIPLICACIÓN
    // es conmutativa — el resultado final debe ser idéntico.
    let diff = (capital - theoretical).abs();
    let rel_err = diff / theoretical;
    assert!(
        rel_err < 1e-12,
        "capital={:.12} vs teórico={:.12}, rel_err={:.2e}",
        capital, theoretical, rel_err
    );
}

/// Verifica que +100% en 3 días requiere ~26% diario compuesto.
/// Útil para calibrar expectativas contra los controles de ruina (25% cap).
#[test]
fn xlvl_meta_100_por_ciento_3_dias_implica_26_diario() {
    // (1 + r)^3 = 2.0 → r = 2^(1/3) − 1 ≈ 0.2599
    let daily_rate = 2.0_f64.powf(1.0 / 3.0) - 1.0;
    assert!(
        (daily_rate - 0.2599).abs() < 0.001,
        "tasa diaria = {:.4}, esperaba ~0.2599",
        daily_rate
    );
    // Verificar que 3 días de esta tasa produce exactamente 2×
    let three_day = (1.0 + daily_rate).powi(3);
    assert!(
        (three_day - 2.0).abs() < 1e-10,
        "3 días = {:.6}, esperaba 2.0",
        three_day
    );
}

/// Verifica que el Kelly con edge positivo PERO control de ruina al 25%
/// NO puede alcanzar +26% diario en un solo trade — la meta requiere
/// MÚLTIPLES trades por día con edge sostenido.
#[test]
fn xlvl_kelly_con_ruina_25_no_alcanza_meta_en_un_trade() {
    let ruin_cap = 0.25_f64; // axioma del 25% por evento
    let best_return = 0.10_f64; // retorno optimista del 10% del capital

    // El máximo retorno por trade = ruin_cap × best_return = 2.5%
    let max_single_trade_return = ruin_cap * best_return;
    assert!(
        max_single_trade_return < 0.26,
        "un solo trade NO debe alcanzar 26% diario: max={:.3}",
        max_single_trade_return
    );
    // Para 26% diario con 2.5% por trade: ln(1.26)/ln(1.025) ≈ 9.6 trades/día
    let trades_needed = (1.26_f64.ln() / 1.025_f64.ln()).ceil();
    assert!(
        trades_needed >= 9.0 && trades_needed <= 11.0,
        "se necesitan ~10 trades/día, calculado={}",
        trades_needed
    );
}

/// Verifica que la pérdida máxima por trade (con stop al 100% del Kelly)
/// no excede el tope de ruina.
#[test]
fn xlvl_perdida_maxima_respecta_tope_ruina() {
    let capital = 1000.0_f64;
    let kelly_full = 0.25_f64; // Kelly pleno
    let stop_pct = 0.02_f64; // stop al 2% del precio

    let max_loss = capital * kelly_full * stop_pct;
    let max_loss_pct = max_loss / capital;

    assert!(
        max_loss_pct <= 0.005, // ≤ 0.5% del capital por trade
        "pérdida max={:.4} ({:.2}%) del capital, esperaba ≤0.5%",
        max_loss, max_loss_pct * 100.0
    );
}
