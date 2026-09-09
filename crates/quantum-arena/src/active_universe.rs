//! 🧬 ACTIVE UNIVERSE: Selección Dinámica de Monedas Basada en Capital
//!
//! QUÉ: Módulo que determina cuántas y cuáles monedas operar según el capital disponible.
//! POR QUÉ: Con $13capital base, operar 30 monedas fragmenta el capital ($0.43/coin) haciendo
//!           imposible cumplir el notional mínimo de Binance ($5) sin leverage suicida.
//! PARA QUÉ: Concentrar capital en las 3-5 monedas con mejor relación
//!            volatilidad/liquidez/fees para maximizar crecimiento compuesto.
//! CÓMO: Ranking multi-criterio (tick_impact, lot_notional, volatility, volume).
//! CUÁNDO: Se recalcula al inicio y cada vez que el capital cambia significativamente.
//! DÓNDE: quantum-arena crate, accesible lock-free desde todos los hilos.
//! QUIÉN: live_trader.rs consulta is_active(coin_id), god_engine filtra señales.

/// Máximo de monedas activas por rango de capital.
/// Estos límites están calculados para garantizar que cada moneda reciba
/// suficiente margen para operar sin leverage suicida (>20x).
/// FASE 1: Modelado Matemático Continuo (Crecimiento Logarítmico Asintótico)
/// Reemplaza el hardcoding arbitrario de escalones. Calcula la distribución
/// óptima del portafolio basada en una curva logarítmica que se ajusta a
/// la ley de rendimientos decrecientes y la volatilidad del capital.
#[inline(always)]
pub fn max_active_coins_for_capital(capital: f64) -> usize {
    // FASE 1 & 14: Modelado Asintótico (Raíz Cuadrada)
    // Distribución óptima que concentra micro-capital en pocas monedas
    // y expande naturalmente con el interés compuesto.
    // Ej: $13 -> ~2 monedas. $100 -> 5 monedas. $1000 -> 15 monedas.
    let safe_capital = if capital.is_finite() {
        capital.max(0.0)
    } else {
        13.0
    };
    let base_coins = (safe_capital.max(1.0).sqrt() * 0.5).ceil() as usize;
    let max_universe = crate::symbols::get_active_universe_size();

    if max_universe == 0 {
        // Universo dinámico aún no registrado (arranque en curso o tests):
        // clamp(1, 0) pánico. Sin tope conocido, devolvemos la base (mínimo 1).
        return base_coins.max(1);
    }
    base_coins.clamp(1, max_universe)
}

/// Score de aptitud para micro-cuenta. Combina:
/// - tick_impact_usd: cuánto cuesta un tick con min_qty (menor = mejor control)
/// - lot_notional_usd: cuánto cuesta el lote mínimo (menor = más accesible)
/// - volume_rank: estimado de liquidez (mayor = mejor slippage)
/// - fee_efficiency: ratio maker/taker (menor = mejor)
#[derive(Debug, Clone)]
pub struct CoinFitness {
    pub coin_id: usize,
    pub symbol: String,
    pub scalp_score: f64,
    pub swing_score: f64,
    pub lot_notional_usd: f64,
    pub tick_impact_usd: f64,
}

/// Calcula el universo activo óptimo para el capital dado.
/// Retorna un vector ordenado de coin_ids que deberían estar activos,
/// y un bitmap de 64 bits para consulta O(1) lock-free.
pub fn calculate_active_universe(
    capital: f64,
    prices: &[f64],
    forced_coin_ids: &[usize],
) -> (Vec<CoinFitness>, u64) {
    let safe_capital = if capital.is_finite() {
        capital.max(0.0)
    } else {
        13.0
    };
    let max_coins = max_active_coins_for_capital(safe_capital);
    let mut candidates: Vec<CoinFitness> = Vec::with_capacity(prices.len());

    for (i, &price) in prices.iter().enumerate() {
        let Some(spec_data) = super::symbol_registry::try_spec(i) else {
            continue; // Spec aún no registrada (arranque): saltar sin pánico
        };
        let spec_ref = &spec_data;
        if price <= 0.0 {
            continue; // No tenemos precio todavía para esta moneda
        }

        let lot_notional = spec_ref.min_qty * price;
        let tick_impact = spec_ref.tick_size * spec_ref.min_qty;

        // FASE 28: Zero-Orphans (Hard-Forcing). Si tenemos posiciones abiertas en esta moneda,
        // le inyectamos una infinidad de score y evadimos el filtro de capital.
        let is_forced = forced_coin_ids.contains(&i);

        if !is_forced {
            // Penalizar monedas cuyo lote mínimo excede el capital disponible por moneda
            let capital_per_coin = safe_capital / max_coins as f64;
            if lot_notional > capital_per_coin * 20.0 {
                // Ni con 20x leverage podemos comprar el lote mínimo con nuestro capital/moneda
                continue;
            }
        }

        let accessibility_score = 1.0 / (lot_notional.max(0.01));
        let granularity_score = 1.0 / (tick_impact.max(0.000001));
        let fee_score = 1.0 / ((spec_ref.maker_fee + spec_ref.taker_fee).max(0.0001));

        let mut scalp_score =
            accessibility_score * 10.0 + granularity_score * 5.0 + fee_score * 1.0;
        let mut swing_score =
            accessibility_score * 20.0 + fee_score * 0.5 + granularity_score * 0.1;

        if is_forced {
            scalp_score += 1_000_000.0;
            swing_score += 1_000_000.0;
        }

        candidates.push(CoinFitness {
            coin_id: i,
            symbol: spec_ref.symbol.clone(),
            scalp_score,
            swing_score,
            lot_notional_usd: lot_notional,
            tick_impact_usd: tick_impact,
        });
    }

    // Sort by joint multi-horizon fitness (max(scalp_score, swing_score))
    // to guarantee both prime Scalp and high-conviction Swing assets are active simultaneously.
    candidates.sort_by(|a, b| {
        let score_a = a.scalp_score.max(a.swing_score);
        let score_b = b.scalp_score.max(b.swing_score);
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Si los forced coins exceden max_coins, igual los incluimos a todos
    // (el mercado manda sobre las reglas logarítmicas de capital).
    let take_count = max_coins.max(forced_coin_ids.len());
    candidates.truncate(take_count);

    let mut bitmap = 0u64;
    for c in &candidates {
        if c.coin_id < 64 {
            bitmap |= 1u64 << c.coin_id;
        }
    }

    (candidates, bitmap)
}

/// 🚀 FASE XXI: Selección Dinámica basada en Momentum y Volumen (L2 Radar)
pub fn calculate_dynamic_universe(
    capital: f64,
    prices: &[f64],
    volumes_usd: &[f64],
    price_changes_pct: &[f64],
    forced_coin_ids: &[usize],
) -> (Vec<CoinFitness>, u64) {
    let max_coins = max_active_coins_for_capital(capital);
    let mut candidates: Vec<CoinFitness> = Vec::with_capacity(prices.len());

    for i in 0..prices.len() {
        let Some(spec_data) = super::symbol_registry::try_spec(i) else {
            continue;
        };
        let spec_ref = &spec_data;
        let price = prices[i];
        if price <= 0.0 {
            continue;
        }

        let lot_notional = spec_ref.min_qty * price;
        let tick_impact = spec_ref.tick_size * spec_ref.min_qty;
        let capital_per_coin = capital / max_coins as f64;

        let is_forced = forced_coin_ids.contains(&i);
        if !is_forced && lot_notional > capital_per_coin * 20.0 {
            continue;
        }

        let accessibility_score = 1.0 / (lot_notional.max(0.01));
        let granularity_score = 1.0 / (tick_impact.max(0.000001));

        let fee_score = 1.0 / ((spec_ref.maker_fee + spec_ref.taker_fee).max(0.0001));
        // Momentum = |% Change| * log10(Volumecapital base)
        // Monedas que se mueven rápido con alto volumen tendrán un momentum_score altísimo
        let momentum_score = price_changes_pct[i].abs() * volumes_usd[i].max(1.0).log10();

        // Mezclamos la accesibilidad base con el momentum dinámico
        // BIFURCACIÓN DE ESTRATEGIA (SCALP VS SWING)
        // Scalp valora infinitamente más la granularidad (tick size) y fees bajos.
        let mut scalp_score = (accessibility_score * 5.0)
            + (granularity_score * 0.5)
            + (momentum_score * 2.0)
            + fee_score * 1.0;

        // Swing valora más la accesibilidad global (margen) y soporta peor granularidad.
        let mut swing_score = (accessibility_score * 20.0)
            + (momentum_score * 5.0)
            + (granularity_score * 0.1)
            + fee_score * 0.5;

        if is_forced {
            scalp_score += 1_000_000.0;
            swing_score += 1_000_000.0;
        }

        candidates.push(CoinFitness {
            coin_id: i,
            symbol: spec_ref.symbol.clone(),
            scalp_score,
            swing_score,
            lot_notional_usd: lot_notional,
            tick_impact_usd: tick_impact,
        });
    }

    // Sort default by scalp_score since Scalping is the primary HFT engine
    candidates.sort_by(|a, b| {
        b.scalp_score
            .partial_cmp(&a.scalp_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let take_count = max_coins.max(forced_coin_ids.len());
    candidates.truncate(take_count);

    let mut bitmap = 0u64;
    for c in &candidates {
        if c.coin_id < 64 {
            bitmap |= 1u64 << c.coin_id;
        }
    }

    (candidates, bitmap)
}

/// Consulta O(1) lock-free si una moneda está en el universo activo.
#[inline(always)]
pub fn is_coin_active(bitmap: u64, coin_id: usize) -> bool {
    if coin_id >= 64 {
        return false;
    }
    bitmap & (1u64 << coin_id) != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Registra un universo de 30 símbolos para que el clamp superior sea determinista.
    fn setup_universe_30() {
        let symbols: Vec<String> = (0..30).map(|i| format!("SYM{i}USDT")).collect();
        crate::symbols::update_dynamic_universe(symbols);
    }

    #[test]
    fn test_micro_account_selects_few_coins() {
        let _guard = crate::symbols::UNIVERSE_TEST_MUTEX.lock().unwrap();
        setup_universe_30();
        // Fórmula: ceil(sqrt(15) * 0.5) = ceil(1.94) = 2
        let max = max_active_coins_for_capital(15.0);
        assert_eq!(max, 2, "micro-capital debe concentrarse en pocas monedas");
    }

    #[test]
    fn test_medium_account_selects_more() {
        let _guard = crate::symbols::UNIVERSE_TEST_MUTEX.lock().unwrap();
        setup_universe_30();
        // Fórmula: ceil(sqrt(100) * 0.5) = ceil(5.0) = 5
        let max = max_active_coins_for_capital(100.0);
        assert_eq!(max, 5, "capital medio expande el universo activo");
    }

    #[test]
    fn test_large_account_full_universe() {
        let _guard = crate::symbols::UNIVERSE_TEST_MUTEX.lock().unwrap();
        setup_universe_30();
        // Fórmula: ceil(sqrt(5000) * 0.5) = 36, clamped al universo (30)
        let max = max_active_coins_for_capital(5000.0);
        assert_eq!(max, 30, "capital grande saturado al universo disponible");
    }

    #[test]
    fn test_empty_universe_does_not_panic() {
        let _guard = crate::symbols::UNIVERSE_TEST_MUTEX.lock().unwrap();
        crate::symbols::update_dynamic_universe(Vec::new());
        // Universo sin registrar: debe devolver la base sin clamping, nunca pánico.
        assert_eq!(max_active_coins_for_capital(15.0), 2);
        assert_eq!(max_active_coins_for_capital(0.0), 1);
    }

    #[test]
    fn test_bitmap_query() {
        let bitmap = 0b101; // coins 0 and 2 active
        assert!(is_coin_active(bitmap, 0));
        assert!(!is_coin_active(bitmap, 1));
        assert!(is_coin_active(bitmap, 2));
    }

    #[test]
    fn test_active_universe_with_prices() {
        setup_universe_30();
        // Registrar specs para los 8 símbolos con precio (el resto queda sin spec → skip)
        let specs: Vec<crate::symbol_registry::SymbolSpec> = (0..8)
            .map(|i| crate::symbol_registry::SymbolSpec {
                symbol: format!("SYM{i}USDT"),
                step_size: 0.001,
                tick_size: 0.01,
                min_qty: 0.01,
                min_notional: 5.0,
                max_leverage: 20,
                maker_fee: 0.0002,
                taker_fee: 0.0004,
                is_shadow: false,
            })
            .collect();
        crate::symbol_registry::update_registry(specs);

        // Precios aproximados realistas
        let mut prices = vec![0.0f64; 100];
        prices[0] = 60000.0; // BTC
        prices[1] = 3500.0; // ETH
        prices[2] = 600.0; // BNB
        prices[3] = 150.0; // SOL
        prices[4] = 0.50; // XRP
        prices[5] = 0.40; // ADA
        prices[6] = 35.0; // AVAX
        prices[7] = 0.15; // DOGE

        let (selected, _bitmap) = calculate_active_universe(50.0, &prices, &[]);
        assert!(
            !selected.is_empty(),
            "con specs registradas debe seleccionar monedas"
        );
        assert!(
            selected.len() <= 30,
            "Should select up to 30 coins, got {}",
            selected.len()
        );
        telemetry_engine::telemetry!("Selected coins for $50:");
        for c in &selected {
            telemetry_engine::telemetry!(
                "  {} score={:.2} lot_notional=${:.4}",
                c.symbol,
                c.scalp_score,
                c.lot_notional_usd
            );
        }
    }
}
