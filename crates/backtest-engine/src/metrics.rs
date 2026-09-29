//! MÉTRICAS EX-POST DEL PANEL DE LA META RE-ENCUADRADA (Ola XLVIII·A).
//!
//! Doctrina del operador (2026-09-29): la meta no es un número fijo de
//! crecimiento — es MAXIMIZAR EL CRECIMIENTO GEOMÉTRICO sujeto a riesgo,
//! drawdown, costos y capacidad. Sin estas métricas, cualquier promesa de
//! crecimiento exponencial es una ilusión. Este módulo es el panel mínimo
//! exigido: CAGR, Sharpe y Sortino anualizados, Calmar, MaxDD, CVaR
//! empírico, turnover, win-rate y profit factor.
//!
//! Convenciones (explícitas, sin ambigüedad):
//! - **Escala anual**: 365,25 días; el tiempo se mide por el SPAN real de
//!   la muestra (último − primer evento), no por suposición de calendario.
//! - **Sharpe/Sortino**: por TRADE, anualizados con √(trades/año) — la
//!   frecuencia medida de la propia muestra. Sin tasa libre de riesgo
//!   (convención cripto 24/7, r_f ≈ 0 en USDT-margen).
//! - **Sortino**: desviación a la baja contra objetivo 0 (loss-only);
//!   sin pérdidas ⇒ +∞ se reporta como `f64::INFINITY` (no se clamp-ea:
//!   esconder el caso degenerado sería deshonesto).
//! - **CVaR 95**: media empírica del peor 5 % de los PnL por trade
//!   (magnitud de pérdida, positiva). Exige ≥ 20 trades: con menos, la
//!   cola no es estimable y se reporta NaN (contrato explícito).
//! - **Turnover**: nocional operado / capital / día — presión de capacity
//!   y costos, no un mérito.
//! - **Calmar**: CAGR/|MaxDD|; MaxDD 0 con CAGR > 0 ⇒ INFINITY (perfecto
//!   e irreal — que se vea).
//! - Entradas vacías o capital ≤ 0: todo el panel a 0/NaN según campo —
//!   jamás fabricar métricas sin muestra.

/// Panel de métricas ex-post. Todos los campos son del cálculo puro sobre
/// la muestra dada; ningún campo se "rellena" sin evidencia.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ExPostMetrics {
    pub n_trades: usize,
    pub span_days: f64,
    /// Retorno geométrico anualizado: (fin/inicio)^(365,25/d) − 1.
    /// Capital final ≤ 0 ⇒ −1 (ruina total es el piso).
    pub cagr: f64,
    /// Sharpe por trade anualizado: media/σ · √(trades/año).
    pub sharpe_ann: f64,
    /// Sortino por trade anualizado: media/σ_a la baja · √(trades/año).
    pub sortino_ann: f64,
    /// CAGR/|MaxDD| — eficiencia del crecimiento por unidad de caída.
    pub calmar: f64,
    /// Máxima caída fraccional del capital (0 = sin caídas).
    pub max_dd: f64,
    /// CVaR 95 empírico por trade (magnitud de pérdida, ≥0). NaN si
    /// n_trades < 20: la cola no es estimable con menos muestra.
    pub cvar_95: f64,
    /// Nocional operado / capital / día.
    pub turnover_per_day: f64,
    /// Trades ganadores / total (PnL > 0).
    pub win_rate: f64,
    /// Σ ganancias / |Σ pérdidas|; sin pérdidas ⇒ INFINITY.
    pub profit_factor: f64,
}

impl ExPostMetrics {
    /// Panel de una línea para telemetría/informes.
    pub fn panel_line(&self) -> String {
        format!(
            "n={} d={:.1} CAGR={:.1}% Sharpe={:.2} Sortino={:.2} Calmar={:.2} MaxDD={:.1}% CVaR95={:.4} turnover/d={:.1} WR={:.1}% PF={:.2}",
            self.n_trades,
            self.span_days,
            self.cagr * 100.0,
            self.sharpe_ann,
            self.sortino_ann,
            self.calmar,
            self.max_dd * 100.0,
            self.cvar_95,
            self.turnover_per_day,
            self.win_rate * 100.0,
            self.profit_factor
        )
    }
}

/// Días por año de la convención (cripto 24/7).
const DIAS_POR_ANIO: f64 = 365.25;
/// Muestra mínima para estimar la cola del 5 %.
pub const MIN_TRADES_CVAR: usize = 20;

/// Calcula el panel ex-post.
///
/// - `pnls`: PnL NETO por trade cerrado (después de fees), en unidades de
///   cuenta.
/// - `notional_total`: nocional bruto operado acumulado (entrada+salida a
///   criterio del llamador; documentar cuál en el consumidor).
/// - `capital_inicial`, `capital_final`: para CAGR y MaxDD.
/// - `max_dd`: máxima caída fraccional ya medida sobre la curva de equity
///   (el llamador la tiene de su bucle; no se re-deriva de PnL).
/// - `span_ms`: último − primer evento de la muestra.
pub fn ex_post_metrics(
    pnls: &[f64],
    notional_total: f64,
    capital_inicial: f64,
    capital_final: f64,
    max_dd: f64,
    span_ms: u64,
) -> ExPostMetrics {
    let mut m = ExPostMetrics {
        n_trades: pnls.len(),
        max_dd: if max_dd.is_finite() && max_dd > 0.0 { max_dd } else { 0.0 },
        ..ExPostMetrics::default()
    };
    let dias = span_ms as f64 / 86_400_000.0;
    m.span_days = dias;

    if pnls.is_empty() || !(capital_inicial > 0.0) {
        return m; // sin muestra no hay panel: ceros explícitos
    }

    // CAGR geométrico (piso −1 en ruina).
    if capital_final > 0.0 && dias > 0.0 {
        m.cagr = (capital_final / capital_inicial).powf(DIAS_POR_ANIO / dias) - 1.0;
    } else {
        m.cagr = -1.0;
    }

    let n = pnls.len() as f64;
    let media = pnls.iter().sum::<f64>() / n;
    let varianza = pnls.iter().map(|p| (p - media) * (p - media)).sum::<f64>() / n;
    let sigma = varianza.sqrt();
    // Frecuencia medida → factor de anualización por trade.
    let trades_por_anio = if dias > 0.0 { n / dias * DIAS_POR_ANIO } else { 0.0 };
    let raiz_anual = trades_por_anio.sqrt();

    m.sharpe_ann = if sigma > 1e-12 { media / sigma * raiz_anual } else { 0.0 };

    // Sortino: desviación a la baja contra objetivo 0 (sólo pérdidas).
    let perdidas = pnls.iter().filter(|p| **p < 0.0);
    let hay_perdidas = perdidas.clone().count() > 0;
    if hay_perdidas {
        let dd = (perdidas.map(|p| p * p).sum::<f64>() / n).sqrt(); // semidesviación
        m.sortino_ann = if dd > 1e-12 { media / dd * raiz_anual } else { 0.0 };
    } else {
        m.sortino_ann = if media > 0.0 { f64::INFINITY } else { 0.0 };
    }

    // Calmar.
    m.calmar = if m.max_dd > 1e-12 {
        m.cagr / m.max_dd
    } else if m.cagr > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    // CVaR 95 empírico: media del peor 5 % de trades (magnitud de pérdida).
    if pnls.len() >= MIN_TRADES_CVAR {
        let mut ordenados = pnls.to_vec();
        ordenados.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cola = (ordenados.len() as f64 * 0.05).ceil() as usize;
        let cola = cola.max(1);
        let media_cola = ordenados[..cola].iter().sum::<f64>() / cola as f64;
        m.cvar_95 = (-media_cola).max(0.0); // magnitud positiva de pérdida
    } else {
        m.cvar_95 = f64::NAN; // cola no estimable: contrato explícito
    }

    // Turnover diario.
    m.turnover_per_day = if dias > 0.0 {
        notional_total / capital_inicial / dias
    } else {
        0.0
    };

    // Win-rate y profit factor.
    let ganancias: f64 = pnls.iter().filter(|p| **p > 0.0).sum();
    let perdidas_abs: f64 = -pnls.iter().filter(|p| **p < 0.0).sum::<f64>();
    m.win_rate = pnls.iter().filter(|p| **p > 0.0).count() as f64 / n;
    m.profit_factor = if perdidas_abs > 1e-12 {
        ganancias / perdidas_abs
    } else if ganancias > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    m
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIA_MS: u64 = 86_400_000;

    /// Fixture determinista con valores calculables a mano: 10 trades,
    /// 6 ganan +1, 4 pierden −0.5, en 10 días, capital 100→104.
    #[test]
    fn panel_de_fixture_determinista_es_exacto() {
        let mut pnls = vec![1.0; 6];
        pnls.extend(vec![-0.5; 4]);
        let m = ex_post_metrics(&pnls, 2000.0, 100.0, 104.0, 0.02, 10 * DIA_MS);
        assert_eq!(m.n_trades, 10);
        assert!((m.span_days - 10.0).abs() < 1e-12);
        // media = (6·1 − 4·0.5)/10 = 0.4
        assert!((m.win_rate - 0.6).abs() < 1e-12);
        // PF = 6/2 = 3
        assert!((m.profit_factor - 3.0).abs() < 1e-9);
        // CAGR = (1.04)^(36.525) − 1 (calculado por la propia fórmula)
        let cagr_manual = 1.04_f64.powf(365.25 / 10.0) - 1.0;
        assert!((m.cagr - cagr_manual).abs() < 1e-9 * cagr_manual.abs());
        // Calmar = CAGR/0.02
        assert!((m.calmar - cagr_manual / 0.02).abs() < 1e-6);
        // Turnover = 2000/100/10 = 2/día
        assert!((m.turnover_per_day - 2.0).abs() < 1e-12);
        // CVaR: n=10 < 20 ⇒ NaN (cola no estimable)
        assert!(m.cvar_95.is_nan());
        // Sharpe: media 0.4, σ = sqrt(E[p²]−0.16) = sqrt(0.7−0.16) = sqrt(0.54)
        let sigma = 0.54_f64.sqrt();
        let esperado = 0.4 / sigma * (10.0_f64 / 10.0 * 365.25).sqrt();
        assert!((m.sharpe_ann - esperado).abs() < 1e-9);
        // Sortino: semidesviación = sqrt(4·0.25/10) = sqrt(0.1)
        let dd = 0.1_f64.sqrt();
        assert!((m.sortino_ann - 0.4 / dd * (10.0_f64 / 10.0 * 365.25).sqrt()).abs() < 1e-9);
    }

    /// Contornos honestos: sin muestra no se fabrica panel; sin pérdidas
    /// el Sortino/PF son infinito EXPLÍCITO; ruina pisa el CAGR a −1.
    #[test]
    fn contornos_sin_muestra_sin_perdidas_y_ruina() {
        let vacio = ex_post_metrics(&[], 0.0, 100.0, 100.0, 0.0, DIA_MS);
        assert_eq!(vacio.n_trades, 0);
        assert_eq!(vacio.cagr, 0.0);

        let sin_perdidas = ex_post_metrics(&[1.0; 30], 0.0, 100.0, 130.0, 0.0, 30 * DIA_MS);
        assert!(sin_perdidas.sortino_ann.is_infinite());
        assert!(sin_perdidas.profit_factor.is_infinite());
        assert!(sin_perdidas.calmar.is_infinite(), "MaxDD 0 con CAGR>0");

        let ruina = ex_post_metrics(&[-10.0; 30], 0.0, 100.0, 0.0, 1.0, 30 * DIA_MS);
        assert_eq!(ruina.cagr, -1.0);
    }

    /// CVaR empírico con muestra suficiente: la cola del 5 % de 40 trades
    /// son los 2 peores; la media de éstos es la métrica.
    #[test]
    fn cvar_empirico_es_la_media_de_la_cola_del_5pct() {
        let mut pnls = vec![1.0; 38];
        pnls.extend(vec![-8.0, -12.0]); // los 2 peores
        let m = ex_post_metrics(&pnls, 0.0, 100.0, 118.0, 0.05, 40 * DIA_MS);
        assert_eq!(m.n_trades, 40);
        assert!((m.cvar_95 - 10.0).abs() < 1e-12, "cola=(−12+−8)/2=−10 ⇒ magnitud 10");
        // Todos-ganadoras con n≥20: cola son las propias ganancias ⇒ pérdida
        // media negativa ⇒ magnitud clamp a 0 (no hay cola de pérdida).
        let m2 = ex_post_metrics(&[1.0; 25], 0.0, 100.0, 125.0, 0.0, 25 * DIA_MS);
        assert_eq!(m2.cvar_95, 0.0);
    }

    /// El panel de una línea contiene los campos que la doctrina exige.
    #[test]
    fn linea_de_panel_nombra_las_metricas() {
        let m = ex_post_metrics(&[1.0, -0.5, 1.0, -0.5], 100.0, 10.0, 11.0, 0.01, 4 * DIA_MS);
        let line = m.panel_line();
        for campo in ["CAGR=", "Sharpe=", "Sortino=", "Calmar=", "MaxDD=", "CVaR95=", "turnover/d=", "WR=", "PF="] {
            assert!(line.contains(campo), "falta {campo} en {line}");
        }
    }
}
