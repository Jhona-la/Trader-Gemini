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
//! - **Sharpe/Sortino**: proxies de PnL MONETARIO por trade, no ratios de
//!   retornos de cartera. La escala √(trades/año) presupone incrementos
//!   comparables, estacionarios y sin correlación; no corrige dependencia,
//!   solapamiento ni duración variable. Benchmark/objetivo 0 por política,
//!   no porque operar 24/7 implique una tasa libre de riesgo nula.
//! - **Sortino**: semidesviación contra 0 con denominador n TOTAL, no sólo
//!   el número de pérdidas. Sin pérdidas y media positiva ⇒ +∞; 0/0 ⇒ NaN.
//! - **CVaR 95**: parte positiva del Expected Shortfall empírico de pérdidas
//!   monetarias por trade, con masa fraccional en el cuantil. El piso en 0
//!   es una convención de presentación, no el ES firmado. Exige ≥20 trades
//!   por política de cobertura mínima; NO garantiza precisión estadística.
//!   Con menos se reporta NaN, aunque el funcional empírico sea definible.
//! - **Turnover**: nocional operado / capital / día — presión de capacity
//!   y costos, no un mérito.
//! - **Calmar**: CAGR/MaxDD; MaxDD 0 con CAGR > 0 ⇒ INFINITY, no evidencia
//!   de perfección. CAGR = MaxDD = 0 ⇒ NaN (denominador no identificado).
//! - **Ausencia/invalidez**: NaN, nunca un cero de rendimiento inventado.
//!   Cada métrica valida sus dependencias: un capital inválido no elimina
//!   el win-rate conocido; un PnL no finito invalida TODA la muestra de
//!   trades sin descartar filas, pero no altera el crecimiento de extremos.
//!   Duración cero invalida anualización y turnover, no prueba ruina.
//! - No se usan epsilons monetarios: ratios adimensionales deben conservarse
//!   al cambiar la unidad de cuenta. Escalado previo evita cuadrados/sumas
//!   intermedios fuera de rango; resultados verdaderamente no representables
//!   aún pueden ser infinitos o subdesbordar en f64.

/// Panel de métricas ex-post. Todos los campos son del cálculo puro sobre
/// la muestra dada; ningún campo se "rellena" sin evidencia.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExPostMetrics {
    pub n_trades: usize,
    pub span_days: f64,
    /// Retorno geométrico anualizado: (fin/inicio)^(365,25/d) − 1.
    /// Capital final finito ≤ 0 ⇒ −1 con capital inicial/tiempo válidos.
    /// Presupone ausencia de aportes/retiros; no ajusta flujos externos.
    pub cagr: f64,
    /// Proxy de PnL por trade: media/σ poblacional · √(trades/año).
    /// σ = 0 ⇒ NaN. No es Sharpe de retornos de cartera.
    pub sharpe_ann: f64,
    /// Sortino por trade anualizado: media/σ_a la baja · √(trades/año).
    pub sortino_ann: f64,
    /// CAGR/|MaxDD| — eficiencia del crecimiento por unidad de caída.
    pub calmar: f64,
    /// Máxima caída fraccional del capital (0 = sin caídas).
    pub max_dd: f64,
    /// CVaR 95 empírico por trade (magnitud de pérdida, ≥0). NaN si
    /// n_trades < 20 por política, o muestra con algún PnL no finito.
    pub cvar_95: f64,
    /// Nocional operado / capital / día.
    pub turnover_per_day: f64,
    /// Trades ganadores / total (PnL > 0).
    pub win_rate: f64,
    /// Σ ganancias / |Σ pérdidas|; ganancias sin pérdidas ⇒ INFINITY;
    /// sin ganancias ni pérdidas ⇒ NaN.
    pub profit_factor: f64,
}

impl Default for ExPostMetrics {
    fn default() -> Self {
        Self {
            n_trades: 0,
            span_days: 0.0,
            cagr: f64::NAN,
            sharpe_ann: f64::NAN,
            sortino_ann: f64::NAN,
            calmar: f64::NAN,
            max_dd: f64::NAN,
            cvar_95: f64::NAN,
            turnover_per_day: f64::NAN,
            win_rate: f64::NAN,
            profit_factor: f64::NAN,
        }
    }
}

impl ExPostMetrics {
    /// Panel de una línea para telemetría/informes.
    pub fn panel_line(&self) -> String {
        format!(
            "n={} d={:.1} CAGR={:.1}% Sharpe={:.2} Sortino={:.2} Calmar={:.2} MaxDD={:.1}% CVaR95={:.4} turnover/d={:.1} WR={:.1}% PF={:.2} basis=cash_pnl_per_trade annualization=iid_proxy",
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
/// Política mínima: masa de al menos una observación en la cola del 5 %.
/// No es una garantía de precisión ni un límite matemático del estimador.
pub const MIN_TRADES_CVAR: usize = 20;

/// Calcula el panel ex-post.
///
/// - `pnls`: PnL NETO por trade cerrado (después de fees), en unidades de
///   cuenta.
/// - `notional_total`: nocional bruto operado acumulado (entrada+salida a
///   criterio del llamador; documentar cuál en el consumidor).
/// - `capital_inicial`, `capital_final`: extremos para CAGR, sin flujos
///   externos. Son evidencia independiente de la lista de trades.
/// - `max_dd`: máxima caída fraccional ya medida sobre la curva de equity
///   (el llamador la tiene de su bucle; no se re-deriva de PnL).
/// - `span_ms`: último − primer evento del período efectivamente observado.
///   Cero no permite anualizar. El caller debe validar orden y cobertura.
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
        max_dd: if max_dd.is_finite() && max_dd >= 0.0 {
            max_dd.abs() // canonical +0: -0 must not reverse Calmar's sign
        } else {
            f64::NAN
        },
        ..ExPostMetrics::default()
    };
    let dias = span_ms as f64 / 86_400_000.0;
    m.span_days = dias;

    // Capital/clock domains are independent of the trade sample domain.
    if capital_inicial.is_finite() && capital_inicial > 0.0 && dias > 0.0 {
        if capital_final.is_finite() {
            m.cagr = if capital_final <= 0.0 {
                -1.0 // preserved convention: finite insolvency floors growth
            } else {
                // ln_1p preserves small relative changes; log difference avoids
                // overflow/underflow of final/initial for extreme finite ratios.
                let relative = (capital_final - capital_inicial) / capital_inicial;
                let log_growth = if relative.abs() <= 0.5 {
                    relative.ln_1p()
                } else {
                    capital_final.ln() - capital_inicial.ln()
                };
                (log_growth * (DIAS_POR_ANIO / dias)).exp_m1()
            };
        }
        if notional_total.is_finite() && notional_total >= 0.0 {
            let direct = notional_total / capital_inicial / dias;
            m.turnover_per_day = if notional_total > 0.0 && (!direct.is_finite() || direct == 0.0) {
                (notional_total.ln() - capital_inicial.ln() - dias.ln()).exp()
            } else {
                direct
            };
        }
    }
    // IEEE division represents missing input and 0/0 as NaN, and positive/0
    // as infinity. Do not turn a tiny but measured drawdown into zero risk.
    m.calmar = m.cagr / m.max_dd;

    // Reject the whole contaminated sample, not individual inconvenient rows.
    if pnls.is_empty() || pnls.iter().any(|p| !p.is_finite()) {
        return m;
    }
    let n = pnls.len() as f64;
    m.win_rate = pnls.iter().filter(|p| **p > 0.0).count() as f64 / n;
    let scale = pnls.iter().map(|p| p.abs()).fold(0.0_f64, f64::max);
    if scale > 0.0 {
        let media = compensated_sum(pnls.iter().map(|p| p / scale)) / n;
        let sigma = (compensated_sum(pnls.iter().map(|p| (p / scale - media).powi(2))) / n).sqrt();
        if dias > 0.0 {
            let raiz_anual = (n / dias * DIAS_POR_ANIO).sqrt();
            if sigma > 0.0 {
                m.sharpe_ann = media / sigma * raiz_anual;
            }
            // hypot avoids squaring away a small downside while the resulting
            // ratio is still representable (e.g. PnLs [1, -1e-200]).
            let downside = pnls
                .iter()
                .filter(|p| **p < 0.0)
                .fold(0.0_f64, |norm, p| norm.hypot(p / scale))
                / n.sqrt();
            m.sortino_ann = media / downside * raiz_anual;
        }
        let ganancias = compensated_sum(pnls.iter().filter(|p| **p > 0.0).map(|p| p / scale));
        let perdidas = compensated_sum(pnls.iter().filter(|p| **p < 0.0).map(|p| -p / scale));
        m.profit_factor = ganancias / perdidas;
    }

    if pnls.len() >= MIN_TRADES_CVAR {
        let mut ordenados = pnls.to_vec();
        // All observations validated above.
        ordenados.sort_by(f64::total_cmp);
        // Integrate the empirical quantile over exactly 5% probability mass.
        // Integer division/remainder avoid rounding ceil(.05*n) at boundaries.
        let completos = pnls.len() / 20;
        let fraccion = (pnls.len() % 20) as f64 / 20.0;
        let usados = completos + usize::from(fraccion > 0.0);
        let escala_cola = ordenados[..usados]
            .iter()
            .map(|p| p.abs())
            .fold(0.0_f64, f64::max);
        m.cvar_95 = if escala_cola == 0.0 {
            0.0
        } else {
            let suma = compensated_sum(
                ordenados[..completos]
                    .iter()
                    .map(|p| p / escala_cola)
                    .chain(
                        (fraccion > 0.0).then(|| fraccion * (ordenados[completos] / escala_cola)),
                    ),
            );
            (-(suma / (completos as f64 + fraccion)) * escala_cola).max(0.0)
        };
    }

    m
}

/// Neumaier summation on bounded, normalized inputs; reduces cancellation
/// without allowing sums of raw monetary values to overflow first.
fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0_f64;
    let mut correction = 0.0_f64;
    for value in values {
        let next = sum + value;
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    sum + correction
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
        assert!(
            (m.cvar_95 - 10.0).abs() < 1e-12,
            "cola=(−12+−8)/2=−10 ⇒ magnitud 10"
        );
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
        for campo in [
            "CAGR=",
            "Sharpe=",
            "Sortino=",
            "Calmar=",
            "MaxDD=",
            "CVaR95=",
            "turnover/d=",
            "WR=",
            "PF=",
        ] {
            assert!(line.contains(campo), "falta {campo} en {line}");
        }
    }
}
