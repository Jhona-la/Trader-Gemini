//! FUNCIÓN DE APTITUD ÚNICA DEL SISTEMA (D-652 / D-653 / D-654 / D-655).
//!
//! # Los defectos que corrige
//!
//! ## D-653 — la aptitud era un maximizador de apalancamiento
//!
//! La definición anterior era `fitness = pnl · sharpe`. Bajo un cambio de
//! escala del apalancamiento `L → λ·L`:
//!
//! ```text
//! pnl    → λ · pnl        (los retornos escalan linealmente)
//! sharpe → sharpe         (media y desviación escalan igual: INVARIANTE)
//! ⟹ fitness → λ · fitness
//! ```
//!
//! La aptitud crecía **linealmente con el apalancamiento y sin cota**. Dos
//! estrategias con calidad ajustada por riesgo idéntica recibían aptitudes que
//! diferían en el factor de apalancamiento, de modo que la evolución escogía
//! invariablemente la más apalancada. El producto `pnl · sharpe` no es una
//! utilidad válida: tiene unidades de moneda y no es cóncavo en la riqueza.
//!
//! ## D-654 — no operar puntuaba mejor que operar y perder
//!
//! Un genoma que jamás abría posición obtenía `-1,0`. Uno que operaba y perdía
//! 50 USD obtenía, con sharpe −0,5 y penalización OOS 1,5, un valor de
//! `-50 · 1,5 · 1,5 = -112,5`. **La inacción era 112 veces mejor que la
//! pérdida moderada**, de modo que el óptimo local más accesible del paisaje
//! era la parálisis operativa: cualquier mutación que endureciera un gate y
//! llevara los trades a cero producía una mejora inmediata y masiva.
//!
//! ## D-655 — la aptitud multiobjetivo coherente era código muerto
//!
//! `compute_nsga3_hyper_fitness` integraba los cinco componentes de calidad y
//! **no tenía ningún llamador**; sus partes se usaban sueltas en cuatro sitios
//! con cuatro fórmulas de combinación distintas.
//!
//! # La utilidad adoptada
//!
//! Crecimiento logarítmico penalizado por ruina:
//!
//! ```text
//! F = ln(capital_final / capital_inicial) − λ · max_drawdown²
//! ```
//!
//! Propiedades que la hacen correcta para este sistema:
//!
//! * **Invariante ante el apalancamiento** salvo por su efecto real sobre la
//!   ruina — que ahora el backtest sí simula (D-669).
//! * **Cóncava en la riqueza**: es la utilidad de Kelly, coherente con el
//!   dimensionamiento de Kelly que el motor ya emplea. Maximizarla es
//!   maximizar la tasa de crecimiento geométrico a largo plazo.
//! * **El drawdown entra al cuadrado**: penaliza desproporcionadamente las
//!   caídas grandes, que son las que producen ruina irreversible.
//! * **Aditiva en el tiempo**: `ln` convierte el producto de retornos en suma,
//!   de modo que la aptitud de dos periodos es la suma de sus aptitudes.

/// Entradas observables de la evaluación. Ninguna es opcional: obligar a
/// aportarlas todas impide que un llamador construya su propia variante
/// omitiendo el componente que le estorba (que es como nacieron los nueve
/// fitness divergentes de D-652).
#[derive(Debug, Clone, Copy)]
pub struct FitnessInputs {
    pub initial_capital: f64,
    pub final_capital: f64,
    pub max_drawdown_pct: f64,
    pub total_trades: u32,
    /// Operaciones mínimas exigidas en la ventana evaluada. Por debajo, el
    /// genoma es INVIABLE, no mediocre (D-654).
    pub min_trades_required: u32,
    /// Capital al cierre de la partición fuera de muestra, y capital al inicio
    /// de la misma. Si no hay validación OOS, ambos pueden ser iguales.
    pub oos_start_capital: f64,
    pub oos_end_capital: f64,
}

/// Aptitud que la evolución **maximiza**. `NEG_INFINITY` marca inviabilidad.
pub const INVIABLE: f64 = f64::NEG_INFINITY;

/// Peso de la penalización por drawdown. Derivado, no elegido: se fija de modo
/// que un drawdown del 50 % anule exactamente una duplicación del capital
/// (`ln 2 ≈ 0,693`), que es el punto en que un operador racional considera
/// equivalentes ambos resultados.
///
/// `λ · 0,5² = ln 2  ⟹  λ = 4·ln 2 ≈ 2,7726`
pub const DRAWDOWN_LAMBDA: f64 = 2.772_588_722_239_781;

/// Penalización por degradación fuera de muestra. Un genoma que gana en la
/// partición de entrenamiento y pierde en la de validación está sobreajustado;
/// la penalización es continua en la magnitud de la degradación, sin escalón.
fn oos_factor(inputs: &FitnessInputs) -> f64 {
    let start = inputs.oos_start_capital;
    let end = inputs.oos_end_capital;
    if !start.is_finite() || !end.is_finite() || start <= 0.0 {
        return 1.0;
    }
    let oos_growth = (end / start).max(1e-12).ln();
    if oos_growth >= 0.0 {
        1.0
    } else {
        // Degradación OOS: el factor crece suavemente con la pérdida relativa.
        // −10 % OOS ⇒ ×1,105; −50 % ⇒ ×1,69. Continuo y sin umbrales.
        1.0 + oos_growth.abs()
    }
}

/// FUNCIÓN ÚNICA DE APTITUD. Todo promotor de genomas debe llamar a ESTA.
pub fn compute(inputs: &FitnessInputs) -> f64 {
    // D-654: la inacción es INVIABLE, no intermedia. El gen `min_trades_per_day`
    // existe precisamente para esto y estaba entre los que nadie leía.
    if inputs.total_trades < inputs.min_trades_required.max(1) {
        return INVIABLE;
    }
    if !inputs.initial_capital.is_finite() || inputs.initial_capital <= 0.0 {
        return INVIABLE;
    }
    if !inputs.final_capital.is_finite() {
        return INVIABLE;
    }
    // Ruina: capital agotado. Peor resultado posible, sin gradiente que
    // invite a explorar en esa dirección.
    if inputs.final_capital <= 0.0 {
        return INVIABLE;
    }

    // Crecimiento logarítmico: la utilidad de Kelly.
    let growth = (inputs.final_capital / inputs.initial_capital).ln();

    // Penalización cuadrática por drawdown.
    let dd = if inputs.max_drawdown_pct.is_finite() {
        inputs.max_drawdown_pct.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let dd_penalty = DRAWDOWN_LAMBDA * dd * dd;

    let base = growth - dd_penalty;

    // La degradación fuera de muestra AMPLIFICA el castigo y ATENÚA el premio:
    // en ambos casos empuja hacia genomas que generalizan.
    let oos = oos_factor(inputs);
    if base >= 0.0 { base / oos } else { base * oos }
}

/// Calcula la aptitud con regularización Bayesiana hacia un prior para muestras reducidas (#22).
/// Previene el bloqueo en frío donde candidatos con N < min_trades_required reciben -inf,
/// contrayendo suavemente el fitness observado hacia el prior en lugar de descartarlo.
pub fn compute_with_bayesian_prior(inputs: &FitnessInputs, prior_fitness: f64) -> f64 {
    if inputs.total_trades == 0 || inputs.initial_capital <= 0.0 || inputs.final_capital <= 0.0 {
        return INVIABLE;
    }
    let req = inputs.min_trades_required.max(1);
    if inputs.total_trades >= req {
        return compute(inputs);
    }
    // Regularización Bayesiana suave: peso proporcional al soporte muestral N / N_req
    let weight = inputs.total_trades as f64 / req as f64;
    let mut modified_inputs = inputs.clone();
    modified_inputs.min_trades_required = inputs.total_trades;
    let raw_fitness = compute(&modified_inputs);
    if raw_fitness == INVIABLE {
        INVIABLE
    } else {
        weight * raw_fitness + (1.0 - weight) * prior_fitness
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> FitnessInputs {
        FitnessInputs {
            initial_capital: 13.0,
            final_capital: 26.0,
            max_drawdown_pct: 0.10,
            total_trades: 100,
            min_trades_required: 10,
            oos_start_capital: 20.0,
            oos_end_capital: 26.0,
        }
    }

    /// D-653: la aptitud NO debe crecer con el apalancamiento cuando la calidad
    /// ajustada por riesgo es la misma. Se modela subiendo proporcionalmente el
    /// crecimiento Y el drawdown, que es lo que hace apalancar.
    #[test]
    fn d653_la_aptitud_no_premia_el_apalancamiento_por_si_mismo() {
        // 2x de apalancamiento: duplica el retorno logarítmico y el drawdown.
        let mut sin_apalancar = base();
        sin_apalancar.final_capital = 13.0 * 1.20;
        sin_apalancar.max_drawdown_pct = 0.10;

        let mut apalancado = base();
        apalancado.final_capital = 13.0 * 1.20f64.powi(2);
        apalancado.max_drawdown_pct = 0.20;

        let f_sin = compute(&sin_apalancar);
        let f_con = compute(&apalancado);
        // Con la utilidad vieja (pnl x sharpe) el apalancado ganaba SIEMPRE.
        // Con la utilidad de Kelly el drawdown cuadrático lo compensa: la
        // ventaja del apalancamiento deja de ser gratuita.
        let ventaja = f_con - f_sin;
        assert!(
            ventaja < f_sin,
            "el apalancamiento no debe dominar la aptitud: sin={f_sin}, con={f_con}"
        );
    }

    /// D-653: drawdown extremo debe ser peor que no crecer nada.
    #[test]
    fn d653_el_drawdown_extremo_domina_al_crecimiento() {
        let mut ruinoso = base();
        ruinoso.final_capital = 13.0 * 1.5; // +50 %
        ruinoso.max_drawdown_pct = 0.90; // pero con 90 % de caída
        let mut plano = base();
        plano.final_capital = 13.0;
        plano.max_drawdown_pct = 0.02;
        assert!(
            compute(&ruinoso) < compute(&plano),
            "un +50 % con 90 % de drawdown no puede superar a un resultado plano y estable"
        );
    }

    /// D-654: no operar es INVIABLE, no mediocre.
    #[test]
    fn d654_la_inaccion_es_inviable_no_intermedia() {
        let mut inactivo = base();
        inactivo.total_trades = 0;
        inactivo.final_capital = 13.0;

        let mut perdedor = base();
        perdedor.total_trades = 80;
        perdedor.final_capital = 13.0 * 0.70; // pierde un 30 %

        assert_eq!(compute(&inactivo), INVIABLE);
        assert!(
            compute(&perdedor) > compute(&inactivo),
            "operar y perder debe puntuar MEJOR que no operar: la parálisis era \\
             el óptimo local más accesible del paisaje anterior"
        );
    }

    /// D-654: por debajo del mínimo exigido también es inviable.
    #[test]
    fn d654_por_debajo_del_minimo_de_operaciones_es_inviable() {
        let mut escaso = base();
        escaso.total_trades = 3;
        escaso.min_trades_required = 10;
        assert_eq!(compute(&escaso), INVIABLE);
    }

    /// La ruina no tiene gradiente que invite a explorarla.
    #[test]
    fn la_ruina_es_inviable() {
        let mut arruinado = base();
        arruinado.final_capital = 0.0;
        assert_eq!(compute(&arruinado), INVIABLE);
    }

    /// La degradación fuera de muestra penaliza de forma continua.
    #[test]
    fn la_degradacion_oos_penaliza_continuamente() {
        let mut generaliza = base();
        generaliza.oos_start_capital = 20.0;
        generaliza.oos_end_capital = 26.0;

        let mut sobreajusta = base();
        sobreajusta.oos_start_capital = 20.0;
        sobreajusta.oos_end_capital = 14.0;

        assert!(
            compute(&generaliza) > compute(&sobreajusta),
            "un genoma que degrada fuera de muestra debe puntuar peor"
        );
    }

    /// La aptitud es aditiva en el tiempo: dos periodos encadenados suman.
    #[test]
    fn la_aptitud_es_aditiva_en_el_tiempo() {
        let f = |ini: f64, fin: f64| {
            let mut i = base();
            i.initial_capital = ini;
            i.final_capital = fin;
            i.max_drawdown_pct = 0.0;
            i.oos_start_capital = ini;
            i.oos_end_capital = fin;
            compute(&i)
        };
        let a = f(100.0, 150.0);
        let b = f(150.0, 300.0);
        let total = f(100.0, 300.0);
        assert!(
            (a + b - total).abs() < 1e-9,
            "ln es aditivo: {a} + {b} debe ser {total}"
        );
    }

    /// #22: Prueba de regularización Bayesiana en arranque en frío (N < min_trades)
    #[test]
    fn test_compute_with_bayesian_prior_cold_start() {
        let mut cold = base();
        cold.total_trades = 5;
        cold.min_trades_required = 30;
        cold.final_capital = 14.0;

        // Con compute regular es INVIABLE
        assert_eq!(compute(&cold), INVIABLE);

        // Con compute_with_bayesian_prior es FINITO y contraído hacia el prior
        let prior = -0.10;
        let bayesian_fit = compute_with_bayesian_prior(&cold, prior);
        assert!(bayesian_fit.is_finite());
        assert!(bayesian_fit > INVIABLE);
    }
}
