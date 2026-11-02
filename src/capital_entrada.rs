//! Compuerta de entrada de capital (§25 del operador, línea CL-15).
//!
//! GLM-178 introdujo la validación fail-closed inline en
//! `continuous_evolution_backtest`; GLM-179 la unifica aquí (mandato
//! anti-«esquizofrenia estructural») para que TODOS los bins del paquete
//! raíz compartan UNA implementación: sin ella cada bin re-implementaba la
//! puerta — y los que no la tenían caían al 13.0 literal prohibido.
//!
//! Doctrina: el capital inicial de una corrida NO tiene default. Ausente o
//! inválido ⇒ `exit(2)` con diagnóstico explícito ANTES de cualquier
//! efecto secundario (ni truncar CSVs ajenos ni crear config_dir). El
//! capital real vive en la wallet y entra por INITIAL_CAPITAL.
//!
//! Para librerías y daemons (que no deben matar el proceso) existe
//! [`validar`], que devuelve `Option` y deja la decisión al llamador.

/// Valida un texto de capital (CLI o env) sin tocar el proceso:
/// `Some(c)` sólo si parsea a un número finito y estrictamente positivo.
pub fn validar(texto: &str) -> Option<f64> {
    let valor: f64 = texto.trim().parse().ok()?;
    if valor.is_finite() && valor > 0.0 {
        Some(valor)
    } else {
        None
    }
}

/// Exige INITIAL_CAPITAL válido (finito > 0). Ausente o ilegible ⇒
/// diagnóstico a stderr + `exit(2)`. Es una puerta de BINARIO: llamar como
/// primera sentencia de `main`, antes de crear archivos o directorios.
pub fn exigir_inicial() -> f64 {
    match std::env::var("INITIAL_CAPITAL") {
        Ok(crudo) => match validar(&crudo) {
            Some(c) => c,
            None => {
                eprintln!(
                    "ERROR: INITIAL_CAPITAL ilegible ({crudo:?}): se exige un número finito > 0. \
                     Sin capital válido no hay simulación (CL-15, paridad §25)."
                );
                std::process::exit(2);
            }
        },
        Err(_) => {
            eprintln!(
                "ERROR: INITIAL_CAPITAL ausente: exporta el balance real de la wallet \
                 (paridad §25, CL-15). Sin defaults silenciosos."
            );
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::validar;

    #[test]
    fn valida_solo_finitos_positivos() {
        assert_eq!(validar("37.0"), Some(37.0));
        assert_eq!(validar("  42 "), Some(42.0));
        assert_eq!(validar("1e3"), Some(1000.0));
        for ilegal in ["", "abc", "NaN", "inf", "-inf", "0", "-1", "1e400"] {
            assert_eq!(validar(ilegal), None, "debe rechazar {ilegal:?}");
        }
    }
}
