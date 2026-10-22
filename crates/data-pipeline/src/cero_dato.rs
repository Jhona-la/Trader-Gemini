//! F1-091 — Un cero que no es dato.
//!
//! # La regla
//!
//! En el camino de dinero (saldo, precio, cantidad, vela, estadística de
//! símbolo) **un valor que no se puede leer es DESCONOCIDO, nunca cero.**
//!
//! El cero es un dato con significado propio: «no hay saldo», «el libro está
//! vacío», «esta vela no negoció». Inventarlo cuando la lectura falló convierte
//! una ausencia de información en una afirmación falsa, y el sistema la usa
//! para dimensionar, rankear o entrenar. Con `panic = "abort"` y motores que
//! dividen por el capital, un `0.0` fabricado no es un error de redondeo: es
//! una posición dimensionada con dinero que no existe.
//!
//! # Por qué `parse::<f64>()` no basta
//!
//! `"NaN".parse::<f64>()` **tiene éxito** y devuelve `NaN`; igual `"inf"` y
//! `"-inf"`. Un `unwrap_or(0.0)` sobre eso entrega un cero silencioso, y un
//! `NaN` que se cuela en un tensor se propaga. Estas funciones cierran las dos
//! puertas: rechazan lo no finito y exigien el signo del dominio.
//!
//! # Dominios
//!
//! | Función | Dominio | Uso |
//! |---|---|---|
//! | [`monto`] | finito y `>= 0` | saldos, volúmenes |
//! | [`cantidad`] | finito y `> 0` | precios y cantidades del libro |
//! | [`monto_de_json`] | finito y `>= 0` | campos numéricos de un JSON |
//! | [`estadistica_legible`] | volumen `> 0`, variación `>= 0`, ambos finitos | rankers de símbolos |
//!
//! Ninguna función tiene un valor por defecto: quien llama decide si descarta
//! la muestra o falla en cerrojo. Eso es deliberado — un default silencioso es
//! exactamente lo que este módulo existe para eliminar.

use serde_json::Value;

/// Parsea un monto (saldo, volumen): finito y no negativo.
///
/// `Ok`-free a propósito: devuelve `None` cuando el texto está vacío, no es
/// un número, es `NaN`/`inf`, o es negativo. Un saldo negativo en futuros
/// puede ser real (deuda), pero un *parseo* que lo produce no es fiable: se
/// descarta la muestra en vez de admitirla.
#[inline]
pub fn monto(texto: &str) -> Option<f64> {
    let v: f64 = texto.trim().parse().ok()?;
    if v.is_finite() && v >= 0.0 {
        Some(v)
    } else {
        None
    }
}

/// Parsea una cantidad de libro (precio, tamaño): finita y estrictamente positiva.
///
/// Un precio o un tamaño de nivel en `0` no es un nivel vivo: es un libro
/// vacío o una muestra corrupta.
#[inline]
pub fn cantidad(texto: &str) -> Option<f64> {
    let v: f64 = texto.trim().parse().ok()?;
    if v.is_finite() && v > 0.0 {
        Some(v)
    } else {
        None
    }
}

/// Igual que [`monto`], sobre un campo de `serde_json::Value`.
///
/// Binance manda los números como **strings** (`"balance":"13.5"`), así que se
/// acepta tanto el string como el número JSON nativo.
pub fn monto_de_json(v: &Value) -> Option<f64> {
    match v {
        Value::String(s) => monto(s),
        Value::Number(n) => {
            let f = n.as_f64()?;
            if f.is_finite() && f >= 0.0 {
                Some(f)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Estadística de símbolo legible: volumen de cotización `> 0` y variación
/// diaria finita (se usa su valor absoluto), ambos finitos.
///
/// Es la fuente ÚNICA del filtro de los rankers (`asset_selector` y
/// `dynamic_ranker` tenían cada uno el suyo con `unwrap_or(0.0)`). Un símbolo
/// cuya estadística no se puede leer se EXCLUYE del universo operable: no se
/// le inventa un volumen para que compita, ni se le adjudica una volatilidad
/// que no tiene.
///
/// La variación admite signo negativo (una caída del 3,5 % es un dato perfectamente
/// legible) y se consume en magnitud, que es lo que la banda de volatilidad mide.
pub fn estadistica_legible(volumen: &str, variacion: &str) -> Option<(f64, f64)> {
    let vol = monto(volumen).filter(|v| *v > 0.0)?;
    let pct_con_signo: f64 = variacion.trim().parse().ok()?;
    if !pct_con_signo.is_finite() {
        return None;
    }
    Some((vol, pct_con_signo.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monto_acepta_lo_legible_y_rechaza_lo_que_no() {
        assert_eq!(monto("13.5"), Some(13.5));
        assert_eq!(monto("0"), Some(0.0));
        assert_eq!(monto("  2.0e3 "), Some(2000.0));
        // Lo que un `unwrap_or(0.0)` habría convertido en cero:
        assert_eq!(monto(""), None);
        assert_eq!(monto("no-un-numero"), None);
        assert_eq!(monto("null"), None);
        assert_eq!(monto("NaN"), None, "`parse` ACEPTA \"NaN\" y devuelve NaN");
        assert_eq!(monto("inf"), None);
        assert_eq!(monto("-inf"), None);
        assert_eq!(monto("infinity"), None);
        assert_eq!(
            monto("-5"),
            None,
            "un saldo negativo no es un monto legible"
        );
    }

    #[test]
    fn cantidad_exige_estrictamente_positivo() {
        assert_eq!(cantidad("82100.50"), Some(82100.50));
        assert_eq!(cantidad("0"), None, "un nivel en 0 no es un nivel vivo");
        assert_eq!(cantidad("-1.0"), None);
        assert_eq!(cantidad("NaN"), None);
        assert_eq!(cantidad(""), None);
    }

    #[test]
    fn monto_de_json_acepta_string_y_numero() {
        assert_eq!(
            monto_de_json(&serde_json::json!("13.5")),
            Some(13.5),
            "Binance manda el saldo como string"
        );
        assert_eq!(monto_de_json(&serde_json::json!(13.5)), Some(13.5));
        assert_eq!(monto_de_json(&serde_json::json!("NaN")), None);
        assert_eq!(monto_de_json(&serde_json::json!(null)), None);
        assert_eq!(monto_de_json(&serde_json::json!(-1.0)), None);
    }

    #[test]
    fn estadistica_legible_es_la_fuente_unica_de_los_rankers() {
        assert_eq!(
            estadistica_legible("1000000.0", "0.1"),
            Some((1_000_000.0, 0.1))
        );
        assert_eq!(
            estadistica_legible("1000000.0", "-3.5"),
            Some((1_000_000.0, 3.5)),
            "la variación se usa en valor absoluto"
        );
        assert_eq!(
            estadistica_legible("0.0", "0.1"),
            None,
            "volumen 0 no es estadística"
        );
        assert_eq!(estadistica_legible("NaN", "0.1"), None);
        assert_eq!(estadistica_legible("1000000.0", "NaN"), None);
        assert_eq!(estadistica_legible("", "0.1"), None);
    }
}
