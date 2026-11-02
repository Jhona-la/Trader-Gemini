//! Compuerta de entrada de capital (§25 del operador, línea CL-15).
//!
//! Historia: GLM-178 introdujo la puerta fail-closed inline en el
//! backtester (contrato de Kimi V33); GLM-179 la unificó en el lib raíz y
//! descubrió el bin dual-compilado (raíz + backtest-engine); GLM-180 la
//! muda AQUÍ (quantum-arena, hoja visible para todos) y le añade la
//! cadena wallet→engine→bin: sin red, sin mocks, el balance REAL que el
//! engine vivo extrajo de la wallet (fail-closed CL-15) y publicó en
//! `outputs/economia_live.json` (ReciboEconomico §52, GLM-144).
//!
//! Cadena de fuentes, en orden:
//! 1. `INITIAL_CAPITAL` (env) válido — lo explícito manda. Inválido ⇒
//!    error (no se sustituye en silencio un valor explícito roto).
//! 2. Recibo vivo del engine: `outputs/economia_live.json` con
//!    `capital_inicial` válido y FRESCO (mtime). Umbral de frescura:
//!    `TG_RECIBO_MAX_EDAD_SECS` (default 86400; `0` deshabilita la fuente
//!    — así aíslan los contratos). Un recibo rancio es un dato zombi y no
//!    se usa.
//! 3. Nada de lo anterior ⇒ `exit(2)` con diagnóstico. Sin defaults
//!    silenciosos: el 13.0 prohibido no vuelve por ninguna puerta.
//!
//! Para librerías y daemons (que no deben matar el proceso) existe
//! [`validar`], que devuelve `Option` y deja la decisión al llamador.

use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

/// Ruta del recibo económico vivo publicado por el worker §52 del engine.
pub const RUTA_RECIBO: &str = "outputs/economia_live.json";

/// Frescura por defecto del recibo (24 h): el engine reinicia a diario y
/// reescribe al arrancar (registrar_capital en el arranque).
const RECIBO_MAX_EDAD_DEFAULT_SECS: u64 = 86_400;

#[derive(serde::Deserialize)]
struct ReciboMinimo {
    capital_inicial: Option<f64>,
}

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

/// Exige capital inicial real. Puerta de BINARIO: llamar como primera
/// sentencia de `main`, antes de crear archivos o directorios.
pub fn exigir_inicial() -> f64 {
    if let Ok(crudo) = std::env::var("INITIAL_CAPITAL") {
        return match validar(&crudo) {
            Some(c) => c,
            None => {
                eprintln!(
                    "ERROR: INITIAL_CAPITAL ilegible ({crudo:?}): se exige un número finito > 0. \
                     Un valor explícito roto no se sustituye en silencio (CL-15, §25)."
                );
                std::process::exit(2);
            }
        };
    }
    let max_edad = max_edad_recibo();
    if max_edad > Duration::ZERO {
        let ruta = Path::new(RUTA_RECIBO);
        if let Ok(meta) = fs::metadata(ruta) {
            let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            if es_fresco(mtime, SystemTime::now(), max_edad) {
                if let Some(capital) = capital_del_recibo(ruta) {
                    let edad_s = SystemTime::now()
                        .duration_since(mtime)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    println!(
                        "💰 INITIAL_CAPITAL no fijado: usando el recibo vivo {ruta:?} \
                         (capital ${capital:.2}, edad {edad_s}s) — balance real extraído \
                         por el engine de la wallet (§52, CL-15)."
                    );
                    return capital;
                }
                eprintln!(
                    "⚠️ {ruta:?} presente sin capital_inicial válido; se exige INITIAL_CAPITAL."
                );
            } else {
                let edad_s = SystemTime::now()
                    .duration_since(mtime)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                eprintln!(
                    "⚠️ {ruta:?} rancio (edad {edad_s}s > máximo {}s): dato zombi, no se usa.",
                    max_edad.as_secs()
                );
            }
        }
    }
    eprintln!(
        "ERROR: INITIAL_CAPITAL ausente y sin recibo vivo fresco ({RUTA_RECIBO}): \
         exporta el balance real de la wallet o arranca el engine para que lo publique \
         (paridad §25, CL-15). Sin defaults silenciosos."
    );
    std::process::exit(2);
}

/// Capital publicado en el recibo, si existe y es válido.
fn capital_del_recibo(ruta: &Path) -> Option<f64> {
    let contenido = fs::read_to_string(ruta).ok()?;
    let recibo: ReciboMinimo = serde_json::from_str(&contenido).ok()?;
    recibo.capital_inicial.filter(|c| c.is_finite() && *c > 0.0)
}

/// Frescura por mtime. Un mtime FUTURO (sesgo de reloj) cuenta como fresco:
/// el valor sigue viniendo del engine y castigarlo por skew rompería la
/// línea del reloj coherente (I-5).
pub fn es_fresco(mtime: SystemTime, ahora: SystemTime, max_edad: Duration) -> bool {
    ahora
        .duration_since(mtime)
        .map(|edad| edad <= max_edad)
        .unwrap_or(true)
}

/// Umbral de frescura con override `TG_RECIBO_MAX_EDAD_SECS`.
/// Ilegible ⇒ default documentado con aviso (nunca 0 en silencio).
fn max_edad_recibo() -> Duration {
    match std::env::var("TG_RECIBO_MAX_EDAD_SECS") {
        Ok(txt) => match txt.trim().parse::<u64>() {
            Ok(secs) => Duration::from_secs(secs),
            Err(_) => {
                eprintln!(
                    "⚠️ TG_RECIBO_MAX_EDAD_SECS ilegible ({txt:?}): usando default \
                     {RECIBO_MAX_EDAD_DEFAULT_SECS}s."
                );
                Duration::from_secs(RECIBO_MAX_EDAD_DEFAULT_SECS)
            }
        },
        Err(_) => Duration::from_secs(RECIBO_MAX_EDAD_DEFAULT_SECS),
    }
}

#[cfg(test)]
mod tests {
    use super::{capital_del_recibo, es_fresco, validar};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{Duration, SystemTime};

    fn fixture(nombre: &str, contenido: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir =
            std::env::temp_dir().join(format!("capital-arena-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&dir).expect("fixture dir");
        let ruta = dir.join(nombre);
        fs::write(&ruta, contenido).expect("fixture write");
        ruta
    }

    #[test]
    fn valida_solo_finitos_positivos() {
        assert_eq!(validar("37.0"), Some(37.0));
        assert_eq!(validar("  42 "), Some(42.0));
        assert_eq!(validar("1e3"), Some(1000.0));
        for ilegal in ["", "abc", "NaN", "inf", "-inf", "0", "-1", "1e400"] {
            assert_eq!(validar(ilegal), None, "debe rechazar {ilegal:?}");
        }
    }

    #[test]
    fn recibo_entrega_solo_capital_valido() {
        let bueno = fixture(
            "bueno.json",
            "{\"trades\":3,\"capital_inicial\":37.5,\"wr_actual\":0.5}",
        );
        assert_eq!(capital_del_recibo(&bueno), Some(37.5));
        for (nombre, contenido) in [
            ("nulo.json", "{\"capital_inicial\":null}"),
            ("ausente.json", "{\"trades\":1}"),
            ("roto.json", "{no json"),
            ("negativo.json", "{\"capital_inicial\":-5.0}"),
            ("nan.json", "{\"capital_inicial\":\"NaN\"}"),
        ] {
            let ruta = fixture(nombre, contenido);
            assert_eq!(
                capital_del_recibo(&ruta),
                None,
                "{nombre} no debe dar capital"
            );
        }
        let inexistente = PathBuf::from("no_existe_ningun_recibo.json");
        assert_eq!(capital_del_recibo(&inexistente), None);
    }

    #[test]
    fn frescura_rechaza_viejos_y_tolerta_futuro() {
        let ahora = SystemTime::now();
        let max = Duration::from_secs(60);
        let hace_10s = ahora - Duration::from_secs(10);
        let hace_120s = ahora - Duration::from_secs(120);
        assert!(es_fresco(hace_10s, ahora, max));
        assert!(!es_fresco(hace_120s, ahora, max));
        // mtime futuro (sesgo de reloj): cuenta como fresco (doc del módulo).
        let en_30s = ahora + Duration::from_secs(30);
        assert!(es_fresco(en_30s, ahora, max));
    }
}
