//! Store de λ̂ (dependencia de cola por par, cópula t) para el veto
//! same-bet (Ola LXXII).
//!
//! La medición LXXI (TRIAGE_TEORICO §LXXI: 100/108 par-horizonte con
//! λ̂≥0.10, ν̂ 2-6; BTC-SOL ρ̂ 0.77 → λ̂ 0.51 donde la gaussiana daría 0)
//! demostró que la ρ̄ lineal SUBESTIMA el stop-out conjunto del grupo.
//! Este store entrega el λ̂ medido por par; `dependency_exposure` lo
//! consume como tercera etapa de inflado hacia 1 (después de la media
//! XLVI·D y del curl_share² de Hodge, ola 6) componiendo sobre el
//! COMPLEMENTO de independencia:
//!
//!   1 − ρ_final = (1 − base) · (1 − curl²) · (1 − λ̂)
//!
//! **Disciplina de arranque frío (D-754)**: sin `config_dir/copulas_
//! manifest.json`, archivo ilegible, o par ausente ⇒ `lambda_entre` =
//! None ⇒ el inflado se OMITE y el veto es BIT A BIT el legado — la
//! medición no significa nada hasta existir, exactamente como V-RISK-006
//! con R de Lundberg ausente. El manifest lo genera `copulas_manifest`
//! (committable; diff = changelog de dependencia de cola del roster).

use std::collections::HashMap;
use std::sync::OnceLock;

/// λ̂ por par de coin_ids (clave canónica: min,max). Vecío = legado.
struct StorePares {
    mapa: HashMap<(usize, usize), f64>,
    /// pares descartados por símbolo fuera del universo vivo (telemetría
    /// de carga, no consumo): "A|B" → λ del manifest no aplicable.
    fuera_de_roster: Vec<String>,
}

static STORE: OnceLock<StorePares> = OnceLock::new();

/// Resuelve símbolo → coin_id vía el universo dinámico del arena.
/// Depende de quantum_arena::symbols (la única fuente de verdad del
/// índice, D-725).
fn coin_id_de_simbolo(sym: &str) -> Option<usize> {
    quantum_arena::symbols::get_coin_id(sym)
}

/// Carga explícita (arranque del god_engine o tests). Devuelve el número
/// de pares aplicables. Llamada doble es no-op (el OnceLock gana).
pub fn cargar_pares_desde_json(texto: &str) -> Result<usize, String> {
    let inner = parsear(texto)?;
    let _ = STORE.set(inner);
    Ok(STORE
        .get()
        .map(|s| s.mapa.len())
        .unwrap_or(0))
}
/// Carga lazy desde `config_dir/copulas_manifest.json` (cwd = raíz del
/// workspace, la convención de replay y de arranque). Idempotente:
/// archivo ausente ⇒ store VACÍO (legado bit-exact), y NO se reintenta
/// por proceso — el watcher de modelos no aplica a este insumo estático
/// (se regenerará por ola, no por sesión).
fn asegurar_cargado() {
    if STORE.get().is_some() {
        return;
    }
    let Ok(texto) = std::fs::read_to_string("config_dir/copulas_manifest.json") else {
        // Sin manifest ⇒ store vacío PERMANENTE del proceso: legado
        // bit-exact, sin reintentos (el insumo es estático por ola).
        let _ = STORE.set(StorePares {
            mapa: HashMap::new(),
            fuera_de_roster: Vec::new(),
        });
        return;
    };
    match parsear(&texto) {
        Ok(inner) => {
            // Si TODOS los pares quedaron fuera de roster porque el
            // universo dinámico aún no fue poblado (llamada temprana),
            // NO sellar el store: diferir y reintentar en la próxima
            // llamada (el bootloader puebla el universo en fase 1/2).
            if inner.mapa.is_empty()
                && !inner.fuera_de_roster.is_empty()
                && quantum_arena::symbols::get_active_universe_size() == 0
            {
                return;
            }
            let _ = STORE.set(inner);
        }
        Err(_) => {
            let _ = STORE.set(StorePares {
                mapa: HashMap::new(),
                fuera_de_roster: Vec::new(),
            });
        }
    }
}

/// λ̂ medido del par (a,b) — None ⇒ sin medición ⇒ inflado omitido.
pub fn lambda_entre(a: usize, b: usize) -> Option<f64> {
    if a == b {
        return None; // mismo símbolo: la ρ base ya lo cubre
    }
    asegurar_cargado();
    let clave = (a.min(b), a.max(b));
    STORE.get()?.mapa.get(&clave).copied().filter(|l| {
        l.is_finite() && *l > 0.0 && *l <= 1.0
    })
}

/// Pares del manifest cuya λ no se aplicó (símbolo fuera del universo).
/// Telemetría de honestidad de carga: un manifest con símbolos muertos
/// debe ser visible, no silencioso.
pub fn pares_fuera_de_roster() -> Vec<String> {
    asegurar_cargado();
    STORE
        .get()
        .map(|s| s.fuera_de_roster.clone())
        .unwrap_or_default()
}

fn parsear(texto: &str) -> Result<StorePares, String> {
    // JSON mínimo a mano (el repo no depende de serde_json en risk-engine
    // salvo donde ya existe; aquí el formato es propio y acotado).
    let mut mapa = HashMap::new();
    let mut fuera = Vec::new();
    let mut dentro_de_pares = false;
    for linea in texto.lines() {
        let l = linea.trim();
        if l.contains("\"pares\"") {
            dentro_de_pares = true;
            // soporta `{"pares": [` inline: continuar con la MISMA línea
            // es innecesario — las entradas siempre vienen en líneas
            // propias en el formato del generador y de los tests.
            continue;
        }
        if !dentro_de_pares {
            continue;
        }
        if l.starts_with(']') {
            break;
        }
        // campos: "a": "...", "b": "...", ..., "lambda": 0.51, "n": 2976
        let Some(campo_a) = extraer_str(l, "\"a\"") else {
            continue;
        };
        let Some(campo_b) = extraer_str(l, "\"b\"") else {
            continue;
        };
        let Some(lambda_txt) = extraer_num(l, "\"lambda\"") else {
            return Err(format!("par {campo_a}-{campo_b} sin lambda"));
        };
        let lambda: f64 = lambda_txt
            .parse()
            .map_err(|_| format!("lambda no numérico: {lambda_txt}"))?;
        if !(0.0..=1.0).contains(&lambda) || !lambda.is_finite() {
            return Err(format!("lambda fuera de [0,1]: {lambda}"));
        }
        match (coin_id_de_simbolo(&campo_a), coin_id_de_simbolo(&campo_b)) {
            (Some(ida), Some(idb)) => {
                mapa.insert((ida.min(idb), ida.max(idb)), lambda);
            }
            _ => fuera.push(format!("{campo_a}|{campo_b}")),
        }
    }
    Ok(StorePares {
        mapa,
        fuera_de_roster: fuera,
    })
}

fn extraer_str(linea: &str, campo: &str) -> Option<String> {
    let pos = linea.find(campo)?;
    let resto = &linea[pos + campo.len()..];
    let inicio = resto.find('"')? + 1;
    let fin = resto[inicio..].find('"')? + inicio;
    Some(resto[inicio..fin].to_string())
}

fn extraer_num(linea: &str, campo: &str) -> Option<String> {
    let pos = linea.find(campo)?;
    let resto = &linea[pos + campo.len()..];
    let inicio = resto.find(':')? + 1;
    let num: String = resto[inicio..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e' || *c == 'E' || *c == '+')
        .collect();
    if num.is_empty() {
        None
    } else {
        Some(num)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parseo_campos_minimos() {
        let linea = r#"    {"a": "BTCUSDT", "b": "SOLUSDT", "tau": 0.562300, "rho": 0.772800, "nu": 3.0, "lambda": 0.514000, "n": 2976},"#;
        assert_eq!(extraer_str(linea, "\"a\"").as_deref(), Some("BTCUSDT"));
        assert_eq!(extraer_str(linea, "\"b\"").as_deref(), Some("SOLUSDT"));
        assert_eq!(extraer_num(linea, "\"lambda\"").as_deref(), Some("0.514000"));
        assert_eq!(extraer_num(linea, "\"nu\"").as_deref(), Some("3.0"));
    }

    #[test]
    fn lambda_fuerade_rango_rechaza_el_parseo() {
        let mal = r#"{"pares": [
    {"a": "BTCUSDT", "b": "SOLUSDT", "lambda": 1.5, "n": 10}
  ]}"#;
        assert!(parsear(mal).is_err());
        let nan = r#"{"pares": [
    {"a": "BTCUSDT", "b": "SOLUSDT", "lambda": NaN, "n": 10}
  ]}"#;
        assert!(parsear(nan).is_err());
    }

    // Nota: los tests de continuidad bit-exact y monotonicidad viven en
    // correlation_guard (sobre dependency_exposure), donde está el inflado;
    // aquí sólo el contrato del store.
}
