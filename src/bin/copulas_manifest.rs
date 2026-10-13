//! Generador de `config_dir/copulas_manifest.json` (Ola LXXII).
//!
//! Insumo del consumo de λ̂ (dependencia de cola de la cópula t) en el veto
//! same-bet. Mide por par del roster: τ de Kendall → ρ̂ (inversión elíptica)
//! → ν̂ (MLE sobre grid) → λ̂ implícita, sobre retornos log del precio medio
//! en rejilla de 5 min (el horizonte de trading donde la medición LXXI dio
//! mediana λ̂ 0.32), usando los tapes REAL de un mes. El manifest es
//! committable: diff = changelog de la dependencia de cola del roster.
//!
//! La matemática vive en `feature_engine::copulas` (9 contratos); este bin
//! sólo orquesta la medición y escribe. Lector de tapes idéntico al de
//! tests/copula_real.rs (mismo formato TGMTICK1; ver tick_replayer.rs).
//!
//! Uso: copulas_manifest <mes AAAA-MM> [--sin-escribir] [--forzar-mes-anterior].
//! Escribe config_dir/copulas_manifest.json salvo con --sin-escribir
//! (medición sin mutar el insumo del veto — incidente LXXV).
//!
//! CL-124 (T3-04 de la Línea G): el mes es obligatorio. Antes el defecto era
//! 2026-08, el mes al que sustituyó la renovación de septiembre: sin
//! argumento se reescribía el manifiesto vivo con el mes viejo (el incidente
//! LXXV otra vez). Tampoco se escribe un mes anterior al del manifiesto
//! vigente salvo con --forzar-mes-anterior.

use feature_engine::copulas::{lambda_cola_t, nu_mle, rho_desde_tau, tau_kendall};

/// Registro TGMTICK1: header 8B + registros de 40B (ts u64 ms + 4×f64;
/// bid=price−½ ask=price+½ ⇒ mid = precio del aggTrade).
fn mid_de_tape(ruta: &std::path::Path) -> Option<Vec<(u64, f64)>> {
    let bytes = std::fs::read(ruta).ok()?;
    const REG: usize = 40;
    if bytes.len() <= 8 || &bytes[..8] != b"TGMTICK1" || (bytes.len() - 8) % REG != 0 {
        return None;
    }
    let n = (bytes.len() - 8) / REG;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = 8 + i * REG;
        let mut b8 = [0u8; 8];
        b8.copy_from_slice(&bytes[off..off + 8]);
        let ts = u64::from_le_bytes(b8);
        b8.copy_from_slice(&bytes[off + 8..off + 16]);
        let bid = f64::from_le_bytes(b8);
        b8.copy_from_slice(&bytes[off + 16..off + 24]);
        let ask = f64::from_le_bytes(b8);
        if out.last().is_some_and(|&(t, _)| ts < t) {
            return None;
        }
        let mid = (bid + ask) * 0.5;
        if !mid.is_finite() || mid <= 0.0 {
            return None;
        }
        out.push((ts, mid));
    }
    Some(out)
}

/// Log-retornos del precio vigente (último trade ≤ t_k) en rejilla común.
fn retornos_en_rejilla(tape: &[(u64, f64)], inicio: u64, fin: u64, paso_ms: u64) -> Vec<f64> {
    let mut rets = Vec::new();
    let mut i = 0usize;
    let mut prev: Option<f64> = None;
    let mut t = inicio;
    while t <= fin {
        while i < tape.len() && tape[i].0 <= t {
            i += 1;
        }
        if i > 0 {
            let p = tape[i - 1].1;
            if let Some(p0) = prev {
                let r = (p / p0).ln();
                if r.is_finite() {
                    rets.push(r);
                }
            }
            prev = Some(p);
        }
        t += paso_ms;
    }
    rets
}

fn scores_por_rango(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let ranks = |v: &[f64]| -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
        let mut r = vec![0usize; v.len()];
        for (pos, &i) in idx.iter().enumerate() {
            r[i] = pos + 1;
        }
        r.iter().map(|&p| p as f64 / (v.len() + 1) as f64).collect()
    };
    (ranks(x), ranks(y))
}

/// CL-123: el roster vivo, la MISMA lista del bootloader (antes una copia,
/// que decía SHIBUSDT como el bootloader). XMR está FUERA del roster (sólo
/// existe en el simulador forense) — no se mide.
const SIMBOLOS: &[&str] = god_engine_core::bootloader::ROSTER;

const HORIENTE_MS: u64 = 300_000; // 5 min: mediana λ̂ de LXXI y horizonte de trading
const TOPE_N: usize = 4_000;
const MIN_N: usize = 1_500;
/// λ mínimo para entrar al manifest: pares sin estructura de cola medible
/// no aportan inflado (y λ<0.05 es ruido del ajuste sobre grid coarse).
const LAMBDA_MIN: f64 = 0.05;

/// Nombre de la cinta de un símbolo del roster para el mes dado.
/// - En 2026-08, BTC y ETH llevan el nombre histórico `{SYM}_AUG_REAL.bin`
///   (antes sólo BTC: ETH quedaba fuera del manifiesto de agosto).
/// - SHIB cotiza en futuros USDⓈ-M como `1000SHIBUSDT` (mil unidades por
///   contrato), que es su nombre en el roster desde CL-123; su cinta lleva
///   ese nombre. Los retornos logarítmicos no cambian con la escala.
fn nombre_de_cinta(simbolo: &str, mes: &str) -> String {
    match (simbolo, mes) {
        ("BTCUSDT" | "ETHUSDT", "2026-08") => format!("{simbolo}_AUG_REAL.bin"),
        _ => format!("{simbolo}_{mes}_REAL.bin"),
    }
}

/// CL-124 — el mes a medir y escribir: obligatorio, con forma AAAA-MM, y no
/// anterior al del manifiesto vigente (`mes_vigente`) salvo con
/// `--forzar-mes-anterior`. Al medir sin escribir no se compara con el
/// vigente.
fn mes_a_escribir(args: &[String], mes_vigente: Option<&str>) -> Result<String, String> {
    let mes = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .ok_or("falta el mes (AAAA-MM): sin él se reescribía el manifiesto con un mes viejo")?;
    let forma = mes.len() == 7
        && mes.as_bytes()[4] == b'-'
        && mes[..4].bytes().all(|c| c.is_ascii_digit())
        && mes[5..].bytes().all(|c| c.is_ascii_digit())
        && (1..=12).contains(&mes[5..].parse::<u32>().unwrap_or(0));
    if !forma {
        return Err(format!("mes «{mes}» sin la forma AAAA-MM"));
    }
    let sin_escribir = args.iter().any(|a| a == "--sin-escribir");
    let forzar = args.iter().any(|a| a == "--forzar-mes-anterior");
    if let Some(vigente) = mes_vigente {
        // AAAA-MM se ordena igual como texto que como fecha.
        if !sin_escribir && !forzar && mes.as_str() < vigente {
            return Err(format!(
                "el manifiesto vigente es de {vigente}; {mes} es anterior (usa --forzar-mes-anterior si es a propósito)"
            ));
        }
    }
    Ok(mes)
}

/// Mes del manifiesto vigente, si se puede leer.
fn mes_del_manifiesto(ruta: &std::path::Path) -> Option<String> {
    let texto = std::fs::read_to_string(ruta).ok()?;
    let i = texto.find("\"mes\": \"")? + "\"mes\": \"".len();
    Some(texto[i..i + texto[i..].find('"')?].to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sin_escribir = args.iter().any(|a| a == "--sin-escribir");
    let ruta_vigente = std::path::Path::new("config_dir").join("copulas_manifest.json");
    let mes = match mes_a_escribir(&args, mes_del_manifiesto(&ruta_vigente).as_deref()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("❌ copulas_manifest: {e}");
            std::process::exit(2);
        }
    };
    let base = std::path::Path::new("data");
    let mut tapes = Vec::new();
    for s in SIMBOLOS {
        let nombre = nombre_de_cinta(s, &mes);
        match mid_de_tape(&base.join(&nombre)) {
            Some(t) => {
                println!("✓ {s}: {} ticks", t.len());
                tapes.push((*s, t));
            }
            None => println!("— {s}: sin tape {nombre} — par excluido del manifest"),
        }
    }
    if tapes.len() < 4 {
        eprintln!("sin pares medibles suficientes ({})", tapes.len());
        std::process::exit(2);
    }

    let mut entradas = String::new();
    let mut n_pares = 0usize;
    for ia in 0..tapes.len() {
        for ib in (ia + 1)..tapes.len() {
            let (na, ta) = &tapes[ia];
            let (nb, tb) = &tapes[ib];
            let inicio = ta.first().unwrap().0.max(tb.first().unwrap().0);
            let fin = ta.last().unwrap().0.min(tb.last().unwrap().0);
            if fin <= inicio {
                continue;
            }
            let t0 = (inicio + HORIENTE_MS - 1) / HORIENTE_MS * HORIENTE_MS;
            let mut ra = retornos_en_rejilla(ta, t0, fin, HORIENTE_MS);
            let mut rb = retornos_en_rejilla(tb, t0, fin, HORIENTE_MS);
            let n = ra.len().min(rb.len());
            if n < MIN_N {
                continue;
            }
            let sa = ra.split_off(ra.len() - n);
            let sb = rb.split_off(rb.len() - n);
            let stride = (sa.len() + TOPE_N - 1) / TOPE_N;
            let xa: Vec<f64> = sa.iter().step_by(stride).copied().collect();
            let xb: Vec<f64> = sb.iter().step_by(stride).copied().collect();
            let Some(tau) = tau_kendall(&xa, &xb) else { continue };
            let rho = rho_desde_tau(tau);
            let (u, v) = scores_por_rango(&xa, &xb);
            let Some((nu, _ll)) = nu_mle(&u, &v, rho) else { continue };
            let lambda = lambda_cola_t(rho, nu);
            if lambda < LAMBDA_MIN {
                continue; // sin estructura de cola medible: no entra
            }
            entradas.push_str(&format!(
                "    {{\"a\": \"{na}\", \"b\": \"{nb}\", \"tau\": {tau:.6}, \"rho\": {rho:.6}, \"nu\": {nu:.1}, \"lambda\": {lambda:.6}, \"n\": {}}},\n",
                xa.len()
            ));
            n_pares += 1;
        }
    }
    let entradas = entradas.trim_end_matches(",\n").to_string();
    let manifest = format!(
        "{{\n  \"horizonte_ms\": {HORIENTE_MS},\n  \"mes\": \"{mes}\",\n  \
         \"fuente\": \"copulas_manifest LXXII: tapes {mes} REAL, rejilla 5m, τ→ρ sin(πτ/2), ν MLE grid, λ cerrada; ver feature_engine::copulas\",\n  \
         \"lambda_min\": {LAMBDA_MIN},\n  \"pares\": [\n{entradas}\n  ]\n}}\n"
    );
    if sin_escribir {
        println!("👁️  --sin-escribir: {n_pares} pares con λ̂≥{LAMBDA_MIN} a 5m ({mes}) — manifest vivo INTACTO");
        return;
    }
    let ruta = std::path::Path::new("config_dir").join("copulas_manifest.json");
    std::fs::write(&ruta, manifest).expect("escribir copulas_manifest.json");
    println!(
        "✅ config_dir/copulas_manifest.json: {n_pares} pares con λ̂≥{LAMBDA_MIN} a 5m ({mes})"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// CL-124. Verdad: sin mes no hay manifiesto (antes, el defecto 2026-08).
    #[test]
    fn cl124_el_mes_es_obligatorio() {
        assert!(mes_a_escribir(&a(&[]), Some("2026-09")).is_err());
        assert!(mes_a_escribir(&a(&["--sin-escribir"]), None).is_err());
        assert!(mes_a_escribir(&a(&["2026-13"]), None).is_err());
        assert!(mes_a_escribir(&a(&["sept"]), None).is_err());
        assert_eq!(mes_a_escribir(&a(&["2026-10"]), Some("2026-09")).unwrap(), "2026-10");
    }

    /// CL-124. Verdad: no se escribe un mes anterior al vigente, salvo a
    /// propósito o sin escribir.
    #[test]
    fn cl124_no_se_escribe_un_mes_anterior_al_vigente() {
        assert!(mes_a_escribir(&a(&["2026-08"]), Some("2026-09")).is_err());
        assert!(mes_a_escribir(&a(&["2026-08", "--forzar-mes-anterior"]), Some("2026-09")).is_ok());
        assert!(mes_a_escribir(&a(&["2026-08", "--sin-escribir"]), Some("2026-09")).is_ok());
        assert!(mes_a_escribir(&a(&["2026-09"]), Some("2026-09")).is_ok(), "regenerar el mismo mes");
    }

    /// Verdad: el mes vigente se lee del manifiesto con su formato.
    #[test]
    fn el_mes_del_manifiesto_vivo() {
        let ruta = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config_dir/copulas_manifest.json");
        assert_eq!(mes_del_manifiesto(&ruta).as_deref(), Some("2026-09"));
    }
}
