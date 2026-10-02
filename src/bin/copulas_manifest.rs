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
//! Uso: copulas_manifest [mes] [--sin-escribir]  (default 2026-08).
//! Escribe config_dir/copulas_manifest.json salvo con --sin-escribir
//! (medición sin mutar el insumo del veto — incidente LXXV).

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

/// Roster vivo (bootloader.rs:301-328) con tape del mes dado. XMR está
/// FUERA del roster (sólo existe en el simulador forense) — no se mide.
const SIMBOLOS: &[&str] = &[
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
    "SHIBUSDT", "DOTUSDT", "LINKUSDT", "POLUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT",
    "UNIUSDT", "XLMUSDT", "NEARUSDT", "ICPUSDT", "FILUSDT", "VETUSDT", "AVAXUSDT",
    "OPUSDT", "APTUSDT", "ARBUSDT", "RENDERUSDT", "LDOUSDT",
];

const HORIENTE_MS: u64 = 300_000; // 5 min: mediana λ̂ de LXXI y horizonte de trading
const TOPE_N: usize = 4_000;
const MIN_N: usize = 1_500;
/// λ mínimo para entrar al manifest: pares sin estructura de cola medible
/// no aportan inflado (y λ<0.05 es ruido del ajuste sobre grid coarse).
const LAMBDA_MIN: f64 = 0.05;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sin_escribir = args.iter().any(|a| a == "--sin-escribir");
    let mes = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "2026-08".into());
    let base = std::path::Path::new("data");
    let mut tapes = Vec::new();
    for s in SIMBOLOS {
        // BTC de agosto usa el nombre histórico
        let nombre = if *s == "BTCUSDT" && mes == "2026-08" {
            "BTCUSDT_AUG_REAL.bin".to_string()
        } else {
            format!("{s}_{mes}_REAL.bin")
        };
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
