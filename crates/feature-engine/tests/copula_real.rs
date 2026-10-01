//! MEDICIÓN DE CÓPULAS t EN TAPES REALES (Ola LXXI).
//!
//! Doctrina Fisher/D-754 y portón ADR-0006 (como XLVIII·E para transfer
//! entropy): medir la dependencia de cola ANTES de conectar cualquier
//! consumidor. El veto same-bet usa hoy ρ̄ (Hayashi-Yoshiana×signo +
//! curl_share²); la pregunta del portón es si hay estructura de cola que
//! ese ρ lineal NO ve (ν̂ pequeño, λ̂>0) en los horizontes de trading.
//!
//! Test manual (--ignored): pares del roster con tape de agosto-2026,
//! retornos log del precio medio en rejilla espectral {1m, 5m, 15m, 1h},
//! τ de Kendall → ρ̂ por inversión elíptica → ν̂ MLE sobre grid → λ̂
//! implícita + λ empírica al 95/5 como contraste no paramétrico.
//!
//! Correr (release por el τ O(n²)):
//! cargo test --release -p feature-engine --test copula_real -- --ignored --nocapture

use feature_engine::copulas::{lambda_cola_t, lambda_empirica, nu_mle, rho_desde_tau, tau_kendall};

/// Registro de un tape TGMTICK1: header 8 bytes + registros de 40 bytes
/// (timestamp u64 ms + 4×f64: bid=price−½, ask=price+½, qty venta/compra).
/// El precio medio (bid+ask)/2 ES el precio del aggTrade.
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
            return None; // tape desordenado: medición no realizada
        }
        let mid = (bid + ask) * 0.5;
        if !mid.is_finite() || mid <= 0.0 {
            return None;
        }
        out.push((ts, mid));
    }
    Some(out)
}

/// Precio del símbolo en cada punto de la rejilla: último trade ≤ t_k
/// (dos punteros sobre el tape ordenado). Devuelve los índices k con
/// precio disponible y su log-retorno contra el punto previo disponible.
fn retornos_en_rejilla(
    tape: &[(u64, f64)],
    inicio: u64,
    fin: u64,
    paso_ms: u64,
) -> Vec<f64> {
    let mut rets = Vec::new();
    let mut i = 0usize;
    let mut prev: Option<f64> = None;
    let mut t = inicio;
    while t <= fin {
        while i < tape.len() && tape[i].0 <= t {
            i += 1;
        }
        // precio vigente en t = tape[i−1] (si existe)
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

/// Tope de muestras para el τ O(n²): submuestreo uniforme (τ no requiere
/// contigüidad). 4_000 ⇒ 8M comparaciones por par-horizonte.
const TOPE_N: usize = 4_000;
const MIN_N: usize = 1_500;

#[test]
#[ignore = "medición manual: ~2 min en release sobre tapes de agosto en data/"]
fn lxxi_copulas_t_pares_reales_agosto() {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    // Símbolos del roster con tape de agosto (BTC usa el nombre histórico).
    let simbolos: &[(&str, &str)] = &[
        ("BTC", "BTCUSDT_AUG_REAL.bin"),
        ("BNB", "BNBUSDT_2026-08_REAL.bin"),
        ("NEAR", "NEARUSDT_2026-08_REAL.bin"),
        ("ATOM", "ATOMUSDT_2026-08_REAL.bin"),
        ("ADA", "ADAUSDT_2026-08_REAL.bin"),
        ("SOL", "SOLUSDT_2026-08_REAL.bin"),
        ("XRP", "XRPUSDT_2026-08_REAL.bin"),
        ("LTC", "LTCUSDT_2026-08_REAL.bin"),
        ("LINK", "LINKUSDT_2026-08_REAL.bin"),
    ];
    let horizontes: &[(&str, u64)] = &[
        ("1m", 60_000),
        ("5m", 300_000),
        ("15m", 900_000),
        ("1h", 3_600_000),
    ];

    let mut tapes = Vec::new();
    for (nombre, archivo) in simbolos {
        let ruta = base.join(archivo);
        match mid_de_tape(&ruta) {
            Some(t) => {
                println!("✓ {nombre}: {} ticks", t.len());
                tapes.push((*nombre, t));
            }
            None => println!("✗ {nombre}: tape ausente/inválido — EXCLUIDO de la medición"),
        }
    }
    assert!(tapes.len() >= 4, "sin pares medibles: medición no realizada");

    println!("\n═══ MEDICIÓN CÓPULAS t — agosto 2026, rejilla espectral ═══");
    println!("par          h    n      τ̂      ρ̂=sin(πτ/2)  ν̂(MLE)  λ̂impl   λU_emp  λL_emp");
    let mut resumen: Vec<(f64, f64, usize)> = Vec::new(); // (λ̂impl, ν̂, es_par_fuera)
    for (h_nombre, paso) in horizontes {
        let mut lambdas_h = Vec::new();
        for ia in 0..tapes.len() {
            for ib in (ia + 1)..tapes.len() {
                let (na, ta) = &tapes[ia];
                let (nb, tb) = &tapes[ib];
                // ventana común del par en esta rejilla
                let inicio = ta.first().unwrap().0.max(tb.first().unwrap().0);
                let fin = ta.last().unwrap().0.min(tb.last().unwrap().0);
                if fin <= inicio {
                    continue;
                }
                // alinear por MISMA rejilla (t0 múltiplo del paso)
                let t0 = (inicio + paso - 1) / paso * paso;
                let mut ra = retornos_en_rejilla(ta, t0, fin, *paso);
                let mut rb = retornos_en_rejilla(tb, t0, fin, *paso);
                // los primeros retornos pueden diferir en disponibilidad:
                // truncar al mínimo común ALINEADO (misma rejilla ⇒ mismo k)
                let n = ra.len().min(rb.len());
                if n < MIN_N {
                    continue;
                }
                let sa = ra.split_off(ra.len() - n);
                let sb = rb.split_off(rb.len() - n);
                // submuestreo uniforme al tope
                let stride = (sa.len() + TOPE_N - 1) / TOPE_N;
                let xa: Vec<f64> = sa.iter().step_by(stride).copied().collect();
                let xb: Vec<f64> = sb.iter().step_by(stride).copied().collect();

                let tau = match tau_kendall(&xa, &xb) {
                    Some(t) => t,
                    None => continue,
                };
                let rho = rho_desde_tau(tau);
                let (u, v) = scores_por_rango(&xa, &xb);
                let (nu, _ll) = match nu_mle(&u, &v, rho) {
                    Some(x) => x,
                    None => continue,
                };
                let lambda = lambda_cola_t(rho, nu);
                let (lu, ll) = lambda_empirica(&xa, &xb, 0.95).unwrap_or((f64::NAN, f64::NAN));
                println!(
                    "{na:<5}-{nb:<5} {h_nombre:<3} {:>5} {:>8.4} {:>10.4} {:>7.1} {:>7.3} {:>7.3} {:>7.3}",
                    xa.len(), tau, rho, nu, lambda, lu, ll
                );
                lambdas_h.push(lambda);
                resumen.push((lambda, nu, 0));
            }
        }
        if !lambdas_h.is_empty() {
            lambdas_h.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mediana = lambdas_h[lambdas_h.len() / 2];
            println!("  → {h_nombre}: mediana λ̂ = {mediana:.4} sobre {} pares", lambdas_h.len());
        }
    }

    // ── Veredicto del portón (cualitativo, el DATO es el entregable) ──
    let estructurales = resumen
        .iter()
        .filter(|(l, nu, _)| *l >= 0.10 && *nu <= 12.0)
        .count();
    let total = resumen.len();
    println!(
        "\nPORTÓN: {estructurales}/{total} par-horizonte con λ̂≥0.10 y ν̂≤12 \
         (estructura de cola que ρ lineal no ve)."
    );
    println!(
        "Lectura: si la mediana de λ̂ es ≈0 y ν̂ grande domina, el tratamiento\
         gaussiano/EWMA del veto same-bet BASTA — decoración cerrada (como TE)."
    );
}
