//! MEDICIÓN DE TRANSFER ENTROPY EN TAPES REALES (Ola XLVIII·E).
//!
//! Continúa XLVIII·D bajo la doctrina Fisher/D-754: medir la evidencia
//! ANTES de conectar la TE a cualquier consumidor. Test manual (--ignored):
//! carga dos tapes reales del mismo mes (TGMTICK1, aggTrades), simboliza
//! sobre la rejilla común y reporta el flujo de información direccional.
//!
//! Requiere data/{A,B}USDT_2026-08_REAL.bin. Correr con:
//! cargo test -p feature-engine --test transfer_entropy_real -- --ignored --nocapture

use feature_engine::transfer_entropy::{VENTANA_MS_DEFAULT, transfer_entropy_eventos};

/// Extrae los timestamps de un tape TGMTICK1: header 8 bytes + registros
/// de 40 bytes (5×f64 little-endian; timestamp es el primero). El layout
/// es el contrato del formato (tick_replayer::BinTick repr(C)).
fn timestamps_de_tape(ruta: &std::path::Path) -> Option<Vec<u64>> {
    let bytes = std::fs::read(ruta).ok()?;
    const REG: usize = 40; // 5 campos × 8 bytes
    if bytes.len() < 8 || &bytes[..8] != b"TGMTICK1" {
        return None;
    }
    let n = (bytes.len() - 8) / REG;
    let mut ts = Vec::with_capacity(n);
    for i in 0..n {
        let off = 8 + i * REG;
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[off..off + 8]);
        ts.push(u64::from_le_bytes(buf));
    }
    Some(ts)
}

/// Pares del mismo mes (agosto-2026): ¿fluye información direccional
/// entre pares a escala sub-segundo? El DATO es el entregable; los
/// asserts pinean sanidad del método, no dirección.
#[test]
#[ignore = "medición manual: ~2 min sobre tapes reales presentes en data/"]
fn xlviiie_te_entre_pares_reales_agosto() {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let pares: &[(&str, &str)] = &[
        ("LTCUSDT", "THETAUSDT"), // major → minor
        ("ADAUSDT", "THETAUSDT"), // major → minor
        ("LTCUSDT", "ADAUSDT"),   // par de majors
    ];
    println!("par                 T(a→b) bits   T(b→a) bits   ratio");
    for (a, b) in pares {
        let ts_a = timestamps_de_tape(&base.join(format!("{a}_2026-08_REAL.bin")));
        let ts_b = timestamps_de_tape(&base.join(format!("{b}_2026-08_REAL.bin")));
        let (Some(ts_a), Some(ts_b)) = (ts_a, ts_b) else {
            println!("{a}→{b}: tape ausente, omitido");
            continue;
        };
        let (tab, tba) = transfer_entropy_eventos(&ts_a, &ts_b, VENTANA_MS_DEFAULT);
        match (tab, tba) {
            (Some(x), Some(y)) => {
                let ratio = if y > 1e-9 { x / y } else { f64::INFINITY };
                println!("{a:<9}→{b:<9} {:>10.4}   {:>10.4}   {:>8.2}", x, y, ratio);
                assert!(x.is_finite() && y.is_finite());
                assert!(x >= 0.0 && y >= 0.0, "TE poblacional ≥ 0");
            }
            _ => println!("{a}→{b}: muestra insuficiente (None)"),
        }
    }
}
