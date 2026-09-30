//! MEDICIÓN DE TRANSFER ENTROPY EN TAPES REALES (Ola XLVIII·E).
//!
//! Continúa XLVIII·D bajo la doctrina Fisher/D-754: medir la evidencia
//! ANTES de conectar la TE a cualquier consumidor. Test manual (--ignored):
//! carga dos tapes reales del mismo mes (TGMTICK1, aggTrades), simboliza
//! sobre la rejilla común y reporta el flujo de información direccional.
//!
//! Requiere data/{A,B}USDT_2026-08_REAL.bin. Correr con:
//! cargo test -p feature-engine --test transfer_entropy_real -- --ignored --nocapture

use feature_engine::transfer_entropy::{transfer_entropy_eventos, VENTANA_MS_DEFAULT};

/// Extrae los timestamps de un tape TGMTICK1: header 8 bytes + registros
/// de 40 bytes (timestamp u64 + cuatro f64 little-endian). Este lector
/// valida cabecera, longitud y orden, NO precios, gaps ni origen autenticado.
fn timestamps_de_tape(ruta: &std::path::Path) -> Option<Vec<u64>> {
    let bytes = std::fs::read(ruta).ok()?;
    timestamps_de_bytes(&bytes)
}

fn timestamps_de_bytes(bytes: &[u8]) -> Option<Vec<u64>> {
    const REG: usize = 40; // 5 campos × 8 bytes
    if bytes.len() <= 8 || &bytes[..8] != b"TGMTICK1" || (bytes.len() - 8) % REG != 0 {
        return None;
    }
    let n = (bytes.len() - 8) / REG;
    let mut ts = Vec::with_capacity(n);
    for i in 0..n {
        let off = 8 + i * REG;
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[off..off + 8]);
        let timestamp = u64::from_le_bytes(buf);
        if ts.last().is_some_and(|&previous| timestamp < previous) {
            return None;
        }
        ts.push(timestamp);
    }
    Some(ts)
}

/// En una medición solicitada explícitamente, falta de evidencia es fallo,
/// no una iteración omitida que deja el test verde. No es un gate de trading.
fn medir_par_obligatorio(ts_a: Option<&[u64]>, ts_b: Option<&[u64]>) -> (f64, f64) {
    let a = ts_a.expect("tape A ausente o inválido: medición no realizada");
    let b = ts_b.expect("tape B ausente o inválido: medición no realizada");
    let (tab, tba) = transfer_entropy_eventos(a, b, VENTANA_MS_DEFAULT);
    let x = tab.expect("sin cobertura/soporte válido A→B: medición no realizada");
    let y = tba.expect("sin cobertura/soporte válido B→A: medición no realizada");
    assert!((0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y));
    (x, y)
}

#[cfg(test)]
mod reader_contract {
    use super::*;

    fn tape(ts: &[u64]) -> Vec<u8> {
        let mut bytes = b"TGMTICK1".to_vec();
        for t in ts {
            bytes.extend(t.to_le_bytes());
            bytes.extend([0_u8; 32]);
        }
        bytes
    }

    #[test]
    fn timestamp_layout_and_duplicates_are_preserved() {
        let ts = [0x0102030405060708_u64, 0x0102030405060708, u64::MAX];
        assert_eq!(timestamps_de_bytes(&tape(&ts)), Some(ts.to_vec()));
    }

    #[test]
    fn partial_record_is_not_silently_discarded() {
        let base = tape(&[10, 20]);
        for trailing in 1..40 {
            let mut bytes = base.clone();
            bytes.extend(vec![0; trailing]);
            assert_eq!(timestamps_de_bytes(&bytes), None, "trailing={trailing}");
        }
    }

    #[test]
    fn empty_payload_is_not_a_valid_observation() {
        assert_eq!(timestamps_de_bytes(b"TGMTICK1"), None);
    }

    #[test]
    fn unordered_tape_is_rejected() {
        assert_eq!(timestamps_de_bytes(&tape(&[20, 10, 30])), None);
    }

    #[test]
    fn wrong_origin_and_short_headers_are_rejected() {
        for n in 0..8 {
            assert_eq!(timestamps_de_bytes(&b"TGMTICK1"[..n]), None);
        }
        let mut bytes = tape(&[10, 20]);
        bytes[..8].copy_from_slice(b"TGMSYNT1");
        assert_eq!(timestamps_de_bytes(&bytes), None);
    }

    #[test]
    fn requested_measurement_cannot_pass_without_both_usable_streams() {
        let valid = [0, 13_000];
        let short = [0, 1];
        for (a, b) in [
            (None, Some(valid.as_slice())),
            (Some(valid.as_slice()), None),
            (None, None),
            (Some(short.as_slice()), Some(short.as_slice())),
        ] {
            assert!(std::panic::catch_unwind(|| medir_par_obligatorio(a, b)).is_err());
        }
    }

    #[test]
    fn valid_measurement_returns_both_directions() {
        let a = [0, 200, 6_000, 13_000];
        let b = [0, 400, 6_200, 13_000];
        let (x, y) = medir_par_obligatorio(Some(&a), Some(&b));
        assert!(x.is_finite() && y.is_finite());
    }
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
    println!("Exploratorio: cobertura inferida de extremos, sin contraste de significancia.");
    println!("par                 T(a→b) bits   T(b→a) bits   ratio");
    for (a, b) in pares {
        let ts_a = timestamps_de_tape(&base.join(format!("{a}_2026-08_REAL.bin")));
        let ts_b = timestamps_de_tape(&base.join(format!("{b}_2026-08_REAL.bin")));
        println!("Leyendo {a}/{b}: ambos tapes y cobertura suficiente son obligatorios.");
        let (x, y) = medir_par_obligatorio(ts_a.as_deref(), ts_b.as_deref());
        let ratio = if y > 0.0 { Some(x / y) } else { None };
        println!("{a:<9}→{b:<9} {x:>12.8}   {y:>12.8}   ratio={ratio:?}");
    }
}
