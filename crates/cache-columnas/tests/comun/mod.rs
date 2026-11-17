//! Fixtures sinteticos (no capturas reales) y utilidades de las pruebas de la cache.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub const HORA: u64 = 3_600_000;
/// 2026-09-14 00:00:00 UTC.
pub const DIA_BASE: u64 = 1_789_344_000_000;

/// Carpeta temporal propia de la prueba (se rehace vacia).
pub fn carpeta(nombre: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("cache_columnas_{nombre}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Cinta `TGMTICK1` con `n` registros desde `ts0`, separados `paso` ms. Los valores llevan bits incomodos
/// (-0.0, NaN con carga, infinito, subnormal) para que solo una copia bit a bit pase.
pub fn cinta(ts0: u64, n: u64, paso: u64, semilla: u64) -> Vec<u8> {
    let raros = [(-0.0f64).to_bits(), 0x7ff8_0000_dead_beef, f64::INFINITY.to_bits(), 1u64];
    let mut v = b"TGMTICK1".to_vec();
    for k in 0..n {
        let p = 100.0 + (semilla as f64) + 0.01 * k as f64;
        let bq = if k % 7 == 3 { raros[(k as usize / 7) % raros.len()] } else { (1.5 + k as f64).to_bits() };
        for w in [ts0 + k * paso, (p - 0.005).to_bits(), (p + 0.005).to_bits(), bq, (2.25 * semilla as f64).to_bits()] {
            v.extend_from_slice(&w.to_le_bytes());
        }
    }
    v
}

pub fn escribir(p: &Path, b: &[u8]) {
    std::fs::write(p, b).unwrap();
}

pub fn sha(b: &[u8]) -> String {
    cache_columnas::sha256_hex(b)
}

/// Todas las rutas (relativas, con `/`) bajo `raiz`.
pub fn listar(raiz: &Path) -> Vec<String> {
    fn rec(base: &Path, d: &Path, out: &mut Vec<String>) {
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                let p = e.path();
                out.push(p.strip_prefix(base).unwrap().to_string_lossy().replace('\\', "/"));
                if p.is_dir() {
                    rec(base, &p, out);
                }
            }
        }
    }
    let mut v = Vec::new();
    rec(raiz, raiz, &mut v);
    v.sort();
    v
}

pub fn opciones(simbolo: &str) -> cache_columnas::construir::Opciones {
    cache_columnas::construir::Opciones {
        fuente: "tgb-reales".into(),
        simbolo: simbolo.into(),
        ..Default::default()
    }
}
