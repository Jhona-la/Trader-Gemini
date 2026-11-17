//! P4 (pedido v3): union de fuentes por momento de PUBLICACION; el lector nunca entrega lo que aun no se conocia.
mod comun;

use cache_columnas::construir::{construir_registros, Opciones, Origen};
use cache_columnas::lector::{leer, rango_pub, Consulta};
use cache_columnas::MS_POR_DIA;
use comun::*;
use std::path::Path;

const MIN: u64 = 60_000;

fn registros(filas: &[(u64, f64)]) -> Vec<u8> {
    let mut v = Vec::new();
    for (ts, p) in filas {
        for w in [*ts, p.to_bits(), p.to_bits(), 1f64.to_bits(), 1f64.to_bits()] {
            v.extend_from_slice(&w.to_le_bytes());
        }
    }
    v
}

fn construir(cache: &Path, fuente: &str, simbolo: &str, filas: &[(u64, f64)], pubs: Option<&[i64]>) -> Result<(), String> {
    let cuerpo = registros(filas);
    let origen = Origen {
        logica: format!("sintetico:{fuente}:{simbolo}"),
        fisica: "sintetico".into(),
        representacion: "sintetico".into(),
        sha256_original: sha(&cuerpo),
        bytes_original: cuerpo.len() as u64,
        sha256_contenedor: None,
    };
    let op = Opciones { fuente: fuente.into(), simbolo: simbolo.into(), ..Default::default() };
    construir_registros(cache, &origen, &cuerpo, pubs, &op).map(|_| ())
}

#[test]
fn union_por_publicacion_sin_futuro() {
    let d = carpeta("p4_union");
    let cache = d.join("cache");
    let (h10, h11) = (DIA_BASE + 10 * HORA, DIA_BASE + 11 * HORA);
    // Captura: X con evento 10:50 publicada a las 11:07 (retraso); Y con evento 10:55 publicada a las 10:56.
    let x = (h10 + 50 * MIN, 1.0);
    let y = (h10 + 55 * MIN, 2.0);
    let pubs = [(h11 + 7 * MIN) as i64, (h10 + 56 * MIN) as i64];
    construir(&cache, "tgl4-captura", "BTCUSDT", &[x, y], Some(&pubs)).unwrap();

    // Parte 1: hasta las 11:00 solo Y (X no era conocible a las 11:00 aunque su evento sea anterior).
    let c = |series: Vec<(&str, &str)>, desde: u64, hasta: u64| Consulta {
        series: series.into_iter().map(|(a, b)| (a.to_string(), b.to_string())).collect(),
        desde_pub_ms: desde as i64,
        hasta_pub_ms: hasta as i64,
    };
    let e = leer(&cache, &c(vec![("tgl4-captura", "BTCUSDT")], DIA_BASE, h11)).unwrap();
    assert_eq!(e.filas.iter().map(|f| f.ts).collect::<Vec<_>>(), vec![y.0]);
    // Hasta las 11:10 entran las dos, ORDENADAS por publicacion (Y antes que X), no por evento.
    let e = leer(&cache, &c(vec![("tgl4-captura", "BTCUSDT")], DIA_BASE, h11 + 10 * MIN)).unwrap();
    assert_eq!(e.filas.iter().map(|f| f.ts).collect::<Vec<_>>(), vec![y.0, x.0]);
    assert_eq!(e.horas[e.filas[1].hora].valor_f64(1, e.filas[1].fila), 1.0);

    // Parte 2: macro t-1. El cierre del dia D-1 se publica al empezar D; el valor del dia D (evento 09:00) no se
    // publica hasta D+1, aunque su evento sea anterior a las 11:10.
    let ayer = (DIA_BASE - 3 * HORA, 5000.0);
    let hoy = (DIA_BASE + 9 * HORA, 5100.0);
    construir(&cache, "macro-yahoo", "SP500", &[ayer, hoy], None).unwrap();
    let e = leer(
        &cache,
        &c(vec![("tgl4-captura", "BTCUSDT"), ("macro-yahoo", "SP500")], DIA_BASE, h11 + 10 * MIN),
    )
    .unwrap();
    let orden: Vec<u64> = e.filas.iter().map(|f| f.ts).collect();
    assert_eq!(orden, vec![ayer.0, y.0, x.0], "merge por pub_ms, sin el macro del dia D");
    assert!(e.filas.windows(2).all(|w| w[0].pub_ms <= w[1].pub_ms));
    assert_eq!(e.filas[0].pub_ms, DIA_BASE as i64);
    // El manifiesto guarda el rango pub_ms por serie.
    assert_eq!(rango_pub(&cache, "macro-yahoo", "SP500").unwrap(), Some((DIA_BASE as i64, (DIA_BASE + MS_POR_DIA) as i64)));

    // La tabla manda: el macro no acepta pub_ms del productor y la captura lo exige; pub_ms < evento se rechaza.
    assert!(construir(&cache, "macro-yahoo", "NASDAQ", &[hoy], Some(&[(DIA_BASE + MS_POR_DIA) as i64])).is_err());
    assert!(construir(&cache, "tgl4-captura", "ETHUSDT", &[x], None).is_err());
    assert!(construir(&cache, "tgl4-captura", "ETHUSDT", &[x], Some(&[(x.0 - 1) as i64])).is_err());
    assert!(construir(&cache, "inventada", "ETHUSDT", &[x], Some(&[x.0 as i64])).is_err());
    let _ = std::fs::remove_dir_all(&d);
}
