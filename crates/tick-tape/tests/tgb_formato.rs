//! TGB-A: el decodificador restituye byte a byte los contenedores de la herramienta del coordinador
//! (`tests/fixtures`, generados con `tgb comprimir`; informe de cada uno en `tgb_fixtures.jsonl`) y los de un
//! codificador de prueba con bloques pequenos, que obligan a reiniciar el delta temporal en cada bloque.

mod comun;

use comun::*;
use tick_tape::tgb::{restituir_tgb, Presupuesto};

const FIXTURES: &[(&str, &[u8], &[u8], u8)] = &[
    ("f1_real", include_bytes!("fixtures/f1_real.bin"), include_bytes!("fixtures/f1_real.bin.tgb"), 1),
    ("f2_legado", include_bytes!("fixtures/f2_legado.bin"), include_bytes!("fixtures/f2_legado.bin.tgb"), 2),
    ("f3_sintetico", include_bytes!("fixtures/f3_sintetico.bin"), include_bytes!("fixtures/f3_sintetico.bin.tgb"), 1),
    ("f4_otro", include_bytes!("fixtures/f4_otro.bin"), include_bytes!("fixtures/f4_otro.bin.tgb"), 0),
    ("f5_vacio", include_bytes!("fixtures/f5_vacio.bin"), include_bytes!("fixtures/f5_vacio.bin.tgb"), 0),
    ("f6_solo_cabecera", include_bytes!("fixtures/f6_solo_cabecera.bin"), include_bytes!("fixtures/f6_solo_cabecera.bin.tgb"), 1),
    ("f7_tgm_ajena", include_bytes!("fixtures/f7_tgm_ajena.bin"), include_bytes!("fixtures/f7_tgm_ajena.bin.tgb"), 1),
];

const INFORME: &str = include_str!("fixtures/tgb_fixtures.jsonl");

fn campo<'a>(linea: &'a str, clave: &str) -> &'a str {
    let k = format!("\"{clave}\":");
    let a = linea.find(&k).unwrap() + k.len();
    let resto = linea[a..].trim_start_matches('"');
    let b = resto.find(|c: char| c == '"' || c == ',' || c == '}').unwrap();
    &resto[..b]
}

/// Compara registro a registro (40 B) y luego el archivo entero; el primer registro distinto sale en el mensaje.
fn igual_por_registro(nombre: &str, esperado: &[u8], obtenido: &[u8], cabecera: usize) {
    assert_eq!(obtenido.len(), esperado.len(), "{nombre}: largo");
    assert_eq!(&obtenido[..cabecera], &esperado[..cabecera], "{nombre}: cabecera");
    for (k, (a, b)) in esperado[cabecera..].chunks(REC).zip(obtenido[cabecera..].chunks(REC)).enumerate() {
        assert_eq!(a, b, "{nombre}: registro {k} distinto");
    }
    assert!(obtenido == esperado, "{nombre}: bytes distintos");
}

#[test]
fn tgb_roundtrip_preserva_bits_y_bloques() {
    // 1. Contenedores de la herramienta: identidad del fixture y restitucion exacta.
    let lineas: Vec<&str> = INFORME.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lineas.len(), FIXTURES.len());
    for (nombre, orig, tgb, modo) in FIXTURES {
        let linea = lineas.iter().find(|l| l.contains(&format!("\"fixture\":\"{nombre}\""))).unwrap();
        assert_eq!(campo(linea, "sha256_tgb"), hex(&sha(tgb)), "{nombre}: el fixture no es el que escribio tgb");
        assert_eq!(campo(linea, "sha256_original"), hex(&sha(orig)), "{nombre}: original del fixture");
        assert_eq!(campo(linea, "modo").parse::<u8>().unwrap(), *modo);
        let r = restituir_tgb(tgb, Presupuesto::sin_tope())
            .unwrap_or_else(|e| panic!("{nombre}: la herramienta lo escribio y el lector lo rechaza: {e}"));
        assert_eq!(r.modo(), *modo, "{nombre}: modo");
        assert_eq!(r.sha256_original(), sha(orig), "{nombre}: SHA del original");
        let cab = if *modo == 1 { 8 } else { 0 };
        igual_por_registro(nombre, orig, r.as_bytes(), if *modo == 0 { orig.len() } else { cab });
    }

    // 2. Bits que no deben tocarse: -0.0, NaN con payload, infinitos y el salto modular del tiempo.
    let f1 = FIXTURES[0].1;
    let r = restituir_tgb(FIXTURES[0].2, Presupuesto::sin_tope()).unwrap();
    let reg = |k: usize, f: usize| u64::from_le_bytes(r.as_bytes()[8 + k * REC + f * 8..8 + k * REC + f * 8 + 8].try_into().unwrap());
    assert_eq!(reg(10, 1), 0x8000_0000_0000_0000);
    assert_eq!(reg(11, 2), 0x7ff8_0000_0000_1234);
    assert_eq!(reg(14, 1), 0xfff0_0000_0000_0001);
    assert_eq!(reg(101, 0), u64::MAX);
    assert_eq!(reg(102, 0), 0);
    assert_eq!(&r.as_bytes()[..], f1);

    // 3. Varios bloques: el delta del tiempo vuelve a 0 al empezar cada bloque (tambien con un bloque por registro).
    let orig = cinta(b"TGMTICK1", 1000, 9);
    for bloque in [1usize, 7, 97, 999, 1000, 4096] {
        let c = codificar(&orig, 1, bloque);
        let r = restituir_tgb(&c, Presupuesto::sin_tope()).unwrap_or_else(|e| panic!("bloque {bloque}: {e}"));
        igual_por_registro(&format!("bloque {bloque}"), &orig, r.as_bytes(), 8);
    }
    let sin_cab = registros(333, 4);
    let r = restituir_tgb(&codificar(&sin_cab, 2, 100), Presupuesto::sin_tope()).unwrap();
    igual_por_registro("modo 2", &sin_cab, r.as_bytes(), 0);
    let otro: Vec<u8> = (0..5000u32).map(|k| (k * 7 % 256) as u8).collect();
    let r = restituir_tgb(&codificar(&otro, 0, 1), Presupuesto::sin_tope()).unwrap();
    assert!(r.as_bytes() == &otro[..]);
}
