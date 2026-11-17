//! TGB-G: firma de generacion y admision para entrenamiento (pedido CODEX-CLI, Parte 4, "Pruebas futuras de la
//! marca"). Solo la cinta relativa con recibo de procedencia e identidad vigentes llega a la preparacion de
//! muestras; los demas casos devuelven una causa verificable.

mod comun;

use comun::{hex, sha};
use tick_tape::logica::{open_logical_tape, TapeBytes};
use tick_tape::procedencia::{admitir_para_entrenamiento, firma_de_cinta, EstadoGeneracion, IndiceFijado, R_RELATIVA, SUELO_ANTIGUO};
use tick_tape::tgb::Presupuesto;

fn registro(ts: u64, bid: f64, ask: f64) -> [u8; 40] {
    let mut r = [0u8; 40];
    r[..8].copy_from_slice(&ts.to_le_bytes());
    r[8..16].copy_from_slice(&bid.to_le_bytes());
    r[16..24].copy_from_slice(&ask.to_le_bytes());
    r[24..32].copy_from_slice(&1.0f64.to_le_bytes());
    r[32..40].copy_from_slice(&2.0f64.to_le_bytes());
    r
}

fn rel(p: f64) -> (f64, f64) {
    let h = p * R_RELATIVA;
    (p - h, p + h)
}

fn ant(p: f64) -> (f64, f64) {
    let h = (p * R_RELATIVA).max(SUELO_ANTIGUO);
    (p - h, p + h)
}

fn medio_tick(p: f64, tick: f64) -> (f64, f64) {
    let h = (p * R_RELATIVA).max(tick * 0.5);
    (p - h, p + h)
}

fn cinta(pares: &[(f64, f64)]) -> Vec<u8> {
    let mut v = b"TGMTICK1".to_vec();
    for (k, (b, a)) in pares.iter().enumerate() {
        v.extend_from_slice(&registro(1_700_000_000_000 + k as u64, *b, *a));
    }
    v
}

fn precios(desde: f64, paso: f64, n: usize) -> Vec<f64> {
    (0..n).map(|k| desde + paso * k as f64).collect()
}

fn estado(bytes: &[u8]) -> EstadoGeneracion {
    firma_de_cinta(bytes).unwrap().estado()
}

struct Caso {
    dir: std::path::PathBuf,
}
impl Caso {
    fn nuevo(n: &str) -> Caso {
        let dir = std::env::temp_dir().join(format!("tick_tape_proc_{}_{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Caso { dir }
    }
    /// Un archivo por caso: en Windows no se sobrescribe un archivo con un mapeo abierto.
    fn tape(&self, nombre: &str, bytes: &[u8]) -> TapeBytes {
        let p = self.dir.join(format!("{nombre}_REAL.bin"));
        std::fs::write(&p, bytes).unwrap();
        open_logical_tape(&p, Presupuesto::sin_tope()).unwrap()
    }
}
impl Drop for Caso {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[test]
fn tgb_firma_distingue_recetas_sin_adivinar() {
    let bajos = precios(0.45, 0.013, 200); // por debajo del cruce 0.01 / 0.00005
    let altos = precios(60_000.0, 0.7, 200); // muy por encima: las dos recetas dan los mismos bytes
    let cruce = SUELO_ANTIGUO / R_RELATIVA;
    let r_bajos: Vec<_> = bajos.iter().map(|p| rel(*p)).collect();
    let a_bajos: Vec<_> = bajos.iter().map(|p| ant(*p)).collect();
    assert_eq!(estado(&cinta(&r_bajos)), EstadoGeneracion::Relativa, "relativa de precio bajo");
    assert_eq!(estado(&cinta(&a_bajos)), EstadoGeneracion::Antigua, "antigua de precio bajo");
    let r_altos: Vec<_> = altos.iter().map(|p| rel(*p)).collect();
    assert_eq!(estado(&cinta(&r_altos)), EstadoGeneracion::Indeterminada, "donde las formulas coinciden");
    // Entorno del cruce y sus vecinos f64: nunca se inventa una firma exclusiva por encima del cruce.
    let mut cerca = Vec::new();
    let mut p = cruce;
    for _ in 0..8 {
        p = f64::from_bits(p.to_bits() + 1);
        cerca.push(rel(p));
    }
    assert_eq!(estado(&cinta(&cerca)), EstadoGeneracion::Indeterminada, "por encima del cruce");
    // Mezcla de firmas.
    let mut mezcla = r_bajos.clone();
    mezcla.extend_from_slice(&a_bajos);
    assert_eq!(estado(&cinta(&mezcla)), EstadoGeneracion::Mixta);
    // Datos no finitos, cruzados o no positivos: ninguna receta.
    for malo in [(f64::NAN, 1.0), (1.0, f64::INFINITY), (2.0, 1.0), (0.0, 1.0)] {
        let mut v = r_bajos.clone();
        v.push(malo);
        assert_eq!(estado(&cinta(&v)), EstadoGeneracion::OtraOInvalida, "registro invalido {malo:?}");
    }
    // La transformacion de medio tick de tape_spread_fix no se confunde con ninguna de las dos.
    let fix: Vec<_> = bajos.iter().map(|p| medio_tick(*p, 0.001)).collect();
    assert_eq!(estado(&cinta(&fix)), EstadoGeneracion::OtraOInvalida, "medio tick");
    // Testigos: el primer registro discriminante queda anotado.
    let f = firma_de_cinta(&cinta(&mezcla)).unwrap();
    assert_eq!(f.primer_solo_relativa, Some(0));
    assert_eq!(f.primer_solo_antigua, Some(bajos.len() as u64));
    // Cabecera REAL de 8 bytes exactos (no basta TGM).
    let mut ajena = cinta(&r_bajos);
    ajena[3..8].copy_from_slice(b"XXXX1");
    assert!(firma_de_cinta(&ajena).is_err());
}

/// Cadena tg-procedencia-2 integra para `bytes`: (recibo, evidencia del productor, indice revisado).
fn cadena(bytes: &[u8], procedencia: &str, cobertura: &str) -> (String, String, String) {
    let n = (bytes.len() - 8) / 40;
    let evidencia = format!(
        "formato=tg-evidencia-productor-1\nproductor=binance_vision_sync\ncommit_productor={}\nsha256_fuentes_productor={}\nsha256_ejecutable={}\nreceta=relativa\nconstante_r_bits={:016x}\norigen_receta=src/bin/binance_vision_sync.rs:222\ncomando=binance_vision_sync\nentrada_sha256={}\nsha256_salida={}\nbytes_salida={}\nsha256_registro_salida={}\ntransformaciones=ninguna\n",
        "ab".repeat(20),
        hex(&sha(b"fuentes")),
        hex(&sha(b"exe")),
        R_RELATIVA.to_bits(),
        hex(&sha(b"zip")),
        hex(&sha(bytes)),
        bytes.len(),
        hex(&sha(b"stdout"))
    );
    let ev_sha = hex(&sha(evidencia.as_bytes()));
    let indice = format!("formato=tg-indice-procedencia-1\nrevisado_por=prueba de contrato\nevidencia={ev_sha};binance_vision_sync;{};relativa\n", "ab".repeat(20));
    let recibo = format!(
        "formato=tg-procedencia-2\nversion_analizador=tick-tape-procedencia-2\nruta_logica=x_REAL.bin\nformato_cinta=TGMTICK1+40B_LE\nsha256_original={}\nbytes_original={}\nregistros_esperados={n}\nregistros_analizados={n}\ncobertura={cobertura}\nrecetas_compatibles=relativa\nestado_firma=relativa\nprimer_testigo_relativa=0\nprocedencia_acreditada={procedencia}\nevidencia_productor_sha256={ev_sha}\nanalisis=intervalos_exhaustivos\n",
        hex(&sha(bytes)),
        bytes.len()
    );
    (recibo, evidencia, indice)
}
#[test]
fn tgb_solo_la_relativa_acreditada_entra_a_entrenamiento() {
    // Contrato de ARCHIVOS (cinta en un temporal, por la fachada sobre TapeBytes); los PUROS estan en tgb_g_contract.
    let caso = Caso::nuevo("admision");
    let bajos = precios(0.45, 0.013, 200);
    let relativa = cinta(&bajos.iter().map(|p| rel(*p)).collect::<Vec<_>>());
    let antigua = cinta(&bajos.iter().map(|p| ant(*p)).collect::<Vec<_>>());
    let ambigua = cinta(&precios(60_000.0, 0.7, 50).iter().map(|p| rel(*p)).collect::<Vec<_>>());
    let admitir = |t: &TapeBytes, c: &(String, String, String)| {
        let ev = c.1.clone().into_bytes();
        let mut res = |_: &[u8; 32]| -> Result<Option<Vec<u8>>, String> { Ok(Some(ev.clone())) };
        let ix = IndiceFijado { bytes: c.2.as_bytes(), sha256_esperado: sha(c.2.as_bytes()) };
        admitir_para_entrenamiento(t, Some(c.0.as_bytes()), Some(ix), &mut res)
    };

    // Verde: relativa con su cadena integra.
    let t = caso.tape("verde", &relativa);
    let ok = admitir(&t, &cadena(&relativa, "relativa", "completa"));
    assert!(ok.is_ok(), "{ok:?}");
    assert_eq!(t.as_bytes(), &relativa[..], "la admitida conserva sus bytes");

    let rechazos: Vec<(&str, Vec<u8>, (String, String, String), &str)> = vec![
        ("marca de otro SHA", relativa.clone(), cadena(&antigua, "relativa", "completa"), "recibo_de_otro_original"),
        ("procedencia no acreditada", relativa.clone(), cadena(&relativa, "indeterminada", "completa"), "procedencia_no_acreditada"),
        ("analisis incompleto", relativa.clone(), cadena(&relativa, "relativa", "parcial"), "cobertura_incompleta"),
        ("firma antigua", antigua.clone(), cadena(&antigua, "relativa", "completa"), "firma_antigua"),
        ("sin testigos discriminantes", ambigua.clone(), cadena(&ambigua, "relativa", "completa"), "firma_indeterminada"),
    ];
    for (nombre, bytes, c, causa) in rechazos {
        let t = caso.tape(&nombre.replace(' ', "_"), &bytes);
        match admitir(&t, &c) {
            Ok(_) => panic!("{nombre}: admitida sin evidencia suficiente"),
            Err(e) => assert!(e.contains(causa), "{nombre}: causa no verificable: {e}"),
        }
    }
    let t = caso.tape("sin_marca", &relativa);
    let mut nada = |_: &[u8; 32]| -> Result<Option<Vec<u8>>, String> { Ok(None) };
    let e = admitir_para_entrenamiento(&t, None, None, &mut nada).err().unwrap_or_default();
    assert!(e.contains("recibo_ausente"), "marca ausente: {e}");
}