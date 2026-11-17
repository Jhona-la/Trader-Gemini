//! TGB-G: firma de generacion y admision para entrenamiento (pedido CODEX-CLI, Parte 4, "Pruebas futuras de la
//! marca"). Solo la cinta relativa con recibo de procedencia e identidad vigentes llega a la preparacion de
//! muestras; los demas casos devuelven una causa verificable.

mod comun;

use comun::{hex, sha};
use tick_tape::logica::{open_logical_tape, TapeBytes};
use tick_tape::procedencia::{admitir_para_entrenamiento, firma_de_cinta, EstadoGeneracion, R_RELATIVA, SUELO_ANTIGUO};
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

fn recibo(bytes: &[u8], procedencia: &str, cobertura: &str) -> String {
    format!(
        "formato=tg-procedencia-1\nsha256_original={}\nbytes_original={}\nprocedencia_acreditada={procedencia}\nevidencia_productor=binance_vision_sync D-723, fuentes data.binance.vision por SHA\ncobertura={cobertura}\n",
        hex(&sha(bytes)),
        bytes.len()
    )
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

#[test]
fn tgb_solo_la_relativa_acreditada_entra_a_entrenamiento() {
    let caso = Caso::nuevo("admision");
    let bajos = precios(0.45, 0.013, 200);
    let relativa = cinta(&bajos.iter().map(|p| rel(*p)).collect::<Vec<_>>());
    let antigua = cinta(&bajos.iter().map(|p| ant(*p)).collect::<Vec<_>>());
    let ambigua = cinta(&precios(60_000.0, 0.7, 50).iter().map(|p| rel(*p)).collect::<Vec<_>>());

    // Verde: relativa con recibo de su SHA, procedencia acreditada y cobertura completa.
    let t = caso.tape("verde", &relativa);
    let ok = admitir_para_entrenamiento(&t, Some(&recibo(&relativa, "relativa", "completa")));
    assert!(ok.is_ok(), "{ok:?}");
    assert_eq!(t.as_bytes(), &relativa[..], "la admitida conserva sus bytes");

    let rechazos: Vec<(&str, Vec<u8>, Option<String>, &str)> = vec![
        ("marca ausente", relativa.clone(), None, "ausente"),
        ("marca de otro SHA", relativa.clone(), Some(recibo(&antigua, "relativa", "completa")), "otro original"),
        ("procedencia no acreditada", relativa.clone(), Some(recibo(&relativa, "indeterminada", "completa")), "no acreditada"),
        ("analisis incompleto", relativa.clone(), Some(recibo(&relativa, "relativa", "parcial")), "incompleto"),
        ("firma antigua", antigua.clone(), Some(recibo(&antigua, "relativa", "completa")), "antigua"),
        ("sin testigos discriminantes", ambigua.clone(), Some(recibo(&ambigua, "relativa", "completa")), "indeterminada"),
    ];
    for (nombre, bytes, rec, causa) in rechazos {
        let t = caso.tape(&nombre.replace(' ', "_"), &bytes);
        match admitir_para_entrenamiento(&t, rec.as_deref()) {
            Ok(_) => panic!("{nombre}: admitida sin evidencia suficiente"),
            Err(e) => assert!(e.contains(causa), "{nombre}: causa no verificable: {e}"),
        }
    }
}
