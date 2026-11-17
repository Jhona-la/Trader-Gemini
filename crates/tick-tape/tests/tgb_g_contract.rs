//! TGB-G (cierre, `pedido_tgb_g_v1.md` §4): contratos PUROS de la firma de generacion y de la admision.
//! Buffers y fixtures en memoria; sin temporales, sin reloj, sin red. Los pares bid/ask se fabrican aqui con las
//! recetas (oraculo independiente del clasificador); nunca se usa el clasificador para generar lo esperado.

use sha2::{Digest, Sha256};
use tick_tape::procedencia::{
    admitir_original, firma_de_cinta, firma_de_registros, Admision, Causa, EstadoGeneracion, Firma, IndiceFijado, Rechazo,
    R_RELATIVA, SUELO_ANTIGUO,
};
use tick_tape::tgb::{empaquetar, restituir_tgb, Presupuesto};

fn registro_q(ts: u64, bid: f64, ask: f64, bq: f64, aq: f64) -> [u8; 40] {
    let mut r = [0u8; 40];
    r[..8].copy_from_slice(&ts.to_le_bytes());
    r[8..16].copy_from_slice(&bid.to_le_bytes());
    r[16..24].copy_from_slice(&ask.to_le_bytes());
    r[24..32].copy_from_slice(&bq.to_le_bytes());
    r[32..40].copy_from_slice(&aq.to_le_bytes());
    r
}

fn registro(ts: u64, bid: f64, ask: f64) -> [u8; 40] {
    registro_q(ts, bid, ask, 1.0, 2.0)
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
        v.extend_from_slice(&registro(1_780_300_800_000 + k as u64, *b, *a));
    }
    v
}

fn precios(desde: f64, paso: f64, n: usize) -> Vec<f64> {
    (0..n).map(|k| desde + paso * k as f64).collect()
}

fn firma(bytes: &[u8]) -> Firma {
    firma_de_cinta(bytes).expect("cinta REAL bien formada")
}

/// Clase de UN registro: (solo_relativa, solo_antigua, ambas, ninguna, invalido).
fn clase(b: f64, a: f64) -> (u64, u64, u64, u64, u64) {
    let f = firma_de_registros(&registro(1, b, a)).expect("40 B");
    (f.solo_relativa, f.solo_antigua, f.ambas, f.ninguna, f.invalidos)
}

fn mismos_bits(x: (f64, f64), y: (f64, f64)) -> bool {
    x.0.to_bits() == y.0.to_bits() && x.1.to_bits() == y.1.to_bits()
}

#[test]
fn tgb_g01_recetas() {
    let bajos = precios(0.45, 0.013, 200); // por debajo del cruce 0.01 / 0.00005
    let altos = precios(60_000.0, 0.7, 200); // muy por encima: las dos recetas dan los mismos bytes
    let r_bajos = cinta(&bajos.iter().map(|p| rel(*p)).collect::<Vec<_>>());
    let a_bajos = cinta(&bajos.iter().map(|p| ant(*p)).collect::<Vec<_>>());
    let r_altos = cinta(&altos.iter().map(|p| rel(*p)).collect::<Vec<_>>());
    let copia = r_bajos.clone();

    let f = firma(&r_bajos);
    assert_eq!(f.estado(), EstadoGeneracion::Relativa);
    assert_eq!((f.registros, f.solo_relativa, f.solo_antigua, f.ambas, f.ninguna, f.invalidos, f.sin_decision), (200, 200, 0, 0, 0, 0, 0));
    assert_eq!((f.primer_solo_relativa, f.primer_excluye_antigua, f.primer_excluye_relativa), (Some(0), Some(0), None));
    assert_eq!(f.recetas_compatibles(), "relativa");

    let f = firma(&a_bajos);
    assert_eq!(f.estado(), EstadoGeneracion::Antigua);
    assert_eq!((f.solo_antigua, f.solo_relativa, f.primer_solo_antigua, f.primer_excluye_relativa), (200, 0, Some(0), Some(0)));
    assert_eq!(f.recetas_compatibles(), "antigua");

    let f = firma(&r_altos);
    assert_eq!(f.estado(), EstadoGeneracion::Indeterminada, "donde las recetas coinciden");
    assert_eq!((f.ambas, f.solo_relativa, f.solo_antigua), (200, 0, 0));
    assert_eq!(f.recetas_compatibles(), "relativa,antigua");

    assert_eq!(r_bajos, copia, "la firma no altera los bytes");
}

#[test]
fn tgb_g02_cruce_y_vecinos() {
    // Cruce exacto c/r y sus vecinos f64 a AMBOS lados, la mitad y el doble del cruce. Oraculo: si las dos recetas
    // dan los mismos bits, el registro admite las dos (nunca se inventa exclusividad); si no, cada par es exclusivo
    // de su receta.
    let cruce = SUELO_ANTIGUO / R_RELATIVA;
    let mut ps = vec![cruce, cruce * 0.5, cruce * 2.0];
    let (mut abajo, mut arriba) = (cruce, cruce);
    for _ in 0..16 {
        abajo = abajo.next_down();
        arriba = arriba.next_up();
        ps.push(abajo);
        ps.push(arriba);
    }
    // f64 extremos validos: subnormal, normal minimo, muy pequeno y muy grande.
    ps.extend([f64::from_bits(1000), f64::MIN_POSITIVE, 1e-300, 1e300]);
    let (mut identicos, mut distintos) = (0, 0);
    for p in ps {
        let (r, a) = (rel(p), ant(p));
        let valido = |x: (f64, f64)| x.0.is_finite() && x.1.is_finite() && x.0 > 0.0 && x.0 <= x.1;
        if mismos_bits(r, a) {
            identicos += 1;
            assert_eq!(clase(r.0, r.1), (0, 0, 1, 0, 0), "p={p:e}: pares identicos deben admitir las dos");
        } else {
            distintos += 1;
            assert_eq!(clase(r.0, r.1), (1, 0, 0, 0, 0), "p={p:e}: par relativo distinto del antiguo");
            if valido(a) {
                assert_eq!(clase(a.0, a.1), (0, 1, 0, 0, 0), "p={p:e}: par antiguo distinto del relativo");
            }
        }
    }
    assert!(identicos >= 10 && distintos >= 4, "control: hay casos de los dos tipos ({identicos}, {distintos})");

    // Cruces de binade: justo por debajo de 1.0 el ask pasa a la binade de arriba y el punto medio calculado a veces
    // NO es p (desde unos 11 000 f64 por debajo). El testigo debe encontrarse igual (recorrido entero del intervalo, no solo el medio).
    let mut p = 1.0f64;
    let (mut medio_distinto, mut n) = (0, 0);
    for _ in 0..30_000 {
        p = p.next_down();
        let (b, a) = rel(p);
        if (b * 0.5 + a * 0.5).to_bits() != p.to_bits() {
            medio_distinto += 1;
        }
        n += 1;
        assert_eq!(clase(b, a), (1, 0, 0, 0, 0), "p={p:e} (binade): relativa exclusiva");
    }
    assert!(medio_distinto > 0, "control: el barrido contiene medios distintos de p ({medio_distinto} de {n})");
}

#[test]
fn tgb_g07_mixta_invalida_y_medio_tick() {
    let bajos = precios(0.45, 0.013, 200);
    let r_bajos: Vec<_> = bajos.iter().map(|p| rel(*p)).collect();
    let a_bajos: Vec<_> = bajos.iter().map(|p| ant(*p)).collect();

    let mut mezcla = r_bajos.clone();
    mezcla.extend_from_slice(&a_bajos);
    let f = firma(&cinta(&mezcla));
    assert_eq!(f.estado(), EstadoGeneracion::Mixta);
    assert_eq!((f.primer_solo_relativa, f.primer_solo_antigua), (Some(0), Some(200)));
    assert_eq!(f.recetas_compatibles(), "ninguna");

    // Precios no finitos, cruzados o no positivos.
    for malo in [(f64::NAN, 1.0), (1.0, f64::INFINITY), (2.0, 1.0), (0.0, 1.0), (-1.0, 1.0)] {
        let mut v = r_bajos.clone();
        v.push(malo);
        let f = firma(&cinta(&v));
        assert_eq!(f.estado(), EstadoGeneracion::OtraOInvalida, "precio invalido {malo:?}");
        assert_eq!((f.invalidos, f.primer_invalido), (1, Some(200)), "precio invalido {malo:?}");
    }
    // Cantidades invalidas: se validan crudas, sin saneamiento, aunque el precio sea relativo exacto.
    for (bq, aq) in [(f64::NAN, 1.0), (1.0, f64::INFINITY), (-1.0, 1.0), (1.0, -0.5)] {
        let mut v = cinta(&r_bajos);
        let (b, a) = rel(0.9);
        v.extend_from_slice(&registro_q(9, b, a, bq, aq));
        let copia = v.clone();
        let f = firma(&v);
        assert_eq!(f.estado(), EstadoGeneracion::OtraOInvalida, "cantidad invalida ({bq}, {aq})");
        assert_eq!(f.primer_invalido, Some(200), "cantidad invalida ({bq}, {aq})");
        assert_eq!(v, copia, "sin saneamiento");
    }
    // Ninguna receta: la transformacion de medio tick no se atribuye a ninguna de las dos.
    let fix: Vec<_> = bajos.iter().map(|p| medio_tick(*p, 0.001)).collect();
    let f = firma(&cinta(&fix));
    assert_eq!(f.estado(), EstadoGeneracion::OtraOInvalida, "medio tick");
    assert!(f.ninguna > 0 && f.invalidos == 0, "medio tick: ninguna receta, no invalido");

    // Cinta vacia (solo cabecera): indeterminada, sin registros. Truncada: error.
    let f = firma(b"TGMTICK1");
    assert_eq!((f.estado(), f.registros), (EstadoGeneracion::Indeterminada, 0));
    let mut truncada = cinta(&r_bajos[..3]);
    truncada.pop();
    assert!(firma_de_cinta(&truncada).is_err(), "registro truncado");
}

// ───────────────────────────── cadena de procedencia (G03-G06, G08, G12) ─────────────────────────────

fn sha(b: &[u8]) -> [u8; 32] {
    Sha256::digest(b).into()
}

fn hx(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

const PRODUCTOR: &str = "binance_vision_sync";
const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const OTRO_COMMIT: &str = "89abcdef0123456789abcdef0123456789abcdef";

fn relativa_baja(n: usize, desde: f64) -> Vec<u8> {
    cinta(&precios(desde, 0.013, n).iter().map(|p| rel(*p)).collect::<Vec<_>>())
}

fn evidencia_de(original: &[u8], productor: &str, commit: &str, receta: &str) -> String {
    format!(
        "formato=tg-evidencia-productor-1\nproductor={productor}\ncommit_productor={commit}\nsha256_fuentes_productor={}\nsha256_ejecutable={}\nreceta={receta}\nconstante_r_bits={:016x}\norigen_receta=src/bin/binance_vision_sync.rs:222\ncomando=binance_vision_sync --symbol XRPUSDT --month 2026-06\nentrada_sha256={}\nentrada_sha256={}\nsha256_salida={}\nbytes_salida={}\nsha256_registro_salida={}\ntransformaciones=ninguna\n",
        hx(&sha(b"fuentes del productor")),
        hx(&sha(b"ejecutable efectivo")),
        R_RELATIVA.to_bits(),
        hx(&sha(b"zip 1")),
        hx(&sha(b"zip 2")),
        hx(&sha(original)),
        original.len(),
        hx(&sha(b"salida original del productor")),
    )
}

fn recibo_de(original: &[u8], evidencia_sha: &[u8; 32], ruta: &str) -> String {
    let n = (original.len() - 8) / 40;
    format!(
        "formato=tg-procedencia-2\nversion_analizador=tick-tape-procedencia-2\nruta_logica={ruta}\nformato_cinta=TGMTICK1+40B_LE\nsha256_original={}\nbytes_original={}\nregistros_esperados={n}\nregistros_analizados={n}\ncobertura=completa\nrecetas_compatibles=relativa\nestado_firma=relativa\nprimer_testigo_relativa=0\nprocedencia_acreditada=relativa\nevidencia_productor_sha256={}\nanalisis=intervalos_exhaustivos\n",
        hx(&sha(original)),
        original.len(),
        hx(evidencia_sha),
    )
}

fn indice_de(entradas: &[(String, &str, &str, &str)]) -> String {
    let mut s = "formato=tg-indice-procedencia-1\nrevisado_por=integrador, registro revisado 2026-11-17\n".to_string();
    for (h, p, c, r) in entradas {
        s.push_str(&format!("evidencia={h};{p};{c};{r}\n"));
    }
    s
}

/// Cambia el valor de la PRIMERA linea `clave=...` de un documento.
fn cambiar(doc: &str, clave: &str, valor: &str) -> String {
    let pref = format!("{clave}=");
    let mut hecho = false;
    let lineas: Vec<String> = doc
        .lines()
        .map(|l| {
            if !hecho && l.starts_with(&pref) {
                hecho = true;
                format!("{pref}{valor}")
            } else {
                l.to_string()
            }
        })
        .collect();
    assert!(hecho, "el fixture no tiene {clave}");
    lineas.join("\n") + "\n"
}

#[derive(Clone)]
struct Cadena {
    original: Vec<u8>,
    recibo: String,
    evidencia: String,
    indice: String,
    pin: [u8; 32],
}

impl Cadena {
    /// Cadena integra: evidencia del productor, indice revisado que la reconoce y recibo que la cita.
    fn nueva(original: Vec<u8>) -> Cadena {
        let ev = evidencia_de(&original, PRODUCTOR, COMMIT, "relativa");
        Cadena::con_evidencia(original, ev, (PRODUCTOR, COMMIT, "relativa"))
    }

    fn con_evidencia(original: Vec<u8>, evidencia: String, entrada: (&str, &str, &str)) -> Cadena {
        let h = sha(evidencia.as_bytes());
        let indice = indice_de(&[(hx(&h), entrada.0, entrada.1, entrada.2)]);
        let recibo = recibo_de(&original, &h, "data/XRPUSDT_2026-06_REAL.bin");
        let pin = sha(indice.as_bytes());
        Cadena { original, recibo, evidencia, indice, pin }
    }

    fn admitir(&self) -> Result<Admision, Rechazo> {
        let ev = self.evidencia.clone().into_bytes();
        let mut resolver = |h: &[u8; 32]| -> Result<Option<Vec<u8>>, String> { Ok((*h == sha(&ev)).then(|| ev.clone())) };
        admitir_original(
            &self.original,
            Some(self.recibo.as_bytes()),
            Some(IndiceFijado { bytes: self.indice.as_bytes(), sha256_esperado: self.pin }),
            &mut resolver,
        )
    }
}

fn rechaza(r: Result<Admision, Rechazo>, causa: Causa, caso: &str) -> Rechazo {
    match r {
        Ok(a) => panic!("{caso}: ADMITIDA sin evidencia suficiente (sha {})", hx(&a.sha256_original)),
        Err(e) => {
            assert_eq!(e.causa, causa, "{caso}: causa distinta ({e})");
            e
        }
    }
}

#[test]
fn tgb_g03_rechaza_autodeclaracion() {
    let original = relativa_baja(200, 0.45);
    let ok = Cadena::nueva(original.clone());

    // Evidencia de texto libre ("prueba"), aunque el indice la reconozca por su hash: no es una cadena.
    let libre = Cadena::con_evidencia(original.clone(), "prueba".to_string(), (PRODUCTOR, COMMIT, "relativa"));
    let e = rechaza(libre.admitir(), Causa::EvidenciaInvalida, "evidencia de texto libre");
    assert!(e.to_string().starts_with("procedencia_no_acreditada"), "{e}");

    // El recibo de H (tg-procedencia-1, evidencia_productor=prueba) ya no acredita nada.
    let h = format!(
        "formato=tg-procedencia-1\nsha256_original={}\nbytes_original={}\nprocedencia_acreditada=relativa\nevidencia_productor=prueba\ncobertura=completa\n",
        hx(&sha(&original)),
        original.len()
    );
    let mut sin_ev = |_: &[u8; 32]| -> Result<Option<Vec<u8>>, String> { Ok(None) };
    let r = admitir_original(&original, Some(h.as_bytes()), Some(IndiceFijado { bytes: ok.indice.as_bytes(), sha256_esperado: ok.pin }), &mut sin_ev);
    rechaza(r, Causa::ReciboInvalido, "recibo de H");

    // Recibo integro sin indice revisado: no acreditado.
    let ev = ok.evidencia.clone().into_bytes();
    let mut res = |_: &[u8; 32]| -> Result<Option<Vec<u8>>, String> { Ok(Some(ev.clone())) };
    let e = rechaza(admitir_original(&original, Some(ok.recibo.as_bytes()), None, &mut res), Causa::IndiceAusente, "sin indice");
    assert!(e.causa.es_procedencia_no_acreditada());

    // Recibo que declara indeterminada/antigua como procedencia: no acreditada.
    let mut c = ok.clone();
    c.recibo = cambiar(&c.recibo, "procedencia_acreditada", "declarada");
    rechaza(c.admitir(), Causa::ProcedenciaNoAcreditada, "procedencia declarada");

    // Solo la cadena integra y reconocida acredita.
    let a = ok.admitir().expect("cadena integra");
    assert_eq!(a.sha256_original, sha(&original));
    assert_eq!(a.bytes_original, original.len() as u64);
    assert_eq!(a.evidencia_sha256, sha(ok.evidencia.as_bytes()));
    assert_eq!(a.sha256_indice, ok.pin);
    assert_eq!(a.firma.estado(), EstadoGeneracion::Relativa);
    assert_eq!(a.revisado_por, "integrador, registro revisado 2026-11-17");
}

#[test]
fn tgb_g04_cadena_rota() {
    let original = relativa_baja(200, 0.45);
    let otro = relativa_baja(200, 0.46);
    let ok = Cadena::nueva(original.clone());
    assert!(ok.admitir().is_ok(), "control positivo: {:?}", ok.admitir().err());

    // La evidencia citada no esta en el indice revisado.
    let mut c = ok.clone();
    c.indice = indice_de(&[(hx(&sha(b"otra evidencia")), PRODUCTOR, COMMIT, "relativa")]);
    c.pin = sha(c.indice.as_bytes());
    rechaza(c.admitir(), Causa::EvidenciaNoReconocida, "evidencia fuera del indice");

    // Evidencia ausente, o el resolutor devuelve otros bytes que los citados.
    let fijo = IndiceFijado { bytes: ok.indice.as_bytes(), sha256_esperado: ok.pin };
    let mut nada = |_: &[u8; 32]| -> Result<Option<Vec<u8>>, String> { Ok(None) };
    rechaza(admitir_original(&original, Some(ok.recibo.as_bytes()), Some(fijo), &mut nada), Causa::EvidenciaAusente, "evidencia ausente");
    // Evidencia valida de ESTE original pero con otros bytes (otro comando): no es la citada por el recibo.
    let falsa = cambiar(&evidencia_de(&original, PRODUCTOR, COMMIT, "relativa"), "comando", "otro comando").into_bytes();
    let mut otra = |_: &[u8; 32]| -> Result<Option<Vec<u8>>, String> { Ok(Some(falsa.clone())) };
    rechaza(admitir_original(&original, Some(ok.recibo.as_bytes()), Some(fijo), &mut otra), Causa::CadenaRota, "hash de evidencia distinto");

    // Indice sin pin o ajeno: SHA esperado distinto, o indice alterado con el pin del original.
    let ajeno = IndiceFijado { bytes: ok.indice.as_bytes(), sha256_esperado: sha(b"otro registro revisado") };
    let ev = ok.evidencia.clone().into_bytes();
    let mut res = |_: &[u8; 32]| -> Result<Option<Vec<u8>>, String> { Ok(Some(ev.clone())) };
    rechaza(admitir_original(&original, Some(ok.recibo.as_bytes()), Some(ajeno), &mut res), Causa::IndiceNoFijado, "pin ajeno");
    let mut c = ok.clone();
    c.indice.push_str(&format!("evidencia={};{PRODUCTOR};{COMMIT};relativa\n", hx(&sha(b"colada"))));
    rechaza(c.admitir(), Causa::IndiceNoFijado, "indice alterado bajo el mismo pin");

    // Cada rotura de la evidencia, por separado (el indice reconoce la evidencia rota por su hash).
    let roturas: Vec<(&str, String, (&str, &str, &str))> = vec![
        ("salida de otro original", evidencia_de(&otro, PRODUCTOR, COMMIT, "relativa"), (PRODUCTOR, COMMIT, "relativa")),
        ("tamano de salida", cambiar(&evidencia_de(&original, PRODUCTOR, COMMIT, "relativa"), "bytes_salida", "9"), (PRODUCTOR, COMMIT, "relativa")),
        ("productor distinto del indice", evidencia_de(&original, "tape_spread_fix", COMMIT, "relativa"), (PRODUCTOR, COMMIT, "relativa")),
        ("commit distinto del indice", evidencia_de(&original, PRODUCTOR, OTRO_COMMIT, "relativa"), (PRODUCTOR, COMMIT, "relativa")),
        ("receta antigua", evidencia_de(&original, PRODUCTOR, COMMIT, "antigua"), (PRODUCTOR, COMMIT, "antigua")),
        ("constante distinta", cambiar(&evidencia_de(&original, PRODUCTOR, COMMIT, "relativa"), "constante_r_bits", &format!("{:016x}", 0.0001f64.to_bits())), (PRODUCTOR, COMMIT, "relativa")),
        ("transformacion intermedia", cambiar(&evidencia_de(&original, PRODUCTOR, COMMIT, "relativa"), "transformaciones", "tape_spread_fix"), (PRODUCTOR, COMMIT, "relativa")),
    ];
    for (caso, ev, entrada) in roturas {
        let c = Cadena::con_evidencia(original.clone(), ev, entrada);
        rechaza(c.admitir(), Causa::CadenaRota, caso);
    }
}

#[test]
fn tgb_g05_identidad_original() {
    let original = relativa_baja(200, 0.45);
    let ok = Cadena::nueva(original.clone());
    let a = ok.admitir().expect("control positivo");

    let mut c = ok.clone();
    c.recibo = cambiar(&c.recibo, "sha256_original", &hx(&sha(b"otro")));
    rechaza(c.admitir(), Causa::OtroOriginal, "solo el hash");
    let mut c = ok.clone();
    c.recibo = cambiar(&c.recibo, "bytes_original", &(original.len() + 40).to_string());
    rechaza(c.admitir(), Causa::OtroOriginal, "solo el tamano");

    // Alias: otro nombre logico con el mismo hash no cambia la generacion de los mismos bytes.
    let mut c = ok.clone();
    c.recibo = cambiar(&c.recibo, "ruta_logica", "data/alias_REAL.bin");
    let b = c.admitir().expect("alias");
    assert_eq!((b.sha256_original, b.firma), (a.sha256_original, a.firma));

    // Un recibo valido de un original no aplica a otro original (tambien relativo).
    let mut c = Cadena::nueva(relativa_baja(200, 0.46));
    c.original = original;
    rechaza(c.admitir(), Causa::OtroOriginal, "recibo antiguo sobre original nuevo");
}

#[test]
fn tgb_g06_cobertura_y_cola() {
    let original = relativa_baja(200, 0.45);
    let ok = Cadena::nueva(original.clone());

    let mut c = ok.clone();
    c.recibo = cambiar(&c.recibo, "cobertura", "parcial");
    rechaza(c.admitir(), Causa::CoberturaIncompleta, "cobertura parcial");
    for (clave, valor) in [("registros_esperados", "199"), ("registros_analizados", "100"), ("registros_esperados", "201")] {
        let mut c = ok.clone();
        c.recibo = cambiar(&c.recibo, clave, valor);
        rechaza(c.admitir(), Causa::CuentasDiscordantes, &format!("{clave}={valor}"));
    }
    for (clave, valor) in [("primer_testigo_relativa", "7"), ("estado_firma", "indeterminada"), ("recetas_compatibles", "relativa,antigua")] {
        let mut c = ok.clone();
        c.recibo = cambiar(&c.recibo, clave, valor);
        rechaza(c.admitir(), Causa::FirmaDiscordante, &format!("{clave}={valor}"));
    }

    // Testigo contrario al FINAL del original: se analiza todo; un prefijo relativo no admite la cinta.
    let mut cola = original.clone();
    let (b, a) = ant(0.9);
    cola.extend_from_slice(&registro(9_999, b, a));
    let mut c = Cadena::nueva(cola.clone());
    c.recibo = cambiar(&c.recibo, "registros_esperados", "201");
    c.recibo = cambiar(&c.recibo, "registros_analizados", "201");
    rechaza(c.admitir(), Causa::Firma(EstadoGeneracion::Mixta), "mezcla tardia");
    assert_eq!(firma_de_cinta(&cola).expect("cinta").primer_solo_antigua, Some(200));
}

#[test]
fn tgb_g08_cabecera_y_recibo() {
    let original = relativa_baja(200, 0.45);
    let ok = Cadena::nueva(original.clone());

    // Prefijo TGM con resto ajeno: no es una cinta REAL.
    let mut c = ok.clone();
    c.original[3..8].copy_from_slice(b"XXXX1");
    rechaza(c.admitir(), Causa::Formato, "cabecera TGMXXXX1");

    let sha_ok = hx(&sha(&original));
    let malos: Vec<(&str, String)> = vec![
        ("clave duplicada", ok.recibo.clone() + "cobertura=completa\n"),
        ("clave desconocida", ok.recibo.clone() + "verified=true\n"),
        ("clave ausente", ok.recibo.replace("cobertura=completa\n", "")),
        ("sha en mayusculas", cambiar(&ok.recibo, "sha256_original", &sha_ok.to_uppercase())),
        ("sha con signo", cambiar(&ok.recibo, "sha256_original", &format!("+{}", &sha_ok[1..]))),
        ("sha corto", cambiar(&ok.recibo, "sha256_original", &sha_ok[..63])),
        ("decimal con signo", cambiar(&ok.recibo, "bytes_original", &format!("+{}", original.len()))),
        ("decimal con cero", cambiar(&ok.recibo, "bytes_original", &format!("0{}", original.len()))),
        ("version no acreditable", cambiar(&ok.recibo, "version_analizador", "tick-tape-procedencia-1")),
        ("formato viejo", cambiar(&ok.recibo, "formato", "tg-procedencia-1")),
        ("metodo de ventana", cambiar(&ok.recibo, "analisis", "ventana_16_ulp")),
        ("retorno de carro", ok.recibo.replace('\n', "\r\n")),
        ("espacios", ok.recibo.replace("cobertura=completa", "cobertura = completa")),
        ("comentario", "# recibo\n".to_string() + &ok.recibo),
    ];
    for (caso, recibo) in malos {
        let mut c = ok.clone();
        c.recibo = recibo;
        rechaza(c.admitir(), Causa::ReciboInvalido, caso);
    }

    // Evidencia e indice tambien estrictos.
    let ev = ok.evidencia.clone() + &format!("entrada_sha256={}\n", hx(&sha(b"zip 1")));
    let c = Cadena::con_evidencia(original.clone(), ev, (PRODUCTOR, COMMIT, "relativa"));
    rechaza(c.admitir(), Causa::EvidenciaInvalida, "entrada repetida");
    let mut c = ok.clone();
    let linea = c.indice.lines().last().expect("entrada").to_string();
    c.indice = c.indice.clone() + &linea + "\n";
    c.pin = sha(c.indice.as_bytes());
    rechaza(c.admitir(), Causa::IndiceInvalido, "evidencia repetida en el indice");
}

#[test]
fn tgb_g12_bin_tgb_equivalentes() {
    let original = relativa_baja(300, 0.45);
    let ok = Cadena::nueva(original.clone());
    let directo = ok.admitir().expect("desde el .bin");

    // El mismo original restituido de su contenedor .tgb: misma firma, misma decision, mismo SHA del ORIGINAL.
    let contenedor = empaquetar(&original).expect("empaquetar");
    let restituido = restituir_tgb(&contenedor, Presupuesto::sin_tope()).expect("restituir");
    assert_eq!(restituido.as_bytes(), &original[..]);
    let mut c = ok.clone();
    c.original = restituido.as_bytes().to_vec();
    let via_tgb = c.admitir().expect("desde el .tgb");
    assert_eq!((via_tgb.sha256_original, via_tgb.bytes_original, via_tgb.firma), (directo.sha256_original, directo.bytes_original, directo.firma));
    assert_ne!(sha(&contenedor), directo.sha256_original, "el SHA del transporte va aparte");

    // La identidad del transporte no sustituye a la del original.
    let mut c = ok.clone();
    c.recibo = cambiar(&c.recibo, "sha256_original", &hx(&sha(&contenedor)));
    c.recibo = cambiar(&c.recibo, "bytes_original", &contenedor.len().to_string());
    rechaza(c.admitir(), Causa::OtroOriginal, "SHA de transporte");

    // Un contenedor corrupto no llega a la admision.
    let mut roto = contenedor.clone();
    let medio = roto.len() / 2;
    roto[medio] ^= 0x40;
    let mut admisiones = 0;
    if let Ok(r) = restituir_tgb(&roto, Presupuesto::sin_tope()) {
        admisiones += 1;
        let _ = r;
    }
    assert_eq!(admisiones, 0, "contenedor corrupto restituido");
}
