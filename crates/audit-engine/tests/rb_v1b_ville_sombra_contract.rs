//! CONTRATO RB-V1b (Codex RB, 2026-11-10): el veredicto del libro en sombra es
//! válido en CUALQUIER momento (e-proceso y desigualdad maximal de Ville).
//!
//! QUÉ: cada intención vetada resuelve un r neto de fricción en
//! `[−(sl+f)/sl, (tp−f)/sl]`, de ancho `(tp+sl)/sl` fijado al registrarla. Con
//! `x = r/ancho` (ancho 1), el libro acumula por fuente la riqueza de una
//! rejilla de apuestas contra «lo vetado no paga» y contra «lo vetado no
//! pierde»: `M = (1/J)·Σ_j Π_i exp(±u_j·x_i − u_j²/8)`. Bajo cada hipótesis
//! nula, `M` es una supermartingala no negativa con `M_0 = 1`, y Ville da
//! `P(∃n: M_n ≥ 1/α) ≤ α`.
//! POR QUÉ: el núcleo vuelve a evaluar el veredicto en cada resolución y lo
//! anuncia en cuanto cruza. El veredicto de Hoeffding de RB-V1a valía para un
//! n fijado de antemano: mirando en cada n, su error real supera el nominal
//! (prueba 4, que lo mide en la misma simulación).
//! CÓMO: (1) forma cerrada de la evidencia y de la rejilla; (2) veredictos
//! sobre poblaciones conocidas; (3) potencia: un veto que sólo bloquea
//! perdedoras se certifica en el n que predice la forma cerrada; (4) un juego
//! justo con geometrías distintas, mirando en cada resolución: Ville respeta
//! su nivel y Hoeffding no.

use audit_engine::shadow_ledger::{
    EstadisticaFuente, IntencionVetada, LibroSombra, Veredicto, APUESTAS, REJILLA_APUESTAS,
};

const T0: u64 = 1_000_000;
const TP: f64 = 0.010;
const SL: f64 = 0.005;
const FRICCION: f64 = 0.001;
/// α por cola de producción: 0,05 repartido por Bonferroni entre las 11 fuentes
/// del núcleo (`god_engine_core::sombra_vetos::FUENTES`) y sus dos colas.
const ALFA_PRODUCCION: f64 = 0.05 / 22.0;
/// α por cola de RB-V1a (3 fuentes): la configuración en la que se usó Hoeffding.
const ALFA_RB_V1A: f64 = 0.05 / 6.0;

fn intencion(tp: f64, sl: f64, friccion: f64, t0_ms: u64) -> IntencionVetada {
    IntencionVetada {
        fuente: 0,
        moneda: 0,
        es_largo: true,
        entrada: 100.0,
        tp_pct: tp,
        sl_pct: sl,
        friccion_pct: friccion,
        tau_ms: 60_000,
        t0_ms,
        promesa_r: None,
    }
}

/// Registra la intención y la resuelve con el precio `precio` en el instante
/// `t0 + espera`.
fn resolver_con(l: &mut LibroSombra, i: IntencionVetada, precio: f64, espera: u64) {
    assert!(l.registrar(i));
    assert_eq!(l.observar(i.moneda, precio, i.t0_ms + espera), 1);
}

fn resolver(l: &mut LibroSombra, fuente: u8, k: u64, gana: bool) {
    let i = IntencionVetada {
        fuente,
        ..intencion(TP, SL, FRICCION, T0 + 10 * k)
    };
    resolver_con(l, i, if gana { 101.0 } else { 99.0 }, 1);
}

fn estadistica(l: &LibroSombra, fuente: u8) -> EstadisticaFuente {
    match l.estadistica(fuente) {
        Some(s) => *s,
        None => panic!("la fuente {fuente} existe"),
    }
}

/// `ln((1/J)·Σ_j exp(Σ_i (s·u_j·x_i − u_j²/8)))`, sumando sin trucos.
fn evidencia_directa(xs: &[f64], signo: f64) -> f64 {
    let suma: f64 = REJILLA_APUESTAS
        .iter()
        .map(|&u| {
            xs.iter()
                .map(|&x| signo * u * x - u * u / 8.0)
                .sum::<f64>()
                .exp()
        })
        .sum();
    (suma / APUESTAS as f64).ln()
}

/// xorshift64* determinista: la prueba no depende de un generador externo.
struct Xorshift(u64);
impl Xorshift {
    fn uniforme(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn rb_v1b_la_evidencia_tiene_forma_cerrada() {
    assert_eq!(
        REJILLA_APUESTAS,
        [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    );
    assert_eq!(
        estadistica(&LibroSombra::new(), 0).log_evidencia(),
        (0.0, 0.0),
        "sin resoluciones la riqueza es 1: ninguna evidencia"
    );

    // TP: r = (0,010 − 0,001)/0,005 = 1,8 → x = 0,6; SL: r = −1,2 → x = −0,4;
    // vence a 100,2: r = (0,002 − 0,001)/0,005 = 0,2 → x = 0,2/3.
    let mut l = LibroSombra::new();
    resolver_con(&mut l, intencion(TP, SL, FRICCION, T0), 101.0, 1);
    resolver_con(&mut l, intencion(TP, SL, FRICCION, T0 + 10), 99.0, 1);
    resolver_con(&mut l, intencion(TP, SL, FRICCION, T0 + 20), 101.0, 1);
    resolver_con(&mut l, intencion(TP, SL, FRICCION, T0 + 30), 100.2, 60_000);
    let s = estadistica(&l, 0);
    assert_eq!((s.n, s.tp, s.sl, s.vencidas), (4, 2, 1, 1));
    let xs = [0.6, -0.4, 0.6, 0.2 / 3.0];
    let (paga, pierde) = s.log_evidencia();
    let (paga_esperada, pierde_esperada) =
        (evidencia_directa(&xs, 1.0), evidencia_directa(&xs, -1.0));
    println!("observado: ln M⁺ {paga:.12} frente a {paga_esperada:.12}; ln M⁻ {pierde:.12} frente a {pierde_esperada:.12}");
    assert!((paga - paga_esperada).abs() < 1e-12);
    assert!((pierde - pierde_esperada).abs() < 1e-12);

    for malo in [0.0, 1.0, -0.1, f64::NAN, f64::INFINITY] {
        assert_eq!(
            s.veredicto_ville(malo),
            Veredicto::SinEvidencia,
            "α = {malo}"
        );
    }
}

#[test]
fn rb_v1b_veredicto_sobre_poblaciones_conocidas() {
    let mut l = LibroSombra::new();
    for k in 0..60 {
        resolver(&mut l, 5, k, false); // todo SL: −1,2 R
        resolver(&mut l, 6, k, true); // todo TP: +1,8 R
        resolver(&mut l, 7, k, k % 2 == 0); // mitad y mitad: +0,3 R
    }
    let v = |f: u8| estadistica(&l, f).veredicto_ville(ALFA_PRODUCCION);
    assert_eq!(v(5), Veredicto::AhorraDinero);
    assert_eq!(v(6), Veredicto::BloqueaGanadoras);
    // x̄ = 0,1: crecimiento óptimo 2·x̄² = 0,02 por resolución; con n = 60 la
    // evidencia (≈ e^1,2 menos ln 8) queda muy lejos del umbral e^6,1.
    assert_eq!(v(7), Veredicto::SinEvidencia);
}

#[test]
fn rb_v1b_un_veto_que_solo_bloquea_perdedoras_se_certifica_en_el_n_teorico() {
    // Todo SL: x = −0,4 en cada resolución; ln M⁻_n = ln((1/J)·Σ_j e^(n·g_j))
    // con g_j = 0,4·u_j − u_j²/8. El primer n que alcanza ln(1/α):
    let umbral = (1.0 / ALFA_PRODUCCION).ln();
    let ln_m = |n: f64| {
        let suma: f64 = REJILLA_APUESTAS
            .iter()
            .map(|&u| (n * (0.4 * u - u * u / 8.0)).exp())
            .sum();
        (suma / APUESTAS as f64).ln()
    };
    let Some(n_teorico) = (1..200u64).find(|&n| ln_m(n as f64) >= umbral) else {
        panic!("con 200 resoluciones todo SL el veredicto llega");
    };
    let mut l = LibroSombra::new();
    let mut certificado_en = None;
    for k in 0..200 {
        resolver(&mut l, 9, k, false);
        if certificado_en.is_none()
            && estadistica(&l, 9).veredicto_ville(ALFA_PRODUCCION) == Veredicto::AhorraDinero
        {
            certificado_en = Some(k + 1);
        }
    }
    println!("observado: certificado en n = {certificado_en:?}; forma cerrada n = {n_teorico}");
    assert_eq!(certificado_en, Some(n_teorico));
}

/// Juego justo con geometrías distintas: SL = 1 %, TP ∈ {0,5 %, 1 %, 2 %}, sin
/// fricción, y p(TP) = sl/(tp+sl), así que E[r] = 0 en CADA intención.
/// Devuelve `(tp, gana)`.
fn juego_justo(rng: &mut Xorshift) -> (f64, bool) {
    let tp = [0.005, 0.01, 0.02][(rng.uniforme() * 3.0) as usize % 3];
    (tp, rng.uniforme() < 0.01 / (tp + 0.01))
}

/// Garantía por réplica: ≤ 2α (dos colas). Margen de 3σ binomial.
fn cota_binomial(alfa: f64, replicas: u32) -> f64 {
    2.0 * alfa + 3.0 * (2.0 * alfa * (1.0 - 2.0 * alfa) / f64::from(replicas)).sqrt()
}

#[test]
fn rb_v1b_ville_respeta_su_nivel_mirando_en_cada_resolucion() {
    // El libro real, mirando el veredicto en cada resolución como el núcleo.
    let replicas = 1_500u32;
    let pasos = 1_000u64;
    let mut rng = Xorshift(0x9E37_79B9_7F4A_7C15);
    let (mut ville_rb_v1a, mut ville_produccion) = (0u32, 0u32);
    for _ in 0..replicas {
        let mut l = LibroSombra::new();
        let (mut va, mut vp) = (false, false);
        for k in 0..pasos {
            let (tp, gana) = juego_justo(&mut rng);
            let i = intencion(tp, 0.01, 0.0, T0 + 10 * k);
            // Más allá de cada barrera (el libro cobra la barrera, no el
            // exceso): 100·(1 + tp) puede quedar una ulp por debajo del TP.
            let precio = if gana { 100.0 * (1.0 + 1.5 * tp) } else { 98.5 };
            resolver_con(&mut l, i, precio, 1);
            let s = estadistica(&l, 0);
            va |= s.veredicto_ville(ALFA_RB_V1A) != Veredicto::SinEvidencia;
            vp |= s.veredicto_ville(ALFA_PRODUCCION) != Veredicto::SinEvidencia;
        }
        ville_rb_v1a += u32::from(va);
        ville_produccion += u32::from(vp);
    }
    let tasa = |c: u32| f64::from(c) / f64::from(replicas);
    println!(
        "observado ({replicas} réplicas × {pasos} miradas): algún veredicto falso — Ville {:.4} (≤ 2α = {:.4}); en producción {:.4} (≤ {:.4})",
        tasa(ville_rb_v1a),
        2.0 * ALFA_RB_V1A,
        tasa(ville_produccion),
        2.0 * ALFA_PRODUCCION
    );
    assert!(tasa(ville_rb_v1a) <= cota_binomial(ALFA_RB_V1A, replicas));
    assert!(tasa(ville_produccion) <= cota_binomial(ALFA_PRODUCCION, replicas));
}

#[test]
fn rb_v1b_el_radio_de_hoeffding_de_n_fijo_se_infla_al_mirar_en_cada_n() {
    // Por qué se sustituyó el veredicto de RB-V1a: el mismo juego justo, con el
    // radio de Hoeffding √(ln(1/δ)·Σ ancho²/2)/n evaluado en cada n (sólo
    // aritmética: no depende del libro). Hay veredicto falso si |media| supera
    // el radio en ALGÚN n, es decir, si |Σ r| > √(ln(1/δ)·Σ ancho²/2).
    let replicas = 4_000u32;
    let pasos = 4_000u32;
    let k = (1.0 / ALFA_RB_V1A).ln() / 2.0;
    let mut rng = Xorshift(0xD1B5_4A32_D192_ED03);
    let mut falsos = 0u32;
    for _ in 0..replicas {
        let (mut suma_r, mut suma_ancho2) = (0.0f64, 0.0f64);
        for _ in 0..pasos {
            let (tp, gana) = juego_justo(&mut rng);
            suma_r += if gana { tp / 0.01 } else { -1.0 };
            let ancho = (tp + 0.01) / 0.01;
            suma_ancho2 += ancho * ancho;
            if suma_r.abs() > (k * suma_ancho2).sqrt() {
                falsos += 1;
                break;
            }
        }
    }
    let tasa = f64::from(falsos) / f64::from(replicas);
    let sigma = (tasa * (1.0 - tasa) / f64::from(replicas)).sqrt();
    println!(
        "observado ({replicas} réplicas × {pasos} miradas): Hoeffding de n fijo se equivoca en {tasa:.4} ± {sigma:.4} frente a 2δ = {:.4} nominal",
        2.0 * ALFA_RB_V1A
    );
    assert!(
        tasa - 3.0 * sigma > 2.0 * ALFA_RB_V1A,
        "la inflación es significativa a 3σ: el veredicto de n fijo no vale mirando en cada n"
    );
}
