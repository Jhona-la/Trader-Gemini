//! C-F2.6 (Línea C): «cuántico» y analogías físicas de la sombra espectral
//! (`FeynmanPropagatorEngine`, `NavierStokesManifoldEngine`) contra
//! soluciones conocidas.
//!
//! Teoría que se prueba. (1) Feynman: con fasores idénticos la coherencia
//! C = |Σ A·e^{iθ}|²/(ΣA²·N) vale 1 y el voto es signo(z)·tanh(|z|/2); el
//! espejo z → −z desplaza todas las fases π y deja C igual (la acción es
//! par). Con fases i.i.d. uniformes, E[N·C] = 1 exacto y la cola es la de
//! Rayleigh, P(N·C > x) ≈ e^{−x}: el corte 0,20 con N = 32 es N·C > 6,4.
//! Su documentación promete abstención (C ≤ 0,20) ante fases estocásticas;
//! una caminata aleatoria no tiene estructura direccional, así que el filtro
//! debería abstenerse casi siempre. (2) Navier-Stokes: el «flujo de cascada»
//! Σ (E_{k+1} − E_k) es telescópico (= E_31 − E_0), y la «energía espectral»
//! E_k = ½z_k² debería crecer con la energía inyectada (σ²) y decrecer con la
//! frecuencia, como cualquier espectro físico (Browniano o K41).
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`. Sin
//! dependencias nuevas: LCG propio y Box-Muller.

use quantum_arena::temporal_spectrum::{TemporalSpectrum, SPECTRUM_SCALES_MS};
use signal_engine::feynman_propagator::FeynmanPropagatorEngine;
use signal_engine::navier_stokes_manifold::NavierStokesManifoldEngine;
use signal_engine::voto_espectral::ESCALAS_VOTO;
use std::f64::consts::PI;

/// LCG de 64 bits (constantes de Knuth/MMIX); se usan los 53 bits altos.
struct Lcg(u64);

impl Lcg {
    fn siguiente(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn unif(&mut self) -> f64 {
        ((self.siguiente() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Box-Muller (una normal por par de uniformes).
    fn normal(&mut self) -> f64 {
        let (a, b) = (self.unif(), self.unif());
        (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos()
    }

    fn expo(&mut self, media: f64) -> f64 {
        -media * self.unif().ln()
    }
}

/// Malla como la arma el núcleo (god-engine-core/src/lib.rs, sombra
/// espectral: momentum_z recortado a ±10, no finito ⇒ 0).
fn malla(spec: &TemporalSpectrum) -> [f64; ESCALAS_VOTO] {
    let mut z = [0.0; ESCALAS_VOTO];
    for k in 0..ESCALAS_VOTO {
        let m = spec.scales[k].momentum_z;
        z[k] = if m.is_finite() { m.clamp(-10.0, 10.0) } else { 0.0 };
    }
    z
}

/// Caminata aleatoria geométrica BROWNIANA en tiempo físico (corrección del
/// escéptico de F2.6): ln p += σ·√Δt_s·N(0,1), con σ en unidades de 1/√s,
/// llegadas Poisson de media `media_ms` y sello entero en ms (Δt ≥ 1 ms).
/// `visita(i, espectro, ms transcurridos desde el primer evento)`.
fn caminata(
    semilla: u64,
    sigma_raiz_s: f64,
    media_ms: f64,
    n: usize,
    mut visita: impl FnMut(usize, &TemporalSpectrum, f64),
) {
    let mut rng = Lcg(semilla);
    let mut spec = TemporalSpectrum::new();
    let t0: u64 = 1_700_000_000_000;
    let mut ts = t0;
    let mut lnp = 100f64.ln();
    for i in 0..n {
        let dt = (rng.expo(media_ms).round() as u64).max(1);
        ts += dt;
        lnp += sigma_raiz_s * (dt as f64 / 1_000.0).sqrt() * rng.normal();
        spec.update(lnp.exp(), ts);
        visita(i, &spec, (ts - t0) as f64);
    }
}

/// Media y error estándar por medias de bloques consecutivos de `largo`.
fn media_y_ee_bloques(x: &[f64], largo: usize) -> (f64, f64, usize) {
    let medias: Vec<f64> = x
        .chunks(largo)
        .filter(|c| c.len() == largo)
        .map(|c| c.iter().sum::<f64>() / largo as f64)
        .collect();
    let nb = medias.len();
    let total = x.iter().sum::<f64>() / x.len() as f64;
    let mb = medias.iter().sum::<f64>() / nb as f64;
    let var = medias.iter().map(|m| (m - mb) * (m - mb)).sum::<f64>() / (nb as f64 - 1.0);
    (total, (var / nb as f64).sqrt(), nb)
}

/// Fasores idénticos (z constante ±c): C = 1 y voto_k = signo(c)·tanh(|c|/2);
/// espejo de 2 000 mallas z_k = 3·N(0,1): C(z) = C(−z) y v(−z) = −v(z).
/// Verdadero: identidades exactas. Tolerancia 1e-9: el error esperado es del
/// orden de 32·ε más el término 1e-12 del denominador (≈ 1,4e-12 relativo
/// con c = 0,3).
#[test]
fn cf2_6_feynman_fase_constante_coherencia_unitaria_y_espejo() {
    for &c in &[0.3, 1.0, 2.5, 7.0] {
        let (vp, cp) = FeynmanPropagatorEngine::voto_espectral_con_coherencia(&[c; ESCALAS_VOTO], 1.0, 1.0, 1.0);
        let (vn, cn) = FeynmanPropagatorEngine::voto_espectral_con_coherencia(&[-c; ESCALAS_VOTO], 1.0, 1.0, 1.0);
        assert!((cp - 1.0).abs() < 1e-9 && (cn - 1.0).abs() < 1e-9, "c={c}: C+={cp} C-={cn}");
        let a = (c * 0.5).tanh();
        for k in 0..ESCALAS_VOTO {
            assert!((vp.en_escala(k) - a).abs() < 1e-9, "c={c} k={k}: {} frente a {a}", vp.en_escala(k));
            assert!((vn.en_escala(k) + a).abs() < 1e-9, "c={c} k={k}: {} frente a {}", vn.en_escala(k), -a);
        }
    }
    let mut rng = Lcg(0xF26_0002);
    for _ in 0..2_000 {
        let mut z = [0.0; ESCALAS_VOTO];
        for v in z.iter_mut() {
            *v = 3.0 * rng.normal();
        }
        let mut zn = z;
        for v in zn.iter_mut() {
            *v = -*v;
        }
        let (v, c) = FeynmanPropagatorEngine::voto_espectral_con_coherencia(&z, 1.0, 1.0, 1.0);
        let (w, d) = FeynmanPropagatorEngine::voto_espectral_con_coherencia(&zn, 1.0, 1.0, 1.0);
        assert!((c - d).abs() < 1e-9, "C(z)={c} C(-z)={d}");
        for k in 0..ESCALAS_VOTO {
            assert!(
                (v.en_escala(k) + w.en_escala(k)).abs() < 1e-9,
                "k={k}: v(z)={} v(-z)={}",
                v.en_escala(k),
                w.en_escala(k)
            );
        }
    }
}

/// M = 20 000 réplicas de 32 fases i.i.d. uniformes con amplitud 1.
/// Verdadero: E[N·C] = 1 exacto, Var(N·C) = (N−1)/N = 0,969 ⇒ EE = 0,00696
/// y banda ±0,028 (4 EE). Cola de Rayleigh: P(N·C > 6,4) ≈ e^{−6,4} ≈ 0,0017
/// (≈ 0,0014 con la corrección de N finito); se exige < 0,005 (EE binomial
/// ≈ 0,0003, así que el margen supera 10 EE).
#[test]
fn cf2_6_feynman_coherencia_es_rayleigh_con_fases_uniformes() {
    let mut rng = Lcg(0xF26_0001);
    let m = 20_000usize;
    let amps = [1.0f64; ESCALAS_VOTO];
    let (mut suma, mut cola) = (0.0f64, 0usize);
    for _ in 0..m {
        let mut fases = [(1.0f64, 0.0f64); ESCALAS_VOTO];
        for f in fases.iter_mut() {
            let th = 2.0 * PI * rng.unif();
            *f = (th.cos(), th.sin());
        }
        // compute_coherence_factor devuelve |Σ|²/(ΣA² + 1e-12) = N·C.
        let bruto = FeynmanPropagatorEngine::compute_coherence_factor(&fases, &amps);
        suma += bruto;
        if bruto / ESCALAS_VOTO as f64 > 0.20 {
            cola += 1;
        }
    }
    let media = suma / m as f64;
    let p_cola = cola as f64 / m as f64;
    println!("E[N·C] = {media}; P(C > 0,20) = {p_cola}");
    assert!((media - 1.0).abs() < 0.028, "E[N·C] = {media}, esperado 1 ± 4 EE");
    assert!(p_cola < 0.005, "P(C > 0,20) = {p_cola}, Rayleigh ≈ e^-6,4 ≈ 0,0017");
}

/// Fracción de eventos con C > 0,20 en una caminata aleatoria: (a) con la
/// malla del núcleo; (b) con C recalculado sólo sobre las escalas
/// observables, resolucion_efectiva_ms() ≤ τ_k ≤ tiempo transcurrido, y
/// z_k = 0 en las demás. Devuelve ((frac, EE), (frac, EE)) por bloques de
/// 600 eventos.
fn fraccion_coherente(semilla: u64, media_ms: f64, n: usize, calentamiento: usize) -> ((f64, f64), (f64, f64)) {
    let mut malla_sobre = Vec::with_capacity(n);
    let mut obs_sobre = Vec::with_capacity(n);
    caminata(semilla, 1e-4, media_ms, n, |i, spec, transcurrido_ms| {
        if i < calentamiento {
            return;
        }
        let z = malla(spec);
        let (_v, c) = FeynmanPropagatorEngine::voto_espectral_con_coherencia(&z, 1.0, 1.0, 1.0);
        malla_sobre.push(if c > 0.20 { 1.0 } else { 0.0 });
        let resolucion = spec.resolucion_efectiva_ms();
        let mut zo = z;
        for (k, v) in zo.iter_mut().enumerate() {
            let tau = SPECTRUM_SCALES_MS[k];
            if tau < resolucion || tau > transcurrido_ms {
                *v = 0.0;
            }
        }
        let (_w, co) = FeynmanPropagatorEngine::voto_espectral_con_coherencia(&zo, 1.0, 1.0, 1.0);
        obs_sobre.push(if co > 0.20 { 1.0 } else { 0.0 });
    });
    let (fm, em, _) = media_y_ee_bloques(&malla_sobre, 600);
    let (fo, eo, _) = media_y_ee_bloques(&obs_sobre, 600);
    ((fm, em), (fo, eo))
}

/// Caminata browniana (σ = 1e-4/√s) con llegadas Poisson de media 1 s y, como
/// variante de tasa, de media 100 ms; 20 000 eventos y 2 000 de calentamiento
/// cada una. Verdadero: estructura direccional nula a todo τ; la doc
/// (C ≤ 0,20 ⇒ abstención ante fases estocásticas) exige abstenerse casi
/// siempre. Bajo fases sin estructura la tasa sería e^{−6,4} ≈ 0,2 %; la
/// holgura 0,05 es la especificación de la propiedad prometida, no una banda
/// de muestreo (EE por bloques de 600 eventos en el mensaje). La variante (b)
/// sólo informa: con 5-8 escalas observables y fases i.i.d. ya daría
/// P(N·C > 0,2·N) = e^{−0,2·N} ≈ 0,2-0,37, así que el corte 0,20 no está
/// calibrado para N pequeño. Observado al escribir el test: (a) ≈ 59 % a
/// 1/s y ≈ 56 % a 10/s; (b) ≈ 62 % y ≈ 65 % (≈ 64 % y ≈ 67 % si las escalas
/// excluidas van como NaN en vez de 0). Quitar las escalas no observables NO
/// devuelve el nivel prometido: las fases de escalas vecinas resueltas
/// están correlacionadas en una caminata aleatoria (comparten historia).
#[test]
fn cf2_6_feynman_caminata_aleatoria_decohiere_como_dice_su_doc() {
    let ((f1, e1), (o1, eo1)) = fraccion_coherente(0x5EED_F260, 1_000.0, 20_000, 2_000);
    let ((f10, e10), (o10, eo10)) = fraccion_coherente(0x5EED_F260, 100.0, 20_000, 2_000);
    println!(
        "P(C > 0,20): malla del núcleo 1/s {f1} ± {e1}, 10/s {f10} ± {e10}; \
         sólo observables 1/s {o1} ± {eo1}, 10/s {o10} ± {eo10}"
    );
    assert!(
        f1 <= 0.05 && f10 <= 0.05,
        "caminata aleatoria: C > 0,20 en el {:.1} % (1/s, EE {:.1} %) y el {:.1} % (10/s, EE {:.1} %) de los eventos; \
         la doc promete abstención ante fases estocásticas. Sólo con escalas observables: {:.1} % ± {:.1} % y {:.1} % ± {:.1} %",
        100.0 * f1,
        100.0 * e1,
        100.0 * f10,
        100.0 * e10,
        100.0 * o1,
        100.0 * eo1,
        100.0 * o10,
        100.0 * eo10
    );
}

/// 1 000 mallas con z_k ~ U(−12, 12) (también ejercita el recorte a ±10).
/// Verdadero: Π = Σ_{k=0}^{30} (E_{k+1} − E_k) = E_31 − E_0 =
/// ½(clamp(z_31)² − clamp(z_0)²), identidad telescópica exacta; |E| ≤ 50,
/// así que el recorte ±100 no actúa. Tolerancia 1e-9: el redondeo de 62
/// operaciones sobre valores ≤ 50 es ≲ 1e-12. Es evidencia de trivialidad
/// (F2.6-04), no una falsación; se iría con el módulo si se retira.
#[test]
fn cf2_6_navier_flujo_de_cascada_es_la_diferencia_de_extremos() {
    let mut rng = Lcg(0xF26_0003);
    for _ in 0..1_000 {
        let mut z = [0.0; ESCALAS_VOTO];
        for v in z.iter_mut() {
            *v = 24.0 * rng.unif() - 12.0;
        }
        let flujo = NavierStokesManifoldEngine::turbulent_cascade_flux(&z);
        let (a, b) = (z[0].clamp(-10.0, 10.0), z[ESCALAS_VOTO - 1].clamp(-10.0, 10.0));
        let esperado = 0.5 * b * b - 0.5 * a * a;
        assert!((flujo - esperado).abs() < 1e-9, "Π={flujo} vs E31−E0={esperado}");
    }
}

/// Escalas resueltas de la caminata de 1 s: k = 15..21 (≈ 1 s a 18 min).
const K_RESUELTAS: std::ops::RangeInclusive<usize> = 15..=21;

/// Ē_k medio de una caminata (σ en 1/√s, llegadas de media 1 s), por bloques
/// de `bloque` eventos tras `calentamiento`.
fn energia_por_bloques(semilla: u64, sigma: f64, n: usize, calentamiento: usize, bloque: usize) -> Vec<[f64; ESCALAS_VOTO]> {
    let mut bloques: Vec<[f64; ESCALAS_VOTO]> = Vec::new();
    let mut acum = [0.0f64; ESCALAS_VOTO];
    let mut cuenta = 0usize;
    caminata(semilla, sigma, 1_000.0, n, |i, spec, _| {
        if i < calentamiento {
            return;
        }
        let en = NavierStokesManifoldEngine::kolmogorov_energy_spectrum(&malla(spec));
        for k in 0..ESCALAS_VOTO {
            acum[k] += en[k];
        }
        cuenta += 1;
        if cuenta == bloque {
            let mut m = [0.0f64; ESCALAS_VOTO];
            for k in 0..ESCALAS_VOTO {
                m[k] = acum[k] / bloque as f64;
            }
            bloques.push(m);
            acum = [0.0; ESCALAS_VOTO];
            cuenta = 0;
        }
    });
    bloques
}

/// Dos caminatas con las mismas normales y los mismos sellos (semilla
/// 0x5EED_F261, 12 000 eventos, 2 000 de calentamiento), σ₁ = 1e-5/√s y
/// σ₂ = 1e-4/√s; se promedia E_k en k = 15..21. Verdadero (corrección del
/// escéptico de refutar): E(k) tiene dimensión de u², y en K41 ε ∝ u³/ℓ ⇒
/// ε^{2/3} ∝ σ², así que multiplicar σ por 10 multiplica E por 100 (lo mismo
/// que E = ½u²). Tolerancia: razón > 10, holgura amplia frente a 100. Hoy
/// z = dev/media|dev| es autonormalizada: la razón queda ≈ 1 (observado
/// 1,0001; la corrección es O(σ√τ)).
#[test]
#[ignore = "OPEN F2.6-04 (AGY): la «energía» E_k = ½z² usa un z autonormalizado y no ve la energía inyectada (razón ≈ 1 frente a ≈ 100)"]
fn cf2_6_navier_energia_espectral_crece_con_la_energia_inyectada() {
    let mut e = [0.0f64; 2];
    for (j, &sigma) in [1e-5f64, 1e-4].iter().enumerate() {
        let bloques = energia_por_bloques(0x5EED_F261, sigma, 12_000, 2_000, 1_000);
        let mut s = 0.0;
        for b in &bloques {
            for k in K_RESUELTAS {
                s += b[k];
            }
        }
        e[j] = s / (bloques.len() * K_RESUELTAS.count()) as f64;
    }
    let razon = e[1] / e[0];
    println!("Ē(σ) = {}, Ē(10σ) = {}, razón {razon}", e[0], e[1]);
    assert!(razon > 10.0, "E(10σ)/E(σ) = {razon}: K41 (y ½u²) exige ≈ 100; la «energía» no ve la energía inyectada");
}

/// Pendiente MCO de y sobre x.
fn pendiente(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
    sxy / sxx
}

/// Corrección del escéptico de completitud: contrastar la LEY que el módulo
/// nombra, la pendiente del espectro, sin la analogía de ε. Regresión de
/// ln Ē_k sobre ln(1/τ_k) en k = 15..21 con la caminata browniana
/// (σ = 1e-4/√s, semilla 0x5EED_F262, 12 000 eventos, 2 000 de calentamiento).
/// Verdadero: la caminata tiene solución conocida (incrementos de varianza
/// σ²τ): energía por banda ∝ τ (pendiente −1) y densidad ∝ τ² (−2); la
/// teoría nombrada, K41, da −2/3 por banda y −5/3 como densidad. Toda ley
/// física cae en ≤ −2/3, así que se exige pendiente ≤ −0,5 (el margen de
/// 0,17 frente a −2/3 es > 4 EE de la pendiente por bloques de 1 000
/// eventos, ≈ 0,023, así que 0,17 ≈ 7 EE). Hoy z ~ O(1) en toda escala
/// resuelta: observado −0,02 global y +0,007 ± 0,023 por bloques.
#[test]
#[ignore = "OPEN F2.6-04 (AGY): la «energía espectral» ½z² es plana en la escala (pendiente ≈ 0), no un espectro K41 ni browniano"]
fn cf2_6_navier_energia_espectral_decrece_con_la_frecuencia() {
    let bloques = energia_por_bloques(0x5EED_F262, 1e-4, 12_000, 2_000, 1_000);
    let x: Vec<f64> = K_RESUELTAS.map(|k| (1.0 / SPECTRUM_SCALES_MS[k]).ln()).collect();
    let pendientes: Vec<f64> = bloques
        .iter()
        .map(|b| {
            let y: Vec<f64> = K_RESUELTAS.map(|k| b[k].ln()).collect();
            pendiente(&x, &y)
        })
        .collect();
    let (p_media, ee, nb) = media_y_ee_bloques(&pendientes, 1);
    let mut global = [0.0f64; ESCALAS_VOTO];
    for b in &bloques {
        for k in 0..ESCALAS_VOTO {
            global[k] += b[k] / bloques.len() as f64;
        }
    }
    let y: Vec<f64> = K_RESUELTAS.map(|k| global[k].ln()).collect();
    let p = pendiente(&x, &y);
    println!("pendiente global {p}; media por bloque {p_media} ± {ee} ({nb} bloques); Ē_k = {:?}", &global[15..=21]);
    assert!(
        p <= -0.5,
        "pendiente de ln Ē_k sobre ln(1/τ_k) = {p} (por bloques {p_media} ± {ee}, {nb} bloques); \
         browniano −1 (banda) / −2 (densidad), K41 −2/3 / −5/3"
    );
}
