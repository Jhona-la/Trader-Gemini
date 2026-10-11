//! C-F2.4 (Línea C): Hawkes univariante de `HawkesBesselEngine` contra
//! procesos con solución conocida.
//!
//! Teoría que se prueba. El motor publica λ(t)/μ̂(t) con
//! λ(t) = μ̂ + Σ α·e^{−β(t − t_i)}, α = 0,3 y β = 0,5 fijos, y μ̂ una EWMA de
//! 60 s del ritmo de llegada. Para cualquier proceso estacionario de tasa r
//! leído en FASE UNIFORME (instantes ajenos a los eventos), Campbell da
//! E[Σ α·e^{−β(t − t_i)}] = α·r/β; con μ̂ ≈ r el cociente vale
//! 1 + α/β = 1,6 (`STEADY_STATE_RATIO`) y la excitación normalizada 0. Para
//! Poisson, PASTA dice además que la intensidad predecible λ(t⁻) leída en los
//! eventos tiene la misma ley que en fase uniforme. Un Poisson compuesto
//! (lotes sin excitación propia) también debe dar 1,6 si λ y μ̂ cuentan lo
//! mismo, y la tasa de falsas alarmas bajo Poisson no debe depender del
//! ritmo si el exceso es «escala libre» como declara el código. Un Hawkes
//! simulado por Ogata con los mismos (α, β) debe excitar más que un Poisson
//! de la misma tasa cuando se lee en los eventos.
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`. Sin
//! dependencias nuevas: LCG propio, exponenciales por inversión, Poisson de
//! Knuth y adelgazamiento de Ogata.

use signal_engine::hawkes_bessel::{excitacion_hawkes_norm, HawkesBesselEngine, STEADY_STATE_RATIO};

/// LCG de 64 bits (constantes de Knuth/MMIX); se usan los 53 bits altos.
struct Lcg(u64);

impl Lcg {
    fn u01(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn exp(&mut self, tasa: f64) -> f64 {
        -self.u01().ln() / tasa
    }

    fn poisson(&mut self, media: f64) -> u32 {
        let l = (-media).exp();
        let (mut k, mut p) = (0u32, 1.0f64);
        loop {
            p *= self.u01();
            if p <= l {
                return k;
            }
            k += 1;
        }
    }
}

fn tiempos_poisson(rng: &mut Lcg, tasa: f64, horizonte_s: f64) -> Vec<f64> {
    let (mut t, mut v) = (0.0f64, Vec::new());
    loop {
        t += rng.exp(tasa);
        if t > horizonte_s {
            return v;
        }
        v.push(t);
    }
}

/// Ogata (adelgazamiento) para un Hawkes de kernel exponencial
/// λ(t) = mu0 + Σ a·e^{−b(t − t_i)}. La cota mu0 + exc en el instante actual
/// domina la intensidad hasta el siguiente candidato porque decrece.
fn hawkes(rng: &mut Lcg, mu0: f64, a: f64, b: f64, horizonte_s: f64) -> Vec<f64> {
    let (mut t, mut exc, mut v) = (0.0f64, 0.0f64, Vec::new());
    loop {
        let cota = mu0 + exc;
        let w = rng.exp(cota);
        if t + w > horizonte_s {
            return v;
        }
        t += w;
        exc *= (-b * w).exp();
        if rng.u01() * cota <= mu0 + exc {
            v.push(t);
            exc += a;
        }
    }
}

/// Lotes de K = 1 + Poisson(extra_medio) fills en el MISMO milisegundo
/// (un barrido de varios niveles publica varios @trade con el mismo event
/// time). Es un Poisson compuesto: sin excitación propia.
fn lotes_mismo_ms(rng: &mut Lcg, tasa_lotes: f64, extra_medio: f64, horizonte_s: f64) -> Vec<f64> {
    let mut v = Vec::new();
    for t in tiempos_poisson(rng, tasa_lotes, horizonte_s) {
        let tq = (t * 1000.0).floor() / 1000.0;
        for _ in 0..(1 + rng.poisson(extra_medio)) {
            v.push(tq);
        }
    }
    v
}

/// Lectura en FASE UNIFORME (rejilla fija, ajena a los eventos): registra
/// los eventos con t_i ≤ t y lee λ/μ̂ en t. Devuelve (t, ratio) desde `desde_s`.
fn lecturas_fase_uniforme(ev: &[f64], paso_s: f64, desde_s: f64) -> Vec<(f64, f64)> {
    let mut h = HawkesBesselEngine::new();
    let fin = *ev.last().expect("serie vacía");
    let (mut i, mut j) = (0usize, 1u64);
    let mut out = Vec::new();
    loop {
        // t = j·paso, sin acumular error de redondeo.
        let t = j as f64 * paso_s;
        if t >= fin {
            return out;
        }
        while i < ev.len() && ev[i] <= t {
            h.record_event(ev[i]);
            i += 1;
        }
        if t >= desde_s {
            out.push((t, h.intensity_ratio(t)));
        }
        j += 1;
    }
}

/// Lectura VIVA (Palm): registra y lee en el mismo t, como hace el núcleo
/// con cada trade (record_event seguido de intensity_ratio en el mismo ts).
fn lecturas_en_eventos(ev: &[f64], desde_s: f64) -> Vec<(f64, f64)> {
    let mut h = HawkesBesselEngine::new();
    let mut out = Vec::new();
    for &t in ev {
        h.record_event(t);
        if t >= desde_s {
            out.push((t, h.intensity_ratio(t)));
        }
    }
    out
}

/// Media de f(ratio) y su error estándar por medias de bloques de
/// `bloque_s` segundos (los bloques de 60 s, la memoria de μ̂, son casi
/// independientes entre sí).
fn media_y_ee(lect: &[(f64, f64)], bloque_s: f64, f: impl Fn(f64) -> f64) -> (f64, f64, usize) {
    let mut bloques: Vec<(f64, usize)> = Vec::new();
    let mut actual = i64::MIN;
    for &(t, r) in lect {
        let b = (t / bloque_s).floor() as i64;
        if b != actual {
            bloques.push((0.0, 0));
            actual = b;
        }
        let ult = bloques.last_mut().expect("bloque");
        ult.0 += f(r);
        ult.1 += 1;
    }
    let medias: Vec<f64> = bloques.iter().map(|&(s, n)| s / n as f64).collect();
    let nb = medias.len();
    let total: f64 = lect.iter().map(|&(_, r)| f(r)).sum::<f64>() / lect.len() as f64;
    let mb = medias.iter().sum::<f64>() / nb as f64;
    let var = medias.iter().map(|m| (m - mb) * (m - mb)).sum::<f64>() / (nb as f64 - 1.0);
    (total, (var / nb as f64).sqrt(), nb)
}

fn identidad(r: f64) -> f64 {
    r
}

/// Copia documentada de la rampa viva de `flow_excitation_confluence.rs`
/// (`rampa(hawkes − umbral, 0,30·umbral)` con el gen por defecto 0,55:
/// umbral = 1,6 + 0,1 = 1,7, ancho 0,51): smoothstep de 1,7 a 2,21.
fn peso_vivo(r: f64) -> f64 {
    let t = ((r - 1.7) / 0.51).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Control: Poisson a 10/s leído en fase uniforme. Verdadero: E[λ/μ̂] = 1,6 y
/// excitación 0. Tolerancia ±0,03 = presupuesto de sesgo, no de muestreo:
/// el tope MU_INST_MAX sesga +0,015 a 10/s (L(10) = 2,5 %) y el EE por
/// bloques de 60 s es ≈ 0,002 (55 bloques), así que 0,015 + 5·EE ≤ 0,03.
/// Observado: 1,609 y excitación 0,006.
#[test]
fn cf2_hawkes_poisson_10hz_fase_uniforme_ancla_1_6() {
    let mut rng = Lcg(0xA1A1_0001);
    let ev = tiempos_poisson(&mut rng, 10.0, 3600.0);
    let lect = lecturas_fase_uniforme(&ev, 0.05, 300.0);
    let (r, ee, nb) = media_y_ee(&lect, 60.0, identidad);
    let (e, _, _) = media_y_ee(&lect, 60.0, excitacion_hawkes_norm);
    println!("10/s fase uniforme: ratio {r} (EE {ee}, {nb} bloques), excitación {e}");
    assert!(
        (r - STEADY_STATE_RATIO).abs() <= 0.03,
        "ratio medio {r} (EE {ee} con {nb} bloques; esperado 1,6)"
    );
    assert!(e.abs() <= 0.02, "excitación media {e} (esperado 0)");
}

/// Poisson a 100/s en fase uniforme. Verdadero: 1,6 (identidad de Poisson a
/// cualquier ritmo). Tolerancia ±0,03 como el control (EE ≈ 0,003 con 10
/// bloques de 60 s y 9·10⁴ eventos). Hoy: el tope MU_INST_MAX = 200/s recorta
/// los intervalos de menos de 5 ms, L(100) = 0,213 ⇒ 1 + 0,6/(1 − 0,213) ≈ 1,76.
#[test]
#[ignore = "OPEN F2.4-05 (Qoder/GLM): el tope MU_INST_MAX = 200/s hunde μ̂ un ~21 % a 100/s y el ancla sube a ~1,76"]
fn cf2_hawkes_poisson_100hz_fase_uniforme_ancla_1_6() {
    let mut rng = Lcg(0xA2A2_0002);
    let ev = tiempos_poisson(&mut rng, 100.0, 900.0);
    let lect = lecturas_fase_uniforme(&ev, 0.05, 300.0);
    let (r, ee, nb) = media_y_ee(&lect, 60.0, identidad);
    println!("100/s fase uniforme: ratio {r} (EE {ee}, {nb} bloques)");
    assert!(
        (r - STEADY_STATE_RATIO).abs() <= 0.03,
        "ratio medio {r} (EE {ee} con {nb} bloques; esperado 1,6; con el tope L(100) = 0,213 da ~1,76)"
    );
}

/// Lotes Poisson a 5/s de K = 1 + Poisson(1) fills con el mismo sello en ms
/// (E[K] = 2). Verdadero: un Poisson compuesto no se excita a sí mismo;
/// contando lo mismo en λ y en μ̂ el cociente en fase uniforme es 1,6 y la
/// excitación 0. Tolerancia ±0,05 (EE por bloques ≈ 0,01 con 55 bloques).
/// Hoy λ cuenta los 10 fills/s y μ̂ sólo los 5 sellos distintos: 1 + 0,6·2 = 2,2.
#[test]
#[ignore = "OPEN F2.4-01 (Qoder/GLM): los fills del mismo ms entran en λ pero no en μ̂ y el ancla sube a 1 + 0,6·m ≈ 2,2"]
fn cf2_hawkes_lotes_mismo_ms_fase_uniforme_ancla_1_6() {
    let mut rng = Lcg(0xA3A3_0003);
    let ev = lotes_mismo_ms(&mut rng, 5.0, 1.0, 3600.0);
    // Paso 0,1003 s: no comensurable con la rejilla de 1 ms de los sellos.
    let lect = lecturas_fase_uniforme(&ev, 0.1003, 300.0);
    let (r, ee, nb) = media_y_ee(&lect, 60.0, identidad);
    let (e, _, _) = media_y_ee(&lect, 60.0, excitacion_hawkes_norm);
    println!("lotes mismo ms: ratio {r} (EE {ee}, {nb} bloques), excitación {e}");
    assert!(
        (r - STEADY_STATE_RATIO).abs() <= 0.05,
        "ratio medio {r} (EE {ee} con {nb} bloques; esperado 1,6; hoy ~1 + 0,6·m = 2,2)"
    );
    assert!(e.abs() <= 0.03, "excitación media {e} (esperado 0; hoy ~0,35)");
}

/// Poisson a 1/s leído en cada evento, como en vivo. Verdadero (PASTA): la
/// intensidad predecible λ(t⁻) en los eventos tiene la ley de la fase
/// uniforme ⇒ 1,6 y excitación 0. Tolerancia ±0,05 (EE por bloques ≈ 0,005).
/// Efecto de Palm residual tras el arreglo: μ̂ se lee justo después de su
/// propio salto (+1/(60·r) relativo), así que a 1/s el esperado queda ≈ 1,59,
/// dentro de la banda; NO extrapolar este test a ritmos por debajo de 0,5/s.
/// Hoy la lectura incluye el propio evento (dt = 0 ⇒ +α): 1,6 + α/μ̂ ≈ 1,89.
#[test]
#[ignore = "OPEN F2.4-03 (Qoder/GLM): la intensidad viva incluye el propio evento (dt = 0 ⇒ +α) y en Poisson a 1/s da ~1,89"]
fn cf2_hawkes_poisson_1hz_lectura_en_evento_excitacion_nula() {
    let mut rng = Lcg(0xA4A4_0004);
    let ev = tiempos_poisson(&mut rng, 1.0, 10_800.0);
    let lect = lecturas_en_eventos(&ev, 300.0);
    let (r, ee, nb) = media_y_ee(&lect, 60.0, identidad);
    let (e, _, _) = media_y_ee(&lect, 60.0, excitacion_hawkes_norm);
    println!("1/s en eventos: ratio {r} (EE {ee}, {nb} bloques), excitación {e}");
    assert!(
        (r - STEADY_STATE_RATIO).abs() <= 0.05,
        "ratio en eventos {r} (EE {ee} con {nb} bloques; esperado 1,6; hoy ~1,89)"
    );
    assert!(e.abs() <= 0.03, "excitación en eventos {e} (esperado 0; hoy ~0,18)");
}

/// Hawkes (μ0 = 0,4; α = 0,3; β = 0,5; n = 0,6; tasa 1/s) frente a Poisson a
/// 1/s, leídos en los eventos. Verdadero: con agrupamiento
/// E[λ(t⁻) | evento] = E[λ] + Var(λ)/E[λ] = 1 + 0,225 ⇒ Δ ratio ≈ 0,17 una vez
/// descontada la inflación de Palm de μ̂. Tolerancias: tasa 1 ± 0,08 (≈ 3,3
/// desviaciones de √(μ/(1−n)³/T) = 0,024); Δ ratio > 0,08 y Δ excitación
/// > 0,03, holgadas frente a Δ ≈ 0,17 y ≈ 0,09 (EE de cada media ≈ 0,01).
#[test]
fn cf2_hawkes_n06_excita_mas_que_poisson_en_lectura_viva() {
    let mut rng = Lcg(0xA5A5_0005);
    let h = hawkes(&mut rng, 0.4, 0.3, 0.5, 10_800.0);
    let p = tiempos_poisson(&mut rng, 1.0, 10_800.0);
    let tasa = h.len() as f64 / 10_800.0;
    assert!((tasa - 1.0).abs() <= 0.08, "tasa Hawkes {tasa} (mu/(1-n) = 1)");
    let lh = lecturas_en_eventos(&h, 300.0);
    let lp = lecturas_en_eventos(&p, 300.0);
    let (rh, eeh, _) = media_y_ee(&lh, 600.0, identidad);
    let (rp, eep, _) = media_y_ee(&lp, 600.0, identidad);
    let (eh, _, _) = media_y_ee(&lh, 600.0, excitacion_hawkes_norm);
    let (ep, _, _) = media_y_ee(&lp, 600.0, excitacion_hawkes_norm);
    println!("tasa {tasa}; Hawkes ratio {rh} ± {eeh}, exc {eh}; Poisson ratio {rp} ± {eep}, exc {ep}");
    assert!(rh - rp > 0.08, "delta ratio {} (Hawkes {rh} ± {eeh}, Poisson {rp} ± {eep})", rh - rp);
    assert!(eh - ep > 0.03, "delta excitación {} (Hawkes {eh}, Poisson {ep})", eh - ep);
}

/// Versión monótona (corrección del escéptico de completitud): misma tasa
/// total 1/s y n ∈ {0; 0,3; 0,6; 0,9} (α_sim = n·β, μ0 = 1 − n, β = 0,5).
/// Verdadero: E[S | evento] = r/β + α_sim/(2(β − α_sim)) crece con n, así que
/// la lectura viva media del motor debe crecer estrictamente con n (los
/// incrementos teóricos son ≈ +0,06, +0,16 y +1,1 antes de los efectos de μ̂).
/// Tolerancia: cada salto debe superar 4 EE de la diferencia (EE por bloques
/// de 600 s, 18 bloques por serie). Observado: 1,890 → 1,951 → 2,051 → 2,515,
/// con saltos de 7,5, 7,3 y 6,6 EE.
#[test]
fn cf2_hawkes_lectura_viva_crece_con_n() {
    let beta = 0.5;
    let mut rng = Lcg(0xA5A5_0505);
    let mut medias = Vec::new();
    for &n in &[0.0f64, 0.3, 0.6, 0.9] {
        let ev = hawkes(&mut rng, 1.0 - n, n * beta, beta, 10_800.0);
        let lect = lecturas_en_eventos(&ev, 300.0);
        let (r, ee, _) = media_y_ee(&lect, 600.0, identidad);
        medias.push((n, r, ee, ev.len() as f64 / 10_800.0));
    }
    println!("(n, ratio vivo, EE, tasa): {medias:?}");
    for par in medias.windows(2) {
        let ((n0, r0, e0, _), (n1, r1, e1, _)) = (par[0], par[1]);
        let ee = (e0 * e0 + e1 * e1).sqrt();
        assert!(
            r1 - r0 > 4.0 * ee,
            "n {n0} → {n1}: ratio vivo {r0} → {r1} (Δ {}, 4 EE = {}); todas: {medias:?}",
            r1 - r0,
            4.0 * ee
        );
    }
}

/// Diagnóstico de F2.4-06, NO una prueba de acierto. Hawkes (μ0 = 2; α = 0,3;
/// β = 0,5; tasa 5/s) frente a Poisson a 5/s en fase uniforme. Verdadero
/// (Campbell): E[Σ α·e^{−β(t − t_i)}] = α·r/β para CUALQUIER proceso
/// estacionario, así que el cociente vale 1,6 en los dos y la lectura en
/// fase uniforme es ciega a n. Tolerancia ±0,05 (sesgo de Jensen de μ̂
/// ≈ +0,006 a 5/s; EE por bloques ≈ 0,004; observado 1,601 frente a
/// 1,602). Debe invertirse (o sustituirse
/// por un KS de los residuos reescalados) cuando exista un estimador de
/// (μ, α, β): entonces el motor tendría que distinguir n = 0,6 de n = 0.
#[test]
fn diagnostico_open_f2_4_06_fase_uniforme_ciega_a_n() {
    let mut rng = Lcg(0xA6A6_0006);
    let h = hawkes(&mut rng, 2.0, 0.3, 0.5, 14_400.0);
    let p = tiempos_poisson(&mut rng, 5.0, 14_400.0);
    let (rh, eeh, _) = media_y_ee(&lecturas_fase_uniforme(&h, 0.5, 300.0), 60.0, identidad);
    let (rp, eep, _) = media_y_ee(&lecturas_fase_uniforme(&p, 0.5, 300.0), 60.0, identidad);
    println!("fase uniforme: Hawkes {rh} ± {eeh}, Poisson {rp} ± {eep}");
    assert!((rh - STEADY_STATE_RATIO).abs() <= 0.05, "Hawkes n=0,6: {rh} (EE {eeh})");
    assert!((rp - STEADY_STATE_RATIO).abs() <= 0.05, "Poisson n=0: {rp} (EE {eep})");
    assert!((rh - rp).abs() <= 0.05, "delta {}", rh - rp);
}

/// Bajo Poisson no hay cascadas: el peso vivo de la puerta Hawkes de
/// `flow_excitation_confluence` (rampa de 1,7 a 2,21, corrección del
/// escéptico de completitud: el umbral 2,0 no existe en el código) debe
/// tener la misma media a 0,5/s que a 30/s si el exceso es escala libre.
/// Verdadero: |ΔE[w]| = 0. Se lee en fase uniforme (rejilla de 0,25 s, el
/// reloj de las klines) y en los eventos (los trades). Tolerancia 0,03: con
/// el mismo E[w] a los dos ritmos, la desviación por bloque de 60 s medida a
/// 0,5/s es ≈ 0,061 (EE 0,0024 con 662 bloques); a 30/s con 7 200 s hay 115
/// bloques (EE ≈ 0,0057), así que el EE de la diferencia es ≈ 0,0062 y
/// 0,03 ≈ 4,8 EE. Hoy la dispersión nula α/√(2βr) es 0,42 a 0,5/s y 0,055 a
/// 30/s: E[w] ≈ 0,19 frente a ≈ 0,002 en fase uniforme y ≈ 0,67 frente a
/// ≈ 0,003 en los eventos.
#[test]
#[ignore = "OPEN F2.4-04 (Qoder/GLM): el exceso λ/μ̂ − 1,6 no se normaliza por su dispersión nula α/√(2βr) y la falsa cascada depende del ritmo (la lectura en eventos arrastra además F2.4-03)"]
fn cf2_hawkes_tasa_de_falsa_cascada_no_depende_del_ritmo() {
    let mut rng = Lcg(0xA7A7_0007);
    let lento = tiempos_poisson(&mut rng, 0.5, 40_000.0);
    let rapido = tiempos_poisson(&mut rng, 30.0, 7_200.0);
    let (wl, eel, nbl) = media_y_ee(&lecturas_fase_uniforme(&lento, 0.25, 300.0), 60.0, peso_vivo);
    let (wr, eer, nbr) = media_y_ee(&lecturas_fase_uniforme(&rapido, 0.25, 300.0), 60.0, peso_vivo);
    let (wle, _, _) = media_y_ee(&lecturas_en_eventos(&lento, 300.0), 60.0, peso_vivo);
    let (wre, _, _) = media_y_ee(&lecturas_en_eventos(&rapido[..rapido.len().min(36_000)], 300.0), 60.0, peso_vivo);
    println!("E[w] fase uniforme: lento {wl} ± {eel} ({nbl}), rápido {wr} ± {eer} ({nbr}); eventos: {wle} frente a {wre}");
    assert!(
        (wl - wr).abs() <= 0.03,
        "fase uniforme: E[w] lento {wl} (EE {eel}, {nbl} bloques) frente a rápido {wr} (EE {eer}, {nbr} bloques); \
         en eventos: lento {wle} frente a rápido {wre}"
    );
    assert!(
        (wle - wre).abs() <= 0.03,
        "en eventos: E[w] lento {wle} frente a rápido {wre}"
    );
}
