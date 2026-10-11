//! C-F2.5 (Línea C): spoofing, ballenas y OFI frente a soluciones conocidas.
//!
//! Teoría que se prueba:
//! - `SpoofingDetector`: una cancelación plantada de un muro debe subir el
//!   score, y un libro sin el patrón de spoofing (sin cancelaciones, o con
//!   las cancelaciones ordinarias del modelo de colas de Cont-Stoikov-Talreja)
//!   no debe hacerlo por encima de un tamaño α = 5 %.
//! - `InstitutionalVolumeTracker`: un z ≥ 2,5 tiene una cola gaussiana
//!   nominal de 0,0062; con ballenas plantadas y etiquetas conocidas, el
//!   detector no puede superar al oráculo de Neyman-Pearson (umbral sobre el
//!   nocional) a la misma tasa de falsas alarmas.
//! - `OFIModel`: el OFI evento a evento es el e_n de Cont-Kukanov-Stoikov
//!   (2014) normalizado por la profundidad; la suma de e_n en un intervalo no
//!   cambia con eventos que no mueven el libro.
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`.

use feature_engine::{InstitutionalVolumeTracker, OFIModel, SpoofingDetector};

/// LCG de 64 bits (constantes de Knuth/MMIX) con salida de 53 bits en (0, 1).
struct Lcg(u64);
impl Lcg {
    fn unif(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }
    /// Normal estándar por Box-Muller.
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unif(), self.unif());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    /// Exponencial de tasa `tasa`.
    fn exp(&mut self, tasa: f64) -> f64 {
        -self.unif().ln() / tasa
    }
}

// ─────────────────────────────────────────────────────────────────
// Spoofing
// ─────────────────────────────────────────────────────────────────

/// Diez snapshots con muros de 1000/1000 cada `paso_ms` y, en el undécimo, el
/// muro bid cae a 100 (90 % cancelado). Devuelve el score tras la caída.
fn cancelacion_plantada(paso_ms: u64) -> f64 {
    let mut sp = SpoofingDetector::new(2.0);
    let mut t = 1_000u64;
    for _ in 0..10 {
        sp.evaluate_wall_decay(1_000.0, 1_000.0, t);
        t += paso_ms;
    }
    sp.evaluate_wall_decay(100.0, 1_000.0, t)
}

/// Verdad: cancelación del 90 % del muro bid en 100 ms. Valor exacto 0,5
/// (dt = 0,1 s < 0,25 s y caída > 50 %, microstructure.rs:707-712).
/// Duplica test_spoofing_detector; sirve de ancla del helper.
#[test]
fn cf2_spoof_cancelacion_plantada_a_100ms() {
    let s = cancelacion_plantada(100);
    assert!(s >= 0.5, "cancelación del 90 % a 100 ms: score {s} (exacto 0,5)");
}

/// Verdad: la misma cancelación del 90 % es real a cualquier cadencia; el
/// detector debe verla de forma continua en dt, sin escalón en 250 ms (la
/// cadencia documentada de @depth5 sin sufijo, no verificada en tape). Se
/// exige score > 0 en dt ∈ {100, 249, 250, 251, 500} ms y |s(249) − s(251)|
/// ≤ 0,05. Hoy la ventana es estricta (dt < 0,25 s, microstructure.rs:707):
/// s(249) = 0,5 y s(250) = s(251) = s(500) = 0.
#[test]
#[ignore = "OPEN F2.5-07 (Línea C): la ventana de detección es dt < 0,25 s estricto, igual a la cadencia de @depth5: una cancelación del 90 % a 250 ms da score 0"]
fn cf2_spoof_cancelacion_plantada_a_la_cadencia_de_depth5() {
    let pasos = [100u64, 249, 250, 251, 500];
    let s: Vec<f64> = pasos.iter().map(|&p| cancelacion_plantada(p)).collect();
    assert!(
        s.iter().all(|&v| v > 0.0) && (s[1] - s[3]).abs() <= 0.05,
        "cancelación del 90 % según dt {pasos:?} ms: scores {s:?} (microstructure.rs:707)"
    );
}

/// Verdad: ninguna cancelación. (a) Muro ejecutado por agresores: la
/// profundidad es idéntica a la de la cancelación plantada. (b) Muro de 50
/// unidades en el quinto nivel bid que sigue en el libro: el precio sube un
/// tick y sale de la ventana de 5 niveles (máximo 4997,5 → 100). Se exige
/// score < 0,05 en los dos. Es un contrato para una API nueva (niveles y
/// trades por nivel): con evaluate_wall_decay(max_bid, max_ask, ts) el caso
/// (a) recibe exactamente la entrada de cf2_spoof_cancelacion_plantada_a_100ms,
/// así que ninguna implementación de la firma actual puede pasar los dos.
#[test]
#[ignore = "OPEN F2.5-07 (Línea C): contrato abierto de API; sólo recibe el máximo nocional de 5 niveles sin trades, y ejecución o traslación dan la misma señal que una cancelación"]
fn cf2_spoof_nulos_estructurales_sin_cancelacion() {
    let ejecutado = cancelacion_plantada(100);
    let tick = 0.01;
    let muro = 100.0 - 5.0 * tick;
    let max_ventana = |mid: f64| {
        (0..5)
            .map(|k| {
                let p = mid - (k as f64 + 1.0) * tick;
                p * if (p - muro).abs() < 0.5 * tick { 50.0 } else { 1.0 }
            })
            .fold(0.0f64, f64::max)
    };
    let mut sp = SpoofingDetector::new(2.0);
    let mut t = 1_000u64;
    for _ in 0..5 {
        sp.evaluate_wall_decay(max_ventana(100.0), 100.0, t);
        t += 100;
    }
    let trasladado = sp.evaluate_wall_decay(max_ventana(100.0 + tick), 100.0, t);
    assert!(
        ejecutado < 0.05 && trasladado < 0.05,
        "falsas alarmas sin cancelación: ejecución {ejecutado}, traslación {trasladado}"
    );
}

/// Libro de Cont-Stoikov-Talreja (2010) en una rejilla de ticks: órdenes
/// límite unitarias a distancia i ∈ 1..=10 del mejor precio opuesto con tasa
/// λ(i) = 1,92/i^0,52, órdenes de mercado con tasa μ = 0,94 y cancelaciones a
/// tasa θ(i)·q a distancia i (escaladas por `factor_cancelacion`; 0 = sin
/// cancelaciones). Parámetros del orden de los estimados por CST (tabla 2).
/// Fuera de la ventana de 10 niveles hay liquidez de reserva de 5 unidades por
/// tick que nunca se cancela. Snapshot cada 100 ms: máximo de p·q sobre los 5
/// mejores niveles no vacíos de cada lado, como el host (god_engine.rs:3205-3214).
/// Devuelve (alarmas, snapshots, movimientos del mejor bid).
fn alarmas_en_libro_cst(semilla: u64, factor_cancelacion: f64, snapshots: usize) -> (usize, usize, usize) {
    const L: usize = 4_000;
    const CENTRO: usize = 2_000;
    let lambda: Vec<f64> = (1..=10).map(|i| 1.92 / (i as f64).powf(0.52)).collect();
    let theta: [f64; 10] = [0.71, 0.81, 0.68, 0.56, 0.47, 0.37, 0.30, 0.26, 0.31, 0.21];
    let mu = 0.94;
    let tick = 0.01;
    let precio = |idx: usize| 100.0 + (idx as f64 - CENTRO as f64) * tick;
    let mut rng = Lcg(semilla);
    let mut bid = vec![0u32; L];
    let mut ask = vec![0u32; L];
    for p in 0..CENTRO {
        bid[p] = 5;
    }
    for p in (CENTRO + 1)..L {
        ask[p] = 5;
    }
    // Siguiente nivel no vacío por debajo (bid) o por encima (ask) de `desde`.
    let bajar = |b: &[u32], desde: usize| (0..=desde).rev().find(|&p| b[p] > 0).expect("reserva bid");
    let subir = |a: &[u32], desde: usize| (desde..L).find(|&p| a[p] > 0).expect("reserva ask");
    let max_muro_bid = |b: &[u32], pb: usize| {
        let mut m = 0.0f64;
        let (mut p, mut vistos) = (pb as isize, 0);
        while vistos < 5 && p >= 0 {
            if b[p as usize] > 0 {
                m = m.max(precio(p as usize) * b[p as usize] as f64);
                vistos += 1;
            }
            p -= 1;
        }
        m
    };
    let max_muro_ask = |a: &[u32], pa: usize| {
        let mut m = 0.0f64;
        let (mut p, mut vistos) = (pa, 0);
        while vistos < 5 && p < L {
            if a[p] > 0 {
                m = m.max(precio(p) * a[p] as f64);
                vistos += 1;
            }
            p += 1;
        }
        m
    };
    let mut sp = SpoofingDetector::new(2.0);
    let decaimiento = (-2.0f64 * 0.1).exp();
    let (mut t, mut siguiente) = (0.0f64, 0.1f64);
    let (mut prev, mut alarmas, mut hechos, mut movimientos) = (0.0f64, 0usize, 0usize, 0usize);
    let (mut pb, mut pa) = (CENTRO - 1, CENTRO + 1);
    let mut pb_prev = pb;
    // Calentamiento de 60 s de tiempo simulado antes de contar.
    let calentamiento = 600usize;
    while hechos < snapshots + calentamiento {
        // Tasas de los 42 sucesos posibles.
        let mut tasas = [0.0f64; 42];
        for i in 1..=10 {
            tasas[i - 1] = lambda[i - 1]; // límite compra en pa − i
            tasas[10 + i - 1] = lambda[i - 1]; // límite venta en pb + i
            tasas[20 + i - 1] = factor_cancelacion * theta[i - 1] * bid[pa - i] as f64;
            tasas[30 + i - 1] = factor_cancelacion * theta[i - 1] * ask[pb + i] as f64;
        }
        tasas[40] = mu; // mercado compra: consume el mejor ask
        tasas[41] = mu; // mercado venta: consume el mejor bid
        let total: f64 = tasas.iter().sum();
        let dt = rng.exp(total);
        // Snapshots que caen antes del próximo suceso.
        while siguiente <= t + dt && hechos < snapshots + calentamiento {
            let s = sp.evaluate_wall_decay(
                max_muro_bid(&bid, pb),
                max_muro_ask(&ask, pa),
                (siguiente * 1000.0).round() as u64 + 1_000,
            );
            if hechos > calentamiento {
                if s > prev * decaimiento + 1e-12 {
                    alarmas += 1;
                }
                if pb != pb_prev {
                    movimientos += 1;
                }
            }
            pb_prev = pb;
            prev = s;
            hechos += 1;
            siguiente += 0.1;
        }
        t += dt;
        let mut u = rng.unif() * total;
        let mut k = 0usize;
        while k < 41 && u >= tasas[k] {
            u -= tasas[k];
            k += 1;
        }
        match k {
            0..=9 => {
                let p = pa - (k + 1);
                bid[p] += 1;
                pb = pb.max(p);
            }
            10..=19 => {
                let p = pb + (k - 9);
                ask[p] += 1;
                pa = pa.min(p);
            }
            20..=29 => {
                let p = pa - (k - 19);
                bid[p] = bid[p].saturating_sub(1);
            }
            30..=39 => {
                let p = pb + (k - 29);
                ask[p] = ask[p].saturating_sub(1);
            }
            40 => ask[pa] -= 1,
            _ => bid[pb] -= 1,
        }
        if bid[pb] == 0 {
            pb = bajar(&bid, pb);
        }
        if ask[pa] == 0 {
            pa = subir(&ask, pa);
        }
    }
    (alarmas, snapshots - 1, movimientos)
}

/// Verdad: libro de colas persistentes sin cancelaciones (CST con θ = 0): no
/// hay ninguna cancelación, así que toda alarma es falsa. Tolerancia: tasa
/// por snapshot ≤ α + 4·√(α(1−α)/N) con α = 5 % y N = 20 000 (≤ 0,0562).
/// El detector no declara su α; se usa el 5 % del borrador del auditor.
/// Con las tasas de CST el mejor nivel recibe λ(1) = 1,92 > μ = 0,94, la cola
/// sólo crece y el precio no se mueve: la tasa es 0 de forma casi trivial. El
/// nulo iid del borrador (cantidades re-sorteadas en cada snapshot, ≈ 0,138)
/// implicaba altas y cancelaciones continuas y no era un libro sin
/// cancelaciones.
#[test]
fn cf2_spoof_tasa_de_falsas_alarmas_en_libro_sin_cancelaciones() {
    let n = 20_000usize;
    let (alarmas, medidos, movimientos) = alarmas_en_libro_cst(0x5E0F_0001, 0.0, n);
    let tasa = alarmas as f64 / medidos as f64;
    let cota = 0.05 + 4.0 * (0.05f64 * 0.95 / n as f64).sqrt();
    assert!(
        tasa <= cota,
        "falsas alarmas sin cancelaciones {tasa} > {cota} ({alarmas}/{medidos}; el mejor bid se movió en {movimientos} snapshots)"
    );
}

/// Verdad: libro CST con sus cancelaciones ordinarias (θ(i)·q) y sin el
/// patrón «muro grande retirado antes de que lo toquen»: no hay spoofing que
/// detectar. Misma tolerancia que el caso sin cancelaciones (≤ 0,0562). Aquí
/// el precio sí se mueve (el mejor bid cambia en ≈ 10 % de los snapshots) y
/// las alarmas que quedan son las de traslación y ejecución de
/// cf2_spoof_nulos_estructurales_sin_cancelacion, por debajo del 5 %.
#[test]
fn cf2_spoof_tasa_de_falsas_alarmas_en_libro_cst_ordinario() {
    let n = 20_000usize;
    let (alarmas, medidos, movimientos) = alarmas_en_libro_cst(0x5E0F_0002, 1.0, n);
    let tasa = alarmas as f64 / medidos as f64;
    let cota = 0.05 + 4.0 * (0.05f64 * 0.95 / n as f64).sqrt();
    assert!(
        tasa <= cota,
        "falsas alarmas en libro CST ordinario {tasa} > {cota} ({alarmas}/{medidos}; el mejor bid se movió en {movimientos} snapshots)"
    );
}

// ─────────────────────────────────────────────────────────────────
// Ballenas
// ─────────────────────────────────────────────────────────────────

/// Verdad: N = 200 000 nocionales LN(0,1) iid, sin ballenas. El z ≥ 2,5
/// declara una cola gaussiana nominal de 0,0062; se exige ≤ 2 × 0,0062. Con
/// los momentos verdaderos el umbral es e^(1/2) + 2,5·√((e−1)e) = 7,05 y
/// P(LN ≥ 7,05) = 0,0254 (banda 0,015–0,040 por el error de σ̂ EWMA con
/// n efectivo ≈ 2000 y curtosis 114). Error de Monte Carlo 0,00035.
#[test]
#[ignore = "OPEN F2.5-08 (Línea C): z gaussiano ≥ 2,5 sobre un nocional de cola pesada; con LN(0,1) marca ≈ 4 veces el nominal 0,0062"]
fn cf2_ballena_tasa_de_falsas_alarmas_sin_ballenas() {
    let mut rng = Lcg(0xBA11_E7A0);
    let mut tr = InstitutionalVolumeTracker::default();
    let (mut alarmas, mut medidas) = (0usize, 0usize);
    for i in 0..200_000usize {
        let (_z, _a, burst) = tr.update(rng.normal().exp());
        if i >= 2_000 {
            medidas += 1;
            alarmas += burst as usize;
        }
    }
    let tasa = alarmas as f64 / medidas as f64;
    assert!(
        tasa <= 2.0 * 0.0062,
        "tasa {tasa}: z no calibrado para cola pesada (esperado ~0,025 frente a 0,0062)"
    );
}

/// Verdad: ballenas plantadas (×50 sobre LN(0,1)) con probabilidad 0,01 y
/// etiquetas conocidas. La razón de verosimilitud de la mezcla es monótona en
/// x, así que el detector óptimo a una tasa de falsas alarmas dada es un
/// umbral fijo x* sobre el nocional (Neyman-Pearson). Se construye ese oráculo
/// con la MISMA tasa empírica de falsas alarmas del detector medido (x* = el
/// k-ésimo mayor nocional sin ballena, k = falsos positivos + 1) y se exige
/// recall ≥ recall_oráculo − 4·EE, con EE = √(r(1−r)/n_ballenas) (n ≈ 1980),
/// además de precisión ≥ 0,6 (criterio del plan; esperada ≈ 0,95–0,98). El
/// recall ≥ 0,9 del borrador era un objetivo de diseño, no una tolerancia
/// derivada. Hoy el detector queda ≈ 4,7 EE por debajo del oráculo: el σ̂
/// EWMA inflado por las propias ballenas mueve el umbral en el tiempo y, a la
/// misma tasa de falsas alarmas, pierde recall frente al umbral fijo.
#[test]
#[ignore = "OPEN F2.5-08 (Línea C): las ballenas inflan el σ̂ EWMA y se enmascaran; a la misma tasa de falsas alarmas el recall queda por debajo del oráculo de Neyman-Pearson"]
fn cf2_ballena_plantada_x50_al_uno_por_ciento() {
    let mut rng = Lcg(0xBA11_E7A1);
    let mut tr = InstitutionalVolumeTracker::default();
    let (mut tp, mut fp, mut fneg) = (0usize, 0usize, 0usize);
    let (mut sin_ballena, mut con_ballena) = (Vec::new(), Vec::new());
    for i in 0..200_000usize {
        let ballena = rng.unif() < 0.01;
        let x = rng.normal().exp() * if ballena { 50.0 } else { 1.0 };
        let (_z, _a, burst) = tr.update(x);
        if i < 2_000 {
            continue;
        }
        if ballena {
            con_ballena.push(x);
        } else {
            sin_ballena.push(x);
        }
        match (ballena, burst) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (true, false) => fneg += 1,
            _ => {}
        }
    }
    let precision = tp as f64 / (tp + fp).max(1) as f64;
    let recall = tp as f64 / (tp + fneg).max(1) as f64;
    // Oráculo de Neyman-Pearson a la misma tasa de falsas alarmas.
    sin_ballena.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let umbral = sin_ballena[fp.min(sin_ballena.len() - 1)];
    let recall_oraculo =
        con_ballena.iter().filter(|&&x| x > umbral).count() as f64 / con_ballena.len() as f64;
    let ee = (recall_oraculo * (1.0 - recall_oraculo) / con_ballena.len() as f64).sqrt();
    assert!(
        precision >= 0.6,
        "precisión {precision} (tp {tp}, fp {fp})"
    );
    assert!(
        recall >= recall_oraculo - 4.0 * ee,
        "recall {recall} frente al oráculo de Neyman-Pearson {recall_oraculo} (EE {ee}, umbral {umbral}, fp {fp})"
    );
}

// ─────────────────────────────────────────────────────────────────
// OFI
// ─────────────────────────────────────────────────────────────────

/// Verdad: definición cerrada de Cont-Kukanov-Stoikov (2014) evento a evento,
/// con la normalización del código (max(prof_t, prof_(t−1), 1e-6), con
/// prev_depth = 0 tras la inicialización, microstructure.rs:139-169). Se
/// recupera ofi_norm = (ema_n − 0,9·ema_(n−1))/0,1 y se compara con e_n/prof;
/// error de redondeo de la recuperación ≈ 1e-14, tolerancia 1e-8·(1 + |e|).
#[test]
fn cf2_ofi_coincide_con_cont_kukanov_stoikov_evento_a_evento() {
    let mut rng = Lcg(0x0F1_2014);
    let mut m = OFIModel::new();
    let (mut b, mut a, mut qb, mut qa) = (100.0f64, 100.01f64, 5.0f64, 5.0f64);
    m.update(b, a, qb, qa); // inicializa sin fijar prev_depth (microstructure.rs:139-145)
    let (mut pb, mut pa, mut pqb, mut pqa) = (b, a, qb, qa);
    let (mut ema_prev, mut prof_prev) = (0.0f64, 0.0f64);
    let paso = |u: f64| {
        if u < 0.25 {
            -0.01
        } else if u < 0.5 {
            0.01
        } else {
            0.0
        }
    };
    let mut peor = 0.0f64;
    for n in 0..10_000 {
        b += paso(rng.unif());
        a += paso(rng.unif());
        if a <= b {
            a = b + 0.01;
        }
        qb = (1.0 + rng.normal()).exp();
        qa = (1.0 + rng.normal()).exp();
        let e_bid = if b > pb {
            qb
        } else if b == pb {
            qb - pqb
        } else {
            -pqb
        };
        let e_ask = if a < pa {
            qa
        } else if a == pa {
            qa - pqa
        } else {
            -pqa
        };
        let esperado = ((e_bid - e_ask) / (qb + qa).max(prof_prev).max(1e-6)).clamp(-10.0, 10.0);
        let ema = m.update(b, a, qb, qa);
        let recuperado = (ema - 0.9 * ema_prev) / 0.1;
        let err = (recuperado - esperado).abs();
        peor = peor.max(err / (1.0 + esperado.abs()));
        assert!(
            err <= 1e-8 * (1.0 + esperado.abs()),
            "evento {n}: OFI recuperado {recuperado} frente a CKS {esperado} (peor error relativo {peor:e})"
        );
        ema_prev = ema;
        prof_prev = qb + qa;
        pb = b;
        pa = a;
        pqb = qb;
        pqa = qa;
    }
}

/// Verdad: el libro sube un tick por snapshot (e = q + q y profundidad 2q:
/// OFI normalizado +1 por snapshot). Entre dos snapshots se meten k ∈ {0, 9}
/// eventos de trade con el libro quieto, que es lo que hace el núcleo al
/// llamar update_ofi en todo evento (god-engine-core/src/lib.rs:2806 → :3091).
/// La suma de CKS por intervalo no cambia con eventos que no mueven el libro:
/// se exige que las dos lecturas difieran ≤ 5 %. Hoy la EMA es por evento
/// (microstructure.rs:190) y cada trade le mete un 0: 1,0 sin trades frente a
/// 0,1/(1 − 0,9^10) = 0,1535 con 9.
#[test]
#[ignore = "OPEN F2.5-09 (Línea C): la EMA del OFI es por evento y cada trade con el libro quieto le mete un 0; con 9 trades por snapshot la lectura cae al 15 %"]
fn cf2_ofi_no_depende_del_numero_de_trades_entre_snapshots() {
    fn lectura(trades_entre_libros: usize) -> f64 {
        let mut m = OFIModel::new();
        let (mut bid, tick, q) = (100.0f64, 0.01f64, 10.0f64);
        m.update(bid, bid + tick, q, q);
        let mut ultima = 0.0;
        for _ in 0..200 {
            bid += tick;
            ultima = m.update(bid, bid + tick, q, q);
            for _ in 0..trades_entre_libros {
                m.update(bid, bid + tick, q, q);
            }
        }
        ultima
    }
    let (sin, con) = (lectura(0), lectura(9));
    assert!(
        (con - sin).abs() <= 0.05 * sin.abs(),
        "OFI {sin} sin trades frente a {con} con 9 trades (analítico 1,0 y 0,1535)"
    );
}
