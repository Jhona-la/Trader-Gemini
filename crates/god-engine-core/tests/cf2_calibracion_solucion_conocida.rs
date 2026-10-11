//! C-F2.2 (Línea C): calibración de Platt, predicción conformal y ATR vivo
//! contra soluciones conocidas.
//!
//! Teoría que se prueba:
//! - Platt (`PlattCalibrator`) recupera un mapa logístico conocido cuando el
//!   generador está dentro de su familia, y NO puede identificar la tasa de
//!   puntuaciones que nunca se ejecutaron (selección por ejecución).
//! - El conjunto conformal con ACI (`ConformalCalibrator`) cubre al nominal
//!   en un flujo intercambiable con probabilidades verdaderas; con selección
//!   de qué se actualiza la cobertura en los candidatos se aparta del
//!   nominal, y α no acota la pérdida entre las aceptadas (riesgo selectivo).
//! - El ATR vivo de `StatefulEngine` convertido con `sigma_from_atr` debe dar
//!   la σ de un minuto de un paseo log-normal conocido (rango discreto de 60
//!   ticks: razón teórica 0,906).
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md): el valor verdadero lo fija el
//! generador. Convención: un test que hoy falla por la causa de un hallazgo
//! lleva `#[ignore = "OPEN F2.x-nn (dueño): causa"]` con el assert intacto;
//! documenta el defecto y se activa con el arreglo. Para correrlos:
//! `cargo test ... -- --include-ignored`.

use god_engine_core::calibration::PlattCalibrator;
use god_engine_core::conformal::ConformalCalibrator;
use god_engine_core::diffusion::sigma_from_atr;
use god_engine_core::stateful_engine::StatefulEngine;

/// LCG de 64 bits (constantes de Knuth/MMIX) con salida de 53 bits en (0, 1).
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    /// Normal estándar por Box-Muller.
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

// ─────────────────────────────── Platt ───────────────────────────────

/// Verdad: p(s) = σ(0,5·logit(s) − 0,3), dentro de la familia de Platt.
/// Tolerancia 0,04: el error de 10 × 512 observaciones en la media es
/// ≈ 0,01 (medido entre réplicas: SE 0,006-0,009, así que 0,04 son ≥ 4 SE)
/// y el empuje de las pseudo-observaciones del prior (masa z² ≈ 3,84) es
/// < 0,002. Es la prueba más cara del archivo (≈ 3,4 s en debug bajo wine:
/// cada `update` reajusta Platt sobre la ventana).
/// Nota del escéptico de completitud: pasa por construcción; no prueba el
/// criterio de la fila 26 (Brier/ECE frente a Platt) ni un generador fuera
/// de la familia logística (FA-10, pendiente).
#[test]
fn cf2_platt_recupera_un_mapa_logistico_conocido() {
    let (a0, b0) = (0.5, -0.3);
    let puntos = [0.6, 0.75, 0.9];
    let reps = 10;
    let mut g = Lcg::new(26);
    let mut media = [0.0f64; 3];
    let mut cuad = [0.0f64; 3];
    for _ in 0..reps {
        let mut c = PlattCalibrator::new();
        for _ in 0..512 {
            let s = 0.5 + 0.45 * g.u();
            let gana = g.u() < sig(a0 * logit(s) + b0);
            c.update(s, gana);
        }
        for (i, &s) in puntos.iter().enumerate() {
            let p = c.calibrate(s);
            media[i] += p / reps as f64;
            cuad[i] += p * p / reps as f64;
        }
    }
    for (i, &s) in puntos.iter().enumerate() {
        let se = (cuad[i] - media[i] * media[i]).max(0.0).sqrt() / (reps as f64).sqrt();
        println!(
            "observado: s = {s}: media {:.4}, verdad {:.4}, SE entre réplicas {se:.4}",
            media[i],
            sig(a0 * logit(s) + b0)
        );
    }
    for (i, &s) in puntos.iter().enumerate() {
        let verdad = sig(a0 * logit(s) + b0);
        assert!(
            (media[i] - verdad).abs() <= 0.04,
            "s = {s}: {:.4} frente a la verdad {verdad:.4}",
            media[i]
        );
    }
}

/// Verdad: p(s) = 0,30 si s < 0,7 y 0,60 si s ≥ 0,7; sólo se ejecutan
/// (y alimentan `update_at`, como al cierre en el núcleo) puntuaciones en
/// [0,7; 0,95], una por minuto. Esperado: calibrate_at(0,5) = 0,30 ± 0,05.
/// Hoy, con datos planos, a → 0 y b → logit(0,6): sirve ≈ 0,59.
///
/// Es una puerta del SUCESOR, no de calibration.rs: la verdad en s = 0,5 no
/// es identificable con datos en [0,7; 0,95], así que ningún calibrador
/// entrenado sólo con lo ejecutado puede pasarla. Necesita una entrada nueva
/// (resultado de barrera de intenciones vetadas, con edad: fila 38,
/// shadow_ledger) o que `calibrate_at` declare «sin soporte» fuera del rango
/// ejecutado.
#[test]
#[ignore = "OPEN F2.2-06 (Línea C, C-F2.2): Platt sólo aprende de lo ejecutado y extrapola su tasa (≈0,59) a puntuaciones nunca ejecutadas (verdad 0,30)"]
fn cf2_platt_no_extrapola_fuera_de_lo_ejecutado() {
    let mut g = Lcg::new(37);
    let mut c = PlattCalibrator::new();
    let mut t = 0u64;
    for _ in 0..512 {
        t += 60_000;
        let s = 0.7 + 0.25 * g.u();
        c.update_at(s, g.u() < 0.60, t);
    }
    let p = c.calibrate_at(0.5, t);
    let (a, b) = c.coefficients();
    assert!(
        (p - 0.30).abs() <= 0.05,
        "p(0,5) servida = {p:.4} con (a, b) = ({a:.4}, {b:.4}); verdad 0,30 fuera del soporte ejecutado"
    );
}

// ────────────────────────────── Conformal ─────────────────────────────

/// ¿Contiene el conjunto conformal al nivel efectivo la etiqueta realizada?
/// Misma regla que `ConformalCalibrator::update` (conformal.rs:182-188),
/// evaluada con el estado ANTERIOR a incorporar el dato.
fn en_conjunto(c: &ConformalCalibrator, p: f64, gana: bool) -> bool {
    c.p_value_for(p, gana) > c.effective_alpha().clamp(1e-4, 0.5)
}

/// Resultado de `cobertura_conformal`.
struct Cobertura {
    /// Error del conjunto al nivel EFECTIVO (ACI) en todos los candidatos.
    error: f64,
    /// Error del conjunto al nivel NOMINAL fijo 0,10 (sin ACI) en todos los
    /// candidatos: mide el p-valor de rangos, no la recursión de ACI.
    error_nominal_fijo: f64,
    /// `adaptive_error_rate` de los actualizados.
    error_adaptativo: f64,
    /// Pérdida entre los candidatos que `accepts` admitiría.
    riesgo: f64,
    aceptadas: usize,
}

/// Flujo de 20 000 candidatos con p ~ U(0,3; 0,9) y gana ~ Bernoulli(p)
/// (probabilidades verdaderas por construcción). Sólo llegan a `update` los
/// candidatos que pasan `seleccion`. Desde el paso 1 000 se mide, en TODOS
/// los candidatos: el error del conjunto (al nivel efectivo y al nominal
/// fijo), y la pérdida entre los que `accepts` admitiría.
fn cobertura_conformal(seleccion: impl Fn(f64) -> bool, seed: u64) -> Cobertura {
    let mut g = Lcg::new(seed);
    let mut c = ConformalCalibrator::new();
    c.set_target_alpha(0.10);
    let (mut fuera, mut fuera_fijo, mut n, mut acept, mut perd) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    for t in 0..20_000 {
        let p = 0.3 + 0.6 * g.u();
        let gana = g.u() < p;
        if t >= 1_000 {
            n += 1;
            if !en_conjunto(&c, p, gana) {
                fuera += 1;
            }
            if c.p_value_for(p, gana) <= 0.10 {
                fuera_fijo += 1;
            }
            if c.accepts(p) {
                acept += 1;
                if !gana {
                    perd += 1;
                }
            }
        }
        if seleccion(p) {
            c.update(p, gana);
        }
    }
    Cobertura {
        error: fuera as f64 / n as f64,
        error_nominal_fijo: fuera_fijo as f64 / n as f64,
        error_adaptativo: c.adaptive_error_rate(),
        riesgo: perd as f64 / acept.max(1) as f64,
        aceptadas: acept,
    }
}

/// Verdad: con flujo intercambiable y todo actualizado, ACI cubre al
/// nominal: error 0,10. Tolerancia 0,02: la cota de seguimiento de ACI da
/// |media − α| ≤ 0,5/(γ·T) = 0,5/(0,005·19 000) ≈ 0,005 mientras no muerda el
/// recorte, y el error binomial de 19 000 pasos es √(0,09/19 000) ≈ 0,0022;
/// 0,02 son > 4 veces la suma.
///
/// Corrección del revisor: esas dos aserciones las cumple ACI por su
/// recursión telescópica con CUALQUIER secuencia (Gibbs y Candès 2021),
/// aunque el p-valor de rangos estuviera roto: ACI movería α_eff hasta
/// compensarlo. La solución conocida del flujo intercambiable es otra y se
/// afirma aparte: con la ventana llena (n = 200) y puntuaciones continuas,
/// el rango de la nueva puntuación es uniforme en {1, …, 201}, así que
/// P(p_y ≤ 0,10) = ⌊0,10·201⌋/201 = 20/201 = 0,0995 al nivel NOMINAL fijo,
/// sin ACI. SE binomial √(0,0995·0,9005/19 000) ≈ 0,0022 (las ventanas
/// solapadas apenas correlacionan los rangos); tolerancia 0,01 ≈ 4,6 SE.
#[test]
fn cf2_conformal_cobertura_marginal_al_nominal() {
    let r = cobertura_conformal(|_| true, 42);
    let (err, err_fijo, err_adapt) = (r.error, r.error_nominal_fijo, r.error_adaptativo);
    let exacto = 20.0 / 201.0;
    println!(
        "observado: error en el conjunto {err:.4}, adaptive_error_rate {err_adapt:.4}, \
         error al nominal fijo {err_fijo:.4} (exacto {exacto:.4})"
    );
    assert!(
        (err - 0.10).abs() <= 0.02 && (err_adapt - 0.10).abs() <= 0.02,
        "error en el conjunto {err:.4}, adaptive_error_rate {err_adapt:.4}; nominal 0,10"
    );
    assert!(
        (err_fijo - exacto).abs() <= 0.01,
        "error al nivel nominal fijo {err_fijo:.4} frente al exacto 20/201 = {exacto:.4}"
    );
}

/// Verdad: la cobertura en los CANDIDATOS debe ser la nominal (0,10 ± 0,02,
/// misma justificación que la marginal). Con selección (sólo se actualiza lo
/// ejecutado, aquí p < 0,6, como en vivo sólo cierran las posiciones
/// abiertas) ACI cubre al nominal en las ejecutadas y se aparta en los
/// candidatos: la simulación del auditor da ≈ 0,165.
///
/// Ningún cambio en conformal.rs la pondría verde: hace falta alimentar a
/// los candidatos con su resultado (contrafactuales de la fila 38) en el
/// alimentador del núcleo (lib.rs, actualización al cierre). La regla
/// p < 0,6 es ilustrativa, no la selección viva.
#[test]
#[ignore = "OPEN F2.2-05 (Línea C, C-F2.2): la cobertura conformal sólo se mide y adapta en las ejecutadas; en los candidatos se aparta del nominal"]
fn cf2_conformal_cobertura_en_candidatos_con_seleccion() {
    let r = cobertura_conformal(|p| p < 0.6, 7);
    let (err, err_ejecutadas) = (r.error, r.error_adaptativo);
    assert!(
        (err - 0.10).abs() <= 0.02,
        "error en candidatos {err:.4} frente al nominal 0,10; en ejecutadas {err_ejecutadas:.4}"
    );
}

/// Puerta del sucesor (selección conformal con control del FDR / riesgo
/// selectivo, p. ej. Jin y Candès 2023, o Learn-then-Test): la pérdida entre
/// las aceptadas debe quedar en α + 0,02 = 0,12. El doc de conformal.rs ya
/// avisa de que la regla singleton no lo garantiza; el defecto es usarla
/// como veto de riesgo en vivo, no del módulo. Hoy ≈ 0,216 (ver el testigo
/// analítico de abajo).
#[test]
#[ignore = "OPEN F2.2-05 (Línea C, C-F2.2): α no acota la pérdida entre las aceptadas (≈0,216 con α = 0,10) y el filtro vivo lo usa como veto"]
fn cf2_conformal_riesgo_entre_aceptadas() {
    let r = cobertura_conformal(|_| true, 42);
    let (riesgo, acept) = (r.riesgo, r.aceptadas);
    assert!(
        riesgo <= 0.12,
        "pérdida entre {acept} aceptadas = {riesgo:.4} con α = 0,10"
    );
}

/// Testigo del valor conocido (corrección del escéptico de refutar): con
/// todo actualizado, el conjunto es {gana} cuando la no-conformidad de
/// «pierde» (s = p) cae en la cola superior α de las puntuaciones. Para
/// p ~ U(0,3; 0,9) y gana ~ Bernoulli(p), P(S ≥ x) = (2(1 − x)² − 0,10)/1,2
/// en x ∈ [0,3; 0,7]; P(S ≥ t) = 0,10 da t = 1 − √0,11 ≈ 0,668, y la
/// pérdida entre aceptadas es E[1 − p | p ∈ U(t; 0,9)] = (1,1 − t)/2 ≈ 0,216.
/// Tolerancia 0,02: el umbral de una ventana de 200 tiene SE ≈ 0,019 en t
/// (≈ 0,0096 en la pérdida) por ventana; con ≈ 95 ventanas independientes en
/// 19 000 pasos el SE de la media es ≈ 0,001, y el sesgo por ponderar por
/// aceptación es ≈ +0,001; 0,02 son > 10 SE.
#[test]
fn cf2_conformal_riesgo_entre_aceptadas_valor_analitico() {
    let t = 1.0 - 0.11f64.sqrt();
    let esperado = (1.1 - t) / 2.0;
    let (_, _, riesgo, acept) = cobertura_conformal(|_| true, 42);
    println!("observado: pérdida entre {acept} aceptadas {riesgo:.4}; analítico {esperado:.4}");
    assert!(
        (riesgo - esperado).abs() <= 0.02,
        "pérdida entre {acept} aceptadas = {riesgo:.4} frente al valor analítico {esperado:.4} (umbral {t:.4})"
    );
}

// ──────────────────────────────── ATR ────────────────────────────────

/// Paseo log-normal con σ por minuto `sigma_min`, un tick por segundo
/// (60 por minuto) durante 600 minutos y volumen 0. Devuelve la media de
/// σ_ATR/σ leída en cada tick tras 100 minutos de calentamiento (el arranque
/// de 50 pb decae 0,85 por minuto: 0,85^100 ≈ 1e-7), que es lo que ve un
/// consumidor vivo que lee el ATR en eventos arbitrarios.
fn razon_atr_media(sigma_min: f64, semilla: u64) -> f64 {
    let sigma_tick = sigma_min / 60f64.sqrt();
    let mut g = Lcg::new(semilla);
    let mut e = StatefulEngine::new();
    let mut ln_p = 100f64.ln();
    let (mut suma, mut n) = (0.0f64, 0usize);
    for i in 0..(600 * 60) as u64 {
        ln_p += sigma_tick * g.normal();
        e.process_tick(ln_p.exp(), 0.0, 1_000 * i);
        if i >= 100 * 60 {
            suma += sigma_from_atr(e.get_atr_pct()) / sigma_min;
            n += 1;
        }
    }
    suma / n as f64
}

/// Verdad: un ATR de 1 minuto sin piso ni trinquete da E[R] = σ·(√(8/π) −
/// 2·0,5826/√60) para el rango de 60 ticks (corrección discreta de Broadie,
/// Glasserman y Kou), así que σ_ATR/σ = 1 − 2·0,5826/(√60·√(8/π)) ≈ 0,906.
/// Banda [0,80; 1,10]: una lectura suelta tiene SD ≈ 0,085 (EWMA 0,15 del
/// rango, SD del rango ≈ 0,4755σ), pero la media de 30 000 lecturas con
/// autocorrelación 0,85 por minuto tiene SE ≈ 0,085·√(12,3/500) ≈ 0,013: la
/// banda queda a −8 y +15 SE del valor teórico.
/// Hoy el piso de 0,0010·precio (10 pb) domina: σ_ATR ≥ 10 pb/1,596 =
/// 6,27 pb, razón ≥ 1,566. La banda exige retirar también el trinquete (ver
/// el gemelo de 30 pb).
#[test]
#[ignore = "OPEN F2.2-07 (Línea C, C-F4): el piso de 10 pb y el trinquete del ATR vivo violan σ = ATR/√(8/π) (razón ≥ 1,566 con σ = 4 pb/min)"]
fn cf2_atr_vivo_da_la_sigma_de_un_minuto() {
    let razon = razon_atr_media(0.0004, 55);
    assert!(
        razon > 0.80 && razon < 1.10,
        "σ_ATR/σ medio = {razon:.4} con σ = 4 pb/min; teórico 0,906, piso 10 pb ⇒ ≥ 1,566"
    );
}

/// Gemelo del escéptico de completitud: con σ = 30 pb/min el rango típico
/// (≈ 43 pb) está muy por encima del piso de 10 pb, así que sólo queda el
/// trinquete intra-vela (v_t sube al rango de la vela en curso y al cierre
/// promedia 0,85·max(v_t, R) + 0,15·R). Misma verdad (0,906) y misma banda
/// que la prueba de 4 pb. Así un arreglo que sólo quite el piso sabe por qué
/// sigue fallando. Punto fijo aproximado del trinquete con R gaussiano:
/// v* ≈ E[R] + 0,745·SD(R) ⇒ razón ≈ 1,13.
#[test]
#[ignore = "OPEN F2.2-07 (Línea C, C-F4): el trinquete intra-vela del ATR vivo infla σ_ATR/σ por encima de 1,10 aun sin piso (σ = 30 pb/min)"]
fn cf2_atr_vivo_sin_piso_aisla_el_trinquete() {
    let razon = razon_atr_media(0.0030, 56);
    assert!(
        razon > 0.80 && razon < 1.10,
        "σ_ATR/σ medio = {razon:.4} con σ = 30 pb/min (piso inactivo); teórico 0,906"
    );
}
