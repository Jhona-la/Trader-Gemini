//! C-F2.3a (Línea C): campo de régimen espectral y BOCPD sobre el transporte
//! W1, contra soluciones conocidas (sólo API pública).
//!
//! Teoría que se prueba:
//! - Una caminata aleatoria no tiene régimen, ni caída, ni aceleración de
//!   escala, ni antipersistencia: cualquier densidad de evidencia de caída
//!   (`crash_flux`) debe ser baja en ruido, que es la falsación que declara
//!   el propio módulo (spectral_regime.rs, cabecera: «ruido gaussiano ⇒
//!   crash_flux bajo»).
//! - Una caminata aleatoria estacionaria no tiene ningún cambio de régimen:
//!   toda alarma del BOCPD (`p_transition > 0,5`) es falsa. Un cambio
//!   plantado de gran magnitud debe detectarse.
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`.

use god_engine_core::{GodEngineCore, W1ChangepointObserver, referencia_de_regimen};
use quantum_arena::spectral_regime::SpectralRegimeField;
use quantum_arena::temporal_spectrum::{TemporalSpectrum, W1_LAG_FISICO_MS};
use quantum_arena::{GlobalArena, symbol_registry, symbols};
use std::sync::Mutex;
use std::sync::atomic::Ordering;

/// Los tests con `GodEngineCore` tocan el universo y el registro de símbolos
/// globales del proceso.
static ENTORNO_GLOBAL: Mutex<()> = Mutex::new(());

/// LCG de 64 bits (constantes de Knuth/MMIX) con salida de 53 bits en (0, 1).
struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    /// Normal estándar por Box-Muller.
    fn n(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

const T0: u64 = 1_700_000_000_000;
const DT: u64 = 1_000;
const P0: f64 = 100.0;

/// Réplica del ancla de régimen del núcleo (lib.rs, bloque de
/// `referencia_de_regimen` en `process_event` y en `process_tick_dual`).
fn campo_vivo(spec: &TemporalSpectrum, ancla: &mut (u64, u64), ts: u64) -> SpectralRegimeField {
    let field = spec.spectral_field(true);
    let cur_ln = if field.resonant_tau_ms.is_finite() && field.resonant_tau_ms > 0.0 {
        field.resonant_tau_ms.ln()
    } else {
        0.0
    };
    let (prev, elapsed, actualizar) = referencia_de_regimen(ancla.0, ancla.1, ts);
    if actualizar && cur_ln > 0.0 {
        *ancla = (cur_ln.to_bits(), ts);
    }
    SpectralRegimeField::from_spectrum(spec, prev, elapsed)
}

/// Verdad: en una caminata aleatoria no hay evidencia de caída. Contrato del
/// propio módulo: media de crash_flux < 0,10 y fracción de eventos con la
/// deriva de τ* saturada (|dominant_drift| ≥ 0,99) < 0,10. Los dos umbrales
/// son POLÍTICA (el módulo dice «bajo» sin número), no bandas estadísticas;
/// el nulo natural de la fracción saturada es ≈ 0 (una caminata no mueve su
/// escala dominante). Se reproduce el ancla como en vivo: `process_event`
/// y `process_tick_dual` recalculan el campo en el MISMO evento y se publica
/// el segundo (cuando el ancla se renueva, el segundo ve elapsed = 1 ms y
/// deriva ≈ 0). Hoy: media ≈ 0,41 (deriva ≈ 0,17, marea ≈ 0,13, entropía
/// ≈ 0,09) y deriva saturada en ≈ 91 % de los eventos.
#[test]
#[ignore = "OPEN F2.3-05 (Línea C, C-F2.3): crash_flux vale ≈0,41 en una caminata aleatoria: la deriva de τ* satura en ruido y la marea y el piso de entropía nunca son 0"]
fn cf2_rw_crash_flux_es_bajo_en_ruido() {
    let mut g = Lcg(0xC0F2_0901);
    let mut spec = TemporalSpectrum::new();
    let mut ancla = (0u64, 0u64);
    let mut x = 0.0f64;
    let (mut suma, mut saturada, mut n) = (0.0f64, 0usize, 0usize);
    let (mut deriva, mut marea, mut entropia, mut micro) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..21_600u64 {
        x += 1e-4 * g.n();
        let ts = T0 + i * DT;
        spec.update(P0 * x.exp(), ts);
        let _ = campo_vivo(&spec, &mut ancla, ts); // process_event
        let campo = campo_vivo(&spec, &mut ancla, ts); // process_tick_dual (el publicado)
        if i >= 7_200 {
            suma += campo.crash_flux;
            if campo.dominant_drift.abs() >= 0.99 {
                saturada += 1;
            }
            deriva += 0.35 * (-campo.dominant_drift).clamp(0.0, 1.0);
            marea += 0.30 * campo.carrier_tide.abs().clamp(0.0, 1.0);
            entropia += 0.20 * (1.0 - campo.mass_entropy).clamp(0.0, 1.0);
            micro += 0.15 * (0.5 - campo.hurst_by_band[0]).max(0.0) / 0.5;
            n += 1;
        }
    }
    let nf = n as f64;
    let media = suma / nf;
    println!(
        "observado: crash_flux medio {media:.4}; aportes: deriva {:.4}, marea {:.4}, entropía {:.4}, micro {:.4}; deriva saturada {saturada}/{n}",
        deriva / nf,
        marea / nf,
        entropia / nf,
        micro / nf
    );
    assert!(
        media < 0.10,
        "crash_flux medio en ruido = {media:.4} (aportes: deriva {:.4}, marea {:.4}, entropía {:.4}, micro {:.4})",
        deriva / nf,
        marea / nf,
        entropia / nf,
        micro / nf
    );
    assert!(
        (saturada as f64) < 0.10 * nf,
        "deriva de τ* saturada en {saturada}/{n} eventos de ruido"
    );
}

/// Núcleo real (corrección del escéptico del espectro): el mismo ruido por
/// `GodEngineCore::process_event` (eventos de depth a 1 Hz), leyendo lo que
/// publica el núcleo en `spectral_crash_flux` y `spectral_coherence`, y el
/// factor de margen que aplica el lector vivo a un largo:
/// free_cap ×= 1 − 0,95·flux si la marea es adversa (tide < 0); a un corto
/// si tide > 0. Contrato de efecto neutro en ruido: E[factor] ≥ 0,95 para
/// cada lado (política, como el umbral de 0,10 de la media). Comprobación
/// cruzada hecha al escribir la prueba: con las mismas 21 600 entradas que la
/// réplica de arriba, el núcleo publicó el mismo crash_flux medio (0,4059),
/// así que la réplica reproduce el cableado. Hoy, con 14 400 eventos: media
/// ≈ 0,41, factor medio ≈ 0,84 (largo) y ≈ 0,77 (corto).
#[test]
#[ignore = "OPEN F2.3-05 (Línea C, C-F2.3): el núcleo publica crash_flux ≈0,41 en ruido y recorta el margen libre un 16-23 % de media sin ninguna caída"]
fn cf2_rw_crash_flux_es_bajo_en_ruido_en_el_nucleo() {
    let _guarda = ENTORNO_GLOBAL.lock().unwrap_or_else(|p| p.into_inner());
    symbols::update_dynamic_universe(vec!["CF2RUIDOUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(
        "CF2RUIDOUSDT",
    )]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    let mut g = Lcg(0xC0F2_0901);
    let mut x = 0.0f64;
    let (mut suma, mut f_largo, mut f_corto, mut n) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    // 4 h de eventos (2 h medidas) en lugar de 6 h: el núcleo completo cuesta
    // ≈ 0,18 ms por evento y la prueba debe caber en ~5 s cuando se active.
    for i in 0..14_400u64 {
        x += 1e-4 * g.n();
        let ts = T0 + i * DT;
        let mid = P0 * x.exp();
        let (bid, ask) = (mid - 0.005, mid + 0.005);
        let _ = core.process_event(
            0, false, false, true, mid, 0.0, bid, ask, 5.0, 5.0, 0.0, 0.0, ts, false, &[0.0; 54],
            false,
        );
        if i >= 7_200 {
            let flux = arena.coins[0]
                .spectral_crash_flux
                .load(Ordering::Relaxed)
                .clamp(0.0, 1.0);
            let tide = arena.coins[0].spectral_coherence.load(Ordering::Relaxed);
            suma += flux;
            f_largo += if tide < 0.0 {
                (1.0 - 0.95 * flux).clamp(0.05, 1.0)
            } else {
                1.0
            };
            f_corto += if tide > 0.0 {
                (1.0 - 0.95 * flux).clamp(0.05, 1.0)
            } else {
                1.0
            };
            n += 1;
        }
    }
    let nf = n as f64;
    let (media, fl, fc) = (suma / nf, f_largo / nf, f_corto / nf);
    println!(
        "observado (núcleo): crash_flux medio {media:.4}; factor de margen largo {fl:.4}, corto {fc:.4}"
    );
    assert!(
        media < 0.10 && fl >= 0.95 && fc >= 0.95,
        "núcleo en ruido: crash_flux medio {media:.4}; factor de margen medio largo {fl:.4}, corto {fc:.4}"
    );
}

/// Verdad: ninguna alarma es verdadera en una caminata estacionaria. Se
/// exige una fracción de eventos con p_transition > 0,5 menor que 0,05 (el
/// umbral de falsas alarmas del plan), con > 10 000 observaciones W1.
/// Pipeline del núcleo: W1 con lag físico de 60 s y `observe_at` por evento.
/// Predicción «incierta»; la ejecución da 60/14 400 (0,4 %): pasa. Pero la
/// fracción de eventos no es una métrica de detección de cambios: los
/// mismos datos dan 56 EPISODIOS de alarma en 4 h (≈ 14 por hora, ARL0 ≈
/// 4 min). Esa parte la fija la prueba de cadencia de abajo.
#[test]
fn cf2_rw_bocpd_no_dispara_en_ruido() {
    let mut g = Lcg(0xC0F2_1001);
    let mut spec = TemporalSpectrum::new();
    let mut obs = W1ChangepointObserver::new();
    let mut x = 0.0f64;
    let (mut alarmas, mut n, mut episodios, mut previa) = (0usize, 0usize, 0usize, false);
    let mut w1_suma = 0.0f64;
    let mut w1_cuad = 0.0f64;
    for i in 0..21_600u64 {
        x += 1e-4 * g.n();
        let ts = T0 + i * DT;
        spec.update(P0 * x.exp(), ts);
        if let Some(w1) = spec.spectral_transport_w1_fisico(W1_LAG_FISICO_MS) {
            if let Some(p) = obs.observe_at(ts, w1) {
                if i >= 7_200 {
                    n += 1;
                    w1_suma += w1;
                    w1_cuad += w1 * w1;
                    let alarma = p > 0.5;
                    if alarma {
                        alarmas += 1;
                        if !previa {
                            episodios += 1;
                        }
                    }
                    previa = alarma;
                }
            }
        }
    }
    let w1m = w1_suma / n.max(1) as f64;
    let w1sd = (w1_cuad / n.max(1) as f64 - w1m * w1m).max(0.0).sqrt();
    println!(
        "observado: {alarmas}/{n} eventos con p > 0,5; {episodios} episodios; W1 medio {w1m:.4}, sd {w1sd:.4}"
    );
    assert!(n > 10_000, "pocas observaciones W1 ({n})");
    assert!(
        (alarmas as f64) < 0.05 * n as f64,
        "{alarmas}/{n} eventos con p_transition > 0,5 en ruido ({episodios} episodios)"
    );
}

/// Episodios de alarma (flancos de subida de p_transition > 0,5) por hora
/// en una caminata aleatoria con σ = 1e-4 por √s, servida a `hz` eventos por
/// segundo (pipeline del núcleo: W1 con lag físico de 60 s y `observe_at`).
/// Devuelve (episodios por hora, episodios, observaciones medidas).
fn episodios_por_hora(
    hz: u64,
    calentamiento_s: u64,
    medida_s: u64,
    semilla: u64,
) -> (f64, usize, usize) {
    let mut g = Lcg(semilla);
    let mut spec = TemporalSpectrum::new();
    let mut obs = W1ChangepointObserver::new();
    let dt = 1_000 / hz;
    let sd = 1e-4 / (hz as f64).sqrt();
    let desde = calentamiento_s * hz;
    let total = (calentamiento_s + medida_s) * hz;
    let mut x = 0.0f64;
    let (mut episodios, mut n, mut previa) = (0usize, 0usize, false);
    for i in 0..total {
        x += sd * g.n();
        let ts = T0 + i * dt;
        spec.update(P0 * x.exp(), ts);
        if let Some(w1) = spec.spectral_transport_w1_fisico(W1_LAG_FISICO_MS) {
            if let Some(p) = obs.observe_at(ts, w1) {
                if i >= desde {
                    n += 1;
                    let alarma = p > 0.5;
                    if alarma && !previa {
                        episodios += 1;
                    }
                    previa = alarma;
                }
            }
        }
    }
    (episodios as f64 * 3_600.0 / medida_s as f64, episodios, n)
}

/// Propuesta de F2.3-08 y del escéptico del espectro (ARL0, Lorden 1971):
/// el MISMO proceso físico (σ = 1e-4 por √s) servido a 1 Hz y a 10 Hz debe
/// dar la misma tasa de falsas alarmas por HORA. Contrato:
/// |ln(r10/r1)| ≤ 4·√(1/c1 + 1/c10), el error estándar de Poisson del
/// logaritmo de la razón de tasas con c1 y c10 episodios (corrección de
/// continuidad +0,5 por si alguna cuenta es 0). La agrupación de episodios
/// hace la banda algo optimista; aun así hoy la razón es ≈ 20 (el hazard
/// 1/100 es por OBSERVACIÓN, no por unidad de tiempo).
/// 1 Hz: 1 h de calentamiento y 4 h medidas. 10 Hz: 30 min y 30 min.
#[test]
#[ignore = "OPEN F2.3-08 (Línea C, C-F2.3): el hazard del BOCPD es por evento: a 10 Hz da ~20 veces más falsas alarmas por hora que a 1 Hz en el mismo proceso"]
fn cf2_rw_bocpd_alarmas_por_hora_no_dependen_de_la_cadencia() {
    let (r1, c1, n1) = episodios_por_hora(1, 3_600, 4 * 3_600, 0xC0F2_1003);
    let (r10, c10, n10) = episodios_por_hora(10, 1_800, 1_800, 0xC0F2_1004);
    // Tasas por hora con corrección de continuidad: 4 h medidas a 1 Hz y
    // 0,5 h a 10 Hz.
    let ln_razon = (((c10 as f64 + 0.5) / 0.5) / ((c1 as f64 + 0.5) / 4.0)).ln();
    let se = (1.0 / (c1 as f64 + 0.5) + 1.0 / (c10 as f64 + 0.5)).sqrt();
    println!(
        "observado: 1 Hz {r1:.1}/h ({c1} episodios, {n1} obs); 10 Hz {r10:.1}/h ({c10} episodios, {n10} obs); ln razón {ln_razon:.3}, SE {se:.3}"
    );
    assert!(
        ln_razon.abs() <= 4.0 * se,
        "falsas alarmas por hora: {r1:.1} a 1 Hz ({c1} episodios) frente a {r10:.1} a 10 Hz ({c10} episodios); \
         |ln razón| = {:.3} > 4·SE = {:.3}",
        ln_razon.abs(),
        4.0 * se
    );
}

/// Cambio plantado: volatilidad ×5 (ln 5 ≈ 1,6 de desviación de
/// log-volatilidad) tras 1 h de caminata con σ = 1e-4 por evento a 1 Hz.
/// Ventana de detección N = 120 s (corrección del escéptico de refutar): W1
/// compara contra la masa de hace 60 s, así que el cambio queda entero
/// dentro del lag entre 60 y 120 s. Diseño pareado para separar detección
/// de falsa alarma. El diseño original del auditor (una semilla, cambio a
/// las 4 h, N = 600 s) pasa con 0xC0F2_1002 (detecta a los 11 s), pero con
/// las semillas vecinas detecta a los 63, 85, 142, 336 o 430 s, y con una
/// falsa alarma cada ≈ 4-5 min en ruido, 600 s no discriminan nada. Aquí, en
/// K = 16 semillas, se cuenta
/// (a) alguna alarma en [cambio, cambio + 120 s) y (b) alguna alarma en la
/// ventana nula [cambio − 120 s, cambio) del mismo camino estacionario.
/// Contrato: potencia ≥ 13/16 y nulo ≤ 3/16. Un detector con ARL0 ≥ 1 h
/// tiene P(alarma en 120 s) ≤ 3,3 % (P(nulo ≥ 4/16) < 0,2 %), y uno que
/// detecte con probabilidad 0,98 da P(potencia ≤ 12/16) < 0,1 %.
/// Hoy: 5/16 detecciones frente a 4/16 alarmas en la ventana nula; el
/// detector no separa el cambio de sus falsas alarmas (el escéptico del
/// espectro añade que la señal está normalizada por `ewma_dev_vol`, así que
/// un cambio de escala sólo mueve la masa mientras se adapta).
///
/// Revisión cruzada (CL-65): para separar el observable del detector, se
/// comprueba primero que W1 vea el cambio: en cada camino, la media de W1 en
/// [cambio, cambio + 60 s) (con el desfase físico de 60 s, el único tramo en
/// que se compara un espectro posterior al cambio con uno anterior) debe
/// superar el máximo de W1 en la ventana nula (≈ su cuantil del 99 % con ~120
/// lecturas) en ≥ 13/16 caminos. Si esa aserción falla, el defecto es del
/// observable y no del BOCPD.
///
/// Resultado en el PC (CL-65): W1 ve el cambio en 0/16 caminos. Su media tras
/// el cambio (≈ 0,91 entre caminos) es la de la ventana nula (≈ 0,94): el
/// transporte espectral está normalizado y es invariante a la escala, así que
/// un cambio de volatilidad ×5 no entra en la serie que vigila el BOCPD. Las
/// 5/16 «detecciones» son falsas alarmas. Se registra como F2.3-08b. La prueba
/// se activa cuando el detector reciba un observable que contenga la escala y
/// además separe el cambio de sus falsas alarmas.
#[test]
#[ignore = "OPEN F2.3-08b (Línea C, C-F2.3): W1 es invariante a la escala y no ve un cambio ×5 de volatilidad (0/16 caminos; media 0,91 tras el cambio frente a 0,94 nula); el BOCPD sólo da falsas alarmas (5/16 frente a 4/16 nulas)"]
fn cf2_bocpd_detecta_un_cambio_de_volatilidad_plantado() {
    let cambio = 3_600u64;
    let ventana = 120u64;
    let k = 16u64;
    let (mut detectadas, mut nulas, mut observable_ve) = (0u64, 0u64, 0u64);
    let mut retardos = Vec::new();
    // (media nula, máximo nulo, media tras el cambio) de W1 por camino.
    let mut w1_resumen: Vec<(f64, f64, f64)> = Vec::new();
    for semilla in 0..k {
        let mut g = Lcg(0xC0F2_1002 + semilla);
        let mut spec = TemporalSpectrum::new();
        let mut obs = W1ChangepointObserver::new();
        let mut x = 0.0f64;
        let (mut primera, mut nula) = (None, false);
        let (mut w1_nulo, mut w1_post) = (Vec::new(), Vec::new());
        for i in 0..(cambio + ventana) {
            let sd = if i < cambio { 1e-4 } else { 5e-4 };
            x += sd * g.n();
            let ts = T0 + i * DT;
            spec.update(P0 * x.exp(), ts);
            if let Some(w1) = spec.spectral_transport_w1_fisico(W1_LAG_FISICO_MS) {
                if i >= cambio - ventana && i < cambio {
                    w1_nulo.push(w1);
                } else if i >= cambio && i < cambio + 60 {
                    w1_post.push(w1);
                }
                if let Some(p) = obs.observe_at(ts, w1) {
                    if p > 0.5 {
                        if i >= cambio - ventana && i < cambio {
                            nula = true;
                        }
                        if i >= cambio && primera.is_none() {
                            primera = Some(i - cambio);
                        }
                    }
                }
            }
        }
        let max_nulo = w1_nulo.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let media_post = w1_post.iter().sum::<f64>() / w1_post.len().max(1) as f64;
        if !w1_post.is_empty() && media_post > max_nulo {
            observable_ve += 1;
        }
        let media_nulo = w1_nulo.iter().sum::<f64>() / w1_nulo.len().max(1) as f64;
        w1_resumen.push((
            (media_nulo * 1e6).round() / 1e6,
            (max_nulo * 1e6).round() / 1e6,
            (media_post * 1e6).round() / 1e6,
        ));
        if let Some(r) = primera {
            detectadas += 1;
            retardos.push(r);
        }
        if nula {
            nulas += 1;
        }
    }
    println!(
        "observado: detección {detectadas}/{k} en {ventana} s (retardos {retardos:?}); nulo {nulas}/{k};          W1 ve el cambio en {observable_ve}/{k}"
    );
    assert!(
        observable_ve >= 13,
        "W1 sólo ve el cambio ×5 en {observable_ve}/{k} caminos: la prueba no mide el BOCPD sino el observable;          (media nula, máximo nulo, media tras el cambio) por camino: {w1_resumen:?}"
    );
    assert!(
        detectadas >= 13 && nulas <= 3,
        "cambio ×5 detectado en {ventana} s en {detectadas}/{k} caminos (retardos {retardos:?}); \
         alarmas en la ventana nula previa: {nulas}/{k}"
    );
}
