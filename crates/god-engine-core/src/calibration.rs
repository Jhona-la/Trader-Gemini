//! CALIBRACIÓN EN LÍNEA DE LA CONFIANZA (D-619 — DÉCIMA OLA).
//!
//! # El problema
//!
//! `SignalIntent::confidence` entra en el Kelly y en el valor esperado como si
//! fuera una probabilidad de ganar, pero se fabrica con mapeos heurísticos
//! —`tanh(·)` con suelo 0,55, `0,5 + 0,4·|score|`, fuerza del consenso
//! tensorial— que nunca se contrastaron con resultados. Quitar esos suelos a
//! secas no la convierte en probabilidad: sólo cambia qué número arbitrario
//! llega a los gates. Lo que falta es un mapa aprendido de la puntuación a la
//! frecuencia real de acierto.
//!
//! # El modelo
//!
//! Escalado de Platt: `p = σ(a·logit(s) + b)`, con `s` la puntuación cruda,
//! estimado por máximo a posteriori sobre una ventana acotada de cierres.
//! LH-12: cada resultado fechado pesa `2^(-edad/semivida)` en la
//! verosimilitud; un cierre nuevo no rejuvenece los anteriores. El prior
//! conserva su masa total y el ajuste sólo ocurre al recibir un resultado.
//!
//! * **Sin datos, identidad en [EPS, 1−EPS].** Los extremos se acotan por el
//!   dominio numérico del logit. El sistema arranca con el comportamiento
//!   que ya tenía; la calibración sólo se aparta de él con evidencia.
//! * **Prior en forma de pseudo-observaciones.** El mapa identidad equivale a
//!   observar, en cada puntuación `s`, una frecuencia de acierto `s`. El prior
//!   son `z² ≈ 3,84` pseudo-observaciones en total —el mismo peso que el prior
//!   del win rate (D-680)—, con etiqueta suave igual a la puntuación, repartidas
//!   por igual sobre las puntuaciones observadas. Cuando todas las
//!   observaciones comparten puntuación, el resultado aproxima (por el ridge)
//!   la media posterior Beta de D-680: tres pérdidas a 0,8 dan 0,449, no 0,27.
//!   Un prior sobre los coeficientes `(a, b)` no tiene esa propiedad: su peso
//!   efectivo en la escala de probabilidad depende de dónde caen los datos.
//! * **Pendiente no negativa.** Si la evidencia dice que la puntuación no
//!   informa, el mapa se aplana hacia la tasa base (`a = 0`); nunca invierte
//!   el orden de las señales.
//!
//! Cada ajuste usa Newton factible con búsqueda de descenso sobre una matriz 2×2, y
//! sólo ocurre al incorporar un resultado, no al consultar la probabilidad.

use std::collections::VecDeque;

/// Pseudo-observaciones del prior: `z²` con `z` = 1,959964.
pub const PRIOR_PSEUDO_OBSERVATIONS: f64 = 1.959_963_984_540_054 * 1.959_963_984_540_054;

/// Cota de almacenamiento y trabajo por ajuste. A alta cadencia también
/// recorta la memoria: puede expulsar cierres cuyo peso temporal sigue alto.
pub const WINDOW: usize = 512;

/// Margen que aleja la puntuación de 0 y 1 antes del logit.
const EPS: f64 = 1e-4;

/// Ridge hacia la identidad: hace definida positiva la hessiana incluso con
/// puntuaciones idénticas. También perturba las direcciones identificadas;
/// NO es una regularización que actúe sólo en el espacio nulo.
const RIDGE: f64 = 1e-4;

#[inline]
fn logit(p: f64) -> f64 {
    let p = p.clamp(EPS, 1.0 - EPS);
    (p / (1.0 - p)).ln()
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Clone)]
pub struct PlattCalibrator {
    obs: VecDeque<(f64, bool, Option<u64>)>,
    a: f64,
    b: f64,
    /// Ola XLIV: instante (ms de evento) del último resultado incorporado.
    ultimo_dato_ms: Option<u64>,
}

/// Semivida de la evidencia del calibrador (Ola XLIV): el borde LENTO de la
/// banda operativa (12 h). Es una política de olvido vigente, no una semivida
/// óptima estimada ni una garantía sobre la duración del régimen.
pub const SEMIVIDA_EVIDENCIA_MS: f64 = quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS;

/// Techo de época plausible en milisegundos (2100-01-01T00:00:00Z = 4_102_444_800_000 ms).
/// CSG-10: Impide que una marca de tiempo corrupta en µs (~1,7e15) o ns (~1,7e18)
/// envenene `ultimo_dato_ms` para siempre, lo cual anularía el olvido en `calibrate_at`
/// y colapsaría a ~0 el peso de toda la evidencia histórica.
pub const MAX_PLAUSIBLE_EPOCH_MS: u64 = 4_102_444_800_000;

/// Salto máximo admisible hacia el futuro respecto a un reloj de época real ya anclado (365 días).
pub const MAX_CLOCK_JUMP_MS: u64 = 365 * 86_400_000;

/// Umbral mínimo para considerar que el reloj previo ya está en época Unix real (2000-01-01 UTC).
const MIN_REAL_EPOCH_ANCHOR_MS: u64 = 946_684_800_000;

#[inline]
pub(crate) fn plausible_evidence_timestamp_ms(
    candidate_ms: u64,
    current_clock_ms: Option<u64>,
) -> bool {
    if candidate_ms > MAX_PLAUSIBLE_EPOCH_MS {
        return false;
    }
    if let Some(prev) = current_clock_ms {
        if prev >= MIN_REAL_EPOCH_ANCHOR_MS && candidate_ms > prev.saturating_add(MAX_CLOCK_JUMP_MS) {
            return false;
        }
    }
    true
}

impl Default for PlattCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl PlattCalibrator {
    pub fn new() -> Self {
        Self {
            obs: VecDeque::with_capacity(WINDOW),
            a: 1.0,
            b: 0.0,
            ultimo_dato_ms: None,
        }
    }

    pub fn observations(&self) -> usize {
        self.obs.len()
    }

    /// Reloj máximo de la evidencia, incluso si su fila salió de la ventana.
    pub fn evidence_clock_ms(&self) -> Option<u64> {
        self.ultimo_dato_ms
    }

    /// Fecha que usaría una observación explícita, sin avanzar el reloj.
    /// Una fecha corrupta conserva el ancla propia de este calibrador. Sin
    /// ancla, `None` representa una observación sin fecha, que el formato
    /// persistente `(score, won, u64)` no puede registrar fielmente.
    pub(crate) fn effective_observation_timestamp_ms(&self, now_ms: u64) -> Option<u64> {
        if plausible_evidence_timestamp_ms(now_ms, self.ultimo_dato_ms) {
            Some(now_ms)
        } else {
            self.ultimo_dato_ms
        }
    }

    /// Restaura el reloj máximo guardado; al reconstruir un snapshot, se
    /// establece antes de reproducir sus filas retenidas, que pueden ser viejas.
    /// No admite rebobinado ni marcas de tiempo corruptas en el futuro (CSG-10).
    /// Reajusta fuera del camino caliente de consulta.
    pub fn restore_evidence_clock_ms(&mut self, clock_ms: u64) {
        if !plausible_evidence_timestamp_ms(clock_ms, self.ultimo_dato_ms) {
            return;
        }
        if self.ultimo_dato_ms.is_some_and(|t| clock_ms <= t) {
            return;
        }
        for (_, _, timestamp) in &mut self.obs {
            if timestamp.is_none() {
                *timestamp = Some(clock_ms);
            }
        }
        self.ultimo_dato_ms = Some(clock_ms);
        self.fit();
    }

    /// Coeficientes actuales `(a, b)`.
    pub fn coefficients(&self) -> (f64, f64) {
        (self.a, self.b)
    }

    /// Probabilidad calibrada para una puntuación cruda.
    #[inline]
    pub fn calibrate(&self, score: f64) -> f64 {
        if !score.is_finite() {
            return 0.5;
        }
        sigmoid(self.a * logit(score) + self.b)
    }

    /// Probabilidad calibrada con OLVIDO por tiempo (Ola XLIV).
    ///
    /// El calibrador sólo aprende de operaciones EJECUTADAS. Si su mapa baja
    /// la p lo bastante como para que la puerta de valor esperado (D-751) lo
    /// rechace todo, no vuelve a haber operaciones, el mapa no vuelve a
    /// cambiar y la moneda queda vetada para siempre: un estado ABSORBENTE
    /// (el que D-690 evitó en la puerta de confianza). Con olvido, la
    /// evidencia pesa `w = 2^(−Δt/semivida)` desde el último resultado y la
    /// probabilidad se mezcla con el prior identidad —el comportamiento en
    /// frío del propio calibrador—: `p = w·p_ajustada + (1 − w)·s`. Un
    /// régimen perdedor se olvida gradualmente y el sistema vuelve a sondear
    /// (a tamaño Kelly pequeño, porque el Kelly lee la misma p); un resultado
    /// nuevo reajusta con la edad individual de cada cierre (LH-12).
    /// La mezcla entre cierres es una aproximación barata: no equivale a
    /// reajustar la verosimilitud ponderada en cada consulta. En el camino
    /// caliente no se reajusta el modelo ni se asigna memoria.
    #[inline]
    pub fn calibrate_at(&self, score: f64, now_ms: u64) -> f64 {
        let ajustada = self.calibrate(score);
        let Some(ultimo) = self.ultimo_dato_ms else {
            return ajustada;
        };
        if !score.is_finite() || now_ms <= ultimo {
            return ajustada;
        }
        let dt = (now_ms - ultimo) as f64;
        let w = 0.5f64.powf(dt / SEMIVIDA_EVIDENCIA_MS);
        let prior = score.clamp(EPS, 1.0 - EPS);
        w * ajustada + (1.0 - w) * prior
    }

    /// Probabilidad calibrada con semivida continua respecto al horizonte temporal τ (Ola Ω93).
    /// Permite que la evidencia de corto o largo plazo decaiga al ritmo intrínseco de su escala.
    #[inline]
    pub fn calibrate_at_tau(&self, score: f64, now_ms: u64, tau_ms: f64) -> f64 {
        let ajustada = self.calibrate(score);
        let Some(ultimo) = self.ultimo_dato_ms else {
            return ajustada;
        };
        if !score.is_finite() || now_ms <= ultimo {
            return ajustada;
        };
        let dt = (now_ms - ultimo) as f64;
        let safe_tau = if tau_ms.is_finite() && tau_ms > 0.0 {
            tau_ms.clamp(
                quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS * 4.0,
            )
        } else {
            SEMIVIDA_EVIDENCIA_MS
        };
        let w = 0.5f64.powf(dt / safe_tau);
        let prior = score.clamp(EPS, 1.0 - EPS);
        w * ajustada + (1.0 - w) * prior
    }

    /// Añade un resultado con su instante de evento (Ola XLIV) y reajusta.
    /// CSG-10: Si `now_ms` excede el techo de época plausible (ej. µs leídos como ms, ~1,7e15)
    /// o salta más de `MAX_CLOCK_JUMP_MS` desde un reloj real ya anclado, no envenena
    /// `ultimo_dato_ms` y ancla la observación al último reloj válido conocido.
    pub fn update_at(&mut self, score: f64, won: bool, now_ms: u64) {
        if !valid_training_score(score) {
            return;
        }
        let Some(timestamp_ms) = self.effective_observation_timestamp_ms(now_ms) else {
            self.push_observation(score, won, None);
            return;
        };
        // La API sin fechas no permite reconstruir edades. Al recibir el
        // primer reloj explícito, sus datos se anclan una sola vez aquí.
        if self.ultimo_dato_ms.is_none() {
            for (_, _, timestamp) in &mut self.obs {
                *timestamp = Some(timestamp_ms);
            }
        }
        self.ultimo_dato_ms = Some(self.ultimo_dato_ms.map_or(timestamp_ms, |u| u.max(timestamp_ms)));
        // Un cierre tardío conserva su fecha; el ajuste usa el reloj máximo
        // ya conocido, sin rebobinar ni rejuvenecer esa observación.
        self.push_observation(score, won, Some(timestamp_ms));
    }

    /// Añade un resultado y reajusta el mapa. Puntuaciones fuera de [0,1]
    /// o no finitas se rechazan sin modificar datos, coeficientes ni reloj.
    /// El logit acota los extremos sólo para cómputo; el prior conserva s.
    /// Sin reloj explícito mantiene el ajuste legacy. Después de `update_at`,
    /// la observación se ancla al último reloj conocido y también envejece.
    pub fn update(&mut self, score: f64, won: bool) {
        if !valid_training_score(score) {
            return;
        }
        self.push_observation(score, won, self.ultimo_dato_ms);
    }

    fn push_observation(&mut self, score: f64, won: bool, timestamp: Option<u64>) {
        if self.obs.len() == WINDOW {
            self.obs.pop_front();
        }
        self.obs.push_back((score, won, timestamp));
        self.fit();
    }

    fn fit(&mut self) {
        let n = self.obs.len();
        if n == 0 {
            return;
        }
        let pseudo_weight = PRIOR_PSEUDO_OBSERVATIONS / n as f64;
        // Precalcular el logit fuera del bucle de Newton y de la búsqueda.
        // Cada fila combina su etiqueta real y su pseudo-observación suave.
        // No normalizar los pesos reales a n: eso reviviría evidencia vieja
        // frente al prior. La pseudo-observación sigue pesando κ/n por fila.
        let data: Vec<(f64, f64, f64)> = self.obs.iter().map(|&(s, won, timestamp)| {
            let evidence_weight = match (self.ultimo_dato_ms, timestamp) {
                (Some(now), Some(t)) => 0.5f64.powf(now.saturating_sub(t) as f64 / SEMIVIDA_EVIDENCIA_MS),
                _ => 1.0,
            };
            let weight = evidence_weight + pseudo_weight;
            (logit(s), (evidence_weight * f64::from(won) + pseudo_weight * s) / weight, weight)
        }).collect();
        let objective = |a: f64, b: f64| {
            let ridge = 0.5 * RIDGE * ((a - 1.0).powi(2) + b * b);
            ridge + data.iter().map(|&(x, target, weight)| {
                let z = a * x + b;
                // softplus(z) - target*z, estable incluso si sigmoid satura.
                weight * (z.max(0.0) - target * z + (-z.abs()).exp().ln_1p())
            }).sum::<f64>()
        };
        let (mut a, mut b) = (self.a, self.b);
        // Una ventana nueva puede hacer peor el warm start que la identidad.
        // Partir del mejor de ambos da un respaldo comprobable, no un clamp de p.
        if objective(a, b) > objective(1.0, 0.0) {
            a = 1.0;
            b = 0.0;
        }
        let mut loss = objective(a, b);
        // Límites de trabajo numérico, no umbrales financieros ni evidencia
        // estadística. Al agotarlos se conserva el mejor iterado factible.
        for _ in 0..100 {
            // Gradiente y hessiana del negativo del log-posterior.
            let (mut ga, mut gb) = (RIDGE * (a - 1.0), RIDGE * b);
            let (mut haa, mut hab, mut hbb) = (RIDGE, 0.0, RIDGE);
            for &(x, target, weight) in &data {
                let p = sigmoid(a * x + b);
                let w = p * (1.0 - p);
                let r = weight * (p - target);
                ga += r * x;
                gb += r;
                haa += weight * w * x * x;
                hab += weight * w * x;
                hbb += weight * w;
            }
            // KKT: en a=0 se permite gradiente a positivo, NO un gradiente
            // del intercepto distinto de cero. Proyectar sólo a tras un paso
            // libre no resuelve este problema restringido.
            let projected_ga = if a == 0.0 { ga.min(0.0) } else { ga };
            if projected_ga.abs().max(gb.abs()) < 1e-9 {
                break;
            }
            let det = haa * hbb - hab * hab;
            if !det.is_finite() || det <= 0.0 {
                break;
            }
            let mut da = -(hbb * ga - hab * gb) / det;
            let mut db = -(haa * gb - hab * ga) / det;
            if a + da < 0.0 {
                // Mínimo del modelo cuadrático EN la frontera a+da=0:
                // recalcular b, no conservar el paso libre incompatible.
                da = -a;
                db = -(gb + hab * da) / hbb;
            }
            let mut step = 1.0;
            let mut accepted = None;
            for _ in 0..50 {
                let next_a = (a + step * da).max(0.0);
                let next_b = b + step * db;
                let descent = ga * (next_a - a) + gb * (next_b - b);
                let next_loss = objective(next_a, next_b);
                // Armijo (1e-4) exige descenso de la misma función cuyo
                // gradiente/hessiana se usan; impide explosiones de Newton.
                if next_loss.is_finite() && descent < 0.0
                    && next_loss <= loss + 1e-4 * descent {
                    accepted = Some((next_a, next_b, next_loss));
                    break;
                }
                step *= 0.5;
            }
            let Some((next_a, next_b, next_loss)) = accepted else {
                break;
            };
            a = next_a;
            b = next_b;
            loss = next_loss;
        }
        if a.is_finite() && b.is_finite() {
            self.a = a;
            self.b = b;
        }
    }
}

#[inline]
fn valid_training_score(score: f64) -> bool {
    score.is_finite() && (0.0..=1.0).contains(&score)
}

/// D-693 (DÉCIMA OLA) — PROBABILIDAD DE SUBIDA QUE CONSUMEN LAS DECISIONES.
///
/// Es la probabilidad del ensamble (bosque y red, ponderados por Brier) más el
/// sesgo spot, acotada a [0, 1]. El residuo online del `OnlineLearningModule`
/// queda fuera:
///
/// - Se entrenaba con `realized_ret − ml_at_entry`, un retorno fraccional
///   (≈ ±0,005) menos una probabilidad (≈ 0,3–0,5). El error fue negativo en el
///   100 % de los cierres medidos y el residuo pasó el 99 % del tiempo en su
///   suelo de −0,15, restando 0,15 a toda predicción: `ml_prob < ½` en el 99,5 %
///   de las evaluaciones y el escudo neuronal vetando casi todos los largos
///   (D-692).
/// - Con el error en unidades de probabilidad tampoco converge: la ganancia de
///   Kalman por rasgo, con momentum 0,9 y rasgos correlacionados, oscila con
///   amplitud muy superior a su cota, así que sólo añadiría ruido de ±0,15.
///
/// Reincorporarlo exige un estimador que converja (mínimos cuadrados recursivos
/// sobre el vector completo, con el error del predictor completo) y su propia
/// validación walk-forward. La calibración con resultados reales ya la hace
/// `PlattCalibrator` (D-619).
pub fn compose_ml_prob(ensemble_prob: f64, spot_bias: f64) -> f64 {
    let p = if ensemble_prob.is_finite() { ensemble_prob } else { 0.5 };
    let b = if spot_bias.is_finite() { spot_bias } else { 0.0 };
    (p + b).clamp(0.0, 1.0)
}

/// D-696 (DÉCIMA OLA) — UMBRALES DE ACUERDO DE LA PREDICCIÓN, FUENTE ÚNICA.
///
/// `ml_threshold_long` y `ml_threshold_short` son genes (bandas de mutación
/// [0,50; 0,95] y [0,05; 0,49]). La regla que el motor declara es que la
/// predicción esté DE ACUERDO con la dirección: un largo exige P(sube) por
/// encima del umbral largo y un corto, por debajo del corto. Esa regla sólo
/// tiene sentido si el umbral largo no baja de ½ ni el corto lo supera: con
/// `largo < ½` el sistema abriría largos que su propio modelo considera más
/// probables a la baja.
///
/// Un genoma escrito a mano (o cargado de un almacén antiguo) puede violar las
/// bandas, porque `apply_to_arena` almacena estos dos genes sin acotar. Aquí se
/// impone el invariante una sola vez, para que todos los consumidores lean los
/// MISMOS umbrales: el valor no finito cae al neutro ½.
pub fn ml_gate_thresholds(long_threshold: f64, short_threshold: f64) -> (f64, f64) {
    let long = if long_threshold.is_finite() { long_threshold } else { 0.5 };
    let short = if short_threshold.is_finite() { short_threshold } else { 0.5 };
    (long.clamp(0.5, 1.0), short.clamp(0.0, 0.5))
}

/// K-06 / K-23 — DIVERGENCIA DIRECCIONAL NORMALIZADA SIMÉTRICA (Ola Ω75).
///
/// Mapea la probabilidad del modelo `p \in [0, 1]` relativa a la tasa base `base \in (0, 1)`
/// al intervalo simétrico [-1.0, 1.0]:
///   - Si p == base: devuelve 0.0 (neutralidad exacta).
///   - Si p > base: devuelve +(p - base) / (1.0 - base) \in [0.0, 1.0] (soporte alcista).
///   - Si p < base: devuelve -(base - p) / base = (p - base) / base \in [-1.0, 0.0] (soporte bajista si se niega).
///
/// Propiedades matemáticas demostradas:
/// 1. Coincidencia legacy en base 0.5: Si base == 0.5, (p - 0.5) / 0.5 == (p - 0.5) * 2.0.
/// 2. Simetría de rango: Tanto el soporte alcista (p = 1.0) como el bajista (p = 0.0) alcanzan
///    la magnitud máxima unitaria (+1.0 y -1.0 respectivamente) para CUALQUIER base honesta.
/// 3. Resuelve K-06: Permite vetar órdenes Long con d < -0.80 cuando p < 0.20 * base, e
///    impide vetos prematuros a Short, equiparando los techos de boost a 1.50x en ambos lados.
/// 4. Resuelve K-23: Elimina la parálisis de cortos cuando base <= lift_short, garantizando
///    que el umbral bajista escale proporcionalmente al espacio inferior [0, base].
/// REV-13 / K-06 / K-23: Si `p` o `base` no son finitos (NaN, Inf), la divergencia direccional
/// es estrictamente 0.0 (neutralidad exacta). Previene el sesgo alcista espurio donde `p = NaN`
/// caía a 0.50 produciendo `+1/3` de soporte long contra `base = 0.25`.
#[inline(always)]
pub fn normalized_directional_divergence(p: f64, base: f64) -> f64 {
    if !p.is_finite() || !base.is_finite() {
        return 0.0;
    }
    let p_clamped = p.clamp(0.0, 1.0);
    let base_clamped = base.clamp(0.01, 0.99);
    if p_clamped >= base_clamped {
        (p_clamped - base_clamped) / (1.0 - base_clamped)
    } else {
        (p_clamped - base_clamped) / base_clamped
    }
}

/// K-26 / REV-13 — PROBABILIDAD DIRECCIONAL DE ÉXITO CALIBRADA (Ola Ω76 / REV-13).
///
/// Transforma la probabilidad cruda `p = P(Long TP antes que SL)` y la base honesta `base`
/// a la probabilidad calibrada de que la orden gane, según su dirección:
///   - Para Long: es `p` (o `base` si `p` es NaN/no finito).
///   - Para Short: proyecta simétricamente a través de la divergencia direccional:
///     * Si p == base (mercado neutro): devuelve `base` (ambos lados comparten la misma tasa base honesta).
///     * Si p == 0.0 (máximo impulso bajista): devuelve 1.0 (certeza de acierto bajista).
///     * Si p == 1.0 (máxima oposición bajista): devuelve 0.0.
///     * Si base == 0.5: reduce exactamente a `1.0 - p`.
///     * Si p es no finito (NaN, Inf): devuelve `base` de forma simétrica a Long.
///
/// Erradica el sesgo de K-26 donde evaluar `1.0 - p` con base 0.25 trataba el estado neutro
/// como una probabilidad artificialmente inflada de 0.75 para cortos, y REV-13 donde NaN daba 0.50 a long y 0.17 a short.
#[inline(always)]
pub fn p_win_directional(is_long: bool, p: f64, base: f64) -> f64 {
    let b = if base.is_finite() { base.clamp(0.01, 0.99) } else { 0.5 };
    if !p.is_finite() {
        return b;
    }
    if is_long {
        p.clamp(0.0, 1.0)
    } else {
        let div = normalized_directional_divergence(p, b);
        let div_short = -div;
        if div_short >= 0.0 {
            (b + div_short * (1.0 - b)).clamp(0.0, 1.0)
        } else {
            (b + div_short * b).clamp(0.0, 1.0)
        }
    }
}


/// D-749 — EL VETO DEL MURO COMPARABA UNA RAZÓN CONTRA UNA FRACCIÓN.
///
/// `wall_veto_threshold` es una RAZÓN entre los tamaños de los muros: su banda
/// evolutiva es [5, 50] y su valor base, π·5 ≈ 15,7. La comprobación, en
/// cambio, lo contrastaba contra el desequilibrio NORMALIZADO
/// `(bid − ask)/(bid + ask)`, que por construcción vive en [−1, 1]. Ninguna
/// cantidad en [−1, 1] supera 15,7: el veto de muro del libro no podía
/// dispararse JAMÁS con el genoma base, y con un mutante en su banda tampoco.
/// El operador creía tener armado un freno contra entrar de cara a una pared
/// de liquidez y no existía.
///
/// La comparación se hace ahora donde el gen vive: si el muro CONTRARIO es al
/// menos `gen` veces el muro a favor, la entrada se veta. Un muro a favor nulo
/// con muro contrario positivo es razón infinita: veto.
#[inline]
pub fn muro_en_contra(bid_wall: f64, ask_wall: f64, gen_razon: f64, es_largo: bool) -> bool {
    let razon_gen = if gen_razon.is_finite() && gen_razon > 1.0 {
        gen_razon
    } else {
        // Un gen no utilizable no puede desarmar el freno: se exige al menos
        // que el muro contrario DOBLE al favorable, el mínimo por debajo del
        // cual «pared de liquidez» no significa nada.
        2.0
    };
    let (a_favor, en_contra) = if es_largo {
        (bid_wall, ask_wall)
    } else {
        (ask_wall, bid_wall)
    };
    if !a_favor.is_finite() || !en_contra.is_finite() || en_contra <= 0.0 {
        return false;
    }
    if a_favor <= 0.0 {
        return true;
    }
    en_contra / a_favor >= razon_gen
}

/// Calentamiento legado en número de actualizaciones con alfa constante.
/// No es tamaño efectivo estadístico ni garantía de precisión de la sigma.
pub const EWMA_MUESTRAS_MINIMAS: f64 = 30.0;

/// Actualiza una EWMA con su PESO de corrección de sesgo: `peso` es la EWMA
/// de la constante 1 (arranca en 0), de modo que `peso = 1 − (1 − α)^t`.
#[inline]
pub fn ewma_con_peso(valor: f64, previo: f64, peso: f64, alfa: f64) -> (f64, f64) {
    (previo + alfa * (valor - previo), peso + alfa * (1.0 - peso))
}

/// Media y σ CORREGIDAS por sesgo de una EWMA que arrancó en 0 (Ola XLIV).
///
/// Sin la corrección, tras t eventos la media vale m·(1 − (1 − α)^t) y el
/// segundo momento igual: con α = 0,002, tras UN evento de CVD −0,05 la σ
/// sale ≈ 0,045·|c| y el z ≈ −22 — el veto de flujo se disparaba en cada
/// arranque y al inicio de cada backtest. Devuelve `None` hasta que el peso
/// equivale a [`EWMA_MUESTRAS_MINIMAS`] eventos: sin σ medida no se inventa
/// umbral (el llamador cae a su literal histórico). Exige semilla cero en
/// ambos momentos y peso; un prior requiere transportar su masa también.
/// Sólo se recorta varianza negativa del orden del redondeo, no una entrada
/// materialmente imposible. None conserva la ausencia de evidencia válida.
#[inline]
pub fn momentos_ewma_corregidos(media: f64, segundo: f64, peso: f64, alfa: f64) -> Option<(f64, f64)> {
    if !(alfa > 0.0 && alfa < 1.0)
        || !media.is_finite()
        || !segundo.is_finite()
        || segundo < 0.0
        || !peso.is_finite()
        || peso <= 0.0
        || peso > 1.0
    {
        return None;
    }
    // -expm1(n*log1p(-alfa)) conserva la masa cuando alfa es muy pequeño.
    let peso_minimo = -(EWMA_MUESTRAS_MINIMAS * (-alfa).ln_1p()).exp_m1();
    if peso < peso_minimo {
        return None;
    }
    let m = media / peso;
    let segundo_normalizado = segundo / peso;
    let media_cuadrada = m * m;
    if !m.is_finite() || !segundo_normalizado.is_finite() || !media_cuadrada.is_finite() {
        return None;
    }
    let var = segundo_normalizado - media_cuadrada;
    let roundoff = 64.0 * f64::EPSILON * segundo_normalizado.max(media_cuadrada);
    if var < -roundoff {
        return None;
    }
    Some((m, var.max(0.0).sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d696_los_umbrales_de_la_puerta_nunca_cruzan_el_neutro() {
        // Genoma normal: se respetan tal cual.
        assert_eq!(ml_gate_thresholds(0.5698, 0.4302), (0.5698, 0.4302));
        // Genoma inválido: un largo jamás exige menos de ½ ni un corto más.
        assert_eq!(ml_gate_thresholds(0.30, 0.70), (0.5, 0.5));
        // No finitos: neutro.
        assert_eq!(ml_gate_thresholds(f64::NAN, f64::INFINITY), (0.5, 0.5));
        // Invariante: para cualquier par, largo ≥ ½ ≥ corto.
        for l in [-1.0, 0.0, 0.49, 0.5, 0.7, 0.95, 2.0] {
            for s in [-1.0, 0.0, 0.2, 0.5, 0.51, 0.9, 2.0] {
                let (gl, gs) = ml_gate_thresholds(l, s);
                assert!(gl >= 0.5 && gs <= 0.5, "largo {gl} corto {gs} para ({l}, {s})");
            }
        }
    }

    #[test]
    fn d693_ml_prob_es_el_ensamble_mas_el_sesgo_spot() {
        assert_eq!(compose_ml_prob(0.45, 0.0), 0.45);
        assert_eq!(compose_ml_prob(0.95, 0.15), 1.0);
        assert_eq!(compose_ml_prob(0.05, -0.15), 0.0);
        assert_eq!(compose_ml_prob(f64::NAN, 0.0), 0.5);
        assert_eq!(compose_ml_prob(0.6, f64::NAN), 0.6);
    }

    /// D-693: por qué el residuo online no puede entrar en la probabilidad. Con
    /// el error que usaba el núcleo (retorno − probabilidad), y aunque las
    /// operaciones ganen y pierdan por igual, su predicción se satura en negativo.
    #[test]
    fn d693_el_error_en_unidades_mezcladas_satura_el_residuo() {
        use metacortex_engine::online_learning::OnlineLearningModule;
        let mut features = [0.0f32; 64];
        for f in features.iter_mut().take(12) {
            *f = 0.5;
        }
        let mut learner = OnlineLearningModule::new(0.001, 0.9);
        for i in 0..400 {
            let realized_ret = if i % 2 == 0 { 0.005 } else { -0.005 };
            let ml_at_entry = 0.5 + (learner.predict(&features) as f64).clamp(-0.15, 0.15);
            learner.update_weights_with_kalman_adaptive_vol(
                &features,
                (realized_ret - ml_at_entry) as f32,
                0.1,
                0.001,
            );
        }
        assert!(
            learner.predict(&features) < -0.15,
            "predicción del residuo {}",
            learner.predict(&features)
        );
    }

    /// Media posterior Beta de D-680 para `n` observaciones en una única
    /// puntuación `s` con `wins` aciertos.
    fn beta_mean(s: f64, wins: f64, n: f64) -> f64 {
        (PRIOR_PSEUDO_OBSERVATIONS * s + wins) / (PRIOR_PSEUDO_OBSERVATIONS + n)
    }

    #[test]
    fn sin_datos_es_la_identidad() {
        let c = PlattCalibrator::new();
        for &s in &[0.2, 0.55, 0.7, 0.9] {
            assert!((c.calibrate(s) - s).abs() < 1e-9, "s = {s}");
        }
    }

    #[test]
    fn pocas_operaciones_mueven_el_mapa_como_el_prior_beta() {
        let mut c = PlattCalibrator::new();
        for _ in 0..3 {
            c.update(0.8, false);
        }
        let p = c.calibrate(0.8);
        let esperado = beta_mean(0.8, 0.0, 3.0);
        assert!((p - esperado).abs() < 1e-3, "tres pérdidas a 0,8: {p} frente a {esperado}");
    }

    #[test]
    fn aprende_una_puntuacion_sobreconfiada() {
        // La puntuación dice 0,8, pero sólo se gana la mitad de las veces.
        let mut c = PlattCalibrator::new();
        for i in 0..400 {
            c.update(0.8, i % 2 == 0);
        }
        let p = c.calibrate(0.8);
        let esperado = beta_mean(0.8, 200.0, 400.0);
        assert!((p - esperado).abs() < 1e-3, "400 observaciones: {p} frente a {esperado}");
    }

    #[test]
    fn conserva_el_orden_de_las_senales() {
        let mut c = PlattCalibrator::new();
        for i in 0..300 {
            // Puntuaciones altas ganan más a menudo que las bajas.
            let s = if i % 3 == 0 { 0.9 } else { 0.6 };
            let won = if s > 0.8 { i % 4 != 0 } else { i % 3 == 1 };
            c.update(s, won);
        }
        assert!(c.calibrate(0.9) > c.calibrate(0.6));
        assert!(c.coefficients().0 >= 0.0);
    }

    #[test]
    fn una_puntuacion_sin_informacion_se_aplana_hacia_la_tasa_base() {
        let mut c = PlattCalibrator::new();
        for i in 0..500 {
            // Alta o baja, se gana el 40 % de las veces.
            let s = if i % 2 == 0 { 0.9 } else { 0.55 };
            c.update(s, i % 5 < 2);
        }
        let (hi, lo) = (c.calibrate(0.9), c.calibrate(0.55));
        assert!((hi - lo).abs() < 0.02, "sin información no debe separar: {hi} vs {lo}");
        assert!((hi - 0.4).abs() < 0.02 && (lo - 0.4).abs() < 0.02);
    }
}

#[cfg(test)]
mod tests_d749_muro {
    use super::muro_en_contra;

    /// El razon base vale π·5 ≈ 15,7 y su banda evolutiva es [5, 50]: son
    /// RAZONES entre muros. Contrastarlo contra `(bid−ask)/(bid+ask)` ∈ [−1, 1]
    /// dejaba el veto muerto para cualquier genoma posible.
    #[test]
    fn d749_el_veto_del_muro_es_alcanzable_con_el_gen_base() {
        let razon = std::f64::consts::PI * 5.0;
        // Muro vendedor 99 veces el comprador: un largo entra de cara a la pared.
        assert!(muro_en_contra(1.0, 99.0, razon, true));
        // El mismo libro no estorba a un corto: la pared está a su favor.
        assert!(!muro_en_contra(1.0, 99.0, razon, false));
        // Libro equilibrado: nadie veta.
        assert!(!muro_en_contra(50.0, 50.0, razon, true));
        assert!(!muro_en_contra(50.0, 50.0, razon, false));
        // Justo en el umbral: la razón ES el razon ⇒ veta.
        assert!(muro_en_contra(1.0, razon, razon, true));
        // Un pelo por debajo: no veta.
        assert!(!muro_en_contra(1.0, razon * 0.99, razon, true));
    }

    #[test]
    fn d749_sin_muro_a_favor_y_con_muro_en_contra_se_veta() {
        assert!(muro_en_contra(0.0, 10.0, 15.7, true));
        // Sin muro en contra no hay nada que vetar, aunque falte el favorable.
        assert!(!muro_en_contra(0.0, 0.0, 15.7, true));
    }

    #[test]
    fn d749_un_gen_no_utilizable_no_desarma_el_freno() {
        for razon in [f64::NAN, 0.0, -3.0, 0.5] {
            assert!(
                muro_en_contra(1.0, 10.0, razon, true),
                "razon {razon} dejó pasar una pared 10×"
            );
        }
    }
}

#[cfg(test)]
mod tests_xliv_ewma_corregida {
    use super::*;

    /// El defecto: con la EWMA cruda, un solo evento ya daba σ «válida» y un
    /// z enorme. Con la corrección, no hay σ hasta 30 eventos efectivos.
    #[test]
    fn xliv_un_evento_no_da_sigma() {
        let alfa = 0.002;
        let (m, w) = ewma_con_peso(-0.05, 0.0, 0.0, alfa);
        let (sq, _) = ewma_con_peso(0.0025, 0.0, 0.0, alfa);
        assert!(momentos_ewma_corregidos(m, sq, w, alfa).is_none());
        // La lectura cruda que usaba el código anterior: σ > 1e-4 y |z| ≈ 22.
        let sd_cruda = (sq - m * m).max(0.0).sqrt();
        assert!(sd_cruda > 1e-4 && ((-0.05 - m) / sd_cruda).abs() > 20.0);
    }

    /// Con muestra suficiente, media y σ corregidas recuperan las de la
    /// serie aunque el peso esté lejos de 1 (t ≪ 1/α).
    #[test]
    fn xliv_momentos_corregidos_recuperan_la_distribucion() {
        let alfa = 0.002;
        let (mut m, mut sq, mut w) = (0.0, 0.0, 0.0);
        for i in 0..200 {
            let v = if i % 2 == 0 { 0.10 + 0.02 } else { 0.10 - 0.02 };
            let (nm, nw) = ewma_con_peso(v, m, w, alfa);
            let (nsq, _) = ewma_con_peso(v * v, sq, w, alfa);
            m = nm;
            sq = nsq;
            w = nw;
        }
        let (media, sd) = momentos_ewma_corregidos(m, sq, w, alfa).expect("200 eventos");
        assert!((media - 0.10).abs() < 2e-3, "media {media}");
        assert!((sd - 0.02).abs() < 2e-3, "sd {sd}");
    }
}

#[cfg(test)]
mod tests_xliv_olvido {
    use super::*;

    fn calibrador_perdedor(t0: u64) -> PlattCalibrator {
        let mut c = PlattCalibrator::new();
        for i in 0..40u64 {
            c.update_at(0.80, false, t0 + i);
        }
        c
    }

    /// Sin tiempo transcurrido, el olvido no cambia nada.
    #[test]
    fn xliv_sin_tiempo_no_hay_olvido() {
        let c = calibrador_perdedor(1_000);
        assert_eq!(c.calibrate_at(0.80, 1_039), c.calibrate(0.80));
        // Un calibrador sin fechas (API antigua) se comporta como siempre.
        let mut v = PlattCalibrator::new();
        v.update(0.80, false);
        assert_eq!(v.calibrate_at(0.80, u64::MAX), v.calibrate(0.80));
    }

    /// EL DEFECTO: 40 pérdidas a 0,80 hunden la p; sin operaciones, antes
    /// quedaba hundida para siempre. Con olvido vuelve hacia el prior: la
    /// mitad del camino en una semivida y casi entera en diez.
    #[test]
    fn xliv_la_evidencia_perdedora_se_olvida_sin_operar() {
        let c = calibrador_perdedor(0);
        let hundida = c.calibrate(0.80);
        assert!(hundida < 0.30, "40 pérdidas deben hundir la p: {hundida}");
        let semivida = SEMIVIDA_EVIDENCIA_MS as u64;
        let media = c.calibrate_at(0.80, 39 + semivida);
        assert!((media - (0.5 * hundida + 0.5 * 0.80)).abs() < 1e-9);
        let lejos = c.calibrate_at(0.80, 39 + 10 * semivida);
        assert!((lejos - 0.80).abs() < 1e-3, "tras 10 semividas: {lejos}");
        // Monótona hacia el prior.
        let mut previo = hundida;
        for k in 1..=20u64 {
            let p = c.calibrate_at(0.80, 39 + k * semivida / 2);
            assert!(p >= previo - 1e-15);
            previo = p;
        }
    }

    /// Un resultado nuevo pesa uno; los antiguos conservan su edad.
    #[test]
    fn lh12_un_dato_nuevo_no_restablece_la_evidencia_antigua() {
        let mut c = calibrador_perdedor(0);
        let t = 5 * SEMIVIDA_EVIDENCIA_MS as u64;
        c.update_at(0.80, false, t);
        assert_eq!(c.calibrate_at(0.80, t), c.calibrate(0.80));
        let count = (0..40u64).map(|i| {
            0.5f64.powf((t - i) as f64 / SEMIVIDA_EVIDENCIA_MS)
        }).sum::<f64>() + 1.0;
        let expected = PRIOR_PSEUDO_OBSERVATIONS * 0.80 / (PRIOR_PSEUDO_OBSERVATIONS + count);
        assert!((c.calibrate(0.80) - expected).abs() < 1e-3);
    }

    #[test]
    fn ola75_k06_k23_divergencia_direccional_normalizada_simetrica() {
        // 1. En base 0.50 coincide EXACTAMENTE con la formulación legacy (p - 0.5) * 2.0
        for &p in &[0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0] {
            let legacy = (p - 0.5) * 2.0;
            let div = normalized_directional_divergence(p, 0.5);
            assert!((div - legacy).abs() < 1e-12, "p={p}: legacy={legacy}, div={div}");
        }

        // 2. Neutralidad exacta: en p == base, divergencia es 0.0 para cualquier base
        for &b in &[0.10, 0.25, 0.33, 0.50, 0.75, 0.90] {
            let div = normalized_directional_divergence(b, b);
            assert_eq!(div, 0.0, "base={b}");
        }

        // 3. Simetría de extremos: p = 1.0 -> +1.0 y p = 0.0 -> -1.0 para cualquier base
        for &b in &[0.10, 0.25, 0.33, 0.50, 0.75, 0.90] {
            let max_bull = normalized_directional_divergence(1.0, b);
            let max_bear = normalized_directional_divergence(0.0, b);
            assert!((max_bull - 1.0).abs() < 1e-12, "base={b} max_bull={max_bull}");
            assert!((max_bear - (-1.0)).abs() < 1e-12, "base={b} max_bear={max_bear}");
        }

        // 4. Monotonía estricta: para p1 < p2, div(p1, b) <= div(p2, b)
        let b = 0.25;
        let mut prev = -1.0;
        for i in 0..=100 {
            let p = i as f64 / 100.0;
            let div = normalized_directional_divergence(p, b);
            assert!(div >= prev - 1e-15, "fallo de monotonía en p={p}");
            assert!(div >= -1.0 && div <= 1.0, "fuera de rango: {div}");
            prev = div;
        }

        // 5. Robustez numérica ante NaN / infinitos (REV-13 neutralidad estricta)
        assert_eq!(normalized_directional_divergence(f64::NAN, 0.25), 0.0);
        assert_eq!(normalized_directional_divergence(0.25, f64::NAN), 0.0);
        assert_eq!(normalized_directional_divergence(f64::NAN, f64::NAN), 0.0);
        assert_eq!(normalized_directional_divergence(f64::INFINITY, 0.25), 0.0);
        assert_eq!(normalized_directional_divergence(0.25, f64::NEG_INFINITY), 0.0);
    }

    #[test]
    fn ola76_k26_p_win_directional_symmetry() {
        // 1. Long siempre coincide con p
        for &p in &[0.0, 0.1, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(p_win_directional(true, p, 0.25), p);
            assert_eq!(p_win_directional(true, p, 0.50), p);
        }

        // 2. Short en base = 0.50 coincide EXACTAMENTE con legacy (1.0 - p)
        for &p in &[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] {
            let legacy = 1.0 - p;
            let pw = p_win_directional(false, p, 0.50);
            assert!((pw - legacy).abs() < 1e-12, "p={p}: legacy={legacy}, pw={pw}");
        }

        // 3. Neutralidad: si p == base, ambos lados tienen p_win == base
        for &b in &[0.10, 0.25, 0.33, 0.50, 0.75] {
            assert_eq!(p_win_directional(true, b, b), b);
            assert_eq!(p_win_directional(false, b, b), b);
        }

        // 4. Extremos de señal:
        // p = 0.0 (máximo bajista) -> Short tiene p_win = 1.0
        // p = 1.0 (máximo alcista) -> Short tiene p_win = 0.0
        for &b in &[0.10, 0.25, 0.33, 0.50, 0.75] {
            assert_eq!(p_win_directional(false, 0.0, b), 1.0);
            assert_eq!(p_win_directional(false, 1.0, b), 0.0);
        }

        // 5. Monotonía decreciente para Short respecto a p
        let b = 0.25;
        let mut prev = 1.0;
        for i in 0..=100 {
            let p = i as f64 / 100.0;
            let pw = p_win_directional(false, p, b);
            assert!(pw <= prev + 1e-15, "fallo de monotonía decreciente en p={p}");
            assert!(pw >= 0.0 && pw <= 1.0, "fuera de rango: {pw}");
            prev = pw;
        }

        // 6. Robustez y neutralidad ante NaN / infinitos (REV-13)
        for &b in &[0.10, 0.25, 0.33, 0.50, 0.75] {
            assert_eq!(p_win_directional(true, f64::NAN, b), b);
            assert_eq!(p_win_directional(false, f64::NAN, b), b);
            assert_eq!(p_win_directional(true, f64::INFINITY, b), b);
            assert_eq!(p_win_directional(false, f64::INFINITY, b), b);
            assert_eq!(p_win_directional(true, f64::NEG_INFINITY, b), b);
            assert_eq!(p_win_directional(false, f64::NEG_INFINITY, b), b);
        }
        assert_eq!(p_win_directional(true, f64::NAN, f64::NAN), 0.50);
        assert_eq!(p_win_directional(false, f64::NAN, f64::NAN), 0.50);
    }

    #[test]
    fn csg_10_corrupted_future_timestamp_does_not_poison_evidence_clock() {
        let t0 = 1_700_000_000_000_u64;
        let mut c = calibrador_perdedor(t0);
        assert_eq!(c.evidence_clock_ms(), Some(t0 + 39));

        // Un evento corrupto con marca en microsegundos (~1,7e15) llega a update_at y restore_evidence_clock_ms
        let corrupted_us = 1_700_000_000_000_000_u64;
        c.update_at(0.80, false, corrupted_us);
        c.restore_evidence_clock_ms(corrupted_us);

        // El reloj de evidencia NO debe envenenarse con 1,7e15
        assert_eq!(
            c.evidence_clock_ms(),
            Some(t0 + 39),
            "Un timestamp corrupto en µs no debe envenenar el reloj de evidencia"
        );

        // El olvido temporal sigue operando normalmente tras una semivida
        let hundida = c.calibrate(0.80);
        let semivida = SEMIVIDA_EVIDENCIA_MS as u64;
        let tras_semivida = c.calibrate_at(0.80, t0 + 39 + semivida);
        assert!(
            (tras_semivida - (0.5 * hundida + 0.5 * 0.80)).abs() < 1e-6,
            "El olvido debe seguir funcionando tras rechazar el timestamp corrupto: {tras_semivida}"
        );
    }
}

