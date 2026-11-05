//! Geometría de la información de Fisher–Rao (mandato §3.4, GLM-191):
//! «trata las distribuciones probabilísticas del espectro como puntos en
//! una variedad diferenciable de Riemann equipada con la métrica de
//! Fisher-Rao. Calcula geodésicas [...] en tiempo real para detectar
//! deformaciones extremas de liquidez».
//!
//! Verificación previa (norma GLM-186, con la lección del falso
//! positivo): `fisher_rao` en main sólo aparece como el test
//! `c5_wasserstein_fisher_rao_distance_metric_invariants` de AGY (#166)
//! — un motor WFR sobre PERFILES DISCRETOS (transporte no-balanceado en
//! grillas, Chizat et al.). Este módulo es el objeto COMPLEMENTARIO que
//! el §3.4 pide explícitamente: la GEODÉSICA Fisher-Rao de la familia
//! gaussiana CONTINUA — precedencia del WFR discreto citada, sin
//! solape del núcleo.
//!
//! **Forma cerrada** (verificada a mano en python antes de escribir
//! Rust): la métrica Fisher de N(μ,σ) es ds² = dμ²/σ² + 2·dσ²/σ². Con
//! X = μ/√2 y u = σ se convierte en el semiplano hiperbólico clásico
//! a²(dX² + du²)/u² con a = √2. ERRATA (GLM-193): aquí se leyó
//! «curvatura escalar de Ricci −1/2»; −1/2 es la curvatura GAUSSIANA
//! (seccional) K = −1/a² — el ESCALAR de Ricci es R = 2K = −1 (el
//! resultado clásico de la familia), reproducido numéricamente por
//! `ricci_empirica::curvatura_ricci_2d` a ~2e-6. La distancia
//! geodésica es (sin cambio — la errata era sólo de nomenclatura)
//!
//!   d_FR = √2 · arcosh(1 + (Δμ²/2 + Δσ²)/(2·σ₁·σ₂))
//!
//! Identidades certificadas: d(p,p) = 0 exacto; simetría; el caso σ
//! puro colapsa a √2·|ln(σ₂/σ₁)| EXACTO (geodésica vertical —
//! verificado a 2e-16); d ≤ longitud de cualquier camino (la integral
//! métrica por el camino recto la acota por arriba — verificado).
//!
//! **Deformación en streaming** (`DeformacionFisherRao`, §3.4 «en
//! tiempo real»): dos escalas EWMA de (μ,σ) de los retornos — rápida
//! P=64, lenta P=512, α = 2/(P+1) — y `push` devuelve la distancia FR
//! entre la escala rápida y la lenta: cuánto se DEFORMÓ la distribución
//! de corto plazo respecto a la de fondo. O(1)/tick, sin asignaciones.
//!
//! **Calibración medida sobre el nulo** (8 caminantes LCG deterministas,
//! σ=0.004, 1024 pasos, escalas convergidas): p50 ∈ [0.11, 0.15],
//! p95 ∈ [0.21, 0.31], **máximo 0.372**. Cambios de régimen MEDIDOS:
//! σ×2 ⇒ pico 0.69; σ×4 ⇒ 0.96; drift de media +1σ ⇒ 0.73 — todos
//! ≥1.9× el máximo del nulo. La separación es real pero no infinita:
//! el umbral de «deformación extrema» lo fija el consumo con su
//! corroboración (consejo) — documentado, no endurecido a ciegas.
//!
//! **Honestidad numérica**: σ ≤ 0 / no finito / varianza degenerada ⇒
//! Err o None (fall-closed, jamás un cero fabricado); el warmup exige
//! WARM=512 muestras válidas — menos que eso y la escala LENTA sigue
//! en su propia transición de convergencia, que NO es señal del mercado
//! (medido: con warmup corto la convergencia de la escala lenta
//! contamina la deformación con valores comparables a un cambio real).

/// Período EWMA de la escala rápida (muestras).
pub const PERIODO_RAPIDO: usize = 64;
/// Período EWMA de la escala lenta (muestras).
pub const PERIODO_LENTO: usize = 512;
/// Muestras válidas mínimas antes de publicar (≈ el período lento: la
/// escala de fondo debe estar convergida o la deformación mide la
/// propia convergencia, no el mercado).
pub const WARMUP: usize = 512;

/// Distancia Fisher-Rao geodésica entre N(μ₁,σ₁) y N(μ₂,σ₂) — forma
/// cerrada del semiplano hiperbólico (ver cabecera). Err si algún σ es
/// ≤ 0 o no finito, o si los μ no son finitos (fall-closed, RB-P1).
pub fn dist_fisher_rao(mu1: f64, sig1: f64, mu2: f64, sig2: f64) -> Result<f64, &'static str> {
    if !(mu1.is_finite() && mu2.is_finite()) {
        return Err("medias deben ser finitas");
    }
    if !(sig1.is_finite() && sig2.is_finite()) || sig1 <= 0.0 || sig2 <= 0.0 {
        return Err("desviaciones deben ser finitas y > 0");
    }
    let delta2 = (mu1 - mu2) * (mu1 - mu2) / 2.0 + (sig1 - sig2) * (sig1 - sig2);
    let arg = 1.0 + delta2 / (2.0 * sig1 * sig2);
    let d = std::f64::consts::SQRT_2 * arg.acosh();
    if d.is_finite() {
        Ok(d)
    } else {
        Err("distancia degenerada")
    }
}

/// Una escala EWMA de (μ, var) con la recurrencia de West/Welford
/// exponencial: mean += α·d; var = (1−α)·var + α·d·(x − mean_nuevo).
/// Sin pánico; var puede llegar a 0 (serie constante) y el llamador
/// la trata como ausencia (None), jamás como σ=0 publicable.
struct EscalaEwma {
    alfa: f64,
    mu: f64,
    var: f64,
    iniciada: bool,
}

impl EscalaEwma {
    fn new(periodo: usize) -> Self {
        Self {
            alfa: 2.0 / (periodo as f64 + 1.0),
            mu: 0.0,
            var: 0.0,
            iniciada: false,
        }
    }
    fn push(&mut self, x: f64) {
        if !self.iniciada {
            self.mu = x;
            self.var = 0.0;
            self.iniciada = true;
            return;
        }
        let d = x - self.mu;
        self.mu += self.alfa * d;
        self.var = (1.0 - self.alfa) * self.var + self.alfa * d * (x - self.mu);
    }
}

/// Deformación Fisher-Rao en streaming: distancia FR entre la
/// distribución de corto plazo (escala rápida) y la de fondo (escala
/// lenta) de los retornos — el observable de «deformación extrema de
/// liquidez/régimen» del §3.4. O(1)/tick, sin asignaciones.
pub struct DeformacionFisherRao {
    rapida: EscalaEwma,
    lenta: EscalaEwma,
    validas: usize,
}

impl Default for DeformacionFisherRao {
    fn default() -> Self {
        Self::new()
    }
}

impl DeformacionFisherRao {
    pub fn new() -> Self {
        Self {
            rapida: EscalaEwma::new(PERIODO_RAPIDO),
            lenta: EscalaEwma::new(PERIODO_LENTO),
            validas: 0,
        }
    }

    /// Ingresa un retorno; devuelve la deformación FR instantánea
    /// (rápida ‖ lenta), `None` durante el warmup, con muestra no
    /// finita (se DESCARTA sin tocar el estado — jamás cero) o con
    /// varianza degenerada (serie constante: ausencia honesta).
    pub fn push(&mut self, x: f64) -> Option<f64> {
        if !x.is_finite() {
            return None;
        }
        self.rapida.push(x);
        self.lenta.push(x);
        self.validas += 1;
        if self.validas < WARMUP {
            return None;
        }
        if !(self.rapida.var > 0.0 && self.lenta.var > 0.0) {
            return None;
        }
        dist_fisher_rao(
            self.rapida.mu,
            self.rapida.var.sqrt(),
            self.lenta.mu,
            self.lenta.var.sqrt(),
        )
        .ok()
    }
}
