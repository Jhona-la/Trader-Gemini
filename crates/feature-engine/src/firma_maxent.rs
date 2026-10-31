//! FIRMAS DE CAMINO vs NULO DE MÁXIMA ENTROPÍA — observable σz STREAMING
//! (GLM-175, titularidad §10 firmas — explota la medición GLM-163).
//!
//! # QUÉ
//! Por ventana de L incrementos (log-retorno, fracción de ventana), calcula
//! el cross S¹² del nivel-2 (kernel Stratonovich de `path_signatures`) y lo
//! estandariza contra los momentos EXACTOS del nulo de máxima entropía
//! (permutación uniforme de los retornos contra los dt fijos):
//!
//! - Reducción (GLM-163, contrato por enumeración bruta 720 perms):
//!   bajo el nulo, S¹² = Σ_a v_a·x_σ(a) con
//!   v_a = (Σ_{b>a} y_b) + ½·y_a  (sufijo de dt + ½ propio)
//!   ⇒ estadístico LINEAL DE ASIGNACIÓN.
//! - Momentos cerrados:  E = x̄·Σv   y   Var = Sxx·Svv/(n−1).
//! - Reflexión: v_a + u_a = Σy ⇒ Svv = Suu ⇒ Var(S¹²) = Var(S²¹) siempre.
//! - Anti-simetría: con Σy = 1, E₁₂+E₂₁ = x̄·n = S¹ y S¹²+S²¹ = S¹·S²
//!   ⇒ z(S²¹) ≡ −z(S¹²) EXACTO — la pareja es UN número con signo.
//!
//! # POR QUÉ
//! La medición GLM-163 (10 tapes × 4.000 ventanas de 64) halló media-z
//! NULA en los 10 símbolos (sin acoplamiento direccional precio-tiempo)
//! pero rechazo del nulo en FORMA: σz 0,470 (ADA) → 1,793 (BTC), |z|>2
//! hasta 29,8% — dos familias (sobre-dispersa BTC/BNB/XLM/XMR/XRP =
//! acoplamiento fuerte alternante; sub-dispersa ADA/NEAR/SOL/ATOM =
//! memoria que el nulo rompe). Esa señal de segundo orden es un observable
//! de régimen que hasta ahora sólo existía en el vehículo de medición.
//!
//! # PARA QUÉ
//! Telemetría cuántica por símbolo: σz como feature de régimen (la
//! separación 4× transversal documentada). Este módulo publica el
//! observable; el consumo (gate, ponderación) es decisión del consejo.
//!
//! # CÓMO / coste
//! push: O(1) (ring). medir: O(L) — sólo el cross depende de los prefijos
//! (no incrementable), L=64 ⇒ ~64 multiplicaciones por medición; llamada
//! por CADENCIA (no por tick). Saturación en todos los acumuladores.
//!
//! API algebraica SIN validación de entrada: guardas mínimas de ventana
//! incompleta (None) y varianza degenerada (retornos constantes ⇒ z no
//! medible), sin clamping del observable.

use crate::path_signatures::firma_nivel2_incrementos;

/// Medida σz de una ventana completa.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MedidaSigmaZ {
    /// z = (S¹² − E[S¹²_nulo]) / √Var[S¹²_nulo]. SIN unidades.
    pub z12: f64,
    /// Desviación estándar muestral de los retornos de la ventana
    /// ((Sxx/(n−1))⁰·⁵) — contexto de volatilidad del régimen.
    pub sigma_x: f64,
    /// Número de incrementos de la ventana (== L tras el llenado).
    pub n: usize,
}

/// Anillo de incrementos + observable σz contra el nulo de máxima entropía.
#[derive(Debug)]
pub struct FirmasMaxEnt {
    dx: Vec<f64>,
    dy: Vec<f64>,
    cabeza: usize,
    llenos: usize,
}

impl FirmasMaxEnt {
    /// Ventana fija de `ventana` incrementos. `ventana < 2` no tiene
    /// varianza de permutación (n−1 = 0) ⇒ Err.
    pub fn new(ventana: usize) -> Result<Self, String> {
        if ventana < 2 {
            return Err(format!("ventana {ventana} < 2: sin varianza de permutación"));
        }
        Ok(Self {
            dx: vec![0.0; ventana],
            dy: vec![0.0; ventana],
            cabeza: 0,
            llenos: 0,
        })
    }

    /// Ventana canónica de 64 (la de la medición 163) — constructor
    /// directo SIN Result para el hilo crítico: 64 ≥ 2 por definición y
    /// el trinquete RB-P1 prohíbe sitios de pánico en producción.
    pub fn ventana64() -> Self {
        Self {
            dx: vec![0.0; 64],
            dy: vec![0.0; 64],
            cabeza: 0,
            llenos: 0,
        }
    }

    /// Un incremento: (log-retorno, fracción de ventana). O(1).
    #[inline]
    pub fn push(&mut self, dx: f64, dy: f64) {
        self.dx[self.cabeza] = dx;
        self.dy[self.cabeza] = dy;
        self.cabeza = (self.cabeza + 1) % self.dx.len();
        if self.llenos < self.dx.len() {
            self.llenos += 1;
        }
    }

    /// Ventana en orden cronológico (el más viejo primero).
    fn cronologico(&self) -> (Vec<f64>, Vec<f64>) {
        let l = self.dx.len();
        let orden: Vec<usize> = (0..l).map(|i| (self.cabeza + i) % l).collect();
        (
            orden.iter().map(|&i| self.dx[i]).collect(),
            orden.iter().map(|&i| self.dy[i]).collect(),
        )
    }

    /// σz de la ventana actual. `None` si la ventana no está llena o el
    /// nulo es degenerado (retornos constantes ⇒ Var = 0 ⇒ z indefinido;
    /// o acumuladores no finitos).
    pub fn medir(&self) -> Option<MedidaSigmaZ> {
        if self.llenos < self.dx.len() {
            return None;
        }
        let (xs, ys) = self.cronologico();
        let n = xs.len();
        let nf = n as f64;

        // S¹² real con el kernel exacto del motor (misma aritmética que
        // la medición 163 — sin atajos algebraicos propios).
        let incs: Vec<(f64, f64)> = xs.iter().zip(ys.iter()).map(|(&a, &b)| (a, b)).collect();
        let s12 = firma_nivel2_incrementos(&incs).level2[0][1];

        // Nulo de máxima entropía: momentos cerrados del estadístico
        // lineal de asignación (pesos = sufijo de dt + ½ propio).
        let (sx, sy): (f64, f64) = (xs.iter().sum(), ys.iter().sum());
        let x_bar = sx / nf;
        let sxx: f64 = xs.iter().map(|&a| (a - x_bar) * (a - x_bar)).sum();
        let mut v: Vec<f64> = Vec::with_capacity(n);
        let mut sufijo = sy;
        for &y in &ys {
            sufijo -= y;
            v.push(sufijo + 0.5 * y);
        }
        let sv: f64 = v.iter().sum();
        let v_bar = sv / nf;
        let svv: f64 = v.iter().map(|&a| (a - v_bar) * (a - v_bar)).sum();
        let esperanza = x_bar * sv;
        let varianza = sxx * svv / (nf - 1.0);
        if !(s12.is_finite() && esperanza.is_finite() && varianza.is_finite()) || varianza <= 0.0 {
            return None;
        }
        Some(MedidaSigmaZ {
            z12: (s12 - esperanza) / varianza.sqrt(),
            sigma_x: (sxx / (nf - 1.0)).sqrt(),
            n,
        })
    }

    /// z(S²¹) — por la anti-simetría exacta (S¹²+S²¹=S¹·S² con Σy=1,
    /// E₁₂+E₂₁=S¹, Var₁₂=Var₂₁) es −z12; se RECALCULA con el kernel como
    /// verificación viva de la identidad, no se niega a ciegas.
    pub fn medir_z21(&self) -> Option<f64> {
        if self.llenos < self.dx.len() {
            return None;
        }
        let (xs, ys) = self.cronologico();
        let n = xs.len();
        let nf = n as f64;
        let incs: Vec<(f64, f64)> = xs.iter().zip(ys.iter()).map(|(&a, &b)| (a, b)).collect();
        let s21 = firma_nivel2_incrementos(&incs).level2[1][0];

        let (sx, _sy): (f64, f64) = (xs.iter().sum(), ys.iter().sum());
        let x_bar = sx / nf;
        let sxx: f64 = xs.iter().map(|&a| (a - x_bar) * (a - x_bar)).sum();
        // Pesos de S²¹: prefijo de dt + ½ propio.
        let mut u: Vec<f64> = Vec::with_capacity(n);
        let mut prefijo = 0.0_f64;
        for &y in &ys {
            u.push(prefijo + 0.5 * y);
            prefijo += y;
        }
        let su: f64 = u.iter().sum();
        let u_bar = su / nf;
        let suu: f64 = u.iter().map(|&a| (a - u_bar) * (a - u_bar)).sum();
        let esperanza = x_bar * su;
        let varianza = sxx * suu / (nf - 1.0);
        if !(s21.is_finite() && esperanza.is_finite() && varianza.is_finite()) || varianza <= 0.0 {
            return None;
        }
        Some((s21 - esperanza) / varianza.sqrt())
    }
}
