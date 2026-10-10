//! #609 (Ola 31, Qoder) — SUSTRATO DEL VOTO ESPECTRAL: la representación
//! resolución-en-escala para los motores del consenso.
//!
//! ## Por qué
//!
//! Los 13 votantes del orquestador emiten hoy un ESCALAR en [−1,1] — su
//! opinión no distingue a qué escala τ aplica. Bajo el universo espectral,
//! un motor refactorizado opina POR ESCALA: el oscilador cuántico evalúa su
//! pozo en el desplazamiento x(τ) de cada banda; el proximo motor que se
//! refactorice (soliton, shockwave) hará lo propio con su variable de
//! estado resuelta. `VotoEspectral` es la moneda común de esa refactorización.
//!
//! ## Contrato
//!
//! - `desde_escalar(v)`: el caso degenerado — el voto plano de un motor no
//!   refactorizado, extendido a las 32 escalas (el consenso con votos
//!   planos reduce al escalar: continuidad hacia atrás).
//! - `desde_espectro(x, f)`: aplica `f` a CADA desplazamiento de escala
//!   (f = la fuerza del motor; pura, sin estado).
//! - `consenso(votos, pesos)`: media ponderada por escala, clampeada —
//!   el consenso también es un espectro, y τ* del consejo selecciona la
//!   rebanada operativa.
//! - `dominante()`: la escala de mayor |voto| y su voto.
//!
//! Cero acoplamiento con el bucle vivo: la sombra observacional (core)
//! computa este consenso AL LADO del escalar y lo publica; el día que el
//! consejo ordene el cambio, el orquestador consume `consenso` en su lugar
//! — una sola ola con oráculo propio.

pub const ESCALAS_VOTO: usize = 32;

#[derive(Clone, Copy, Debug)]
pub struct VotoEspectral {
    por_escala: [f64; ESCALAS_VOTO],
}

impl Default for VotoEspectral {
    fn default() -> Self {
        Self { por_escala: [0.0; ESCALAS_VOTO] }
    }
}

impl VotoEspectral {
    /// El voto plano de un motor NO refactorizado, extendido a la malla.
    pub fn desde_escalar(v: f64) -> Self {
        let v = if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 };
        Self { por_escala: [v; ESCALAS_VOTO] }
    }

    /// El voto de un motor REFACTORIZADO: `f` evaluado en el desplazamiento
    /// de cada escala. f no finito ⇒ 0 en esa escala (sin inventar).
    pub fn desde_espectro(x_por_escala: &[f64; ESCALAS_VOTO], f: impl Fn(f64) -> f64) -> Self {
        let mut por_escala = [0.0; ESCALAS_VOTO];
        for (k, &x) in x_por_escala.iter().enumerate() {
            let v = f(x);
            por_escala[k] = if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 };
        }
        Self { por_escala }
    }

    #[inline]
    pub fn en_escala(&self, k: usize) -> f64 {
        self.por_escala.get(k).copied().unwrap_or(0.0)
    }

    #[inline]
    pub fn array(&self) -> &[f64; ESCALAS_VOTO] {
        &self.por_escala
    }

    /// La escala de mayor |voto| y su voto con signo. Un espectro sin
    /// convicción (todo cero) NO tiene dominante — `None` (sin inventar).
    pub fn dominante(&self) -> Option<(usize, f64)> {
        let mut mejor: Option<(usize, f64)> = None;
        for (k, &v) in self.por_escala.iter().enumerate() {
            if mejor.map(|(_, m)| v.abs() > m.abs()).unwrap_or(true) {
                mejor = Some((k, v));
            }
        }
        match mejor {
            Some((k, v)) if v.abs() > 0.0 => Some((k, v)),
            _ => None,
        }
    }

    /// Media de la banda [lo, hi] de escalas (ambos inclusive).
    pub fn media_banda(&self, lo: usize, hi: usize) -> Option<f64> {
        if lo > hi || hi >= ESCALAS_VOTO {
            return None;
        }
        let n = (hi - lo + 1) as f64;
        let suma: f64 = self.por_escala[lo..=hi].iter().sum();
        Some(suma / n)
    }

    /// Media de las escalas activas (con voto no nulo) en la banda [lo, hi].
    /// Resuelve R7-R4-B-2: evita que escalas excluidas por el gate de observabilidad
    /// (anuladas a 0.0) diluyan el denominador a un 32 fijo, lo que imponía un techo
    /// artificial estricto a la coherencia inter-espectral (<= n_activas / 32)
    /// penalizando arbitrariamente a símbolos con menor cadencia de ticks.
    pub fn media_banda_activa(&self, lo: usize, hi: usize) -> Option<f64> {
        if lo > hi || hi >= ESCALAS_VOTO {
            return None;
        }
        let mut suma = 0.0f64;
        let mut n_activas = 0usize;
        for &v in &self.por_escala[lo..=hi] {
            if v.is_finite() && v.abs() > 1e-9 {
                suma += v;
                n_activas += 1;
            }
        }
        if n_activas > 0 {
            Some(suma / n_activas as f64)
        } else {
            None
        }
    }

    /// Consenso espectral: media ponderada por escala, clampeada. Pesos no
    /// finitos o suma ~0 ⇒ espectro plano en 0 (sin inventar convicción).
    pub fn consenso(votos: &[VotoEspectral], pesos: &[f64]) -> VotoEspectral {
        let mut por_escala = [0.0; ESCALAS_VOTO];
        let mut peso_total = 0.0f64;
        for (v, &w) in votos.iter().zip(pesos.iter()) {
            if !w.is_finite() || w <= 0.0 {
                continue;
            }
            peso_total += w;
            for (k, &vk) in v.por_escala.iter().enumerate() {
                por_escala[k] += w * vk;
            }
        }
        if peso_total > 1e-12 {
            for v in &mut por_escala {
                *v = (*v / peso_total).clamp(-1.0, 1.0);
            }
        } else {
            por_escala = [0.0; ESCALAS_VOTO];
        }
        Self { por_escala }
    }

    /// #626 — CONSENSO PONDERADO POR ESCALA: el peso del motor m depende
    /// de la escala k (su habilidad prequential medida ahí). Normalización
    /// por Σw EN CADA escala; pesos no finitos/≤0 se saltan; Σw ~0 en una
    /// escala ⇒ esa escala vota 0 (sin inventar). Con pesos uniformes es
    /// matemáticamente `consenso` (continuidad hacia atrás con #623).
    pub fn consenso_por_escala(
        votos: &[VotoEspectral],
        pesos: &[[f64; ESCALAS_VOTO]],
    ) -> VotoEspectral {
        let mut por_escala = [0.0; ESCALAS_VOTO];
        for (k, salida) in por_escala.iter_mut().enumerate() {
            let mut suma_w = 0.0f64;
            let mut suma_wv = 0.0f64;
            for (v, fila) in votos.iter().zip(pesos.iter()) {
                let w = fila[k];
                if !w.is_finite() || w <= 0.0 {
                    continue;
                }
                suma_w += w;
                suma_wv += w * v.en_escala(k);
            }
            *salida = if suma_w > 1e-12 {
                (suma_wv / suma_w).clamp(-1.0, 1.0)
            } else {
                0.0
            };
        }
        Self { por_escala }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qo_609_escalar_plano_y_dominante() {
        let plano = VotoEspectral::desde_escalar(0.6);
        assert_eq!(plano.en_escala(0), 0.6);
        assert_eq!(plano.en_escala(31), 0.6);
        assert!((plano.media_banda(0, 31).unwrap() - 0.6).abs() < 1e-12);
        // El dominante de un plano es la primera escala con su voto.
        assert_eq!(plano.dominante(), Some((0, 0.6)));
        let vacio = VotoEspectral::default();
        assert_eq!(vacio.dominante(), None);
        // No finito ⇒ 0 (sin inventar).
        assert_eq!(VotoEspectral::desde_escalar(f64::NAN).en_escala(5), 0.0);
    }

    #[test]
    fn qo_609_espectro_aplica_f_por_escala() {
        let mut x = [0.0; ESCALAS_VOTO];
        for (k, v) in x.iter_mut().enumerate() {
            *v = if k % 2 == 0 { 0.5 } else { -0.5 };
        }
        let voto = VotoEspectral::desde_espectro(&x, |xi| -xi); // fuerza = −x
        assert_eq!(voto.en_escala(0), -0.5);
        assert_eq!(voto.en_escala(1), 0.5);
        assert_eq!(voto.dominante(), Some((0, -0.5))); // primer máximo |·|
        // Banda parcial.
        assert_eq!(voto.media_banda(0, 1), Some(0.0));
    }

    #[test]
    fn qo_609_consenso_ponderado_reduce_al_escalar() {
        // DOS votos planos con pesos: el consenso espectral reduce a la
        // media escalar ponderada en TODAS las escalas (continuidad hacia
        // atrás con los motores no refactorizados).
        let a = VotoEspectral::desde_escalar(1.0);
        let b = VotoEspectral::desde_escalar(-1.0);
        let c = VotoEspectral::consenso(&[a, b], &[3.0, 1.0]);
        let esperado = (3.0 * 1.0 + 1.0 * (-1.0)) / 4.0;
        for k in 0..ESCALAS_VOTO {
            assert!((c.en_escala(k) - esperado).abs() < 1e-12);
        }
        // Pesos inválidos ⇒ consenso plano en 0 (sin convicción inventada).
        let c0 = VotoEspectral::consenso(&[a], &[0.0]);
        assert_eq!(c0.en_escala(10), 0.0);
        // Con votos RESUELTOS el consenso conserva la forma por escala.
        let mut x = [0.0; ESCALAS_VOTO];
        x[7] = 1.0;
        let resuelto = VotoEspectral::desde_espectro(&x, |xi| xi);
        let mix = VotoEspectral::consenso(&[resuelto, b], &[1.0, 1.0]);
        assert!((mix.en_escala(7) - 0.0).abs() < 1e-12);
        assert!((mix.en_escala(8) - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn test_r7_r4_b2_media_banda_activa_no_diluye_escalas_gated() {
        let mut x = [0.0; ESCALAS_VOTO];
        // Supongamos que las escalas 0..9 están anuladas por el gate de observabilidad (cadencia lenta)
        // y las escalas 10..31 tienen voto 0.8
        for k in 10..ESCALAS_VOTO {
            x[k] = 0.8;
        }
        let voto = VotoEspectral::desde_arr(&x);
        // media_banda(0, 31) divide por 32 -> 22 * 0.8 / 32 = 0.55
        let media_fija = voto.media_banda(0, 31).unwrap();
        assert!((media_fija - (22.0 * 0.8 / 32.0)).abs() < 1e-12);

        // media_banda_activa(0, 31) divide sólo entre las 22 escalas activas -> exactamente 0.8!
        let media_activa = voto.media_banda_activa(0, 31).unwrap();
        assert!((media_activa - 0.8).abs() < 1e-12, "media_activa debe ser 0.8, got {media_activa}");
    }
}

impl VotoEspectral {
    /// #616 — construye desde un array precomputado por escala (para
    /// motores que necesitan contexto de VECINOS, no solo el valor local).
    pub fn desde_arr(por_escala: &[f64; ESCALAS_VOTO]) -> Self {
        let mut arr = [0.0; ESCALAS_VOTO];
        for (k, &v) in por_escala.iter().enumerate() {
            arr[k] = if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 };
        }
        Self { por_escala: arr }
    }
}
