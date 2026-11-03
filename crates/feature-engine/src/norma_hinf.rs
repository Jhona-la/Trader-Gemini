//! Norma H∞ del error de estimación escalar (mandato §12.4, GLM-185).
//!
//! «Acopla al cálculo de apalancamiento un controlador robusto H∞ que trate
//! al mercado y a las imperfecciones de red como un adversario que busca
//! maximizar nuestro error de seguimiento.»
//!
//! El ladrillo medible de esa torre: para el filtro lineal escalar
//! e_{k+1} = a·e_k + b·u_k con |a| < 1 (estacionario), la norma H∞ de la
//! perturbación u al error e es EXACTA y cerrada:
//!
//! ‖G‖∞ = |b| / √(1 − a²)
//!
//! (energía de salida máxima por unidad de energía de entrada en el peor
//! caso; H₂ sería |b|/√(1−a²) también en escalar — la distinción aparece
//! en MIMO; aquí el punto es que el PEOR CASO es computable, no simulado).
//!
//! Aplicación viva: el estimador de precio del engine (Kalman 1D / EMA) es
//! exactamente ese filtro — su error satisface e_{k+1} = (1−K)·e_k +
//! (1−K)·w_k − K·v_k con w el ruido de proceso y v el de medición, así que
//! la ganancia H∞ de cada canal de perturbación al error del precio justo
//! es (1−K)/√(1−(1−K)²) para el proceso y K/√(1−(1−K)²) para la medición.
//! [`GananciaHinf`] publica esas dos ganancias del peor caso por tick.
//!
//! Medir-antes-de-decidir: observable puro, sin tocar decisiones de
//! apalancamiento; el acople a sizing es decisión del consejo.

/// Ganancias H∞ del peor caso del error de un filtro escalar estacionario.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GananciaHinf {
    /// Ganancia del peor caso del canal de PROCESO w → error.
    pub proceso: f64,
    /// Ganancia del peor caso del canal de MEDICIÓN v → error.
    pub medicion: f64,
    /// a = 1 − K del filtro (factor de contracción del error por tick).
    pub a: f64,
}

/// Norma H∞ del sistema escalar e_{k+1} = a·e + b·u: |b|/√(1−a²).
///
/// |a| ≥ 1 ⇒ Err honesto: el sistema NO contrae (la ganancia del peor caso
/// es infinita — un integrador acumula energía sin cota). NaN/Inf ⇒ Err.
pub fn norma_hinf_escalar(a: f64, b: f64) -> Result<f64, &'static str> {
    if !a.is_finite() || !b.is_finite() {
        return Err("parámetros no finitos");
    }
    if a.abs() >= 1.0 {
        return Err("|a| >= 1: sistema no contractivo, ganancia H∞ infinita");
    }
    let denominador = (1.0 - a * a).sqrt();
    if denominador <= 0.0 || !denominador.is_finite() {
        return Err("denominador degenerado");
    }
    let norma = b.abs() / denominador;
    if !norma.is_finite() {
        return Err("norma no finita");
    }
    Ok(norma)
}

/// Ganancias H∞ de un filtro de ganancia K (0 < K < 1): el error cumple
/// e' = (1−K)·e + (1−K)·w − K·v, así que cada canal es un sistema escalar
/// con a = 1−K y su propia b. Err si K ≤ 0 o K ≥ 1 (el filtro ni escucha
/// ni converge — ganancia infinita o estimador muerto).
pub fn ganancias_hinf_filtro(k: f64) -> Result<GananciaHinf, &'static str> {
    if !k.is_finite() || k <= 0.0 || k >= 1.0 {
        return Err("K debe estar en (0,1): fuera, el filtro no es contractivo");
    }
    let a = 1.0 - k;
    Ok(GananciaHinf {
        proceso: norma_hinf_escalar(a, a)?,
        medicion: norma_hinf_escalar(a, -k)?,
        a,
    })
}

/// Estimador streaming de la ganancia H∞ EMPÍRICA del precio: peyor ratio
/// de energía del error de estimación a energía de la perturbación total
/// (Δ precio − estimado como perturbación compuesta observada), sobre una
/// ventana de anillo. Es la cota inferior observable de la norma verdadera:
/// el peor caso puede superar lo visto, jamás lo contrario.
pub struct GananciaHinfEmpirica {
    /// Anillo de energías por tick (error², perturbación²).
    errores: Vec<f64>,
    perturbaciones: Vec<f64>,
    cabeza: usize,
    llenos: usize,
    suma_error: f64,
    suma_perturbacion: f64,
    /// Peor ratio observado en ventanas completas (energía ventana).
    peor: f64,
}

impl GananciaHinfEmpirica {
    pub fn try_new(ventana: usize) -> Result<Self, &'static str> {
        if ventana < 8 {
            return Err("ventana >= 8 para energías significativas");
        }
        Ok(Self {
            errores: vec![0.0; ventana],
            perturbaciones: vec![0.0; ventana],
            cabeza: 0,
            llenos: 0,
            suma_error: 0.0,
            suma_perturbacion: 0.0,
            peor: 0.0,
        })
    }

    /// Ingresa un tick: error del estimador (precio − estimado) y la
    /// perturbación observada (innovación del filtro). Ninguno finito ⇒
    /// Err y estado intacto.
    pub fn push(&mut self, error: f64, perturbacion: f64) -> Result<(), &'static str> {
        if !error.is_finite() || !perturbacion.is_finite() {
            return Err("muestras no finitas");
        }
        let e2 = error * error;
        let p2 = perturbacion * perturbacion;
        if !e2.is_finite() || !p2.is_finite() {
            return Err("energías no finitas");
        }
        let n = self.errores.len();
        let viejo_e = self.errores[self.cabeza];
        let viejo_p = self.perturbaciones[self.cabeza];
        self.suma_error += e2 - viejo_e;
        self.suma_perturbacion += p2 - viejo_p;
        self.errores[self.cabeza] = e2;
        self.perturbaciones[self.cabeza] = p2;
        self.cabeza = (self.cabeza + 1) % n;
        if self.llenos < n {
            self.llenos += 1;
        } else if self.suma_perturbacion > 0.0 {
            let ratio = self.suma_error / self.suma_perturbacion;
            if ratio.is_finite() && ratio > self.peor {
                self.peor = ratio;
            }
        }
        Ok(())
    }

    /// Peor ganancia cuadrática-media observada (√ del peor ratio de
    /// energías); None hasta llenar la primera ventana.
    pub fn peor_ganancia(&self) -> Option<f64> {
        (self.llenos == self.errores.len() && self.peor.is_finite() && self.peor >= 0.0)
            .then(|| self.peor.sqrt())
    }

    /// Ventana ocupada (ticks).
    pub fn llenos(&self) -> usize {
        self.llenos
    }
}
