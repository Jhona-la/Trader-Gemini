//! Grupo de Renormalización multi-escala de Wilson sobre el espectro de
//! potencias (mandato §12.2 del operador, GLM-183).
//!
//! «Aplica transformaciones de grupo de renormalización en el espacio de
//! frecuencias y escalas temporales: integra los grados de libertad
//! microscópicos (ticks) para extraer las leyes de escala macroscópicas,
//! identificando la invariancia de escala y los puntos fijos críticos
//! donde todas las temporalidades se alinean.»
//!
//! Coarse-graining de Wilson: el espectro de potencias (bins de FFT,
//! eje lineal en frecuencia) se agrupa en OCTAVAS DIÁDICAS [2^k, 2^{k+1});
//! cada octava integra sus grados de libertad microscópicos en la media de
//! log-potencia. El OBSERVABLE del flujo RG es el exponente espectral
//! β_k entre octavas consecutivas (P ~ f^−β):
//!
//! - **Punto fijo**: β_k constante ⇒ invariancia de escala (1/f^β puro).
//!   Para una ley de potencias EXACTA, log P es afín en log f punto a
//!   punto, y promediar preserva la afinidad: la media de log P de
//!   CUALQUIER banda es log C − β·(media de log f de la banda) ⇒ la
//!   pendiente entre octavas es β en TODA escala — el punto fijo es
//!   identidad algebraica para cualquier partición, no aproximación.
//! - **Ruptura**: el salto máximo |β_{k+1} − β_k| localiza la escala
//!   crítica donde el régimen cambia (donde las temporalidades se alinean
//!   o quiebran) — el análogo espectral de un punto fijo que se pierde.
//!
//! Entrada: potencias por bin (positivas, finitas; el bin DC se ignora).
//! Medir-antes-de-decidir: núcleo puro + contrato; el wiring al espectro
//! del engine (Radix-2 cada 64 ticks) cuando el consejo lo consuma.

/// Flujo de renormalización de Wilson sobre un espectro de potencias.
#[derive(Debug, Clone, PartialEq)]
pub struct FlujoRg {
    /// Octavas diádicas identificadas (≥ 2 para un paso de flujo).
    pub n_octavas: usize,
    /// Exponente espectral β_k entre octavas consecutivas
    /// (longitud n_octavas − 1). Punto fijo ⇒ todos iguales.
    pub betas: Vec<f64>,
    /// Máxima desviación del punto fijo: máx |β_{k+1} − β_k|
    /// (0.0 si hay un solo paso — un paso no puede desviarse de sí mismo).
    pub desvio_punto_fijo: f64,
    /// Escala de la mayor ruptura: índice k del β_k con mayor salto al
    /// siguiente (la octava k→k+1 es donde el régimen quiebra). None con
    /// menos de dos pasos de flujo.
    pub escala_rompimiento: Option<usize>,
    /// Magnitud del salto en la escala de ruptura.
    pub salto_rompimiento: Option<f64>,
}

/// Coarse-graining diádico de Wilson y flujo del exponente espectral.
///
/// Errores (CL-15, ausencia honesta): menos de 4 bins (no hay dos octavas
/// completas), potencias no finitas o ≤ 0 (log indefinido — el llamador
/// descarta bins vacíos, jamás se rellenan con ceros ficticios).
pub fn flujo_wilson(potencias: &[f64]) -> Result<FlujoRg, &'static str> {
    // Bin DC (índice 0) fuera: no es escala, es el offset.
    let bins = potencias.get(1..).ok_or("espectro sin bins tras el DC")?;
    if bins.len() < 3 {
        return Err("se requieren >= 4 bins (dos octavas diádicas)");
    }
    if bins.iter().any(|p| !p.is_finite() || *p <= 0.0) {
        return Err("potencias deben ser finitas y > 0 (log indefinido)");
    }
    // Media de log-potencia y de log-frecuencia por octava [2^k, 2^{k+1}).
    let mut media_log_p = Vec::new();
    let mut media_log_f = Vec::new();
    let mut inicio = 0usize; // índice relativo a `bins` (bin i+1 = freq i+1)
    loop {
        let fin = ((inicio + 1).next_power_of_two()).min(bins.len());
        // next_power_of_two(inicio+1): el borde superior diádico.
        if fin <= inicio {
            break;
        }
        let mut suma_lp = 0.0f64;
        let mut suma_lf = 0.0f64;
        for (rel, &p) in bins[inicio..fin].iter().enumerate() {
            let freq = (inicio + rel + 1) as f64;
            suma_lp += p.ln();
            suma_lf += freq.ln();
        }
        let n = (fin - inicio) as f64;
        media_log_p.push(suma_lp / n);
        media_log_f.push(suma_lf / n);
        inicio = fin;
        if inicio >= bins.len() {
            break;
        }
    }
    let n_octavas = media_log_p.len();
    if n_octavas < 2 {
        return Err("no hay dos octavas completas");
    }
    let mut betas = Vec::with_capacity(n_octavas - 1);
    for k in 0..n_octavas - 1 {
        let dlp = media_log_p[k + 1] - media_log_p[k];
        let dlf = media_log_f[k + 1] - media_log_f[k];
        if dlf <= 0.0 {
            return Err("octavas sin separación espectral");
        }
        // P ~ f^−β ⇒ β = −Δlog P / Δlog f.
        betas.push(-dlp / dlf);
    }
    if betas.iter().any(|b| !b.is_finite()) {
        return Err("β no finito");
    }
    let mut desvio = 0.0f64;
    let mut escala_rompimiento = None;
    let mut salto_rompimiento = None;
    for k in 0..betas.len().saturating_sub(1) {
        let salto = (betas[k + 1] - betas[k]).abs();
        if salto > desvio {
            desvio = salto;
            escala_rompimiento = Some(k);
            salto_rompimiento = Some(salto);
        }
    }
    Ok(FlujoRg {
        n_octavas,
        betas,
        desvio_punto_fijo: desvio,
        escala_rompimiento,
        salto_rompimiento,
    })
}
