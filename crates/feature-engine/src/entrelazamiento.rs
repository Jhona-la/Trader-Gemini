//! Entrelazamiento temporal neto (GLM 155 — titularidad §56): el observable
//! de mercado construido SOBRE el núcleo MPS del 151. La cadena cuántica de
//! L sitios es una ventana deslizante de retornos binarizados (↑/↓) del
//! mismo activo; el «estado» ψ sobre las 2^L configuraciones es la raíz de
//! la distribución empírica de ventanas (ψ = √p̂), y la entropía de
//! entrelazamiento de von Neumann del corte medio (S, `mps::
//! entrelazamiento_corte`) mide la dependencia NO lineal entre la primera y
//! la segunda mitad de la ventana — lo que un producto de marginales no
//! puede representar.
//!
//! El valor NETO exige restar el ruido de muestreo: `entrelazamiento_neto`
//! permuta los retornos (Fisher–Yates determinista, semilla fija — rompe el
//! orden temporal preservando la marginal EXACTA) y mide S sobre la
//! permutación. ΔS = S − S_nulo es el entrelazamiento temporal genuino,
//! neto del sesgo finito-muestral: para una serie iid ΔS ≈ 0 sin importar
//! la asimetría de la marginal (los bits iid dan estado producto, S = 0
//! exacto en ambos lados salvo ruido que se cancela al restar).
//!
//! Dos binarizaciones documentadas, ambas reportadas (ninguna es «la»
//! correcta): signo del retorno (física de precio ↑/↓) y mediana adaptativa
//! (auto-calibrada al bloque, simétrica por construcción).
//!
//! # Ausencia honesta (CL-15)
//! Retornos no finitos se excluyen ANTES de ventanear (la serie se
//! comprime; se reportan los descartes). Serie constante en modo mediana ⇒
//! `MedianaDegenerada` (jamás un umbral inventado). Sin ventanas ⇒ error.

use crate::mps::{self, MPS_MAX_L};

/// Modo de binarización del retorno: ↑ (bit 1) o ↓ (bit 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModoBinarizacion {
    /// bit = 1 ⇔ retorno > 0 (física de precio; el 0.0 exacto cuenta como ↓).
    Signo,
    /// bit = 1 ⇔ retorno > mediana del bloque (adaptativo, simétrico).
    Mediana,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EntrelazamientoError {
    /// L fuera de rango (1..=MPS_MAX_L): 2^L debe ser estimable.
    LongitudInvalida { l: usize },
    /// Menos de L+1 retornos finitos: ni una ventana.
    RetornosInsuficientes { recibidos: usize, necesarios: usize },
    /// Serie constante en modo mediana: umbral degenerado, ausencia honesta.
    MedianaDegenerada,
    /// El núcleo MPS declara degeneración (norma/entrada) — se propaga.
    NucleoDegenerado,
    /// Cero ventanas válidas tras los filtros.
    SinVentanas,
}

/// Resultado de una medición de entrelazamiento temporal neto.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MedidaEntrelazamiento {
    /// S del estado empírico (corte medio de la ventana de L sitios).
    pub s: f64,
    /// S del estado permutado (nulo de orden temporal, misma marginal).
    pub s_nulo: f64,
    /// ΔS = S − S_nulo: entrelazamiento temporal genuino, neto del ruido.
    pub delta: f64,
    /// Ventanas válidas que entraron a p̂.
    pub ventanas: usize,
    /// Retornos descartados por no finitos.
    pub descartados: usize,
}

/// Filtra los retornos no finitos y devuelve la serie limpia.
fn filtrar_finitos(rets: &[f64]) -> (Vec<f64>, usize) {
    let mut limpios = Vec::with_capacity(rets.len());
    let mut descartados = 0usize;
    for &r in rets {
        if r.is_finite() {
            limpios.push(r);
        } else {
            descartados += 1;
        }
    }
    (limpios, descartados)
}

/// Mediana de una serie NO vacía (promedio de los dos centrales si par).
fn mediana(vals: &[f64]) -> f64 {
    let mut v = vals.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Bits de la serie bajo el modo elegido. Devuelve la serie de bits y el
/// umbral efectivo usado (0.0 en signo; la mediana en mediana — o error si
/// la serie es constante).
fn binarizar(rets: &[f64], modo: ModoBinarizacion) -> Result<(Vec<bool>, f64), EntrelazamientoError> {
    match modo {
        ModoBinarizacion::Signo => Ok((rets.iter().map(|&r| r > 0.0).collect(), 0.0)),
        ModoBinarizacion::Mediana => {
            let m = mediana(rets);
            // Serie constante ⇒ todos los bits iguales por >: sin información
            // y el umbral es arbitrario — ausencia honesta.
            if rets.iter().all(|&r| r == rets[0]) {
                return Err(EntrelazamientoError::MedianaDegenerada);
            }
            Ok((rets.iter().map(|&r| r > m).collect(), m))
        }
    }
}

/// Estado empírico ψ = √p̂ sobre las 2^L configuraciones de ventanas
/// deslizantes (stride 1) de la serie binarizada. El bit del retorno más
/// antiguo de la ventana es el sitio MÁS significativo (convenio del 151:
/// sitio 0 = bit alto), así el corte medio separa primera y segunda mitad
/// temporales. Devuelve (ψ, ventanas, retornos descartados por no finitos).
pub fn estado_empirico(
    rets: &[f64],
    l: usize,
    modo: ModoBinarizacion,
) -> Result<(Vec<f64>, usize, usize), EntrelazamientoError> {
    if l == 0 || l > MPS_MAX_L {
        return Err(EntrelazamientoError::LongitudInvalida { l });
    }
    let (limpios, descartados) = filtrar_finitos(rets);
    if limpios.len() < l {
        return Err(EntrelazamientoError::RetornosInsuficientes {
            recibidos: limpios.len(),
            necesarios: l,
        });
    }
    let (bits, _umbral) = binarizar(&limpios, modo)?;
    let dim = 1usize << l;
    let mut cuentas = vec![0u64; dim];
    let n_ventanas = bits.len() + 1 - l;
    for i in 0..n_ventanas {
        let mut idx = 0usize;
        for b in &bits[i..i + l] {
            idx = (idx << 1) | (*b as usize);
        }
        cuentas[idx] += 1;
    }
    let total = n_ventanas as f64;
    let mut psi = vec![0.0f64; dim];
    for (c, k) in cuentas.iter().zip(psi.iter_mut()) {
        if *c > 0 {
            *k = (*c as f64 / total).sqrt();
        }
    }
    Ok((psi, n_ventanas, descartados))
}

/// Fisher–Yates determinista (xorshift64*, semilla fija): rompe el orden
/// temporal preservando la marginal EXACTA — el nulo contra el que se resta.
fn permutacion_determinista(rets: &mut [f64]) {
    let mut semilla: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rng = move || {
        semilla ^= semilla >> 12;
        semilla ^= semilla << 25;
        semilla ^= semilla >> 27;
        semilla.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    for i in (1..rets.len()).rev() {
        let j = (rng() % (i as u64 + 1)) as usize;
        rets.swap(i, j);
    }
}

/// Entrelazamiento temporal neto de una serie de retornos: S del estado
/// empírico, S del nulo permutado y ΔS = S − S_nulo. El corte es el medio
/// de la ventana (primera vs segunda mitad temporal). Ausencia honesta en
/// cada degeneración; jamás valores inventados.
pub fn entrelazamiento_neto(
    rets: &[f64],
    l: usize,
    modo: ModoBinarizacion,
) -> Result<MedidaEntrelazamiento, EntrelazamientoError> {
    let (psi, ventanas, descartados) = estado_empirico(rets, l, modo)?;
    if ventanas == 0 {
        return Err(EntrelazamientoError::SinVentanas);
    }
    let corte = l / 2;
    let s = mps::entrelazamiento_corte(&psi, l, corte).map_err(|_| EntrelazamientoError::NucleoDegenerado)?;
    // Nulo: misma marginal, orden destruido — determinista y reproducible.
    let (limpios, _) = filtrar_finitos(rets);
    let mut perm = limpios;
    permutacion_determinista(&mut perm);
    let (psi_nulo, _, _) = estado_empirico(&perm, l, modo)?;
    let s_nulo =
        mps::entrelazamiento_corte(&psi_nulo, l, corte).map_err(|_| EntrelazamientoError::NucleoDegenerado)?;
    Ok(MedidaEntrelazamiento {
        s,
        s_nulo,
        delta: s - s_nulo,
        ventanas,
        descartados,
    })
}
