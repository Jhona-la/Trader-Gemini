//! Redes Tensoriales: Matrix Product States (GLM 151 — titularidad §56 del
//! mandato, ratificada por el consejo en Ω209). Comprime estados de dimensión
//! exponencial 2^L en cadenas de tensores de bajo rango A[l] (D_l, d,
//! D_{l+1}) — O(L·d·D²) en memoria y O(L·d·D³) en contracción, dentro del
//! presupuesto de 16 GB con D acotado (MPS_MAX_D).
//!
//! # Qué provee
//! - `Mps::contraccion_norma`: ⟨ψ|ψ⟩ exacto por transfer-matrix product.
//! - `Mps::contraccion_solape`: ⟨ψ|φ⟩ entre dos MPS (ortogonalidad).
//! - `mps_desde_estado`: SVD-truncation sucesivo — MPS de un estado completo
//!   pequeño (L ≤ MPS_MAX_L) con bond dim acotada + error de truncamiento.
//! - `entrelazamiento_corte`: entropía de von Neumann S = −Σ p·ln p del
//!   espectro de Schmidt — EL observable §56 (entrelazamiento no lineal).
//!
//! # Qué NO hace (límites documentados)
//! No es DMRG (optimización variacional — átomo siguiente); no maneja
//! simetrías U(1)/Z₂ ni MPOs; d=2 físico (retorno discretizado ↑/↓).
//!
//! # Ingeniería numérica
//! SVD vía EVD simétrico de M·Mᵀ con Jacobi cíclico PROPIO (feature-engine
//! no depende de risk-engine, y el Jacobi de allí exige PSD con diagonal
//! unidad). σ = √λ; autovectores acumulando rotaciones.

/// Dimensión física por sitio (qubit: retorno discretizado ↑/↓).
pub const MPS_D_FIS: usize = 2;
/// Dimensión de enlace máxima (la gramiana a diagonalizar es (D·d)² ≤ 256).
pub const MPS_MAX_D: usize = 8;
/// Longitud máxima para construcción desde estado completo (2^L ≤ 4096).
pub const MPS_MAX_L: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MpsError {
    LongitudInvalida { l: usize },
    BondInvalido { d: usize },
    EstadoLongitud { esperado: usize, recibido: usize },
    /// Norma/entrada degenerada: ausencia honesta (CL-15), jamás clamp.
    NormaDegenerada { valor: f64 },
    EntradaNoFinita,
}

/// MPS abierto: L tensores A[l] de forma (D_l, d, D_{l+1}), D_0 = D_L = 1.
/// Layout plano: A[l][((dl·d + s)·D_{l+1} + dnext)].
#[derive(Debug, Clone)]
pub struct Mps {
    pub tensores: Vec<Vec<f64>>,
    pub dims: Vec<usize>, // dims[l] = D_l, largo L+1, dims[0] = dims[L] = 1
    pub largo: usize,
}

impl Mps {
    /// ⟨ψ|ψ⟩ exacto: env evoluciona nueva[j,j'] = Σ_{i,i',s} env[i,i']·A[i,s,j]·A[i',s,j'].
    pub fn contraccion_norma(&self) -> f64 {
        let mut env: Vec<f64> = vec![1.0]; // 1×1
        let mut dl = 1usize;
        for l in 0..self.largo {
            let dnext = self.dims[l + 1];
            let mut nueva = vec![0.0; dnext * dnext];
            for j in 0..dnext {
                for jp in 0..dnext {
                    let mut s_total = 0.0;
                    for i in 0..dl {
                        for ip in 0..dl {
                            let e = env[i * dl + ip];
                            if e == 0.0 {
                                continue;
                            }
                            for fs in 0..MPS_D_FIS {
                                let a = self.tensores[l]
                                    [((i * MPS_D_FIS + fs) * dnext) + j];
                                let ap = self.tensores[l]
                                    [((ip * MPS_D_FIS + fs) * dnext) + jp];
                                s_total += e * a * ap;
                            }
                        }
                    }
                    nueva[j * dnext + jp] = s_total;
                }
            }
            env = nueva;
            dl = dnext;
        }
        env[0]
    }

    /// ⟨ψ|φ⟩: nueva[j_a, j_b] = Σ_{i_a,i_b,s} env[i_a,i_b]·A[i_a,s,j_a]·B[i_b,s,j_b].
    /// El env es RECTANGULAR (D^A_l × D^B_l): cada cadena evoluciona su
    /// propia dimensión de enlace.
    pub fn contraccion_solape(&self, otro: &Mps) -> Result<f64, MpsError> {
        if self.largo != otro.largo {
            return Err(MpsError::LongitudInvalida { l: otro.largo });
        }
        let mut env: Vec<f64> = vec![1.0]; // 1×1
        let mut da = 1usize; // D^A_l (self)
        let mut db = 1usize; // D^B_l (otro)
        for l in 0..self.largo {
            let dna = self.dims[l + 1];
            let dnb = otro.dims[l + 1];
            let mut nueva = vec![0.0; dna * dnb];
            for ja in 0..dna {
                for jb in 0..dnb {
                    let mut s_total = 0.0;
                    for ia in 0..da {
                        for ib in 0..db {
                            let e = env[ia * db + ib];
                            if e == 0.0 {
                                continue;
                            }
                            for fs in 0..MPS_D_FIS {
                                let a = self.tensores[l]
                                    [((ia * MPS_D_FIS + fs) * dna) + ja];
                                let b = otro.tensores[l]
                                    [((ib * MPS_D_FIS + fs) * dnb) + jb];
                                s_total += e * a * b;
                            }
                        }
                    }
                    nueva[ja * dnb + jb] = s_total;
                }
            }
            env = nueva;
            da = dna;
            db = dnb;
        }
        Ok(env[0])
    }

    /// Estado denso 2^L del MPS (contracción sitio a sitio; sitio 0 = bit MÁS
    /// significativo, el convenio de `entrelazamiento_corte` y del
    /// `estado_empirico` del 155). Consumido por la curva de fases S↔g del
    /// 156: S del ground state del DMRG y S del mercado pasan por EL MISMO
    /// funcional — comparación limpia por construcción.
    ///
    /// env[(D_l, configuración)] evoluciona: nueva[D_{l+1}, cfg·2 + σ] =
    /// Σ env[D_l, cfg]·A[l][(D_l·2+σ)·D_{l+1}]. El resultado final vive en
    /// la fila D_L = 0.
    pub fn estado_denso(&self) -> Result<Vec<f64>, MpsError> {
        if self.largo == 0 || self.largo > MPS_MAX_L {
            return Err(MpsError::LongitudInvalida { l: self.largo });
        }
        // (D_actual, coefs sobre configs de sitios ya contraídos)
        let mut fila: Vec<(usize, Vec<f64>)> = vec![(1usize, vec![1.0])];
        for l in 0..self.largo {
            let dnext = self.dims[l + 1];
            let a = &self.tensores[l];
            let nconf = fila[0].1.len();
            let mut nueva: Vec<(usize, Vec<f64>)> =
                (0..dnext).map(|_| (dnext, vec![0.0; nconf * 2])).collect();
            for (aix, (_, coefs)) in fila.iter().enumerate() {
                for fs in 0..MPS_D_FIS {
                    for aj in 0..dnext {
                        let v = a[((aix * MPS_D_FIS + fs) * dnext) + aj];
                        if v == 0.0 {
                            continue;
                        }
                        for (c, coef) in coefs.iter().enumerate() {
                            nueva[aj].1[c * 2 + fs] += v * coef;
                        }
                    }
                }
            }
            fila = nueva;
        }
        Ok(fila[0].1.clone())
    }
}

/// Jacobi cíclico simétrico (n ≤ 256), devuelve el espectro o None.
fn jacobi_simetrico(a: &[f64], n: usize, tolerancia: f64) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    const MAX_SWEEPS: usize = 64;
    for _ in 0..=MAX_SWEEPS {
        let mut off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += m[i * n + j] * m[i * n + j];
            }
        }
        if off <= tolerancia {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Semejanza completa (filas + columnas); la re-simetrización
                // explícita rompería rotaciones previas de la misma pasada.
                for k in 0..n {
                    let mpk = m[p * n + k];
                    let mqk = m[q * n + k];
                    m[p * n + k] = c * mpk - s * mqk;
                    m[q * n + k] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let mkp = m[k * n + p];
                    let mkq = m[k * n + q];
                    m[k * n + p] = c * mkp - s * mkq;
                    m[k * n + q] = s * mkp + c * mkq;
                }
            }
        }
    }
    let espectro: Vec<f64> = (0..n).map(|i| m[i * n + i]).collect();
    if espectro.iter().any(|v| !v.is_finite()) {
        None
    } else {
        Some(espectro)
    }
}

/// Autovectores de matriz simétrica n×n por Jacobi con rotaciones
/// acumuladas. Devuelve (V: n×k principales column-ordered, λ: k).
fn autovectores_jacobi(a: &[f64], n: usize, k: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    const MAX_SWEEPS: usize = 64;
    for _ in 0..=MAX_SWEEPS {
        let mut off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += m[i * n + j] * m[i * n + j];
            }
        }
        if off <= 1e-24 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Semejanza completa M ← JᵀMJ: filas p,q (izquierda) y
                // columnas p,q (derecha). Sólo así m converge a diagonal.
                for kk in 0..n {
                    let mpk = m[p * n + kk];
                    let mqk = m[q * n + kk];
                    m[p * n + kk] = c * mpk - s * mqk;
                    m[q * n + kk] = s * mpk + c * mqk;
                }
                for kk in 0..n {
                    let mkp = m[kk * n + p];
                    let mkq = m[kk * n + q];
                    m[kk * n + p] = c * mkp - s * mkq;
                    m[kk * n + q] = s * mkp + c * mkq;
                }
                for kk in 0..n {
                    let vkp = v[kk * n + p];
                    let vkq = v[kk * n + q];
                    v[kk * n + p] = c * vkp - s * vkq;
                    v[kk * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        m[j * n + j].partial_cmp(&m[i * n + i]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out_v = vec![0.0; n * k];
    let mut out_l = vec![0.0; k];
    for (col, &ix) in idx.iter().enumerate().take(k) {
        for f in 0..n {
            out_v[f * k + col] = v[f * n + ix];
        }
        out_l[col] = m[ix * n + ix];
    }
    if out_v.iter().any(|x| !x.is_finite()) || out_l.iter().any(|x| !x.is_finite()) {
        return None;
    }
    Some((out_v, out_l))
}

/// Construye el MPS de un estado completo (2^L coeficientes, L ≤
/// MPS_MAX_L) por SVD-truncation sucesivo con bond dim objetivo D.
/// Devuelve (MPS, error de truncamiento = Σ σ² descartados).
pub fn mps_desde_estado(
    estado: &[f64],
    l: usize,
    bond_objetivo: usize,
) -> Result<(Mps, f64), MpsError> {
    if l == 0 || l > MPS_MAX_L {
        return Err(MpsError::LongitudInvalida { l });
    }
    if bond_objetivo == 0 || bond_objetivo > MPS_MAX_D {
        return Err(MpsError::BondInvalido { d: bond_objetivo });
    }
    let esperado = 1usize << l;
    if estado.len() != esperado {
        return Err(MpsError::EstadoLongitud { esperado, recibido: estado.len() });
    }
    if estado.iter().any(|v| !v.is_finite()) {
        return Err(MpsError::EntradaNoFinita);
    }

    // Estructura: resto es matriz (D_l × d^{L-l}) con la física de los
    // sitios l+1..L TODAVÍA en las columnas. En cada sitio:
    //   M = resto.reshape(D_l·d, d^{L-l-1})  [bits más significativos = s_l]
    //   SVD(M) → A[l] = U (D_l·d × D_{l+1}), resto = σ·Vᵀ (D_{l+1} × d^{L-l-1})
    // El último sitio toma A[L-1] = resto.reshape(D_{L-1}·d, 1) sin SVD.
    let mut dl = 1usize;
    let mut resto_cols = esperado; // d^{L-l}
    let mut resto: Vec<f64> = estado.to_vec();
    let mut tensores: Vec<Vec<f64>> = Vec::with_capacity(l);
    let mut dims: Vec<usize> = vec![1];
    let mut error_total = 0.0f64;

    for sitio in 0..l {
        let fisicas = if sitio == 0 { 1 } else { MPS_D_FIS };
        // Reshape M: (dl·fisicas... para sitio 0 la física s_0 son los bits
        // más significativos del estado: M = psi.reshape(d, d^{L-1}).
        // Para sitios >0: M[(i,s), k] = resto[i, s·(cols/2) + k].
        let m_cols = if sitio == 0 { resto_cols / MPS_D_FIS } else { resto_cols / MPS_D_FIS };
        let m_filas = dl * MPS_D_FIS;
        let mut m = vec![0.0; m_filas * m_cols];
        for i in 0..dl {
            for fs in 0..MPS_D_FIS {
                for kc in 0..m_cols {
                    m[(i * MPS_D_FIS + fs) * m_cols + kc] =
                        resto[i * resto_cols + fs * m_cols + kc];
                }
            }
        }
        let _ = fisicas;
        if sitio == l - 1 {
            // A[L-1] = M con D_L = 1: m_cols == 1 garantizado.
            tensores.push(m);
            dims.push(1);
            break;
        }
        // Gramiana G = M·Mᵀ (m_filas × m_filas).
        let mut g = vec![0.0; m_filas * m_filas];
        for i in 0..m_filas {
            for j in 0..m_filas {
                let mut s = 0.0;
                for kc in 0..m_cols {
                    s += m[i * m_cols + kc] * m[j * m_cols + kc];
                }
                g[i * m_filas + j] = s;
            }
        }
        // Pedir TODAS las columnas: el error de truncamiento necesita el
        // espectro completo (Σλ descartadas por rango Y por bond).
        let (u, lambdas) = autovectores_jacobi(&g, m_filas, m_filas)
            .ok_or(MpsError::NormaDegenerada { valor: f64::NAN })?;
        let k = lambdas.len();
        let lam_max = lambdas.iter().cloned().fold(0.0f64, f64::max);
        let umbral = lam_max * 1e-24;
        let d_bond = lambdas
            .iter()
            .filter(|&&x| x > umbral)
            .count()
            .min(bond_objetivo)
            .max(1);
        for &lam in lambdas.iter().skip(d_bond) {
            if lam > 0.0 {
                error_total += lam;
            }
        }
        // A[l] = U[(D_l·d + s), α].
        let mut a = vec![0.0; m_filas * d_bond];
        for f in 0..m_filas {
            for alfa in 0..d_bond {
                a[f * d_bond + alfa] = u[f * k + alfa];
            }
        }
        tensores.push(a);
        dims.push(d_bond);
        // resto ← σ·Vᵀ = Uᵀ·M (el σ se cancela: Vᵀ = σ⁻¹UᵀM ⇒ σVᵀ = UᵀM).
        // Con A[l] = U ortonormal y resto = Uᵀ·M se preserva ||M||² = ⟨ψ|ψ⟩.
        let mut nuevo = vec![0.0; d_bond * m_cols];
        for alfa in 0..d_bond {
            for kc in 0..m_cols {
                let mut s = 0.0;
                for f in 0..m_filas {
                    s += u[f * k + alfa] * m[f * m_cols + kc];
                }
                nuevo[alfa * m_cols + kc] = s;
            }
        }
        resto = nuevo;
        dl = d_bond;
        resto_cols = m_cols;
    }

    Ok((Mps { tensores, dims, largo: l }, error_total)
    )
}

/// Entropía de entrelazamiento de von Neumann S = −Σ p·ln p del corte:
/// rho = estado.reshape(2^corte, 2^{L-corte}); p = autovalores de
/// rho·rhoᵀ normalizados por la traza (espectro de Schmidt al cuadrado).
/// Solución conocida: producto ⇒ 0; GHZ ⇒ ln 2; W ⇒ < ln 2.
pub fn entrelazamiento_corte(estado: &[f64], l: usize, corte: usize) -> Result<f64, MpsError> {
    if l == 0 || l > MPS_MAX_L || corte == 0 || corte >= l {
        return Err(MpsError::LongitudInvalida { l });
    }
    let esperado = 1usize << l;
    if estado.len() != esperado {
        return Err(MpsError::EstadoLongitud { esperado, recibido: estado.len() });
    }
    if estado.iter().any(|v| !v.is_finite()) {
        return Err(MpsError::EntradaNoFinita);
    }
    let filas = 1usize << corte;
    let cols = esperado >> corte;
    let mut g = vec![0.0; filas * filas];
    for i in 0..filas {
        for j in 0..filas {
            let mut s = 0.0;
            for k in 0..cols {
                s += estado[i * cols + k] * estado[j * cols + k];
            }
            g[i * filas + j] = s;
        }
    }
    let espectro = jacobi_simetrico(&g, filas, 1e-24)
        .ok_or(MpsError::NormaDegenerada { valor: f64::NAN })?;
    let traza: f64 = espectro.iter().sum();
    if !(traza.is_finite() && traza > 1e-300) {
        return Err(MpsError::NormaDegenerada { valor: traza });
    }
    let mut entropia = 0.0f64;
    for lam in espectro {
        let p = lam / traza;
        if p > 1e-15 {
            entropia -= p * p.ln();
        }
    }
    if !entropia.is_finite() {
        return Err(MpsError::NormaDegenerada { valor: entropia });
    }
    Ok(entropia)
}

// ══════════════════════════════════════════════════════════════════════
// GLM 152→154 — DMRG DE 2 SITIOS (titularidad §56): el optimizador
// variacional sobre el núcleo MPS del 151. Hamiltoniano de certificación:
// Ising transverso H = −J·Σ σᶻσᶻ − h·Σ σˣ (vecino más cercano), MPO bond 3.
// Estado inicial |↑…↑⟩ (D=1); el paso de PARES hace crecer los enlaces
// 1→2→4→…→bond_max — el 1-sitio tenía D' ≤ min(dl·d, dr) ≤ dr y jamás
// crecía desde el producto (los contratos D>1 eran inalcanzables).
// Contratos: h=0 y J=0 exactos con D=1; punto crítico vs diagonalización
// exhaustiva; monotonía variacional E(D=1) ≥ E(D=2) ≥ E(D=4).
// ══════════════════════════════════════════════════════════════════════

/// MPO del Ising transverso: W[l][w_up, s, s', w_down] con w ∈ {0,1,2}:
/// canal 0 = identidad en tránsito, 1 = ZZ pendiente, 2 = término hecho.
/// Matriz de operadores 3×3 (fila = w_up, col = w_down), la MISMA en todo
/// sitio: (0→0)=I, (0→1)=σᶻ abre, (0→2)=−hσˣ campo, (1→2)=−Jσᶻ cierra,
/// (2→2)=I. (1→1)=0 — el ZZ es estrictamente vecino-más-cercano.
pub struct MpoIsing {
    pub largo: usize,
    pub j: f64,
    pub h: f64,
}

impl MpoIsing {
    /// Tensor W del sitio l (0-indexado): layout
    /// W[(w_up·d + s)·d·w_dn + s'·w_dn_idx] ≡ flat
    /// [((w_up·2 + s)·2 + s')·3 + w_dn].
    fn tensor(&self, _l: usize) -> Vec<f64> {
        let dw = 3usize;
        let mut w = vec![0.0; dw * MPS_D_FIS * MPS_D_FIS * dw];
        let idx = |wu: usize, s: usize, sp: usize, wd: usize| {
            ((wu * MPS_D_FIS + s) * MPS_D_FIS + sp) * dw + wd
        };
        // Local (s,s'): I=δ, Z=diag(1,−1), X=antidiag(1,1).
        for s in 0..MPS_D_FIS {
            for sp in 0..MPS_D_FIS {
                let z = if s == sp { if s == 0 { 1.0 } else { -1.0 } } else { 0.0 };
                let x = if s != sp { 1.0 } else { 0.0 };
                // (0→0)=I transporta; (0→2)=−hσˣ añade el campo UNA vez por
                // sitio (fix 153: en la fila 2 se acumulaba multiplicativa-
                // mente como ∏(I−hσˣ)).
                w[idx(0, s, sp, 0)] += if s == sp { 1.0 } else { 0.0 };
                w[idx(0, s, sp, 2)] += -self.h * x;
                // (0→1)=σᶻ ABRE el ZZ en cualquier sitio, SIN coeficiente:
                // el −J vive en el cierre. (1→1)=0 — transportar σᶻ por el
                // canal pendiente generaba cadenas de muchos cuerpos
                // σᶻ₀⋯σᶻ_k con coeficiente J² (GLM 154). Los tests de
                // límite con J=1 eran ciegos a esa deformación: E_min del
                // operador equivocado = −J²(L−1) = −J(L−1) cuando J=1. El
                // punto crítico contra diagonalización exacta sí la caza.
                w[idx(0, s, sp, 1)] += z;
                // (1→2)=−Jσᶻ CIERRA el par con el coeficiente del término.
                w[idx(1, s, sp, 2)] += -self.j * z;
                // (2→2)=I mantiene el canal completado.
                w[idx(2, s, sp, 2)] += if s == sp { 1.0 } else { 0.0 };
            }
        }
        w
    }
}

/// Resultado de un DMRG.
pub struct DmrgResultado {
    pub energia: f64,
    pub mps: Mps,
    pub sweeps: usize,
}

/// DMRG de 2 SITIOS (GLM 154) con barrido completo de pares por sweep.
/// Estado inicial |↑…↑⟩ (D=1 en todos los enlaces); en cada par (s, s+1)
/// se contrae L·W_s·W_{s+1}·R al operador efectivo (dl·d·d·dr)², se toma
/// el autovector del MÍNIMO autovalor y se re-factoriza por SVD: AMBOS
/// tensores del par se re-emiten de la misma factorización (dims[s+1] ==
/// alto real de A[s+1] por construcción — el defecto estructural del
/// 1-sitio del 153 desaparece) y el índice físico extra del par hace
/// crecer el enlace hacia bond_max. Devuelve la energía variacional
/// (≥ E₀ exacto siempre).
pub fn dmrg_ising(
    l: usize,
    j: f64,
    h: f64,
    bond_max: usize,
    sweeps: usize,
) -> Result<DmrgResultado, MpsError> {
    if l == 0 || l > MPS_MAX_L {
        return Err(MpsError::LongitudInvalida { l });
    }
    if !(j.is_finite() && h.is_finite()) || bond_max == 0 || bond_max > MPS_MAX_D {
        return Err(MpsError::BondInvalido { d: bond_max });
    }
    // L=1: sin pares que optimizar — diagonalización exacta del sitio único
    // H = −h·σˣ con las primitivas del módulo (API total, sin atajo físico).
    if l == 1 {
        let mut h1 = vec![0.0f64; 4];
        h1[0 * 2 + 1] = -h;
        h1[1 * 2 + 0] = -h;
        let espectro = jacobi_simetrico(&h1, 2, 1e-18)
            .ok_or(MpsError::NormaDegenerada { valor: f64::NAN })?;
        let energia = espectro.iter().cloned().fold(f64::INFINITY, f64::min);
        let neg: Vec<f64> = h1.iter().map(|x| -x).collect();
        let (v, _) = autovectores_jacobi(&neg, 2, 1)
            .ok_or(MpsError::NormaDegenerada { valor: f64::NAN })?;
        let mps = Mps { tensores: vec![vec![v[0], v[1]]], dims: vec![1, 1], largo: 1 };
        return Ok(DmrgResultado { energia, mps, sweeps: sweeps.max(2) });
    }
    let hpo = MpoIsing { largo: l, j, h };
    let mpos: Vec<Vec<f64>> = (0..l).map(|i| hpo.tensor(i)).collect();

    // Estado inicial: |↑…↑⟩ (s=0) con D=1.
    let mut tensores: Vec<Vec<f64>> = Vec::with_capacity(l);
    for _sitio in 0..l {
        // A[l][(0·2+s)·1+0] = δ_{s,0}
        tensores.push(vec![1.0, 0.0]);
    }
    let dims: Vec<usize> = vec![1; l + 1];
    let mut mps = Mps { tensores, dims, largo: l };

    let dw = 3usize;
    let mut energia = 0.0f64;
    for _ in 0..sweeps.max(2) {
        // ── Barrido izquierda→derecha: pares (0,1), (1,2), …, (L−2,L−1).
        // La norma viaja a la derecha; A[par] queda canónico izquierdo.
        for par in 0..l - 1 {
            let (e, a_izq, a_der, dims_nuevas) =
                optimizar_par(&mps, &mpos, par, bond_max, dw, false)?;
            energia = e;
            mps.tensores[par] = a_izq;
            mps.tensores[par + 1] = a_der;
            mps.dims = dims_nuevas;
        }
        // ── Barrido derecha→izquierda: pares (L−2,L−1), …, (1,2), (0,1).
        // La norma viaja a la izquierda; A[par+1] queda canónico derecho.
        for par in (0..l - 1).rev() {
            let (e, a_izq, a_der, dims_nuevas) =
                optimizar_par(&mps, &mpos, par, bond_max, dw, true)?;
            energia = e;
            mps.tensores[par] = a_izq;
            mps.tensores[par + 1] = a_der;
            mps.dims = dims_nuevas;
        }
    }
    if !energia.is_finite() {
        return Err(MpsError::NormaDegenerada { valor: energia });
    }
    Ok(DmrgResultado { energia, mps, sweeps: sweeps.max(2) })
}

/// Optimiza el par (`par`, `par`+1): contrae L (sitios < par), W_par·W_{par+1}
/// precontraído sobre el canal MPO intermedio y R (sitios > par+1) al
/// operador efectivo Heff[(a,σ,σ₁,b),(a',σ',σ₁',b')] y resuelve el mínimo
/// autovalor. El autovector theta (dl·d, d·dr) se factoriza por SVD y AMBOS
/// tensores del par se re-emiten de esa ÚNICA factorización — la "matriz
/// pendiente" del 1-sitio deja de ser un estado aparte que el vecino debe
/// absorber: σVᵀ (barrido →) o U·σ (barrido ←) ES directamente el tensor
/// vecino, con dims[par+1] = D' por construcción.
///
/// Barrido → : A[par] = U (columnas ortonormales, canónico izq); la norma
///             viaja con σVᵀ = Uᵀ·theta hacia A[par+1] (el σ se cancela,
///             como en `mps_desde_estado`).
/// Barrido ← : A[par+1] = Vᵀ (filas ortonormales, canónico der); la norma
///             viaja con U·σ = theta·V hacia A[par].
/// El factor que carga la norma se re-normaliza por √(Σλ_kept) para que
/// ‖ψ‖ = 1 sobreviva al truncado en cada paso (sin esto la norma deriva
/// ~1e-3 en 12 sweeps y el contrato de autoconsistencia revienta).
fn optimizar_par(
    mps: &Mps,
    mpos: &[Vec<f64>],
    par: usize,
    bond_max: usize,
    dw: usize,
    hacia_izquierda: bool,
) -> Result<(f64, Vec<f64>, Vec<f64>, Vec<usize>), MpsError> {
    let l = mps.largo;
    let dl = mps.dims[par];
    let dr = mps.dims[par + 2];
    // Entorno izquierdo: L[wu, a, ap] — contraer sitios 0..par-1.
    let mut env_l: Vec<f64> = vec![1.0];
    let mut dim_l_mps = 1usize;
    let mut dim_l_mpo = 1usize;
    for s in 0..par {
        let a = &mps.tensores[s];
        let w = &mpos[s];
        env_l = contraer_entorno_izq(&env_l, dim_l_mps, dim_l_mpo, w, a, dw);
        dim_l_mps = mps.dims[s + 1];
        dim_l_mpo = dw;
    }
    // Entorno derecho: R[wd, b, bp] — contraer sitios par+2..l-1 (inverso).
    // Seed en delta del canal COMPLETADO (wd=2) — fix del 153 vigente: el
    // MPO abre en wu=0 (izquierda) y cierra en wd=2 (derecha).
    let mut env_r: Vec<f64> = vec![0.0; dw];
    env_r[(2 * 1 + 0) * 1 + 0] = 1.0; // índice (wd=2, b=0, bp=0)
    let mut dim_r_mps = 1usize;
    let mut dim_r_mpo = dw;
    for s in ((par + 2)..l).rev() {
        let a = &mps.tensores[s];
        let w = &mpos[s];
        let dap = mps.dims[s]; // D_s (lado izquierdo del sitio s)
        env_r = contraer_entorno_der(&env_r, dim_r_mps, dim_r_mpo, w, a, dw);
        dim_r_mps = dap;
        dim_r_mpo = dw;
    }
    // W2 = W_par·W_{par+1} precontraído sobre el canal MPO intermedio:
    // W2[(wu,σ,σ₁),(σ',σ₁',wd)] = Σ_{wm} W_a[(wu,σ,σ'),wm]·W_b[(wm,σ₁,σ₁'),wd]
    // (s = índice físico del sitio par; σ₁ = del sitio par+1; sin prima =
    // lado bra, con prima = lado ket — el mismo convenio de los entornos).
    // Layout: w2[((wu·d+σ)·d+σ₁)·(d·d·dw) + (σ'·d+σ₁')·dw + wd].
    let w_a = &mpos[par];
    let w_b = &mpos[par + 1];
    let n_cols = MPS_D_FIS * MPS_D_FIS * dw; // (σ', σ₁', wd)
    let mut w2 = vec![0.0; dw * MPS_D_FIS * MPS_D_FIS * n_cols];
    for wu in 0..dw {
        for fs in 0..MPS_D_FIS {
            for fs1 in 0..MPS_D_FIS {
                for fsp in 0..MPS_D_FIS {
                    for fsp1 in 0..MPS_D_FIS {
                        for wd in 0..dw {
                            // Σ_{wm} W_a[(wu, σ_BRA, σ'_KET), wm]·W_b[(wm, σ₁_BRA, σ₁'_KET), wd].
                            // OJO al cableado bra/ket (bug GLM 154): el KET del
                            // sitio par es fsp (loop interno), NO fs1 — fs1 es
                            // el BRA del sitio par+1. Cruzarlos arme un Heff no
                            // Hermitico con energias espurias (offset exacto
                            // −1.0 vs el estado denso).
                            let fila = (wu * MPS_D_FIS + fs) * MPS_D_FIS + fs1;
                            let col = (fsp * MPS_D_FIS + fsp1) * dw + wd;
                            let mut acc = 0.0;
                            for wm in 0..dw {
                                let wa = w_a
                                    [((wu * MPS_D_FIS + fs) * MPS_D_FIS + fsp) * dw + wm];
                                if wa == 0.0 {
                                    continue;
                                }
                                let wb = w_b
                                    [((wm * MPS_D_FIS + fs1) * MPS_D_FIS + fsp1) * dw + wd];
                                if wb == 0.0 {
                                    continue;
                                }
                                acc += wa * wb;
                            }
                            w2[fila * n_cols + col] += acc;
                        }
                    }
                }
            }
        }
    }
    // Heff[(a,σ,σ₁,b),(a',σ',σ₁',b')] = Σ_{wu,wd} L[wu,a,a']·W2·R[wd,b,b'].
    // n = dl·2·2·dr ≤ 8·4·8 = 256 (guard heredado del 1-sitio).
    let n = dl * MPS_D_FIS * MPS_D_FIS * dr;
    if n == 0 || n > 256 {
        return Err(MpsError::BondInvalido { d: n });
    }
    let mut heff = vec![0.0; n * n];
    for ia in 0..dl {
        for ipa in 0..dl {
            for fs in 0..MPS_D_FIS {
                for fs1 in 0..MPS_D_FIS {
                    for fsp in 0..MPS_D_FIS {
                        for fsp1 in 0..MPS_D_FIS {
                            // Bloque de índices físicos de W2 (fijo por las
                            // 6 iteraciones exteriores).
                            let fbase = (fsp * MPS_D_FIS + fsp1) * dw;
                            for ib in 0..dr {
                                for ipb in 0..dr {
                                    let fila =
                                        ((ia * MPS_D_FIS + fs) * MPS_D_FIS + fs1) * dr + ib;
                                    let col = ((ipa * MPS_D_FIS + fsp) * MPS_D_FIS + fsp1)
                                        * dr
                                        + ipb;
                                    let mut acc = 0.0;
                                    for wu in 0..dim_l_mpo {
                                        let lv = env_l[(wu * dl + ia) * dl + ipa];
                                        if lv == 0.0 {
                                            continue;
                                        }
                                        let wbase = ((wu * MPS_D_FIS + fs) * MPS_D_FIS + fs1)
                                            * n_cols
                                            + fbase;
                                        for wd in 0..dim_r_mpo {
                                            let rv = env_r[(wd * dr + ib) * dr + ipb];
                                            if rv == 0.0 {
                                                continue;
                                            }
                                            acc += lv * w2[wbase + wd] * rv;
                                        }
                                    }
                                    heff[fila * n + col] = acc;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Mínimo autovalor por Jacobi (Heff simétrico por H†=H) + autovector
    // del mínimo vía −Heff (autovectores_jacobi con k=1 da el MÁXIMO).
    let espectro = jacobi_simetrico(&heff, n, 1e-18)
        .ok_or(MpsError::NormaDegenerada { valor: f64::NAN })?;
    let min_e = espectro.iter().cloned().fold(f64::INFINITY, f64::min);
    let neg: Vec<f64> = heff.iter().map(|x| -x).collect();
    let (v, _) = autovectores_jacobi(&neg, n, 1)
        .ok_or(MpsError::NormaDegenerada { valor: f64::NAN })?;
    // theta como matriz (dl·d, d·dr): fila = (a,σ) del sitio par,
    // columna = (σ₁,b) del sitio par+1.
    let filas = dl * MPS_D_FIS;
    let cols = MPS_D_FIS * dr;
    let mut theta = vec![0.0f64; filas * cols];
    for a in 0..dl {
        for fs in 0..MPS_D_FIS {
            for fs1 in 0..MPS_D_FIS {
                for b in 0..dr {
                    theta[(a * MPS_D_FIS + fs) * cols + (fs1 * dr + b)] = v
                        [((a * MPS_D_FIS + fs) * MPS_D_FIS + fs1) * dr + b];
                }
            }
        }
    }
    let mut nuevos_dims = mps.dims.clone();
    if !hacia_izquierda {
        // Barrido →: A[par] = U (dl·d, D'); σVᵀ = Uᵀ·theta normalizado ES
        // el nuevo A[par+1] (D', d, dr).
        let (u, lambdas) = svp_truncado(&theta, filas, cols, bond_max)
            .ok_or(MpsError::NormaDegenerada { valor: f64::NAN })?;
        let d_new = u.len() / filas;
        // Σλ_kept = norma² del estado tras el truncado (‖theta‖ = 1 antes).
        let masa: f64 = lambdas.iter().take(d_new).sum();
        if !(masa > 0.0 && masa.is_finite()) {
            return Err(MpsError::NormaDegenerada { valor: masa });
        }
        let esc = 1.0 / masa.sqrt();
        let mut a_der = vec![0.0; d_new * cols];
        for c in 0..d_new {
            for k in 0..cols {
                let mut acc = 0.0;
                for f in 0..filas {
                    acc += u[f * d_new + c] * theta[f * cols + k];
                }
                a_der[c * cols + k] = acc * esc;
            }
        }
        nuevos_dims[par + 1] = d_new;
        Ok((min_e, u, a_der, nuevos_dims))
    } else {
        // Barrido ←: V = autovectores de thetaᵀ·theta (= σ²) vía el mismo
        // svp_truncado sobre la traspuesta; A[par+1] = Vᵀ (D'', d, dr)
        // canónico derecho; U·σ = theta·V normalizado ES el nuevo A[par].
        let mut theta_t = vec![0.0f64; filas * cols];
        for f in 0..filas {
            for k in 0..cols {
                theta_t[k * filas + f] = theta[f * cols + k];
            }
        }
        let (vt, mus) = svp_truncado(&theta_t, cols, filas, bond_max)
            .ok_or(MpsError::NormaDegenerada { valor: f64::NAN })?;
        let d_new = vt.len() / cols;
        let masa: f64 = mus.iter().take(d_new).sum();
        if !(masa > 0.0 && masa.is_finite()) {
            return Err(MpsError::NormaDegenerada { valor: masa });
        }
        let esc = 1.0 / masa.sqrt();
        // A[par] = U·σ = theta·V (dl·d, D'').
        let mut a_izq = vec![0.0; filas * d_new];
        for f in 0..filas {
            for c in 0..d_new {
                let mut acc = 0.0;
                for k in 0..cols {
                    acc += theta[f * cols + k] * vt[k * d_new + c];
                }
                a_izq[f * d_new + c] = acc * esc;
            }
        }
        // A[par+1] = Vᵀ: [c,(σ₁,b)] = V[(σ₁,b),c] — filas ortonormales.
        let mut a_der = vec![0.0; d_new * cols];
        for c in 0..d_new {
            for k in 0..cols {
                a_der[c * cols + k] = vt[k * d_new + c];
            }
        }
        nuevos_dims[par + 1] = d_new;
        Ok((min_e, a_izq, a_der, nuevos_dims))
    }
}

fn contraer_entorno_izq(
    env: &[f64],
    dim_mps: usize,
    dim_mpo: usize,
    w: &[f64],
    a: &[f64],
    dw: usize,
) -> Vec<f64> {
    let dap = a.len() / (dim_mps * MPS_D_FIS);
    let mut nueva2 = vec![0.0; dw * dap * dap];
    for wu in 0..dim_mpo {
        for aix in 0..dim_mps {
            for aip in 0..dim_mps {
                let e = env[(wu * dim_mps + aix) * dim_mps + aip];
                if e == 0.0 {
                    continue;
                }
                for fs in 0..MPS_D_FIS {
                    for fsp in 0..MPS_D_FIS {
                        for wup in 0..dw {
                            let wv = w[((wu * MPS_D_FIS + fs) * MPS_D_FIS + fsp) * dw + wup];
                            if wv == 0.0 {
                                continue;
                            }
                            for aj in 0..dap {
                                let av = a[((aix * MPS_D_FIS + fs) * dap) + aj];
                                if av == 0.0 {
                                    continue;
                                }
                                for apj in 0..dap {
                                    let apv = a[((aip * MPS_D_FIS + fsp) * dap) + apj];
                                    nueva2[((wup * dap + aj) * dap) + apj] += e * wv * av * apv;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    nueva2
}

fn contraer_entorno_der(
    env: &[f64],
    dim_mps: usize,
    dim_mpo: usize,
    w: &[f64],
    a: &[f64],
    dw: usize,
) -> Vec<f64> {
    // a tiene forma (D_s·d, D_{s+1}) con D_{s+1} = dim_mps.
    let dl = a.len() / (dim_mps * MPS_D_FIS);
    let mut nueva = vec![0.0; dw * dl * dl];
    for wd in 0..dim_mpo {
        for bix in 0..dim_mps {
            for bip in 0..dim_mps {
                let e = env[(wd * dim_mps + bix) * dim_mps + bip];
                if e == 0.0 {
                    continue;
                }
                for fs in 0..MPS_D_FIS {
                    for fsp in 0..MPS_D_FIS {
                        for wdn in 0..dw {
                            let wv = w[((wdn * MPS_D_FIS + fs) * MPS_D_FIS + fsp) * dw + wd];
                            if wv == 0.0 {
                                continue;
                            }
                            for ai in 0..dl {
                                let av = a[((ai * MPS_D_FIS + fs) * dim_mps) + bix];
                                if av == 0.0 {
                                    continue;
                                }
                                for api in 0..dl {
                                    let apv = a[((api * MPS_D_FIS + fsp) * dim_mps) + bip];
                                    nueva[((wdn * dl + ai) * dl) + api] += e * wv * av * apv;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    nueva
}

/// SVD-truncado directo de una matriz M (filas × cols) a bond dim ≤ b:
/// devuelve (U: filas×d_bond, λ: d_bond) con M ≈ U·λ·(normalizado).
fn svp_truncado(m: &[f64], filas: usize, cols: usize, b: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    if m.len() != filas * cols {
        return None;
    }
    let mut g = vec![0.0; filas * filas];
    for i in 0..filas {
        for j in 0..filas {
            let mut s = 0.0;
            for k in 0..cols {
                s += m[i * cols + k] * m[j * cols + k];
            }
            g[i * filas + j] = s;
        }
    }
    let (u, lambdas) = autovectores_jacobi(&g, filas, filas.min(b))?;
    let k = lambdas.len();
    let lam_max = lambdas.iter().cloned().fold(0.0f64, f64::max);
    let d_bond = lambdas.iter().filter(|&&x| x > lam_max * 1e-24).count().max(1).min(b);
    let mut out_u = vec![0.0; filas * d_bond];
    for f in 0..filas {
        for alfa in 0..d_bond {
            out_u[f * d_bond + alfa] = u[f * k + alfa];
        }
    }
    Some((out_u, lambdas))
}
