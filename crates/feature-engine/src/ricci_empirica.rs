//! Curvatura de Ricci EMPÍRICA de una métrica 2D (mandato §3.4, GLM-193):
//! «trata las distribuciones probabilísticas del espectro como puntos en
//! una variedad... equipada con la métrica de Fisher-Rao. Calcula
//! geodésicas, curvatura escalar de Ricci [...]».
//!
//! Verificación previa (norma 186, con word-boundaries): «ricci» ya vive
//! en main en DOS objetos distintos, ambos citados como precedencia:
//! (a) `fisher_rao.rs` (GLM 191) — la CONSTANTE cerrada de la familia
//! gaussiana; (b) `strategy-core/kahler_hodge.rs` (AGY Ω139) — el escalar
//! de Ricci de una variedad de Kähler construida sobre correlaciones de
//! activos (cointegración multiactivo). Este módulo es el TERCERO y
//! complementario: el ESTIMADOR NUMÉRICO GENÉRICO de la sección escalar
//! para cualquier métrica de Fisher 2D — diferencias finitas → símbolos
//! de Christoffel → tensor de Riemann → contracción.
//!
//! **Convención** (estándar): R^l_{jkm} = ∂_k Γ^l_{mj} − ∂_m Γ^l_{kj} +
//! Γ^l_{kt}Γ^t_{mj} − Γ^l_{mt}Γ^t_{kj}; Ric_jm = R^l_{jlm}; R = g^{jm}Ric.
//! Verificada por TRIPLE forma cerrada en python antes de escribir Rust:
//! euclídea ⇒ 0 EXACTO; hiperbólica (dx²+dy²)/y² ⇒ −2; **gaussiana (μ,σ)
//! diag(1/σ², 2/σ²) ⇒ −1 — que corrige una ERRATA del doc del 191**:
//! ahí se leyó «curvatura escalar de Ricci −1/2», pero −1/2 es la
//! curvatura GAUSSIANA (seccional) K; el escalar de Ricci es R = 2K = −1
//! (el resultado clásico de la familia). El estimador reproduce las tres
//! constantes a ~2e-6 con h=1e-3 (a ~2e-8 con h=1e-4).
//!
//! **Honestidad numérica**: h ≤ 0 / métrica no finita / diagonal no
//! positiva / determinante degenerado (la inversa g^{ij} exige
//! definida-positiva) ⇒ Err — jamás un cero fabricado. Coste: ~17
//! evaluaciones de la métrica + álgebra 2×2 — trivial a cualquier
//! cadencia.

/// Paso de diferenciación por defecto (error ~2e-6 en las tres
/// constantes cerradas verificadas).
pub const H_POR_DEFECTO: f64 = 1e-3;

/// Derivadas centrales ∂_k g_ij en el punto: dg[k][i][j]. None si algún
/// valor no es finito.
fn derivadas_metrica<G>(metrica: &G, t1: f64, t2: f64, h: f64) -> Option<[[[f64; 2]; 2]; 2]>
where
    G: Fn(f64, f64) -> [[f64; 2]; 2],
{
    let mut dg = [[[0.0f64; 2]; 2]; 2];
    for k in 0..2 {
        let gp = if k == 0 {
            metrica(t1 + h, t2)
        } else {
            metrica(t1, t2 + h)
        };
        let gm = if k == 0 {
            metrica(t1 - h, t2)
        } else {
            metrica(t1, t2 - h)
        };
        for i in 0..2 {
            for j in 0..2 {
                let v = (gp[i][j] - gm[i][j]) / (2.0 * h);
                if !v.is_finite() {
                    return None;
                }
                dg[k][i][j] = v;
            }
        }
    }
    Some(dg)
}

/// Inversa 2×2 con validación de métrica riemanniana (diagonal positiva
/// en la base dada + determinante no degenerado).
fn inversa2x2(g: &[[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let (a, b) = (g[0][0], g[0][1]);
    let (c, d) = (g[1][0], g[1][1]);
    if !(a > 0.0 && d > 0.0) {
        return None;
    }
    let det = a * d - b * c;
    if !(det > 1e-30) {
        return None;
    }
    Some([[d / det, -b / det], [-c / det, a / det]])
}

/// Métrica + inversa + símbolos de Christoffel Γ^i_{jk} en un punto.
fn cristo<G>(metrica: &G, t1: f64, t2: f64, h: f64) -> Option<([[f64; 2]; 2], [[[f64; 2]; 2]; 2])>
where
    G: Fn(f64, f64) -> [[f64; 2]; 2],
{
    let dg = derivadas_metrica(metrica, t1, t2, h)?;
    let g = metrica(t1, t2);
    if !g.iter().all(|f| f.iter().all(|v| v.is_finite())) {
        return None;
    }
    let gi = inversa2x2(&g)?;
    let mut gam = [[[0.0f64; 2]; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                let mut s = 0.0;
                for l in 0..2 {
                    s += gi[i][l] * (dg[j][l][k] + dg[k][l][j] - dg[l][j][k]);
                }
                gam[i][j][k] = 0.5 * s;
            }
        }
    }
    Some((gi, gam))
}

/// Estima la sección escalar de Ricci R de una métrica riemanniana 2D
/// g_ij(θ₁,θ₂) dada como closure, en el punto (t1, t2), por diferencias
/// finitas centrales (paso `h`). Err = fall-closed (jamás cero
/// fabricado). Verificada contra tres constantes cerradas (cabecera).
pub fn curvatura_ricci_2d<G>(metrica: G, t1: f64, t2: f64, h: f64) -> Result<f64, &'static str>
where
    G: Fn(f64, f64) -> [[f64; 2]; 2],
{
    if !h.is_finite() || h <= 0.0 {
        return Err("h de diferenciacion debe ser finito y > 0");
    }
    let (gi, gam) = cristo(&metrica, t1, t2, h).ok_or("metrica degenerada o no finita")?;
    // Derivadas de Γ en ±h por eje (recomputando Christoffels).
    let gp1 = cristo(&metrica, t1 + h, t2, h).map(|(_, g)| g);
    let gm1 = cristo(&metrica, t1 - h, t2, h).map(|(_, g)| g);
    let gp2 = cristo(&metrica, t1, t2 + h, h).map(|(_, g)| g);
    let gm2 = cristo(&metrica, t1, t2 - h, h).map(|(_, g)| g);
    let [gp1, gm1, gp2, gm2] = match [gp1, gm1, gp2, gm2] {
        [Some(a), Some(b), Some(c), Some(d)] => [a, b, c, d],
        _ => return Err("entorno del punto degenerado"),
    };
    // d_gam[k][i][j][m] = ∂_k Γ^i_{jm}
    let mut d_gam = [[[[0.0f64; 2]; 2]; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for m in 0..2 {
                d_gam[0][i][j][m] = (gp1[i][j][m] - gm1[i][j][m]) / (2.0 * h);
                d_gam[1][i][j][m] = (gp2[i][j][m] - gm2[i][j][m]) / (2.0 * h);
            }
        }
    }
    // Ric_jm = R^l_{jlm} = ∂_l Γ^l_{mj} − ∂_m Γ^l_{lj} + Γ^l_{lt}Γ^t_{mj} − Γ^l_{mt}Γ^t_{lj}
    let mut escalar = 0.0f64;
    for j in 0..2 {
        for m in 0..2 {
            let mut ric_jm = 0.0f64;
            for l in 0..2 {
                // ∂_l Γ^l_{m j}: derivada en dirección l de Γ^l_{jm}
                let d1 = d_gam[l][l][j][m];
                // ∂_m Γ^l_{l j}: derivada en dirección m de Γ^l_{jl}
                let d2 = d_gam[m][l][l][j];
                let mut r = d1 - d2;
                for t in 0..2 {
                    r += gam[l][l][t] * gam[t][m][j] - gam[l][m][t] * gam[t][l][j];
                }
                ric_jm += r;
            }
            escalar += gi[j][m] * ric_jm;
        }
    }
    if escalar.is_finite() {
        Ok(escalar)
    } else {
        Err("curvatura degenerada")
    }
}
