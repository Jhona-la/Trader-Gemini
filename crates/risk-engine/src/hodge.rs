//! DESCOMPOSICIÓN DE HELMHOLTZ-HODGE SOBRE EL FLUJO DE CONTAGIO (Ola XLVI·C, T05).
//!
//! Cierra el hueco T05 del informe XLI (C4: «Hodge→Hodge discreto en grafo de
//! señales, candidato futuro»). La teoría — misma familia matemática que la
//! conjetura de Hodge (problema del milenio): toda 1-forma de borde sobre un
//! grafo conexo se parte de forma única en componente EXACTA (gradiente de un
//! potencial) + componente COEXACTA (rotacional) + armónica. Sobre grafo
//! completo la parte armónica es trivial y la descomposición es una
//! proyección en mínimos cuadrados (Jiang-Lim-Yao-Ye 2011).
//!
//! ## Contrato de transferencia (protocolo del repo)
//!
//! - **Variable**: flujo antisimétrico de contagio `f_ij = α[i][j] − α[j][i]`
//!   — la excitación cruzada NETA de Hawkes (z-scores, adimensional) que la
//!   matriz XLV ya produce.
//! - **Operador**: `curl_share = 1 − ‖∇φ‖²/‖f‖²` con `φ` resolviendo la
//!   proyección `L·φ = div(f)` (L = laplaciano del grafo completo, un nodo
//!   a tierra para fijar la constante). El gradiente captura la CASCADA
//!   JERÁRQUICA transitiva (existe un potencial de liderazgo); el residual
//!   es el contagio CÍCLICO — la cámara de eco que ninguna jerarquía
//!   explica y que los roles líder/seguidor (XLV·F) NO ven.
//! - **Unidades**: fracción de energía del flujo, adimensional ∈ [0,1].
//! - **Contorno**: N<3 → None (sin ciclos no hay rotacional); flujo
//!   simétrico (‖f‖²=0) → None; cualquier NaN → None. Sin observación no
//!   se afirma estructura (misma honestidad que la matriz XLV).
//! - **Identificabilidad**: grafo completo ⇒ φ único salvo constante (la
//!   tierra la elimina). Sin aristas faltantes no hay parte armónica.
//! - **Coste**: eliminación gaussiana O(N³) sobre (N−1)² — N = roster ≤ 16,
//!   llamado desde el publicador cada 4096 ticks, fuera del hot path.
//! - **Falsación**: (a) cascada transitiva pura `f = ∇φ` con φ=[2,1,0] ⇒
//!   curl_share ≈ 0; (b) 3-ciclo puro ⇒ ≈ 1; (c) ruido ⇒ intermedio. Los
//!   tres son tests de contrato abajo.
//!
//! ## Doctrina de consumo
//!
//! El escalar se PUBLICA como observable (`hawkes_contagion_curl_share`,
//! publicador XLV·H) y NO veta nada todavía: igual que Fisher (C3/XLI) y
//! habilidad R² (D-754), la constante de acoplamiento al veto de exposición
//! estructural exige medición en vivo antes de gobernar. Publicar sin
//! medir sería la decoración que el mandato prohíbe; medir sin publicar,
//! el punto ciego que esta ola cierra.

/// Curl de Hodge sobre el flujo antisimétrico `flow` (matriz N×N, se espera
/// `flow[i][j] = −flow[j][i]`; se tolera asimetría menor re-antisimetrizando).
///
/// None en los contornos del contrato (N<3, flujo nulo, NaN). El valor de
/// retorno es la FRACCIÓN de energía del flujo que NO es explicable por
/// ninguna jerarquía de liderazgo: 0 = cascada transitiva pura,
/// 1 = circulación pura (cámara de eco).
pub fn hodge_curl_share(flow: &[Vec<f64>]) -> Option<f64> {
    let n = flow.len();
    if n < 3 || flow.iter().any(|r| r.len() != n) {
        return None;
    }
    // Antisimetrización explícita + sanidad: un NaN envenena todo el campo.
    let mut f = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let a = flow[i][j];
            let b = flow[j][i];
            if !a.is_finite() || !b.is_finite() {
                return None;
            }
            f[i][j] = (a - b) * 0.5;
        }
    }

    // Energía total del flujo sobre aristas no dirigidas (cada par una vez).
    let mut energia = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            energia += f[i][j] * f[i][j];
        }
    }
    if energia <= 0.0 || !energia.is_finite() {
        return None; // flujo simétrico: nada dirigido que descomponer
    }

    // div(f)_i = Σ_j f_ij (flujo neto saliente del nodo i).
    let mut div = vec![0.0_f64; n];
    for (i, d) in div.iter_mut().enumerate() {
        let s: f64 = f[i].iter().sum();
        if !s.is_finite() {
            return None;
        }
        *d = s;
    }

    // Laplaciano del grafo completo (pesos unitarios): L_ii = n−1, L_ij = −1.
    // Tierra en el nodo n−1: resolver el sistema reducido (n−1)×(n−1).
    let m = n - 1;
    let mut a_mat = vec![vec![0.0_f64; m]; m];
    for i in 0..m {
        for j in 0..m {
            a_mat[i][j] = if i == j { m as f64 } else { -1.0 };
        }
    }
    // Resuelve L·φ = div por eliminación gaussiana con pivoteo parcial.
    let mut rhs = div[..m].to_vec();
    for col in 0..m {
        // Pivoteo parcial: estabilidad numérica sin bibliotecas externas.
        let mut best = col;
        for r in (col + 1)..m {
            if a_mat[r][col].abs() > a_mat[best][col].abs() {
                best = r;
            }
        }
        if a_mat[best][col].abs() < 1e-12 {
            return None; // singular más allá de la tierra: campo degenerado
        }
        a_mat.swap(col, best);
        rhs.swap(col, best);
        let piv = a_mat[col][col];
        for r in (col + 1)..m {
            let factor = a_mat[r][col] / piv;
            if factor == 0.0 {
                continue;
            }
            for c in col..m {
                a_mat[r][c] -= factor * a_mat[col][c];
            }
            rhs[r] -= factor * rhs[col];
        }
    }
    let mut phi = vec![0.0_f64; m];
    for row in (0..m).rev() {
        let mut acc = rhs[row];
        for c in (row + 1)..m {
            acc -= a_mat[row][c] * phi[c];
        }
        phi[row] = acc / a_mat[row][row];
        if !phi[row].is_finite() {
            return None;
        }
    }
    let mut phi_full = phi;
    phi_full.push(0.0); // nodo a tierra

    // Energía del gradiente: Σ_{i<j} (φ_i − φ_j)².
    let mut energia_grad = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = phi_full[i] - phi_full[j];
            energia_grad += d * d;
        }
    }
    // Proyección en norma de frobenius sobre pares: la fracción explicada
    // puede exceder 1 por redondeo en el borde — clamp con holgura física.
    let curl = 1.0 - energia_grad / energia;
    Some(curl.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Falsación (a): cascada transitiva pura — el flujo ES el gradiente de
    /// un potencial de liderazgo φ=[2,1,0]. Ninguna circulación: curl ≈ 0.
    #[test]
    fn cascada_transitiva_es_gradiente_puro() {
        let phi = [2.0_f64, 1.0, 0.0];
        let mut f = vec![vec![0.0; 3]; 3];
        for (i, &pi) in phi.iter().enumerate() {
            for (j, &pj) in phi.iter().enumerate() {
                f[i][j] = pi - pj;
            }
        }
        let curl = hodge_curl_share(&f).expect("campo válido");
        assert!(curl < 1e-9, "cascada transitiva con curl={curl} (≠0)");
    }

    /// Falsación (b): 3-ciclo puro A→B→C→A. Ningún potencial lo explica:
    /// curl ≈ 1. Es la cámara de eco — riesgo conjunto que la jerarquía
    /// líder/seguidor (roles XLV) no representa.
    #[test]
    fn ciclo_puro_es_rotacional_puro() {
        // f_AB=1, f_BC=1, f_CA=1 (f_AC=−1)
        let mut f = vec![vec![0.0; 3]; 3];
        f[0][1] = 1.0;
        f[1][0] = -1.0;
        f[1][2] = 1.0;
        f[2][1] = -1.0;
        f[2][0] = 1.0;
        f[0][2] = -1.0;
        let curl = hodge_curl_share(&f).expect("campo válido");
        assert!(curl > 0.99, "ciclo puro con curl={curl} (≠1)");
    }

    /// Contornos: N<3, flujo simétrico y NaN devuelven None — sin
    /// observación no se afirma estructura.
    #[test]
    fn contornos_sin_observacion_son_none() {
        let dos = vec![vec![0.0, 1.0], vec![-1.0, 0.0]];
        assert_eq!(hodge_curl_share(&dos), None, "N=2 no tiene ciclos");
        let sim = vec![vec![0.0, 0.5, 0.2], vec![0.5, 0.0, 0.4], vec![0.2, 0.4, 0.0]];
        assert_eq!(hodge_curl_share(&sim), None, "flujo simétrico: nada dirigido");
        let nan = vec![
            vec![0.0, 1.0, f64::NAN],
            vec![-1.0, 0.0, 1.0],
            vec![-1.0, -1.0, 0.0],
        ];
        assert_eq!(hodge_curl_share(&nan), None, "NaN envenena el campo");
    }

    /// Cascada + ciclo superpuestos: el curl queda estrictamente entre los
    /// extremos — la medida separa grados de cámara de eco, no es binaria.
    #[test]
    fn mezcla_cascada_ciclo_es_intermedia_y_monotona() {
        let phi = [2.0_f64, 1.0, 0.0];
        let mut grad = vec![vec![0.0; 3]; 3];
        for (i, &pi) in phi.iter().enumerate() {
            for (j, &pj) in phi.iter().enumerate() {
                grad[i][j] = pi - pj;
            }
        }
        let mut ciclo = vec![vec![0.0; 3]; 3];
        ciclo[0][1] = 1.0;
        ciclo[1][0] = -1.0;
        ciclo[1][2] = 1.0;
        ciclo[2][1] = -1.0;
        ciclo[2][0] = 1.0;
        ciclo[0][2] = -1.0;

        let mut prev = -1.0_f64;
        for w in [0.0_f64, 0.25, 0.5, 1.0, 2.0, 8.0] {
            let mix: Vec<Vec<f64>> = (0..3)
                .map(|i| (0..3).map(|j| grad[i][j] + w * ciclo[i][j]).collect())
                .collect();
            let curl = hodge_curl_share(&mix).expect("mezcla válida");
            assert!(curl > prev, "monotonía en el peso del ciclo: {curl} ≤ {prev}");
            prev = curl;
        }
        assert!(
            prev > 0.5 && prev < 1.0,
            "con gradiente dominante el ciclo no satura: {prev}"
        );
    }

    /// Determinismo trivial (sin azar) + asimetría tolerada: el llamador que
    /// pasa la matriz α completa (no antisimetrizada) obtiene el mismo
    /// veredicto que pasando la parte antisimétrica.
    #[test]
    fn antisimetrizacion_del_llamador_no_cambia_el_veredicto() {
        let mut f = vec![vec![0.0; 3]; 3];
        f[0][1] = 1.0;
        f[1][0] = -1.0;
        f[1][2] = 1.0;
        f[2][1] = -1.0;
        f[2][0] = 1.0;
        f[0][2] = -1.0;
        let directo = hodge_curl_share(&f).unwrap();
        // Simétrico superpuesto (se cancela en la antisimetrización):
        let mut con_sim = f.clone();
        con_sim[0][1] += 0.7;
        con_sim[1][0] += 0.7; // parte simétrica: no debe mover el curl
        let indirecto = hodge_curl_share(&con_sim).unwrap();
        assert!(
            (directo - indirecto).abs() < 1e-9,
            "parte simétrica cambió el curl: {directo} vs {indirecto}"
        );
    }
}
