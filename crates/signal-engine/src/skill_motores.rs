//! #626 (Ola 47, Qoder) — HABILIDAD PREQUENTIAL POR MOTOR×ESCALA.
//!
//! ## Por qué
//!
//! #623 compone los 11 motores espectrales con pesos IGUALES y #624 puso
//! ese consenso al timón: un motor sistemáticamente equivocado a su escala
//! dominante vota lo mismo que uno sistemáticamente acertado. Es la clase
//! exacta de defecto que D-752 cerró para las ramas del host (cota de
//! Wilson de su propio historial) — ausente en la vía espectral.
//!
//! ## Contrato
//!
//! - Cada maduración de bloque de la escala k puntúa el voto que cada
//!   motor emitió AL ARMARSE ese bloque (snapshot tomado en la maduración
//!   anterior) contra el retorno realizado: IC prequential
//!   E[v·r]/√(E[v²]·E[r²]) con olvido 1/64 y madurez 30 — la MISMA
//!   maquinaria de #594, elevada de escala→motor×escala.
//! - Peso del motor m a la escala k: `PISO + (1−PISO)·ic` si el IC es
//!   maduro Y significativo (umbral #599: 2/√(n−3)); piso de exploración
//!   en el resto. IC negativo ⇒ piso (se castiga, no se vota al revés).
//! - CAUSALIDAD: el voto puntado es el de ARMADO, jamás el del cierre —
//!   un voto cercano al cierre conocería mecánicamente el retorno del
//!   bloque (look-ahead que inflaría el IC).
//! - Arranque en frío: todos al piso ⇒ composición matemáticamente
//!   equivalente a los pesos iguales de #623 (misma decisión, salvo
//!   redondeo FP) — la ponderación SÓLO entra con evidencia madura.

use crate::voto_espectral::{VotoEspectral, ESCALAS_VOTO};
use quantum_arena::temporal_spectrum::{umbral_ic_significativo, MUESTRAS_SKILL_MADURAS};

/// Los 13 motores con `voto_espectral()` de la composición del consenso
/// (11 originales de #623 + trend-runner y RenyiTsallis, AGY P29).
pub const MOTORES: usize = 13;
pub const PISO_EXPLORACION: f64 = 0.15;
const OLVIDO: f64 = 1.0 / 64.0;
const EPS_VOTO: f64 = 1e-9;

/// Acumuladores EWMA del IC (forma exacta de #594).
#[derive(Clone, Copy)]
struct AcumIc {
    ws: f64,
    wr: f64,
    wsr: f64,
    n: u64,
}

impl AcumIc {
    const fn nuevo() -> Self {
        Self { ws: 0.0, wr: 0.0, wsr: 0.0, n: 0 }
    }

    fn observar(&mut self, v: f64, r: f64) {
        // Abstención (v≈0) o dato no finito: sin información de habilidad.
        if !v.is_finite() || !r.is_finite() || v.abs() < EPS_VOTO {
            return;
        }
        let (ws, wr, wsr) = (v * v, r * r, v * r);
        self.ws += (ws - self.ws) * OLVIDO;
        self.wr += (wr - self.wr) * OLVIDO;
        self.wsr += (wsr - self.wsr) * OLVIDO;
        self.n = self.n.saturating_add(1);
    }

    /// IC maduro y SIGNIFICATIVO (umbral #599 contra el sesgo de
    /// selección de ruido). `None` ⇒ el llamador usa el piso.
    fn ic_significativo(&self) -> Option<f64> {
        if self.n < MUESTRAS_SKILL_MADURAS {
            return None;
        }
        let den = self.ws * self.wr;
        if !den.is_finite() || den <= 0.0 {
            return None;
        }
        let ic = self.wsr / den.sqrt();
        if !ic.is_finite() {
            return None;
        }
        let umbral = umbral_ic_significativo(self.n)?;
        if ic > umbral {
            Some(ic)
        } else {
            None
        }
    }
}

/// Habilidad prequential de los MOTORES votantes por escala, por moneda.
#[derive(Clone)]
pub struct SkillMotores {
    acum: [[AcumIc; ESCALAS_VOTO]; MOTORES],
    /// Voto de cada motor a cada escala AL ARMARSE el bloque en curso.
    voto_armado: [[f64; ESCALAS_VOTO]; MOTORES],
    /// ts del último bloque procesado por escala (dedup: el accesor del
    /// espectro devuelve SIEMPRE el último maduro, no un evento).
    ts_procesado: [u64; ESCALAS_VOTO],
}

impl Default for SkillMotores {
    fn default() -> Self {
        Self::new()
    }
}

impl SkillMotores {
    pub fn new() -> Self {
        Self {
            acum: [[AcumIc::nuevo(); ESCALAS_VOTO]; MOTORES],
            voto_armado: [[0.0; ESCALAS_VOTO]; MOTORES],
            ts_procesado: [0; ESCALAS_VOTO],
        }
    }

    /// Maduró el bloque de `escala` (cerró en `ts` con retorno `r`):
    /// puntúa los votos de ARMADO de cada motor y re-snapshot con los
    /// votos actuales (el voto del bloque NUEVO que acaba de armarse).
    /// Dedup por ts: la misma maduración observada dos veces no puntúa.
    #[inline]
    pub fn observar_maduracion(
        &mut self,
        escala: usize,
        ts: u64,
        r: f64,
        votos_actuales: &[VotoEspectral],
    ) {
        if escala >= ESCALAS_VOTO || ts <= self.ts_procesado[escala] {
            return;
        }
        self.ts_procesado[escala] = ts;
        for m in 0..MOTORES {
            self.acum[m][escala].observar(self.voto_armado[m][escala], r);
            if let Some(v) = votos_actuales.get(m) {
                self.voto_armado[m][escala] = v.en_escala(escala);
            }
        }
    }

    /// Peso del motor `m` a la escala `k`: piso + evidencia significativa.
    #[inline]
    pub fn peso(&self, m: usize, k: usize) -> f64 {
        match self.acum.get(m).and_then(|fila| fila.get(k)) {
            Some(a) => match a.ic_significativo() {
                Some(ic) => PISO_EXPLORACION + (1.0 - PISO_EXPLORACION) * ic.clamp(0.0, 1.0),
                None => PISO_EXPLORACION,
            },
            None => PISO_EXPLORACION,
        }
    }

    /// Matriz completa de pesos [motor][escala] para la composición.
    pub fn pesos(&self) -> [[f64; ESCALAS_VOTO]; MOTORES] {
        let mut w = [[PISO_EXPLORACION; ESCALAS_VOTO]; MOTORES];
        for (m, fila) in w.iter_mut().enumerate() {
            for (k, wk) in fila.iter_mut().enumerate() {
                *wk = self.peso(m, k);
            }
        }
        w
    }

    /// Diagnóstico de banda: (pares motor×escala por encima del piso,
    /// peso máximo observado). Telemetría de adopción para el forense.
    pub fn diagnostico_banda(&self) -> (usize, f64) {
        let mut sobresalientes = 0usize;
        let mut pmax = PISO_EXPLORACION;
        for fila in &self.acum {
            for a in fila {
                if let Some(ic) = a.ic_significativo() {
                    sobresalientes += 1;
                    let p = PISO_EXPLORACION + (1.0 - PISO_EXPLORACION) * ic.clamp(0.0, 1.0);
                    pmax = pmax.max(p);
                }
            }
        }
        (sobresalientes, pmax)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voto_espectral::VotoEspectral;

    fn votos_constantes(v: f64) -> Vec<VotoEspectral> {
        (0..MOTORES).map(|_| VotoEspectral::desde_escalar(v)).collect()
    }

    fn madurar(sm: &mut SkillMotores, escala: usize, ts: u64, r: f64, v: f64) {
        sm.observar_maduracion(escala, ts, r, &votos_constantes(v));
    }

    /// En frío (sin maduraciones) todos los pesos al piso y la composición
    /// por escala es matemáticamente la de pesos iguales (#623).
    #[test]
    fn qo_626_frio_equivalente_a_pesos_iguales() {
        let sm = SkillMotores::new();
        let w = sm.pesos();
        assert_eq!(w[0][0], PISO_EXPLORACION);
        assert_eq!(w[10][31], PISO_EXPLORACION);

        let votos = votos_constantes(0.6);
        let a = VotoEspectral::consenso(&votos, &[1.0; MOTORES]);
        let b = VotoEspectral::consenso_por_escala(&votos, &w);
        for k in 0..ESCALAS_VOTO {
            assert!(
                (a.en_escala(k) - b.en_escala(k)).abs() < 1e-12,
                "escala {k}: {} vs {}",
                a.en_escala(k),
                b.en_escala(k)
            );
        }
    }

    /// CAUSALIDAD: el voto puntado es el de ARMADO. Un motor que votó +1 al
    /// armarse y −1 al cierre puntúa POSITIVO contra un retorno positivo.
    #[test]
    fn qo_626_puntua_el_voto_de_armado_no_el_de_cierre() {
        let mut sm = SkillMotores::new();
        // Bloque 1 arma con voto +1 (la maduración inicial snapshot-ea).
        madurar(&mut sm, 19, 1_000, 0.0, 1.0);
        // El voto CAMBIA a −1 durante el bloque; el bloque cierra +r.
        madurar(&mut sm, 19, 2_000, 0.05, -1.0);
        // Con votos constantes por maduración, el IC refleja v_armado·r:
        // el +1 armado del bloque 2 fue puntuado con r=+0.05.
        let w = sm.pesos();
        // Aún inmaduro (n=2): piso.
        assert_eq!(w[0][19], PISO_EXPLORACION);
        // Maduramos bloques alineados: armado +1, retorno +0.05 constante.
        for i in 3..40u64 {
            madurar(&mut sm, 19, i * 1_000, 0.05, 1.0);
        }
        let w = sm.pesos();
        assert!(
            w[0][19] > PISO_EXPLORACION + 1e-6,
            "evidencia alineada debe subir el peso: {}",
            w[0][19]
        );
    }

    /// Anti-alineado: votos de armado contrarios al retorno ⇒ piso
    /// (castigo, nunca voto invertido).
    #[test]
    fn qo_626_antialineado_se_queda_en_el_piso() {
        let mut sm = SkillMotores::new();
        madurar(&mut sm, 5, 1_000, 0.0, -1.0);
        for i in 2..40u64 {
            madurar(&mut sm, 5, i * 1_000, 0.05, -1.0);
        }
        assert_eq!(sm.peso(0, 5), PISO_EXPLORACION);
        assert_eq!(sm.diagnostico_banda(), (0, PISO_EXPLORACION));
    }

    /// Dedup por ts: la misma maduración observada dos veces puntúa UNA.
    #[test]
    fn qo_626_dedup_por_ts() {
        let mut sm = SkillMotores::new();
        madurar(&mut sm, 7, 1_000, 0.0, 1.0);
        for _ in 0..5 {
            // Mismo ts: no re-puntúa ni re-snapshot-ea.
            madurar(&mut sm, 7, 2_000, 0.05, 0.9);
        }
        madurar(&mut sm, 7, 3_000, 0.05, 1.0);
        // La 1ª maduración puntúa abstención (armado 0 ⇒ no cuenta); la
        // 2ª puntúa el armado +1; las 4 repeticiones dedupeadas; la 3ª
        // puntúa el armado 0.9. n = 2, no 7.
        let n = sm.acum[0][7].n;
        assert_eq!(n, 2, "dedup por ts falló: n={n}");
    }

    /// Significancia #599: maduro pero BAJO el umbral sigue en piso; la
    /// abstención (v≈0) no infla n.
    #[test]
    fn qo_626_abstencion_no_cuenta_muestra() {
        let mut sm = SkillMotores::new();
        // Voto ~0 al armarse: sin información, n no crece.
        madurar(&mut sm, 3, 1_000, 0.0, 0.0);
        for i in 2..40u64 {
            madurar(&mut sm, 3, i * 1_000, 0.05, 0.0);
        }
        assert_eq!(sm.acum[0][3].n, 0);
        assert_eq!(sm.peso(0, 3), PISO_EXPLORACION);
    }
}
