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
use quantum_arena::evalues::EProceso;
use quantum_arena::temporal_spectrum::MUESTRAS_SKILL_MADURAS;

/// Los 13 motores con `voto_espectral()` de la composición del consenso
/// (11 originales de #623 + trend-runner y RenyiTsallis, AGY P29).
pub const MOTORES: usize = 13;
pub const PISO_EXPLORACION: f64 = 0.15;
const OLVIDO: f64 = 1.0 / 64.0;
const EPS_VOTO: f64 = 1e-9;
/// Ola 48 / H5 — tamaño efectivo del estimador EWMA (λ=1/64 ⇒ N_ef ≈
/// 2/λ = 128): el umbral de significancia se ancla AQUÍ, no al conteo
/// crudo n. Con n crudo el umbral decae a 0 en sesiones largas y admite
/// ruido como habilidad — reabriendo el sesgo de selección que #599
/// cerró para el espectro.
pub const N_EFECTIVO_EWMA: u64 = 128;

/// Acumuladores EWMA del IC (forma exacta de #594).
#[derive(Clone, Copy)]
struct AcumIc {
    ws: f64,
    wr: f64,
    wsr: f64,
    n: u64,
    /// #661 — E-PROCESO de Ville sobre el signo de voto·retorno: la
    /// significancia anytime-valid que sustituye al umbral fijo. El IC
    /// sigue midiendo la MAGNITUD (para el tamaño del peso); el
    /// e-proceso decide SI hay habilidad (inmune al optional stopping
    /// de la composición por evento y a la multiplicidad de la
    /// selección del máximo entre escalas).
    e_proceso: EProceso,
}

impl AcumIc {
    fn nuevo() -> Self {
        Self {
            ws: 0.0,
            wr: 0.0,
            wsr: 0.0,
            n: 0,
            e_proceso: EProceso::new(),
        }
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
        // #661: la misma observación alimenta el e-proceso.
        self.e_proceso.observar(v, r);
    }

    /// IC maduro y SIGNIFICATIVO (umbral #599 contra el sesgo de
    /// selección de ruido). `None` ⇒ el llamador usa el piso.
    /// Ola 48/H5: el umbral se calcula con el N EFECTIVO del EWMA (128),
    /// no con n crudo — en sesiones largas n→∞ haría el umbral →0 y
    /// admitiría ruido como habilidad.
    fn ic_significativo(&self) -> Option<f64> {
        if self.n < MUESTRAS_SKILL_MADURAS {
            return None;
        }
        // #661 — Ville REPLAZA el umbral fijo de Fisher: el e-proceso es
        /// anytime-valid (cualquier número de consultas) y corrige la
        /// multiplicidad de la selección del máximo entre escalas — el
        /// consejo diagnosticó que «en ruido el máximo de varias IC suele
        /// ser positivo»; el capital ×20 no se cruza por azar.
        if !self.e_proceso.significativo() {
            return None;
        }
        let den = self.ws * self.wr;
        if !den.is_finite() || den <= 0.0 {
            return None;
        }
        let ic = self.wsr / den.sqrt();
        if ic.is_finite() && ic > 0.0 {
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
    /// Ola 48/H6 — ts del último RE-ARME por escala: el score ocurre al
    /// cierre (camino general), el re-arme con votos frescos en el
    /// siguiente depth. Dedup independiente del de score.
    ts_rearmado: [u64; ESCALAS_VOTO],
    /// #659 (F1-C4) — snapshot del ÚLTIMO depth por escala: es el voto
    /// CAUSAL para armar el bloque que nazca después — el voto del re-arme
    /// diferido (t_d) contenía información de la propia ventana del bloque
    /// nuevo e inflaba el IC.
    voto_ultimo_depth: [[f64; ESCALAS_VOTO]; MOTORES],
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
            ts_rearmado: [0; ESCALAS_VOTO],
            voto_ultimo_depth: [[0.0; ESCALAS_VOTO]; MOTORES],
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
        // #659: el re-arme INLINE de este camino es causal (votos al ts
        // exacto del nacimiento) — marca el dedup para que el re-arme
        // diferido del siguiente depth NO lo pise con un snapshot viejo.
        self.ts_rearmado[escala] = ts;
    }

    /// Ola 48/H6 — puntúa la maduración SIN re-armar: para el camino de
    /// trades, donde no se computan votos frescos (el re-arme ocurre en
    /// el siguiente depth). Así el bloque que cierra entre depths se
    /// puntúa a tiempo contra el voto con el que NACIÓ, en vez de
    /// esperar al siguiente evento de libro.
    #[inline]
    pub fn observar_maduracion_sin_rearmar(&mut self, escala: usize, ts: u64, r: f64) {
        if escala >= ESCALAS_VOTO || ts <= self.ts_procesado[escala] {
            return;
        }
        self.ts_procesado[escala] = ts;
        for m in 0..MOTORES {
            self.acum[m][escala].observar(self.voto_armado[m][escala], r);
        }
    }

    /// Ola 48/H6 — RE-ARME (camino depth): si un bloque cerró desde el
    /// último re-arme, el snapshot de armado pasa a ser el voto con el que
    /// NACE el bloque nuevo. #659 (F1-C4): ese voto es el snapshot del
    /// ÚLTIMO DEPTH ANTERIOR al cierre (causal) — antes se usaba el voto
    /// del propio depth del re-arme (t_d), que arrastra información de la
    /// ventana del bloque nuevo e infla el IC. El snapshot se actualiza
    /// SIEMPRE al final: es el candidato para el próximo nacimiento.
    #[inline]
    pub fn re_amar_con_votos(&mut self, escala: usize, votos_actuales: &[VotoEspectral]) {
        if escala >= ESCALAS_VOTO {
            return;
        }
        if self.ts_procesado[escala] > self.ts_rearmado[escala] {
            self.ts_rearmado[escala] = self.ts_procesado[escala];
            for m in 0..MOTORES {
                self.voto_armado[m][escala] = self.voto_ultimo_depth[m][escala];
            }
        }
        for m in 0..MOTORES {
            if let Some(v) = votos_actuales.get(m) {
                self.voto_ultimo_depth[m][escala] = v.en_escala(escala);
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

/// Ola 48/H3 — TTL del veredicto espectral: sin depth fresco por más de
/// UN período de la escala dominante (mínimo 30 s), el dominante caduca y
/// cae a 0 — el último valor no-cero no puede pisar el veredicto escalar
/// indefinidamente si el stream de depth cae.
#[inline]
pub fn ttl_consenso_ms(tau_dominante_ms: f64) -> u64 {
    if tau_dominante_ms.is_finite() && tau_dominante_ms > 0.0 {
        tau_dominante_ms.max(30_000.0) as u64
    } else {
        30_000
    }
}

/// Ola 48/H1 — GATE DE OBSERVABILIDAD (D-742/CL-32) sobre la matriz de
/// pesos: las escalas con τ por DEBAJO de la resolución efectiva (su
/// masa es copia del último evento, no información) votan con peso 0 en
/// TODOS los motores. La composición resultante vale 0 en esas escalas
/// (Σw=0) y `dominante()` las salta — el consenso deja de poder ser
/// ruido de un tick a una τ sin físico. Devuelve las escalas excluidas.
#[inline]
pub fn aplicar_gate_observabilidad(
    pesos: &mut [[f64; ESCALAS_VOTO]; MOTORES],
    resolucion_efectiva_ms: f64,
) -> usize {
    if !resolucion_efectiva_ms.is_finite() || resolucion_efectiva_ms <= 0.0 {
        return 0;
    }
    let mut excluidas = 0usize;
    for k in 0..ESCALAS_VOTO {
        if (quantum_arena::temporal_spectrum::SPECTRUM_SCALES_MS[k] as f64)
            < resolucion_efectiva_ms
        {
            excluidas += 1;
            for fila in pesos.iter_mut() {
                fila[k] = 0.0;
            }
        }
    }
    excluidas
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

    /// Ola 48/H5 — sesiones largas: el umbral se ancla al N EFECTIVO del
    /// EWMA (≈128), no al conteo crudo. Una señal con IC≈0.05 (ruido
    /// leve sobre retornos ±) NO gana peso aunque n sea enorme — con el
    /// umbral viejo (2/√n → 0) la habría admitido como habilidad.
    #[test]
    fn qo_648_umbral_anclado_al_n_efectivo() {
        let mut sm = SkillMotores::new();
        // v=+1 constante; retornos alternantes +0.02/−0.018:
        // E[r]=0.001, E[r²]≈3.62e-4 ⇒ IC≈0.0526 — bajo el umbral
        // efectivo 2/√125≈0.179, SOBRE el umbral con n crudo (n=5000 ⇒
        // 0.028).
        madurar(&mut sm, 9, 1_000, 0.0, 1.0);
        for i in 2..5000u64 {
            let r = if i % 2 == 0 { 0.02 } else { -0.018 };
            madurar(&mut sm, 9, i * 1_000, r, 1.0);
        }
        assert!(sm.acum[0][9].n > 4000, "n={}", sm.acum[0][9].n);
        assert_eq!(
            sm.peso(0, 9),
            PISO_EXPLORACION,
            "IC≈0.05 debe quedar en piso con umbral efectivo"
        );
        // Control: la MISMA maquinaria sí premia evidencia fuerte.
        let mut fuerte = SkillMotores::new();
        madurar(&mut fuerte, 9, 1_000, 0.0, 1.0);
        for i in 2..200u64 {
            madurar(&mut fuerte, 9, i * 1_000, 0.05, 1.0);
        }
        assert!(fuerte.peso(0, 9) > PISO_EXPLORACION + 1e-6);
    }

    /// Ola 48/H6 — el score sin re-armar puntúa el voto de ARMADO y NO
    /// toca el snapshot: un trade-event intermedio no corrompe el armado.
    #[test]
    fn qo_648_score_sin_rearmar_y_rearme() {
        let mut sm = SkillMotores::new();
        madurar(&mut sm, 4, 1_000, 0.0, 0.7); // arma con 0.7
        // depth d1: el snapshot del último depth pasa a +0.9.
        sm.re_amar_con_votos(4, &votos_constantes(0.9));
        // El bloque cierra en un TRADE: score sin re-armar (arma sigue 0.7).
        sm.observar_maduracion_sin_rearmar(4, 2_000, 0.05);
        assert_eq!(sm.acum[0][4].n, 1);
        // depth d2 con voto −0.3 DESPUÉS del nacimiento del bloque nuevo:
        // #659 (F1-C4) — el re-arme usa el snapshot PREVIO (+0.9), no el
        // voto fresco de t_d (contenía información de la ventana del
        // bloque nuevo — look-ahead que inflaba el IC).
        sm.re_amar_con_votos(4, &votos_constantes(-0.3));
        assert_eq!(sm.acum[0][4].n, 1, "el re-arme no debe puntuar");
        assert_eq!(sm.voto_armado[0][4], 0.9, "arma con el último depth ANTERIOR al nacimiento");
        assert_eq!(sm.voto_ultimo_depth[0][4], -0.3, "el snapshot ya prepara el próximo armado");
        // El siguiente cierre puntúa el ARMADO +0.9 contra su retorno.
        sm.observar_maduracion_sin_rearmar(4, 3_000, -0.05);
        assert_eq!(sm.acum[0][4].n, 2);
        // El cierre en 3_000 dejó maduración pendiente: este depth re-arma
        // el bloque nacido en 3_000 con el snapshot previo (−0.3) y deja
        // el snapshot listo para el próximo (0.4).
        sm.re_amar_con_votos(4, &votos_constantes(0.4));
        assert_eq!(sm.voto_armado[0][4], -0.3, "arma con el último depth previo al nacimiento");
        assert_eq!(sm.voto_ultimo_depth[0][4], 0.4);
        // Sin maduración nueva: no-op de armado.
        sm.re_amar_con_votos(4, &votos_constantes(0.5));
        assert_eq!(sm.voto_armado[0][4], -0.3, "sin maduración nueva el armado no cambia");
    }

    /// Ola 48/H3 — TTL: piso 30 s, escala dominante cuando es mayor.
    #[test]
    fn qo_648_ttl_consenso() {
        assert_eq!(ttl_consenso_ms(0.0), 30_000);
        assert_eq!(ttl_consenso_ms(f64::NAN), 30_000);
        assert_eq!(ttl_consenso_ms(5_000.0), 30_000);
        assert_eq!(ttl_consenso_ms(600_000.0), 600_000);
        assert_eq!(ttl_consenso_ms(43_200_000.0), 43_200_000);
    }

    /// Ola 48/H1 — gate de observabilidad: escalas bajo la resolución
    /// efectiva quedan a peso 0 (la composición vale 0 ahí y el
    /// dominante las salta); las observables conservan su peso.
    #[test]
    fn qo_648_gate_observabilidad() {
        let sm = SkillMotores::new();
        let mut w = sm.pesos();
        // Resolución 30 s: excluye toda escala con τ_k < 30 s (la malla
        // es 4^k ms desde 1 ms), conserva las observables.
        let excluidas = aplicar_gate_observabilidad(&mut w, 30_000.0);
        assert!(excluidas > 0 && excluidas < ESCALAS_VOTO);
        for fila in &w {
            for k in 0..excluidas {
                assert_eq!(fila[k], 0.0, "escala {k} debe quedar a 0");
            }
            assert!(fila[excluidas] > 0.0, "la primera observable conserva peso");
        }
        // La composición con la columna a 0 vota 0 en esa escala.
        let votos = votos_constantes(0.9);
        let c = VotoEspectral::consenso_por_escala(&votos, &w);
        assert_eq!(c.en_escala(0), 0.0);
        assert!((c.en_escala(excluidas) - 0.9).abs() < 1e-12);
        // Resolución inválida ⇒ sin cambios.
        let mut w2 = sm.pesos();
        assert_eq!(aplicar_gate_observabilidad(&mut w2, f64::NAN), 0);
        assert_eq!(w2[0][0], PISO_EXPLORACION);
    }
}
