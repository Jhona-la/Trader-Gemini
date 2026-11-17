//! Contadores de la sección C de `qoder_contraste_inventario.md` (Qoder).
//!
//! Banco de contadores de TELEMETRÍA, en `crates/telemetry-server`: mide,
//! nunca decide. Cada contador es un `AtomicU64`/`AtomicI64` publicado al
//! registry mediante `snapshot()` (visible en telemetría). Ningún método de
//! este módulo cambia un veto, un tamaño, un precio ni un cierre: el encargo
//! del coordinador (12-nov 09:55) prohíbe tocar el motor.
//!
//! Los **puntos de incremento** están en el host/núcleo y se DECLARAN en
//! `PUNTOS_DE_INCREMENTO`; cablearlos requiere autorización explícita del
//! dueño del host y una corrida en la que se pueda tocar el motor. Mientras
//! tanto, los contadores quedan a cero y `snapshot()` los publica igualmente,
//! para que la ausencia de señal sea visible y no se confunda con un cero
//! medido (se acompaña de `registrado=false` por contador).
//!
//! Motivos de veto (lista CORREGIDA de `qoder_veto_host.md` §B, 9 motivos) y
//! motivos de cierre (8 del núcleo + IMMUNE_FLATTEN + RECONCILIATION +
//! BRACKET_EXCHANGE = 11) se toman literalmente del código, no se inventan.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

/// 9 motivos de veto de entrada del host (lista corregida de Qoder §B).
pub const MOTIVOS_VETO_HOST: [&str; 9] = [
    "VETO_MASIVO_MEMORIA_RELOJ", // :5470-5479, silencioso si cancel OK
    "ENVOLVENTE_NO_OPERABLE",    // :5584 (log :5786-5791), inoperante bootstrap n<30
    "FEE_BREAKER",               // :5798-5805, suspensión por símbolo
    "PURGA_PENDIENTE_B36",       // :5810-5819
    "RESERVA_NO_REAJUSTABLE",    // :5857-5867
    "CHECK_ENTRY",               // :5903-5909
    "D631_FILTROS_EXCHANGE",     // :5911-5925
    "MARGIN_GUARD",              // :5845-5852 (Envolvente/SinValidacion/Margen)
    "DEADLINE_RECUPERACION",     // :6054-6065/6116-6128, revocación post-envío
];

/// 11 motivos de cierre: 8 del núcleo (`lib.rs:5418-5443`) + 3 externos.
pub const MOTIVOS_CIERRE: [&str; 11] = [
    "TP",
    "SL",
    "TRAIL_HIT",
    "FORCE_TRAIL",
    "ZOMBIE",
    "TOXIC_FLOW",
    "PEAK_HARVEST",
    "ALPHA_DECAY",
    "IMMUNE_FLATTEN",   // aplanado inmune: executor.rs:1523 devuelve Ok((cancelaciones, cerradas)), un conteo agregado sin PnL neto por cierre
    "RECONCILIATION",   // reconciliation.rs
    "BRACKET_EXCHANGE", // trade_accounting.rs:176, trigger_kind
];

/// 7 etapas del embudo unificado (`direction_diag.rs:37-45`).
pub const ETAPAS_EMBUDO: [&str; 7] = [
    "escudo macro (D-624)",
    "acondicionamiento espectral (X-016)",
    "anti-whiplash (D-463)",
    "convicción bayesiana (D-472)",
    "racha direccional (D-499)",
    "escudo de libro L2 (D-475/D-688)",
    "escudo neuronal (D-473/D-688)",
];

/// Declaración de los sitios de incremento. NO se editan en este encargo.
pub const PUNTOS_DE_INCREMENTO: &str = "\
host_entry_vetos[9]            -> src/bin/god_engine.rs (lista corregida qoder_veto_host.md §B)
bootstrap_orders_sent          -> src/bin/god_engine.rs:5552-5556
bootstrap_envolvente_vetaba    -> ibídem, guardar `operable` antes de la rama
exit_reason_counts[11]         -> god-engine-core/lib.rs:5418-5443 (8), executor.rs:1008,
                                  reconciliation.rs, trade_accounting.rs:176
exit_pnl_net_by_reason[11]     -> trade_ledger.rs al asentar cada cierre (centavos)
ml_shadow_vetos_aprobados      -> god-engine-core/lib.rs:10326-10339
red_team_veto_leido            -> consumir red_team_veto_* (lib.rs:7382-7383)
conformal_gauge_veto_leido     -> consumir conformal_gauge_veto_* (lib.rs:6848-6849)
funnel_veto[d][stage]          -> direction_diag.rs:227-245, SIN la condición de libro plano
";

const D: usize = 2; // direcciones [largo, corto]
const S: usize = ETAPAS_EMBUDO.len();

/// Banco de contadores. `Default` los deja todos en cero y sin registrar.
#[derive(Debug, Default)]
pub struct ContadoresDemo {
    host_entry_vetos: [AtomicU64; MOTIVOS_VETO_HOST.len()],
    bootstrap_orders_sent: AtomicU64,
    bootstrap_orders_sent_envolvente_vetaba: AtomicU64,
    exit_reason_counts: [AtomicU64; MOTIVOS_CIERRE.len()],
    /// Centavos (Qoder pide `AtomicI64`); se acumula con saturación.
    exit_pnl_net_by_reason: [AtomicI64; MOTIVOS_CIERRE.len()],
    ml_shadow_vetos_aprobados: AtomicU64,
    red_team_veto_leido: AtomicU64,
    conformal_gauge_veto_leido: AtomicU64,
    /// `[dirección][etapa]`, sin la condición de libro plano.
    funnel_veto: [[AtomicU64; S]; D],
    /// El host ha cableado al menos un incremento (para distinguir 0 medido
    /// de «sin señal»). Lo pone el dueño del host al registrar el primer dato.
    registrado: AtomicU64,
}

/// Error al referir un motivo/etapa desconocido. No se fabrica un índice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorContador {
    MotivoVetoDesconocido,
    MotivoCierreDesconocido,
    EtapaDesconocida,
    DireccionDesconocida,
    PnlNoRepresentable,
}

impl ContadoresDemo {
    pub fn new() -> Self {
        Self::default()
    }

    fn idx_veto(motivo: &str) -> Result<usize, ErrorContador> {
        MOTIVOS_VETO_HOST
            .iter()
            .position(|m| *m == motivo)
            .ok_or(ErrorContador::MotivoVetoDesconocido)
    }
    fn idx_cierre(motivo: &str) -> Result<usize, ErrorContador> {
        MOTIVOS_CIERRE
            .iter()
            .position(|m| *m == motivo)
            .ok_or(ErrorContador::MotivoCierreDesconocido)
    }

    /// Incrementa un veto del host. No decide: sólo cuenta.
    pub fn registrar_veto_host(&self, motivo: &str) -> Result<u64, ErrorContador> {
        let i = Self::idx_veto(motivo)?;
        self.registrado.fetch_add(1, Ordering::Relaxed);
        Ok(self.host_entry_vetos[i].fetch_add(1, Ordering::Relaxed) + 1)
    }

    pub fn registrar_bootstrap(&self, envolvente_vetaba: bool) -> u64 {
        self.registrado.fetch_add(1, Ordering::Relaxed);
        if envolvente_vetaba {
            self.bootstrap_orders_sent_envolvente_vetaba
                .fetch_add(1, Ordering::Relaxed);
        }
        self.bootstrap_orders_sent.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Asienta un cierre: motivo + PnL neto en USDT, convertido a centavos.
    pub fn registrar_cierre(&self, motivo: &str, pnl_neto_usdt: f64) -> Result<u64, ErrorContador> {
        let i = Self::idx_cierre(motivo)?;
        if !pnl_neto_usdt.is_finite() {
            return Err(ErrorContador::PnlNoRepresentable);
        }
        let centavos = (pnl_neto_usdt * 100.0).round() as i64;
        self.sumar_centavos(i, centavos)?;
        self.registrado.fetch_add(1, Ordering::Relaxed);
        Ok(self.exit_reason_counts[i].fetch_add(1, Ordering::Relaxed) + 1)
    }

    fn sumar_centavos(&self, i: usize, delta: i64) -> Result<(), ErrorContador> {
        // Saturación sin panic: fetch_update con comprobación explícita.
        let r = self.exit_pnl_net_by_reason[i].fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |actual| Some(actual.saturating_add(delta)),
        );
        if r.is_err() {
            return Err(ErrorContador::PnlNoRepresentable);
        }
        Ok(())
    }

    pub fn registrar_shadow_veto_aprobado(&self) -> u64 {
        self.registrado.fetch_add(1, Ordering::Relaxed);
        self.ml_shadow_vetos_aprobados
            .fetch_add(1, Ordering::Relaxed)
            + 1
    }
    pub fn registrar_veto_leido(&self, gauge: bool) -> u64 {
        self.registrado.fetch_add(1, Ordering::Relaxed);
        let a = if gauge {
            &self.conformal_gauge_veto_leido
        } else {
            &self.red_team_veto_leido
        };
        a.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// `direccion`: 0 = largo, 1 = corto. `etapa`: nombre de `ETAPAS_EMBUDO`.
    /// Cuenta TODAS las entradas del embudo, sin la condición de libro plano.
    pub fn registrar_funnel_veto(
        &self,
        direccion: usize,
        etapa: &str,
    ) -> Result<u64, ErrorContador> {
        if direccion >= D {
            return Err(ErrorContador::DireccionDesconocida);
        }
        let s = ETAPAS_EMBUDO
            .iter()
            .position(|e| *e == etapa)
            .ok_or(ErrorContador::EtapaDesconocida)?;
        self.registrado.fetch_add(1, Ordering::Relaxed);
        Ok(self.funnel_veto[direccion][s].fetch_add(1, Ordering::Relaxed) + 1)
    }

    /// `true` si el host llegó a registrar algún incremento.
    pub fn cableado(&self) -> bool {
        self.registrado.load(Ordering::Relaxed) > 0
    }

    /// Publicación al registry: clave plana -> valor. Las claves son estables.
    /// Siempre incluye `_cableado` para no confundir «0 medido» con «sin señal».
    pub fn snapshot(&self) -> BTreeMap<String, i64> {
        let mut m: BTreeMap<String, i64> = BTreeMap::new();
        m.insert("_cableado".to_string(), if self.cableado() { 1 } else { 0 });
        for (i, nombre) in MOTIVOS_VETO_HOST.iter().enumerate() {
            m.insert(
                format!("host_entry_vetos.{}", nombre),
                self.host_entry_vetos[i].load(Ordering::Relaxed) as i64,
            );
        }
        m.insert(
            "bootstrap_orders_sent".to_string(),
            self.bootstrap_orders_sent.load(Ordering::Relaxed) as i64,
        );
        m.insert(
            "bootstrap_orders_sent_envolvente_vetaba".to_string(),
            self.bootstrap_orders_sent_envolvente_vetaba
                .load(Ordering::Relaxed) as i64,
        );
        for (i, nombre) in MOTIVOS_CIERRE.iter().enumerate() {
            m.insert(
                format!("exit_reason_counts.{}", nombre),
                self.exit_reason_counts[i].load(Ordering::Relaxed) as i64,
            );
            m.insert(
                format!("exit_pnl_net_by_reason.{}.centavos", nombre),
                self.exit_pnl_net_by_reason[i].load(Ordering::Relaxed),
            );
        }
        m.insert(
            "ml_shadow_vetos_aprobados".to_string(),
            self.ml_shadow_vetos_aprobados.load(Ordering::Relaxed) as i64,
        );
        m.insert(
            "red_team_veto_leido".to_string(),
            self.red_team_veto_leido.load(Ordering::Relaxed) as i64,
        );
        m.insert(
            "conformal_gauge_veto_leido".to_string(),
            self.conformal_gauge_veto_leido.load(Ordering::Relaxed) as i64,
        );
        for d in 0..D {
            for (s, etapa) in ETAPAS_EMBUDO.iter().enumerate() {
                m.insert(
                    format!("funnel_veto.{}.{}", d, etapa),
                    self.funnel_veto[d][s].load(Ordering::Relaxed) as i64,
                );
            }
        }
        m
    }

    /// Línea de telemetría (una sola línea, fácil de volcar al log).
    pub fn render_registry(&self) -> String {
        let mut s = String::from("[CONTADORES-C] ");
        for (k, v) in self.snapshot() {
            s.push_str(&format!("{}={};", k, v));
        }
        s
    }
}

#[cfg(test)]
mod tests_contadores_demo {
    use super::*;

    #[test]
    fn c_red_banco_nace_en_cero_y_declara_no_cableado() {
        let c = ContadoresDemo::new();
        assert!(!c.cableado(), "sin incrementos del host, NO está cableado");
        let snap = c.snapshot();
        assert_eq!(snap.get("_cableado").copied(), Some(0));
        assert_eq!(snap.get("host_entry_vetos.FEE_BREAKER").copied(), Some(0));
        assert_eq!(snap.get("exit_reason_counts.TP").copied(), Some(0));
        assert!(c.render_registry().starts_with("[CONTADORES-C] "));
    }

    #[test]
    fn c_verde_vetos_del_host_cuentan_solo_su_motivo() {
        let c = ContadoresDemo::new();
        assert_eq!(c.registrar_veto_host("FEE_BREAKER").ok(), Some(1));
        assert_eq!(c.registrar_veto_host("FEE_BREAKER").ok(), Some(2));
        assert_eq!(c.registrar_veto_host("MARGIN_GUARD").ok(), Some(1));
        let s = c.snapshot();
        assert_eq!(s.get("host_entry_vetos.FEE_BREAKER").copied(), Some(2));
        assert_eq!(s.get("host_entry_vetos.MARGIN_GUARD").copied(), Some(1));
        assert_eq!(
            s.get("host_entry_vetos.CHECK_ENTRY").copied(),
            Some(0),
            "ningún otro motivo se toca"
        );
        assert!(c.cableado());
        assert_eq!(s.get("_cableado").copied(), Some(1));
        assert_eq!(
            c.registrar_veto_host("MOTIVO_INVENTADO"),
            Err(ErrorContador::MotivoVetoDesconocido),
            "no se inventa un índice"
        );
    }

    #[test]
    fn c_verde_bootstrap_envolvente_y_11_motivos_de_cierre() {
        let c = ContadoresDemo::new();
        assert_eq!(c.registrar_bootstrap(false), 1);
        assert_eq!(c.registrar_bootstrap(true), 2);
        assert_eq!(c.registrar_cierre("TP", 0.0148).ok(), Some(1));
        assert_eq!(c.registrar_cierre("SL", -0.0051).ok(), Some(1));
        assert_eq!(c.registrar_cierre("IMMUNE_FLATTEN", -0.01).ok(), Some(1));
        let s = c.snapshot();
        assert_eq!(s.get("bootstrap_orders_sent").copied(), Some(2));
        assert_eq!(
            s.get("bootstrap_orders_sent_envolvente_vetaba").copied(),
            Some(1)
        );
        assert_eq!(s.get("exit_reason_counts.TP").copied(), Some(1));
        assert_eq!(
            s.get("exit_pnl_net_by_reason.TP.centavos").copied(),
            Some(1),
            "0.0148 USDT = 1 centavo (redondeo declarado)"
        );
        assert_eq!(
            s.get("exit_pnl_net_by_reason.SL.centavos").copied(),
            Some(-1)
        );
        assert_eq!(MOTIVOS_CIERRE.len(), 11, "la especificación pide 11");
        assert_eq!(MOTIVOS_VETO_HOST.len(), 9, "la especificación pide 9");
        assert_eq!(
            c.registrar_cierre("RAZON_INVENTADA", 1.0),
            Err(ErrorContador::MotivoCierreDesconocido)
        );
        assert_eq!(
            c.registrar_cierre("TP", f64::NAN),
            Err(ErrorContador::PnlNoRepresentable),
            "N/D no se convierte en 0.00"
        );
    }

    #[test]
    fn c_verde_funnel_veto_sin_condicion_de_libro_plano() {
        let c = ContadoresDemo::new();
        assert_eq!(
            c.registrar_funnel_veto(0, "convicción bayesiana (D-472)")
                .ok(),
            Some(1)
        );
        assert_eq!(
            c.registrar_funnel_veto(1, "convicción bayesiana (D-472)")
                .ok(),
            Some(1)
        );
        let s = c.snapshot();
        assert_eq!(
            s.get("funnel_veto.0.convicción bayesiana (D-472)").copied(),
            Some(1)
        );
        assert_eq!(
            s.get("funnel_veto.1.convicción bayesiana (D-472)").copied(),
            Some(1)
        );
        assert_eq!(
            c.registrar_funnel_veto(2, "convicción bayesiana (D-472)"),
            Err(ErrorContador::DireccionDesconocida)
        );
        assert_eq!(
            c.registrar_funnel_veto(0, "etapa que no existe"),
            Err(ErrorContador::EtapaDesconocida)
        );
    }

    #[test]
    fn c_verde_saturacion_de_centavos_sin_panico() {
        let c = ContadoresDemo::new();
        assert!(c.registrar_cierre("ZOMBIE", f64::MAX).is_ok());
        assert!(c.registrar_cierre("ZOMBIE", f64::MAX).is_ok());
        let v = c
            .snapshot()
            .get("exit_pnl_net_by_reason.ZOMBIE.centavos")
            .copied();
        assert_eq!(v, Some(i64::MAX), "saturating_add, no desbordamiento");
    }

    #[test]
    fn c_verde_declaracion_de_puntos_de_incremento() {
        for sitio in [
            "direction_diag.rs:227-245",
            "trade_ledger.rs",
            "executor.rs:1008",
            "trade_accounting.rs:176",
            "lib.rs:5418-5443",
            "lib.rs:10326-10339",
            "lib.rs:7382-7383",
            "lib.rs:6848-6849",
        ] {
            assert!(
                PUNTOS_DE_INCREMENTO.contains(sitio),
                "falta declarar {}",
                sitio
            );
        }
    }
}
