//! Contrato de Evidencia de Ignición (FASE A-A2 del PLAN MAESTRO DE
//! EVIDENCIA, GLM-202/203; diseño en
//! `docs/audit/CONTRATO_EVIDENCIA_IGNICION_A2_2026-11-05.md`).
//!
//! Mata al BLOQUEANTE #1 de la auditoría A1
//! (`docs/audit/IGNICION_A1_2026-11-05.md`): `darwin_approved` se
//! auto-aprobaba tras un entrenamiento NO-OP (GE:1716 + BOOT:287-294) y
//! bastaban 120 ticks. Aquí la aprobación de Darwin es una FUNCIÓN de
//! evidencia medible producida por el arranque — fuente única de verdad,
//! crate neutro sin tocar el bin.
//!
//! Reglas (política, sin literales mágicos de aprobación):
//! 1. Warmup contado de verdad: `klines_warmup_por_simbolo >= minimo`
//!    (el mínimo lo pasa el host como config; el A1 midió 1000 fijos).
//! 2. Entrenamiento REAL: `epochs_efectivas >= 1` Y `metrica` presente y
//!    finita. Training NO-OP ⇒ RECHAZADO — el sistema NO se habilita a
//!    sí mismo.
//! 3. OMS coherente: `oms_reconciled` debe venir del snapshot conjunto
//!    (en vivo sólo lo escribe `apply_snapshot`, OMS:619-629) Y ser
//!    coherente con el modo: **el papel local NUNCA reconcilia** —
//!    `es_cuenta_viva=false` con `oms_reconciled=true` es incoherencia
//!    (el papel no consulta nada: A1 MEDIO #6).
//! 4. Cuenta viva (Testnet/Mainnet) reconciliada: `oms_reconciled=true`
//!    exigido para aprobar.
//!
//! **Integración prevista** (host — dueños AGY/QRA; ver doc A2):
//! GE:1716 `darwin_approved.store(matches!(ev.aprueba_darwin(),
//! VeredictoDarwin::Aprobado))` + log de la razón; BOOT:287-294 deja de
//! ser no-op y devuelve métricas reales; ORCH:74-83 sin cambio (ya exige
//! darwin — ahora darwin SIGNIFICA algo); OMS:199 sin cambio en vivo.
//! Compatibilidad con la corrida A3: puerta explícita y auditable
//! `TG_IGNICION_SIN_DARWIN=1` (decisión del consejo, nunca default).

/// Venue del calentamiento y la ejecución — del discriminador real
/// (GE:1586: `set_paper_trading(is_demo_mode && !is_env_testnet)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Venue {
    /// Binance Testnet o Mainnet — cuenta viva que consulta la API.
    CuentaViva,
    /// Papel local — no consulta nada; NUNCA puede reconciliar.
    PapelLocal,
}

/// Evidencia medible producida por el arranque. Todos los campos los
/// llena el host con lo que REALMENTE hizo (contado, no supuesto).
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenciaIgnicion {
    /// Klines de warmup cargados POR SÍMBOLO (BOOT:255 hoy fija 1000 —
    /// el contrato CERTIFICA el número contado, no lo supone).
    pub klines_warmup_por_simbolo: u32,
    /// Mínimo exigido de klines por símbolo (config del host; el
    /// contrato no esconde un literal mágico de aprobación).
    pub minimo_klines_por_simbolo: u32,
    /// Venue del warmup/ejecutor (BOOT:28-35 garantiza paridad; el
    /// contrato la exige coherente con `es_cuenta_viva`).
    pub venue: Venue,
    /// Epochs de entrenamiento EJECUTADAS de verdad (BOOT:287-294 hoy
    /// es no-op y las ignora — quien llame debe contarlas; 0 = no-op).
    pub epochs_entrenamiento_efectivas: u32,
    /// Métrica del entrenamiento (loss/IC del modelo calentado);
    /// `None` = no se produjo métrica (training no-op). NaN/±Inf son
    /// inválidos (fall-closed, jamás se aprueban).
    pub metrica_entrenamiento: Option<f64>,
    /// `oms_reconciled` del SNAPSHOT CONJUNTO (OMS:619-629), no del
    /// constructor. En papel local este campo debe ser `false` — si el
    /// host pasa `true`, es incoherencia y el veredicto la rechaza.
    pub oms_reconciled: bool,
}

/// Veredicto de la aprobación de Darwin: la razón del rechazo es
/// estática para que el log del arranque la citen sin pérdida.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VeredictoDarwin {
    /// Todo cumplido: warmup contado suficiente, entrenamiento real,
    /// venue coherente y (en cuenta viva) OMS del snapshot.
    Aprobado,
    /// Rechazado con la razón auditable.
    Rechazado(&'static str),
}

impl EvidenciaIgnicion {
    /// Política ÚNICA de aprobación de Darwin — sustituye al timer de
    /// 120 ticks. Fall-closed total: sin panic, cada rechazo con razón.
    pub fn aprueba_darwin(&self) -> VeredictoDarwin {
        // Regla 2 (primera: la que mata al BLOQUEANTE #1): entrenamiento
        // REAL. El no-op ya no aprueba aunque todo lo demás esté perfecto.
        if self.epochs_entrenamiento_efectivas == 0 {
            return VeredictoDarwin::Rechazado("training_no_op");
        }
        let metrica_valida = match self.metrica_entrenamiento {
            Some(m) => m.is_finite(),
            None => false,
        };
        if !metrica_valida {
            return VeredictoDarwin::Rechazado("metrica_entrenamiento_invalida");
        }
        // Regla 1: warmup suficiente CONTADO (no supuesto).
        if self.minimo_klines_por_simbolo == 0 {
            return VeredictoDarwin::Rechazado("config_minimo_klines_cero");
        }
        if self.klines_warmup_por_simbolo < self.minimo_klines_por_simbolo {
            return VeredictoDarwin::Rechazado("warmup_insuficiente");
        }
        // Regla 3: coherencia venue/OMS. El papel local NUNCA reconcilia
        // (A1 MEDIO #6: TRUE sin consultar nada era indistinguible de
        // reconciliado). Venue y boolean deben contar la misma historia.
        let es_cuenta_viva = self.venue == Venue::CuentaViva;
        match (es_cuenta_viva, self.oms_reconciled) {
            (false, true) => {
                return VeredictoDarwin::Rechazado("papel_local_no_reconcilia");
            }
            (true, false) => {
                return VeredictoDarwin::Rechazado("oms_no_reconciliado");
            }
            _ => {}
        }
        // Regla 4 implícita en el match: cuenta viva ⇒ exigimos
        // oms_reconciled=true (rama (true,false) ya rechazó). Papel
        // coherente (false,false) continúa.
        VeredictoDarwin::Aprobado
    }
}
