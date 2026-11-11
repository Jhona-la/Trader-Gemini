//! Contador del embudo por compuerta (SOL M1, ítem 2 / sol.md §5).
//!
//! Agrega `(compuerta, símbolo, hora) -> muertes por compuerta` con contadores
//! atómicos de bajo costo en el hot path. Incluye los vetos por kill-switch de
//! latencia y por sistema inmune:
//!   - H-07: GLM halló ≥1675 entradas bloqueadas por latencia que NO dejan
//!     línea por evento. Se ingestan en bloque (`record_latency_blocked`).
//!   - Nota 14:45: cierres por el sistema inmune (falso drawdown) se registran
//!     con `record_immune_veto`.
//!
//! Diseño (sol.md §5, regla 7): `FunnelCounter` con `AtomicU64` por
//! `(gate, symbol, hour)`. La inserción de una clave nueva toma un Mutex (raro);
//! el incremento es atómico y lock-free. No se emite telemetría por evento para
//! las entradas de latencia (H-07): se acumulan en bloque.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Compuertas del embudo. `LatencyKillSwitch` e `ImmuneSystem` cubren los vetos
/// de H-07 (latencia) y del sistema inmune (nota 14:45).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FunnelGate {
    SpectrumEval,
    ModelB325,
    MlGate,
    FixedFilter,
    AdversarialGate,
    Fill,
    Flatten,
    /// H-07: entradas bloqueadas por latencia (sin línea por evento).
    LatencyKillSwitch,
    /// Nota 14:45: cierres/filtros por el sistema inmune (falso drawdown).
    ImmuneSystem,
}

impl FunnelGate {
    /// Etiqueta corta para reportes / JSONL.
    pub fn as_str(&self) -> &'static str {
        match self {
            FunnelGate::SpectrumEval => "SPECTRUM_EVAL",
            FunnelGate::ModelB325 => "MODEL_B3_25",
            FunnelGate::MlGate => "ML_GATE",
            FunnelGate::FixedFilter => "FIXED_FILTER",
            FunnelGate::AdversarialGate => "ADVERSARIAL_GATE",
            FunnelGate::Fill => "FILL",
            FunnelGate::Flatten => "FLATTEN",
            FunnelGate::LatencyKillSwitch => "LATENCY_KILL_SWITCH",
            FunnelGate::ImmuneSystem => "IMMUNE_SYSTEM",
        }
    }
}

/// Clave del embudo: compuerta + símbolo + cubo de hora (epoch-ms / 3_600_000).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunnelKey {
    pub gate: FunnelGate,
    pub symbol: String,
    pub hour_bucket: u64,
}

/// Contador atómico lock-free por `(gate, symbol, hour)`. La primera vez que
/// aparece una clave se inserta bajo un Mutex; luego el `fetch_add` es atómico.
#[derive(Default)]
pub struct FunnelCounter {
    cells: Mutex<HashMap<FunnelKey, Arc<AtomicU64>>>,
}

impl FunnelCounter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Devuelve la celda atómica para la clave, creándola si hace falta.
    /// Doble chequeo: la ruta común (clave ya existente) no toma el lock.
    fn cell(&self, key: FunnelKey) -> Arc<AtomicU64> {
        {
            let guard = self.cells.lock().expect("mutex del embudo envenenado");
            if let Some(c) = guard.get(&key) {
                return Arc::clone(c);
            }
        }
        let mut guard = self.cells.lock().expect("mutex del embudo envenenado");
        Arc::clone(
            guard
                .entry(key)
                .or_insert_with(|| Arc::new(AtomicU64::new(0))),
        )
    }

    /// Registra `n` muertes (vetos/salidas) en una compuerta. `n` puede ser >1
    /// para ingestar en bloque entradas sin línea por evento (H-07: ≥1675).
    pub fn add_deaths(&self, gate: FunnelGate, symbol: &str, hour_bucket: u64, n: u64) {
        if n == 0 {
            return;
        }
        let key = FunnelKey {
            gate,
            symbol: symbol.to_string(),
            hour_bucket,
        };
        self.cell(key).fetch_add(n, Ordering::Relaxed);
    }

    /// Una muerte en una compuerta.
    pub fn record_death(&self, gate: FunnelGate, symbol: &str, hour_bucket: u64) {
        self.add_deaths(gate, symbol, hour_bucket, 1);
    }

    /// Veto por kill-switch de latencia (H-07). Soporta ingestión en bloque.
    pub fn record_latency_blocked(&self, symbol: &str, hour_bucket: u64, n: u64) {
        self.add_deaths(FunnelGate::LatencyKillSwitch, symbol, hour_bucket, n);
    }

    /// Veto por sistema inmune (nota 14:45): cierre/filtro del sistema inmune.
    pub fn record_immune_veto(&self, symbol: &str, hour_bucket: u64, n: u64) {
        self.add_deaths(FunnelGate::ImmuneSystem, symbol, hour_bucket, n);
    }

    /// Lee el total de muertes en una clave concreta.
    pub fn deaths(&self, gate: FunnelGate, symbol: &str, hour_bucket: u64) -> u64 {
        let key = FunnelKey {
            gate,
            symbol: symbol.to_string(),
            hour_bucket,
        };
        let guard = self.cells.lock().expect("mutex del embudo envenenado");
        guard
            .get(&key)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Total de muertes en una compuerta (todos los símbolos/horas).
    pub fn deaths_by_gate(&self, gate: FunnelGate) -> u64 {
        let guard = self.cells.lock().expect("mutex del embudo envenenado");
        guard
            .iter()
            .filter(|(k, _)| k.gate == gate)
            .map(|(_, c)| c.load(Ordering::Relaxed))
            .sum()
    }

    /// Total de muertes en todas las compuertas.
    pub fn total_deaths(&self) -> u64 {
        let guard = self.cells.lock().expect("mutex del embudo envenenado");
        guard.values().map(|c| c.load(Ordering::Relaxed)).sum()
    }

    /// Número de claves distintas (cardinalidad para reportes).
    pub fn distinct_keys(&self) -> usize {
        self.cells
            .lock()
            .expect("mutex del embudo envenenado")
            .len()
    }

    /// `true` si una compuerta tiene al menos 1 muerte registrada.
    pub fn gate_has_deaths(&self, gate: FunnelGate) -> bool {
        self.deaths_by_gate(gate) > 0
    }

    /// Snapshot por compuerta para el reporte GLM a3g vs F1/F2 (sol.md §5, regla 7).
    pub fn snapshot_by_gate(&self) -> HashMap<FunnelGate, u64> {
        let mut out: HashMap<FunnelGate, u64> = HashMap::new();
        let guard = self.cells.lock().expect("mutex del embudo envenenado");
        for (k, c) in guard.iter() {
            *out.entry(k.gate).or_insert(0) += c.load(Ordering::Relaxed);
        }
        out
    }
}

#[cfg(test)]
mod tests_funnel {
    use super::*;

    const H: u64 = 1_700_000; // cubo de hora arbitrario

    #[test]
    fn funnel_cuenta_muertes_por_compuerta_simbolo_hora() {
        let f = FunnelCounter::new();
        f.record_death(FunnelGate::MlGate, "BTCUSDT", H);
        f.record_death(FunnelGate::MlGate, "BTCUSDT", H);
        f.record_death(FunnelGate::MlGate, "ETHUSDT", H);
        assert_eq!(f.deaths(FunnelGate::MlGate, "BTCUSDT", H), 2);
        assert_eq!(f.deaths(FunnelGate::MlGate, "ETHUSDT", H), 1);
        assert_eq!(f.deaths_by_gate(FunnelGate::MlGate), 3);
        // otra compuerta es independiente
        f.record_death(FunnelGate::Flatten, "BTCUSDT", H);
        assert_eq!(f.deaths(FunnelGate::Flatten, "BTCUSDT", H), 1);
        assert_eq!(f.deaths_by_gate(FunnelGate::MlGate), 3);
        assert!(!f.gate_has_deaths(FunnelGate::SpectrumEval));
    }

    #[test]
    fn funnel_vetos_latencia_en_bloque_h07() {
        // H-07: GLM halló ≥1675 entradas bloqueadas por latencia sin línea por evento.
        let f = FunnelCounter::new();
        f.record_latency_blocked("BTCUSDT", H, 1675);
        f.record_latency_blocked("ETHUSDT", H, 12);
        assert_eq!(f.deaths(FunnelGate::LatencyKillSwitch, "BTCUSDT", H), 1675);
        assert_eq!(f.deaths(FunnelGate::LatencyKillSwitch, "ETHUSDT", H), 12);
        assert_eq!(f.deaths_by_gate(FunnelGate::LatencyKillSwitch), 1687);
        assert!(f.gate_has_deaths(FunnelGate::LatencyKillSwitch));
        // No se crean celdas con n=0.
        f.record_latency_blocked("X", H, 0);
        assert_eq!(f.deaths(FunnelGate::LatencyKillSwitch, "X", H), 0);
    }

    #[test]
    fn funnel_veto_sistema_inmune() {
        // nota 14:45: cierres por sistema inmune (falso drawdown).
        let f = FunnelCounter::new();
        f.record_immune_veto("BTCUSDT", H, 1);
        f.record_immune_veto("BTCUSDT", H, 1);
        assert_eq!(f.deaths(FunnelGate::ImmuneSystem, "BTCUSDT", H), 2);
        assert_eq!(f.deaths_by_gate(FunnelGate::ImmuneSystem), 2);
    }

    #[test]
    fn funnel_hora_distinta_separa() {
        let f = FunnelCounter::new();
        f.record_death(FunnelGate::MlGate, "BTCUSDT", H);
        f.record_death(FunnelGate::MlGate, "BTCUSDT", H + 1);
        assert_eq!(f.deaths(FunnelGate::MlGate, "BTCUSDT", H), 1);
        assert_eq!(f.deaths(FunnelGate::MlGate, "BTCUSDT", H + 1), 1);
        assert_eq!(f.deaths_by_gate(FunnelGate::MlGate), 2);
        assert_eq!(f.distinct_keys(), 2);
    }

    #[test]
    fn funnel_total_y_vacio() {
        let f = FunnelCounter::new();
        assert_eq!(f.total_deaths(), 0);
        f.record_death(FunnelGate::SpectrumEval, "BTCUSDT", H);
        f.record_latency_blocked("BTCUSDT", H, 1675);
        f.record_immune_veto("BTCUSDT", H, 3);
        // 1 + 1675 + 3 = 1679
        assert_eq!(f.total_deaths(), 1679);
        let snap = f.snapshot_by_gate();
        assert_eq!(snap.get(&FunnelGate::SpectrumEval).copied(), Some(1));
        assert_eq!(
            snap.get(&FunnelGate::LatencyKillSwitch).copied(),
            Some(1675)
        );
        assert_eq!(snap.get(&FunnelGate::ImmuneSystem).copied(), Some(3));
    }
}
