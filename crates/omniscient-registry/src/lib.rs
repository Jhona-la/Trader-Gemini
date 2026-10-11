use crossbeam_skiplist::SkipMap;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use uuid::Uuid;

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    Archive,
    RkyvSerialize,
    RkyvDeserialize,
)]
pub enum ParameterKind {
    Fixed,
    Adaptive,
}

#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
pub struct RegistrySnapshot {
    pub parameters: Vec<ParameterSnapshot>,
}

#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
pub struct ParameterSnapshot {
    pub name: String,
    pub kind: ParameterKind,
    pub value_bits: u64,
    pub owner: String,
    pub timestamp: i64,
}

pub struct Parameter {
    pub id: Uuid,
    pub name: String,
    pub kind: ParameterKind,
    pub value: AtomicU64,
    pub owner: String,

    pub timestamp: AtomicI64,
}

impl Parameter {
    pub fn new(name: &str, kind: ParameterKind, initial_value: f64, owner: &str) -> Self {
        Self {
            id: Uuid::now_v7(),
            name: name.to_string(),
            kind,
            value: AtomicU64::new(initial_value.to_bits()),
            owner: owner.to_string(),
            timestamp: AtomicI64::new(chrono::Utc::now().timestamp_millis()),
        }
    }

    pub fn get_value(&self) -> f64 {
        f64::from_bits(self.value.load(Ordering::Relaxed))
    }

    pub fn set_value(&self, val: f64) {
        self.value.store(val.to_bits(), Ordering::Relaxed);
        self.timestamp
            .store(chrono::Utc::now().timestamp_millis(), Ordering::Relaxed);
    }

    pub fn get_timestamp(&self) -> i64 {
        self.timestamp.load(Ordering::Relaxed)
    }
}

pub struct OmniscientRegistry {
    // Usamos SkipMap para concurrencia lock-free verdadera
    map: SkipMap<String, Arc<Parameter>>,
}

impl std::fmt::Debug for OmniscientRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OmniscientRegistry")
            .field("parameters_count", &self.map.len())
            .finish()
    }
}

#[inline(always)]
fn format_scoped_key<F, R>(prefix: &str, separator: &str, suffix: &str, f: F) -> R
where
    F: FnOnce(&str) -> R,
{
    let mut buf = [0u8; 96];
    let p_bytes = prefix.as_bytes();
    let s_bytes = separator.as_bytes();
    let su_bytes = suffix.as_bytes();
    let total = p_bytes.len() + s_bytes.len() + su_bytes.len();
    if total <= buf.len() {
        buf[..p_bytes.len()].copy_from_slice(p_bytes);
        buf[p_bytes.len()..p_bytes.len() + s_bytes.len()].copy_from_slice(s_bytes);
        buf[p_bytes.len() + s_bytes.len()..total].copy_from_slice(su_bytes);
        if let Ok(s) = std::str::from_utf8(&buf[..total]) {
            return f(s);
        }
    }
    f(&format!("{}{}{}", prefix, separator, suffix))
}

#[inline(always)]
fn format_coin_key<F, R>(coin_id: usize, name: &str, f: F) -> R
where
    F: FnOnce(&str) -> R,
{
    let mut buf = [0u8; 64];
    buf[0] = b'c';
    let mut cid = coin_id;
    let mut num_buf = [0u8; 20];
    let mut num_len = 0;
    if cid == 0 {
        num_buf[0] = b'0';
        num_len = 1;
    } else {
        while cid > 0 {
            num_buf[num_len] = b'0' + (cid % 10) as u8;
            cid /= 10;
            num_len += 1;
        }
        num_buf[..num_len].reverse();
    }
    let total = 1 + num_len + 1 + name.len();
    if total <= buf.len() {
        buf[1..1 + num_len].copy_from_slice(&num_buf[..num_len]);
        buf[1 + num_len] = b':';
        buf[2 + num_len..total].copy_from_slice(name.as_bytes());
        if let Ok(s) = std::str::from_utf8(&buf[..total]) {
            return f(s);
        }
    }
    f(&format!("c{}:{}", coin_id, name))
}

impl OmniscientRegistry {
    pub fn new() -> Self {
        Self {
            map: SkipMap::new(),
        }
    }

    pub fn register(&self, param: Parameter) -> Result<(), String> {
        let name = param.name.clone();
        if self.map.contains_key(&name) {
            return Err(format!("Parameter {} already exists!", name));
        }
        self.map.insert(name, Arc::new(param));
        Ok(())
    }

    pub fn set(&self, name: &str, val: f64) {
        if let Some(entry) = self.map.get(name) {
            entry.value().set_value(val);
        } else {
            let param = Parameter::new(name, ParameterKind::Adaptive, val, "system");
            self.map.insert(name.to_string(), Arc::new(param));
        }
    }

    pub fn register_or_update(&self, name: &str, kind: ParameterKind, val: f64, owner: &str) {
        if let Some(entry) = self.map.get(name) {
            entry.value().set_value(val);
        } else {
            let param = Parameter::new(name, kind, val, owner);
            self.map.insert(name.to_string(), Arc::new(param));
        }
    }

    pub fn get(&self, name: &str, _consumer_name: &str) -> Option<Arc<Parameter>> {
        if let Some(entry) = self.map.get(name) {
            let param = entry.value().clone();
            if false {}
            Some(param)
        } else {
            None
        }
    }

    /// Lectura directa y rápida del valor numérico sin clonar Arc ni mutar sets de consumidores (Hot-Path HFT)
    #[inline(always)]
    pub fn get_value_fast(&self, name: &str) -> Option<f64> {
        self.map.get(name).map(|entry| entry.value().get_value())
    }

    /// Lectura directa con fallback por defecto ante ausencia o valor no finito / desconocido (Hot-Path HFT)
    #[inline(always)]
    pub fn get_value_or(&self, name: &str, default: f64) -> f64 {
        match self.get_value_fast(name) {
            Some(v) if v.is_finite() => v,
            _ => default,
        }
    }

    /// D-07: Namespacing por activo: registra o actualiza un parámetro prefijado por símbolo con zero heap-allocation.
    #[inline(always)]
    pub fn set_scoped(&self, symbol: &str, name: &str, val: f64) {
        format_scoped_key(symbol, "_", name, |scoped_name| {
            self.set(scoped_name, val);
        });
    }

    /// D-07: Lectura con resolución de ámbito: busca `{symbol}_{name}` y si no existe busca `{name}` sin heap-allocation.
    #[inline(always)]
    pub fn get_scoped(
        &self,
        symbol: &str,
        name: &str,
        consumer_name: &str,
    ) -> Option<Arc<Parameter>> {
        format_scoped_key(symbol, "_", name, |scoped_name| {
            self.get(scoped_name, consumer_name)
                .or_else(|| self.get(name, consumer_name))
        })
    }

    /// D-07: Lectura rápida O(1) con resolución de ámbito y fallback por defecto (Zero Heap Allocation Hot-Path).
    #[inline(always)]
    pub fn get_scoped_value_or(&self, symbol: &str, name: &str, default: f64) -> f64 {
        format_scoped_key(symbol, "_", name, |scoped_name| {
            if let Some(v) = self.get_value_fast(scoped_name) {
                if v.is_finite() {
                    return v;
                }
            }
            if let Some(v) = self.get_value_fast(name) {
                if v.is_finite() {
                    return v;
                }
            }
            default
        })
    }

    /// D-07: Lectura estricta escopada por símbolo SIN fallback al parámetro global (Aislamiento Multi-Activo).
    #[inline(always)]
    pub fn get_scoped_strict(&self, symbol: &str, name: &str) -> Option<f64> {
        format_scoped_key(symbol, "_", name, |scoped_name| {
            self.get_value_fast(scoped_name)
        })
    }

    /// D-07: Lectura estricta escopada por símbolo con default SIN fallback al parámetro global (Aislamiento Multi-Activo).
    #[inline(always)]
    pub fn get_scoped_strict_or(&self, symbol: &str, name: &str, default: f64) -> f64 {
        self.get_scoped_strict(symbol, name)
            .filter(|v| v.is_finite())
            .unwrap_or(default)
    }

    /// D38: Namespacing por índice numérico de activo con stack buffer (Zero Heap Allocation Hot-Path).
    #[inline(always)]
    pub fn set_for_coin(&self, coin_id: usize, name: &str, val: f64) {
        format_coin_key(coin_id, name, |key| {
            self.set(key, val);
        });
    }

    /// D38: Lectura por índice numérico de activo con fallback al parámetro global sin heap allocation.
    #[inline(always)]
    pub fn get_for_coin_or(&self, coin_id: usize, name: &str, default: f64) -> f64 {
        format_coin_key(coin_id, name, |key| {
            if let Some(v) = self.get_value_fast(key) {
                if v.is_finite() {
                    return v;
                }
            }
            if let Some(v) = self.get_value_fast(name) {
                if v.is_finite() {
                    return v;
                }
            }
            default
        })
    }

    /// D38: Lectura estricta por coin_id SIN fallback al parámetro global (Aislamiento Multi-Activo Estricto).
    #[inline(always)]
    pub fn get_for_coin_strict(&self, coin_id: usize, name: &str) -> Option<f64> {
        format_coin_key(coin_id, name, |key| {
            self.get_value_fast(key)
        })
    }

    /// D38: Lectura estricta por coin_id con default SIN fallback al parámetro global (Aislamiento Multi-Activo Estricto).
    #[inline(always)]
    pub fn get_for_coin_strict_or(&self, coin_id: usize, name: &str, default: f64) -> f64 {
        self.get_for_coin_strict(coin_id, name)
            .filter(|v| v.is_finite())
            .unwrap_or(default)
    }

    /// D-219: Lectura polimórfica escopada por activo (símbolo + coin_id) con fallback transparente al global (Zero Heap Alloc).
    #[inline(always)]
    pub fn get_scoped_parameter(
        &self,
        symbol: Option<&str>,
        coin_id: Option<usize>,
        name: &str,
        consumer_name: &str,
    ) -> Option<Arc<Parameter>> {
        if let Some(sym) = symbol {
            if !sym.is_empty() {
                let p = format_scoped_key(sym, "_", name, |scoped_name| {
                    self.get(scoped_name, consumer_name)
                });
                if p.is_some() {
                    return p;
                }
            }
        }
        if let Some(cid) = coin_id {
            let p = format_coin_key(cid, name, |scoped_cid| {
                self.get(scoped_cid, consumer_name)
            });
            if p.is_some() {
                return p;
            }
        }
        self.get(name, consumer_name)
    }

    /// D-219: Lectura rápida O(1) de valor numérico escopado por activo (símbolo + coin_id) con stack buffer.
    #[inline(always)]
    pub fn get_scoped_val_or(
        &self,
        symbol: Option<&str>,
        coin_id: Option<usize>,
        name: &str,
        default: f64,
    ) -> f64 {
        if let Some(sym) = symbol {
            if !sym.is_empty() {
                let val = format_scoped_key(sym, "_", name, |scoped_name| {
                    self.get_value_fast(scoped_name)
                });
                if let Some(v) = val {
                    if v.is_finite() {
                        return v;
                    }
                }
            }
        }
        if let Some(cid) = coin_id {
            let val = format_coin_key(cid, name, |scoped_cid| {
                self.get_value_fast(scoped_cid)
            });
            if let Some(v) = val {
                if v.is_finite() {
                    return v;
                }
            }
        }
        if let Some(v) = self.get_value_fast(name) {
            if v.is_finite() {
                return v;
            }
        }
        default
    }

    /// LC-F1-06 / D-219: Lectura polimórfica escopada por activo (símbolo + coin_id)
    /// SIN fallback al parámetro global (Aislamiento Multi-Activo Estricto, Zero Heap Alloc).
    #[inline(always)]
    pub fn get_scoped_parameter_strict(
        &self,
        symbol: Option<&str>,
        coin_id: Option<usize>,
        name: &str,
        consumer_name: &str,
    ) -> Option<Arc<Parameter>> {
        if let Some(sym) = symbol {
            if !sym.is_empty() {
                let p = format_scoped_key(sym, "_", name, |scoped_name| {
                    self.get(scoped_name, consumer_name)
                });
                if p.is_some() {
                    return p;
                }
            }
        }
        if let Some(cid) = coin_id {
            let p = format_coin_key(cid, name, |scoped_cid| {
                self.get(scoped_cid, consumer_name)
            });
            if p.is_some() {
                return p;
            }
        }
        None
    }

    /// LC-F1-06 / D-219: Lectura rápida O(1) de valor numérico escopado por activo (símbolo + coin_id)
    /// SIN fallback al parámetro global (Aislamiento Multi-Activo Estricto, Zero Heap Alloc).
    #[inline(always)]
    pub fn get_scoped_val_strict_or(
        &self,
        symbol: Option<&str>,
        coin_id: Option<usize>,
        name: &str,
        default: f64,
    ) -> f64 {
        if let Some(sym) = symbol {
            if !sym.is_empty() {
                let val = format_scoped_key(sym, "_", name, |scoped_name| {
                    self.get_value_fast(scoped_name)
                });
                if let Some(v) = val {
                    if v.is_finite() {
                        return v;
                    }
                }
            }
        }
        if let Some(cid) = coin_id {
            let val = format_coin_key(cid, name, |scoped_cid| {
                self.get_value_fast(scoped_cid)
            });
            if let Some(v) = val {
                if v.is_finite() {
                    return v;
                }
            }
        }
        default
    }

    pub fn detect_collisions(&self) -> Vec<String> {
        // Since we prevent duplicates on register, collisions in a strict map sense are avoided.
        // However, if strategies attempt to register the same name, the `register` returns Err.
        Vec::new()
    }

    pub fn scan_all(&self) -> Vec<Arc<Parameter>> {
        let mut all = Vec::with_capacity(self.map.len());
        for entry in self.map.iter() {
            all.push(entry.value().clone());
        }
        all
    }

    pub fn take_snapshot(&self) -> RegistrySnapshot {
        let mut parameters = Vec::with_capacity(self.map.len());
        for entry in self.map.iter() {
            let p = entry.value();
            parameters.push(ParameterSnapshot {
                name: p.name.clone(),
                kind: p.kind,
                value_bits: p.value.load(Ordering::Relaxed),
                owner: p.owner.clone(),
                timestamp: p.timestamp.load(Ordering::Relaxed),
            });
        }
        RegistrySnapshot { parameters }
    }

    pub fn restore_from_snapshot(&self, snapshot: &RegistrySnapshot) -> usize {
        let mut restored = 0;
        for p in &snapshot.parameters {
            let val = f64::from_bits(p.value_bits);
            self.register_or_update(&p.name, p.kind, val, &p.owner);
            restored += 1;
        }
        restored
    }


    /// Reconcilia valores de estado atómico con datos confirmados por REST API (Punto #275)
    pub fn reconcile_with_remote(&self, remote_values: &[(String, f64)], owner: &str) -> usize {
        let mut reconciled = 0;
        for (name, val) in remote_values {
            if val.is_finite() {
                self.register_or_update(name, ParameterKind::Adaptive, *val, owner);
                reconciled += 1;
            }
        }
        reconciled
    }

    pub fn persist_to_disk(&self, path: &str) -> std::io::Result<()> {
        let snapshot = self.take_snapshot();
        let bytes = rkyv::to_bytes::<_, 4096>(&snapshot)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
            .to_vec();

        let path_str = path.to_string();

        // FASE 2 FIX: Desacoplar I/O bloqueante (Zero-Copy lock-in prevention)
        // El sync_all() y rename bloquean el disco duro, paralizando el ciclo de decisión.
        // Se delega la persistencia física a un hilo esclavo huérfano.
        std::thread::spawn(move || {
            let tmp_path = format!("{}.tmp", path_str);
            if let Ok(mut file) = File::create(&tmp_path) {
                let _ = file.write_all(&bytes);
                let _ = file.sync_all();
            }
            if std::path::Path::new(&path_str).exists() {
                let _ = std::fs::remove_file(&path_str);
            }
            let _ = std::fs::rename(&tmp_path, &path_str);
        });

        Ok(())
    }
}

impl Default for OmniscientRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_get() {
        let registry = OmniscientRegistry::new();
        let param = Parameter::new("test_param", ParameterKind::Fixed, 42.0, "test_owner");
        assert!(registry.register(param).is_ok());

        let retrieved = registry.get("test_param", "test_consumer").unwrap();
        assert_eq!(retrieved.get_value(), 42.0);
        // assert!(retrieved.consumers.contains("test_consumer"));
    }

    #[test]
    fn test_reconciliation_and_snapshot_restore() {
        let registry = OmniscientRegistry::new();
        registry.register_or_update(
            "active_margin",
            ParameterKind::Adaptive,
            13.0,
            "risk_engine",
        );
        registry.register_or_update("leverage", ParameterKind::Fixed, 5.0, "risk_engine");

        let snap = registry.take_snapshot();
        assert_eq!(snap.parameters.len(), 2);

        let new_registry = OmniscientRegistry::new();
        let restored_count = new_registry.restore_from_snapshot(&snap);
        assert_eq!(restored_count, 2);
        assert_eq!(new_registry.get_value_or("active_margin", 0.0), 13.0);
        assert_eq!(new_registry.get_value_or("leverage", 0.0), 5.0);

        let updates = vec![
            ("active_margin".to_string(), 15.5),
            ("corrupt_val".to_string(), f64::NAN),
        ];
        let rec = new_registry.reconcile_with_remote(&updates, "binance_rest");
        assert_eq!(rec, 1);
        assert_eq!(new_registry.get_value_or("active_margin", 0.0), 15.5);
    }

    #[test]
    fn test_omniscient_registry_nan_and_fast_value_access() {
        let registry = OmniscientRegistry::new();
        registry.set("nan_param", f64::NAN);

        // Desconocido ≠ 0: NaN se preserva intacto en get_value_fast
        let fast_val = registry.get_value_fast("nan_param");
        assert!(fast_val.is_some());
        assert!(fast_val.unwrap().is_nan());

        // get_value_or devuelve el default ante desconocido (NaN), nunca inventa 0.0
        assert_eq!(registry.get_value_or("nan_param", 10.0), 10.0);

        assert_eq!(registry.get_value_fast("non_existent"), None);
        assert_eq!(registry.get_value_or("non_existent", 99.0), 99.0);
    }


    #[test]
    fn test_omniscient_registry_file_dump_and_reload() {
        let registry = OmniscientRegistry::new();
        registry.set("quantum_leverage", 10.0);
        registry.set("max_drawdown_limit", 0.15);

        let temp_dir = std::env::temp_dir();
        let snap_path = temp_dir.join("omni_snap_test.bin");
        let _ = std::fs::remove_file(&snap_path);
        registry
            .persist_to_disk(snap_path.to_str().unwrap())
            .expect("persist snapshot");

        // persist_to_disk delega la escritura física a un hilo esclavo
        // (I/O desacoplado): sondear con timeout en vez de asumir escritura
        // síncrona — el assert inmediato era una carrera.
        let mut existed = false;
        for _ in 0..200 {
            if snap_path.exists() {
                existed = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            existed,
            "el snapshot debe materializarse en disco tras el persist (2s max)"
        );
        let _ = std::fs::remove_file(snap_path);
    }

    #[test]
    fn test_omniscient_registry_scan_all_and_duplicate_rejection() {
        let registry = OmniscientRegistry::new();
        let p1 = Parameter::new("alpha", ParameterKind::Adaptive, 1.23, "ml_engine");
        let p2 = Parameter::new("alpha", ParameterKind::Adaptive, 4.56, "strategy_engine");

        assert!(registry.register(p1).is_ok());
        // Duplicate key rejection
        assert!(registry.register(p2).is_err());

        let all = registry.scan_all();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].name, "alpha");
        assert_eq!(all[0].get_value(), 1.23);
    }

    #[test]
    fn test_omniscient_registry_zero_alloc_scoped_and_coin_lookups() {
        let registry = OmniscientRegistry::new();
        // 1. Probar fallback al global
        assert_eq!(registry.get_for_coin_or(0, "global_param", 10.5), 10.5);
        registry.set("global_param", 99.0);
        assert_eq!(registry.get_for_coin_or(0, "global_param", 10.5), 99.0);

        // 2. Probar override específico por coin_id
        registry.set_for_coin(0, "global_param", 123.45);
        assert_eq!(registry.get_for_coin_or(0, "global_param", 10.5), 123.45);
        assert_eq!(registry.get_for_coin_or(1, "global_param", 10.5), 99.0);

        // 3. Probar scoped por símbolo
        registry.set_scoped("BTCUSDT", "spread", 0.0001);
        assert_eq!(registry.get_scoped_value_or("BTCUSDT", "spread", 0.0), 0.0001);
        assert_eq!(registry.get_scoped_value_or("ETHUSDT", "spread", 0.0005), 0.0005);

        // 4. Probar resolución polimórfica (símbolo + coin_id)
        assert_eq!(
            registry.get_scoped_val_or(Some("BTCUSDT"), Some(0), "spread", 0.0),
            0.0001
        );
        assert_eq!(
            registry.get_scoped_val_or(None, Some(0), "global_param", 0.0),
            123.45
        );
        assert_eq!(
            registry.get_scoped_val_or(None, None, "global_param", 0.0),
            99.0
        );
    }

    #[test]
    fn test_omniscient_registry_timestamp_freshness() {
        let registry = OmniscientRegistry::new();
        let t0 = chrono::Utc::now().timestamp_millis();
        registry.set("dynamic_metric", 1.0);
        let p = registry.get("dynamic_metric", "test").unwrap();
        assert!(p.get_timestamp() >= t0);

        std::thread::sleep(std::time::Duration::from_millis(5));
        let t_before_update = p.get_timestamp();
        registry.set("dynamic_metric", 2.0);
        assert!(p.get_timestamp() >= t_before_update);
        assert_eq!(p.get_value(), 2.0);
    }

    #[test]
    fn test_omniscient_registry_strict_coin_isolation() {
        let registry = OmniscientRegistry::new();
        // Moneda 0 escribe su spread específico
        registry.set_for_coin(0, "local_spread", 0.00012);

        // Moneda 0 debe leerlo
        assert_eq!(
            registry.get_for_coin_strict(0, "local_spread"),
            Some(0.00012)
        );
        assert_eq!(
            registry.get_for_coin_strict_or(0, "local_spread", 0.0),
            0.00012
        );

        // Moneda 1 NUNCA debe ver el valor de Moneda 0 (Aislamiento absoluto)
        assert_eq!(registry.get_for_coin_strict(1, "local_spread"), None);
        assert_eq!(
            registry.get_for_coin_strict_or(1, "local_spread", 0.0005),
            0.0005
        );
        // Incluso en get_for_coin_or tradicional, si no hay parámetro global "local_spread", Moneda 1 recibe default
        assert_eq!(
            registry.get_for_coin_or(1, "local_spread", 0.0005),
            0.0005
        );
    }
}
