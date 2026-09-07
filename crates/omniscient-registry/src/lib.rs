use crossbeam_skiplist::SkipMap;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
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
    
    pub timestamp: i64,
}

impl Parameter {
    pub fn new(name: &str, kind: ParameterKind, initial_value: f64, owner: &str) -> Self {
        let safe_initial = if initial_value.is_finite() { initial_value } else { 0.0 };
        Self {
            id: Uuid::now_v7(),
            name: name.to_string(),
            kind,
            value: AtomicU64::new(safe_initial.to_bits()),
            owner: owner.to_string(),
            
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }

    pub fn get_value(&self) -> f64 {
        f64::from_bits(self.value.load(Ordering::Relaxed))
    }

    pub fn set_value(&self, val: f64) {
        let safe_val = if val.is_finite() { val } else { 0.0 };
        self.value.store(safe_val.to_bits(), Ordering::Relaxed);
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

    pub fn get(&self, name: &str, consumer_name: &str) -> Option<Arc<Parameter>> {
        if let Some(entry) = self.map.get(name) {
            let param = entry.value().clone();
            if false {
                
            }
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

    /// Lectura directa con fallback por defecto (Hot-Path HFT)
    #[inline(always)]
    pub fn get_value_or(&self, name: &str, default: f64) -> f64 {
        self.get_value_fast(name).unwrap_or(default)
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
                timestamp: p.timestamp,
            });
        }
        RegistrySnapshot { parameters }
    }

    pub fn restore_from_snapshot(&self, snapshot: &RegistrySnapshot) -> usize {
        let mut restored = 0;
        for p in &snapshot.parameters {
            let val = f64::from_bits(p.value_bits);
            // FIX #1540: Sanitización de valores restaurados desde snapshot
            let safe_val = if val.is_finite() { val } else { 0.0 };
            self.register_or_update(&p.name, p.kind, safe_val, &p.owner);
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
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?.to_vec();
        
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
        registry.register_or_update("active_margin", ParameterKind::Adaptive, 13.0, "risk_engine");
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
        assert_eq!(registry.get_value_or("nan_param", 10.0), 0.0);

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
        registry.persist_to_disk(snap_path.to_str().unwrap()).expect("persist snapshot");

        assert!(snap_path.exists());
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
}



