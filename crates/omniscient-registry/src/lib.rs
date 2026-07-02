use crossbeam_skiplist::SkipMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use rkyv::{Archive, Serialize as RkyvSerialize, Deserialize as RkyvDeserialize};
use std::fs::File;
use std::io::{Write, Read};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
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
        Self {
            id: Uuid::now_v7(),
            name: name.to_string(),
            kind,
            value: AtomicU64::new(initial_value.to_bits()),
            owner: owner.to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }

    pub fn get_value(&self) -> f64 {
        f64::from_bits(self.value.load(Ordering::Relaxed))
    }

    pub fn set_value(&self, val: f64) {
        self.value.store(val.to_bits(), Ordering::Relaxed);
    }
}

pub struct OmniscientRegistry {
    // Usamos SkipMap para concurrencia lock-free verdadera
    map: SkipMap<String, Arc<Parameter>>,
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

    pub fn get(&self, name: &str) -> Option<Arc<Parameter>> {
        self.map.get(name).map(|entry| entry.value().clone())
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

    pub fn persist_to_disk(&self, path: &str) -> std::io::Result<()> {
        let snapshot = self.take_snapshot();
        let bytes = rkyv::to_bytes::<_, 256>(&snapshot).unwrap();
        let mut file = File::create(path)?;
        file.write_all(&bytes)?;
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
        
        let retrieved = registry.get("test_param").unwrap();
        assert_eq!(retrieved.get_value(), 42.0);
    }
}
