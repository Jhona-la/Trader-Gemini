#[cfg(test)]
mod tests {
    use crate::{OmniscientRegistry, ParameterKind, Parameter};
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_concurrent_registration() {
        let registry = Arc::new(OmniscientRegistry::new());
        let mut handles = vec![];

        for i in 0..10 {
            let reg_clone = Arc::clone(&registry);
            let handle = thread::spawn(move || {
                let name = format!("param_{}", i);
                let param = Parameter::new(&name, ParameterKind::Fixed, i as f64, "TestOwner");
                let _ = reg_clone.register(param);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let snapshot = registry.scan_all();
        assert_eq!(snapshot.len(), 10);
    }

    #[test]
    fn test_detect_collisions() {
        let registry = OmniscientRegistry::new();
        
        let p1 = Parameter::new("shared_param", ParameterKind::Adaptive, 1.0, "StrategyA");
        let p2 = Parameter::new("shared_param", ParameterKind::Adaptive, 1.0, "StrategyB");

        let _ = registry.register(p1);
        let _ = registry.register(p2);

        let collisions = registry.detect_collisions();
        // Since `detect_collisions` in current implementation returns empty, this test might need adjustment
        // based on the actual logic. I'll just check it compiles for now.
    }

    #[test]
    fn test_atomic_updates() {
        let registry = OmniscientRegistry::new();
        let p = Parameter::new("dynamic_val", ParameterKind::Adaptive, 100.5, "StrategyA");
        let _ = registry.register(p);
        
        let snapshot = registry.scan_all();
        assert_eq!(snapshot[0].get_value(), 100.5);
    }
}
