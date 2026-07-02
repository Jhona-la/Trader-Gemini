use crate::{BaseStrategy, Signal};
use omniscient_registry::{OmniscientRegistry, ParameterKind, Parameter};
use std::sync::Arc;
use uuid::Uuid;

pub struct ScalpingStrategy {
    registry: Option<Arc<OmniscientRegistry>>,
    fast_ma_period_id: Option<Uuid>,
    slow_ma_period_id: Option<Uuid>,
    // Internal state
    fast_ma: f64,
    slow_ma: f64,
}

impl ScalpingStrategy {
    pub fn new() -> Self {
        Self {
            registry: None,
            fast_ma_period_id: None,
            slow_ma_period_id: None,
            fast_ma: 0.0,
            slow_ma: 0.0,
        }
    }
}

impl BaseStrategy for ScalpingStrategy {
    fn initialize(&mut self, registry: Arc<OmniscientRegistry>) {
        let fast_param = Parameter::new("scalping_fast_ma", ParameterKind::Adaptive, 9.0, "ScalpingStrategy");
        let slow_param = Parameter::new("scalping_slow_ma", ParameterKind::Adaptive, 21.0, "ScalpingStrategy");
        
        self.fast_ma_period_id = Some(fast_param.id);
        self.slow_ma_period_id = Some(slow_param.id);
        
        let _ = registry.register(fast_param);
        let _ = registry.register(slow_param);
        
        self.registry = Some(registry);
    }

    fn on_tick(&mut self, price: f64) -> Option<Signal> {
        // Mock EMA calculation for HFT Scalping
        self.fast_ma = (price * 0.2) + (self.fast_ma * 0.8);
        self.slow_ma = (price * 0.1) + (self.slow_ma * 0.9);

        if self.fast_ma > self.slow_ma {
            Some(Signal { direction: 1, confidence: 0.85 })
        } else if self.fast_ma < self.slow_ma {
            Some(Signal { direction: -1, confidence: 0.85 })
        } else {
            None
        }
    }
}
