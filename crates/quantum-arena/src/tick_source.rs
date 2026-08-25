#[derive(Debug, Clone, Copy)]
pub struct TickEvent {
    pub coin_id: usize,
    pub timestamp: u64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_qty: f64,
    pub ask_qty: f64,
}

/// Axioma VII: Unificación Backtest = Producción (El Espejo Perfecto)
/// Este trait permite inyectar ticks al motor sin importar su origen.
pub trait TickSource: Send + Sync {
    #[allow(async_fn_in_trait)]
    async fn next_tick(&mut self) -> Option<TickEvent>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    struct MockTickSource {
        ticks: Vec<TickEvent>,
        idx: usize,
    }

    impl TickSource for MockTickSource {
        async fn next_tick(&mut self) -> Option<TickEvent> {
            if self.idx < self.ticks.len() {
                let t = self.ticks[self.idx];
                self.idx += 1;
                Some(t)
            } else {
                None
            }
        }
    }

    fn dummy_waker() -> Waker {
        unsafe fn clone(_: *const ()) -> RawWaker { RawWaker::new(std::ptr::null(), &VTABLE) }
        unsafe fn wake(_: *const ()) {}
        unsafe fn wake_by_ref(_: *const ()) {}
        unsafe fn drop(_: *const ()) {}
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop);
        unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
    }

    fn block_on<F: Future>(mut fut: F) -> F::Output {
        let waker = dummy_waker();
        let mut cx = Context::from_waker(&waker);
        let mut fut = unsafe { Pin::new_unchecked(&mut fut) };
        loop {
            match fut.as_mut().poll(&mut cx) {
                Poll::Ready(res) => return res,
                Poll::Pending => {}
            }
        }
    }

    #[test]
    fn test_mock_tick_source() {
        let mut src = MockTickSource {
            ticks: vec![
                TickEvent {
                    coin_id: 0,
                    timestamp: 1000,
                    bid_price: 60000.0,
                    ask_price: 60001.0,
                    bid_qty: 1.0,
                    ask_qty: 1.0,
                },
            ],
            idx: 0,
        };

        let tick = block_on(src.next_tick());
        assert!(tick.is_some());
        assert_eq!(tick.unwrap().bid_price, 60000.0);
        assert!(block_on(src.next_tick()).is_none());
    }
}
