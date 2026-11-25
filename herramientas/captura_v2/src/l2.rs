//! Arbitraje de una pareja de sockets para un único flujo de un símbolo.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    mem::size_of,
    time::{Duration, Instant},
};

pub(crate) const DEPTH_WAIT: Duration = Duration::from_millis(250);
pub(crate) const DEPTH_ROWS: usize = 128;
pub(crate) const DEPTH_BYTES: usize = 1024 * 1024;
pub(crate) const TICKER_WAIT: Duration = Duration::from_millis(75);
pub(crate) const TICKER_ROWS: usize = 128;
pub(crate) const TICKER_BYTES: usize = 256 * 1024;
pub(crate) const TICKER_HISTORY: usize = 8192;

const MISSING_PREDECESSOR: &str = "depth predecessor deadline exceeded";
const DEPTH_CAPACITY: &str = "depth reorder buffer capacity exhausted";
const STOPPED_WITH_GAP: &str = "depth predecessor missing at shutdown";
const LOST_BOTH: &str = "both L2 connections lost";
const PRE_OUTAGE_GAP: &str = "depth predecessor missing when both L2 connections failed";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DepthId {
    pub first: u64,
    pub last: u64,
    pub previous: u64,
}

impl DepthId {
    fn order(self) -> (u64, u64, u64) {
        (self.last, self.previous, self.first)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DepthGap {
    pub expected: u64,
    pub reason: &'static str,
}

#[derive(Debug)]
pub(crate) struct DepthOutput<T> {
    pub id: DepthId,
    pub frame: T,
    pub gap: Option<DepthGap>,
}

struct Held<T> {
    frame: T,
    since: Instant,
    cost: usize,
}

struct DepthPending<T> {
    id: DepthId,
    held: Held<T>,
}

pub(crate) struct Depth<T> {
    last: Option<u64>,
    pending: Vec<DepthPending<T>>,
    bytes: usize,
    lost_both: bool,
}

impl<T> Default for Depth<T> {
    fn default() -> Self {
        Self {
            last: None,
            pending: Vec::new(),
            bytes: 0,
            lost_both: false,
        }
    }
}

impl<T> Depth<T> {
    pub(crate) fn offer(
        &mut self,
        id: DepthId,
        frame: T,
        cost: usize,
        now: Instant,
    ) -> Vec<DepthOutput<T>> {
        let mut output = self.expire(now);
        if self.is_stale(id) || self.pending.iter().any(|pending| pending.id == id) {
            return output;
        }
        let incoming = DepthPending {
            id,
            held: Held {
                frame,
                since: now,
                cost: cost.saturating_add(size_of::<DepthPending<T>>()),
            },
        };

        loop {
            if self.lost_both {
                self.lost_both = false;
                self.commit(incoming, Some(LOST_BOTH), &mut output);
                break;
            }
            if self.last.is_none_or(|last| id.previous == last) {
                self.commit(incoming, None, &mut output);
                break;
            }
            if self.pending.len() < DEPTH_ROWS && incoming.held.cost <= DEPTH_BYTES - self.bytes {
                self.bytes += incoming.held.cost;
                self.pending.push(incoming);
                break;
            }

            // El límite obliga a decidir. Elegir primero el menor u permite conservar
            // los sucesores que ya están en el búfer, además de declarar el salto.
            let first = self.first_matching(|_| true);
            let Some(index) = first.filter(|&index| self.pending[index].id.order() < id.order())
            else {
                self.commit(incoming, Some(DEPTH_CAPACITY), &mut output);
                break;
            };
            let pending = self.remove(index);
            self.commit(pending, Some(DEPTH_CAPACITY), &mut output);
            if self.is_stale(id) {
                break;
            }
        }
        output
    }

    pub(crate) fn expire(&mut self, now: Instant) -> Vec<DepthOutput<T>> {
        let mut output = Vec::new();
        while self
            .pending
            .iter()
            .any(|p| now.saturating_duration_since(p.held.since) >= DEPTH_WAIT)
        {
            self.reanchor(MISSING_PREDECESSOR, &mut output);
        }
        output
    }

    /// SALIDAFALLO-12 (C3): `u` de las tramas retenidas (solo lectura, testigo del ensayo P8).
    pub(crate) fn pendientes(&self) -> Vec<u64> {
        self.pending.iter().map(|p| p.id.last).collect()
    }

    pub(crate) fn finish(&mut self) -> Vec<DepthOutput<T>> {
        self.drain_pending(STOPPED_WITH_GAP)
    }

    fn drain_pending(&mut self, reason: &'static str) -> Vec<DepthOutput<T>> {
        let mut output = Vec::new();
        while !self.pending.is_empty() {
            self.reanchor(reason, &mut output);
        }
        output
    }

    /// Conserva las tramas anteriores al corte y separa su generación de la primera trama nueva.
    pub(crate) fn force_gap_on_next(&mut self) -> Vec<DepthOutput<T>> {
        let output = self.drain_pending(PRE_OUTAGE_GAP);
        self.lost_both = self.last.is_some();
        output
    }

    fn is_stale(&self, id: DepthId) -> bool {
        self.last.is_some_and(|last| id.last <= last)
    }

    fn first_matching(&self, filter: impl Fn(&DepthPending<T>) -> bool) -> Option<usize> {
        self.pending
            .iter()
            .enumerate()
            .filter(|(_, p)| filter(p))
            .min_by_key(|(_, p)| p.id.order())
            .map(|(index, _)| index)
    }

    fn remove(&mut self, index: usize) -> DepthPending<T> {
        let p = self.pending.swap_remove(index);
        self.bytes -= p.held.cost;
        p
    }

    fn reanchor(&mut self, reason: &'static str, output: &mut Vec<DepthOutput<T>>) {
        if let Some(index) = self.first_matching(|_| true) {
            let pending = self.remove(index);
            self.commit(pending, Some(reason), output);
        }
    }

    fn commit(
        &mut self,
        first: DepthPending<T>,
        reason: Option<&'static str>,
        output: &mut Vec<DepthOutput<T>>,
    ) {
        let mut next = Some((first, reason));
        while let Some((item, reason)) = next {
            let gap = reason
                .zip(self.last)
                .map(|(reason, expected)| DepthGap { expected, reason });
            let last = item.id.last;
            self.last = Some(last);
            output.push(DepthOutput {
                id: item.id,
                frame: item.held.frame,
                gap,
            });
            self.pending.retain(|p| p.id.last > last);
            self.bytes = self.pending.iter().map(|p| p.held.cost).sum();
            next = self
                .first_matching(|p| p.id.previous == last)
                .map(|index| (self.remove(index), None));
        }
    }
}

pub(crate) struct TickerOffer<T> {
    pub ready: Vec<T>,
    pub retired: Option<T>,
}

pub(crate) struct Ticker<T> {
    pending: BTreeMap<u64, Held<T>>,
    bytes: usize,
    recent: BTreeSet<u64>,
    emitted: VecDeque<u64>,
    retired_max: Option<u64>,
}

impl<T> Default for Ticker<T> {
    fn default() -> Self {
        Self {
            pending: BTreeMap::new(),
            bytes: 0,
            recent: BTreeSet::new(),
            emitted: VecDeque::new(),
            retired_max: None,
        }
    }
}

impl<T> Ticker<T> {
    pub(crate) fn offer(&mut self, id: u64, frame: T, cost: usize, now: Instant) -> TickerOffer<T> {
        let mut output = TickerOffer {
            ready: self.expire(now),
            retired: None,
        };
        if self.recent.contains(&id) || self.pending.contains_key(&id) {
            return output;
        }
        // Detrás del límite hay IDs ya retirados y posibles IDs únicos tardíos.
        // Conservar el payload para diagnóstico fuera de la secuencia canónica.
        if self.retired_max.is_some_and(|limit| id <= limit) {
            output.retired = Some(frame);
            return output;
        }
        let cost = cost.saturating_add(size_of::<Held<T>>() + size_of::<u64>());
        while self.pending.len() >= TICKER_ROWS || cost > TICKER_BYTES - self.bytes {
            if self
                .pending
                .first_key_value()
                .is_none_or(|(&smallest, _)| id < smallest)
            {
                self.emit(id, frame, &mut output.ready);
                return output;
            }
            self.emit_first(&mut output.ready);
        }
        self.bytes += cost;
        self.pending.insert(
            id,
            Held {
                frame,
                since: now,
                cost,
            },
        );
        output
    }

    pub(crate) fn expire(&mut self, now: Instant) -> Vec<T> {
        let due = self
            .pending
            .iter()
            .rev()
            .find(|(_, p)| now.saturating_duration_since(p.since) >= TICKER_WAIT)
            .map(|(&id, _)| id);
        let mut output = Vec::new();
        while let (Some(last_due), Some((&first, _))) = (due, self.pending.first_key_value()) {
            if first > last_due {
                break;
            }
            self.emit_first(&mut output);
        }
        output
    }

    /// SALIDAFALLO-12 (C3): IDs retenidos (solo lectura, testigo del ensayo P8).
    pub(crate) fn pendientes(&self) -> Vec<u64> {
        self.pending.keys().copied().collect()
    }

    pub(crate) fn flush(&mut self) -> Vec<T> {
        let mut output = Vec::new();
        while !self.pending.is_empty() {
            self.emit_first(&mut output);
        }
        output
    }

    fn emit_first(&mut self, output: &mut Vec<T>) {
        if let Some((id, p)) = self.pending.pop_first() {
            self.bytes -= p.cost;
            self.emit(id, p.frame, output);
        }
    }

    fn emit(&mut self, id: u64, frame: T, output: &mut Vec<T>) {
        if self.emitted.len() == TICKER_HISTORY
            && let Some(old) = self.emitted.pop_front()
        {
            self.recent.remove(&old);
            self.retired_max = Some(self.retired_max.map_or(old, |limit| limit.max(old)));
        }
        self.recent.insert(id);
        self.emitted.push_back(id);
        output.push(frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(last: u64, previous: u64) -> DepthId {
        DepthId {
            first: last,
            last,
            previous,
        }
    }

    #[test]
    fn depth_emits_each_tuple_once_after_its_predecessor() {
        let now = Instant::now();
        let mut depth = Depth::default();
        assert_eq!(depth.offer(id(100, 99), "seed", 4, now)[0].frame, "seed");
        assert!(depth.offer(id(103, 101), "ahead", 5, now).is_empty());
        assert!(depth.offer(id(103, 101), "mirror", 6, now).is_empty());
        let output = depth.offer(id(101, 100), "predecessor", 11, now);
        assert_eq!(
            output.iter().map(|x| x.frame).collect::<Vec<_>>(),
            ["predecessor", "ahead"]
        );
        assert!(output.iter().all(|x| x.gap.is_none()));
        assert!(depth.offer(id(103, 101), "again", 5, now).is_empty());
    }

    #[test]
    fn depth_due_uses_smallest_forward_id_and_keeps_its_successors() {
        let now = Instant::now();
        let mut depth = Depth::default();
        depth.offer(id(100, 99), 100, 0, now);
        depth.offer(id(110, 105), 110, 0, now);
        depth.offer(id(105, 104), 105, 0, now + Duration::from_millis(10));
        assert!(
            depth
                .expire(now + DEPTH_WAIT - Duration::from_millis(1))
                .is_empty()
        );
        let output = depth.expire(now + DEPTH_WAIT);
        assert_eq!(
            output.iter().map(|x| x.id.last).collect::<Vec<_>>(),
            [105, 110]
        );
        assert_eq!(
            output[0].gap,
            Some(DepthGap {
                expected: 100,
                reason: MISSING_PREDECESSOR
            })
        );
        assert!(output[1].gap.is_none());
        assert!(
            depth
                .offer(id(104, 100), 104, 0, now + DEPTH_WAIT)
                .is_empty()
        );
    }

    #[test]
    fn depth_count_and_bytes_cap_are_explicit() {
        let now = Instant::now();
        let mut depth = Depth::default();
        depth.offer(id(100, 99), 100, 0, now);
        for n in 102..102 + DEPTH_ROWS as u64 {
            assert!(depth.offer(id(n, n - 1), n, 0, now).is_empty());
        }
        let n = 102 + DEPTH_ROWS as u64;
        let full = depth.offer(id(n, n - 1), n, 0, now);
        assert_eq!(full.len(), DEPTH_ROWS + 1);
        assert_eq!(full[0].gap.as_ref().map(|g| g.reason), Some(DEPTH_CAPACITY));
        assert!(full[1..].iter().all(|x| x.gap.is_none()));
        let oversized = depth.offer(id(n + 2, n + 1), n + 2, DEPTH_BYTES, now);
        assert_eq!(
            oversized[0].gap.as_ref().map(|g| g.reason),
            Some(DEPTH_CAPACITY)
        );
        assert_eq!(depth.bytes, 0);
    }

    #[test]
    fn depth_finishes_old_frames_and_marks_dual_loss_on_fresh_id() {
        let now = Instant::now();
        let mut depth = Depth::default();
        depth.offer(id(100, 99), 100, 0, now);
        depth.offer(id(105, 104), 105, 0, now);
        let prior = depth.force_gap_on_next();
        assert_eq!(
            prior[0].gap.as_ref().map(|g| g.reason),
            Some(PRE_OUTAGE_GAP)
        );
        assert!(depth.offer(id(105, 104), 105, 0, now).is_empty());
        let resumed = depth.offer(id(106, 105), 106, 0, now);
        assert_eq!(
            resumed[0].gap,
            Some(DepthGap {
                expected: 105,
                reason: LOST_BOTH
            })
        );
        assert!(depth.offer(id(107, 106), 107, 0, now)[0].gap.is_none());
    }

    #[test]
    fn ticker_orders_buffered_ids_and_preserves_unique_late_ids() {
        let now = Instant::now();
        let mut ticker = Ticker::default();
        assert!(ticker.offer(108, "high", 4, now).ready.is_empty());
        let mirror = ticker.offer(108, "mirror", 6, now);
        assert!(mirror.ready.is_empty() && mirror.retired.is_none());
        assert!(
            ticker
                .offer(102, "low", 3, now + Duration::from_millis(30))
                .ready
                .is_empty()
        );
        assert_eq!(ticker.expire(now + TICKER_WAIT), ["low", "high"]);
        let mirror = ticker.offer(102, "low mirror", 10, now + TICKER_WAIT);
        assert!(mirror.ready.is_empty() && mirror.retired.is_none());
        let late = ticker.offer(105, "late unique", 11, now + TICKER_WAIT);
        assert!(late.ready.is_empty() && late.retired.is_none());
        assert_eq!(ticker.flush(), ["late unique"]);
    }

    #[test]
    fn ticker_buffer_capacity_preserves_order_and_bounds_history() {
        let now = Instant::now();
        let mut ticker = Ticker::default();
        for n in 1..=TICKER_ROWS as u64 {
            assert!(ticker.offer(n, n, 0, now).ready.is_empty());
        }
        assert_eq!(ticker.offer(0, 0, TICKER_BYTES, now).ready, [0]);
        assert_eq!(
            ticker.offer(200, 200, TICKER_BYTES, now).ready,
            (1..=TICKER_ROWS as u64).chain([200]).collect::<Vec<_>>(),
        );
        assert_eq!(ticker.bytes, 0);
        for n in 201..=(TICKER_HISTORY as u64 + 201) {
            ticker.offer(n, n, TICKER_BYTES, now);
        }
        assert_eq!(ticker.recent.len(), TICKER_HISTORY);
        let mirror = ticker.offer(202, 202, TICKER_BYTES, now);
        assert!(mirror.ready.is_empty() && mirror.retired.is_none());
    }

    #[test]
    fn ticker_retired_ids_stay_out_of_canonical_frames() {
        let now = Instant::now();
        let mut ticker = Ticker::default();
        let base = 10_000;
        for id in (base..base + 2 * TICKER_HISTORY as u64).step_by(2) {
            assert_eq!(ticker.offer(id, id, TICKER_BYTES, now).ready, [id]);
        }
        assert!(ticker.offer(5, 5, 0, now).ready.is_empty());
        assert_eq!(ticker.offer(4, 4, TICKER_BYTES, now).ready, [4]);
        assert_eq!(ticker.retired_max, Some(base));

        for duplicate in [5, base + 2] {
            let output = ticker.offer(duplicate, 999, 0, now);
            assert!(output.ready.is_empty() && output.retired.is_none());
        }
        for at_or_below_frontier in [base, 6] {
            let output = ticker.offer(at_or_below_frontier, 999, 0, now);
            assert!(output.ready.is_empty());
            assert_eq!(output.retired, Some(999));
        }
        let open_id = ticker.offer(base + 1, base + 1, 0, now);
        assert!(open_id.ready.is_empty() && open_id.retired.is_none());
        assert_eq!(ticker.flush(), [5, base + 1]);
    }
}
