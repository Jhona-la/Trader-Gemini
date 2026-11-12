use flight_recorder::mmap_flight_recorder::MmapFlightRecordRing;

#[test]
fn zero_capacity_ignores_event_and_returns_none() {
    let mut ring = MmapFlightRecordRing::<0>::new();
    ring.record_event(1, 2, 3, 4, 100.0, 1.0);
    assert!(ring.latest_event().is_none());
}
