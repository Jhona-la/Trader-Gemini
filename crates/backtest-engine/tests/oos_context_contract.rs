//! Contrato ejecutable con `rustc --test`: importa la función de producción
//! directamente, sin compilar el workspace ni construir motores o modelos.

#[path = "../src/oos_context.rs"]
mod oos_context;

use oos_context::prepend_is_context;
use oos_context::{validate_is_oos_split, PartitionError};

#[test]
fn empty_tape_has_a_typed_error() {
    assert_eq!(validate_is_oos_split(0, 0), Err(PartitionError::EmptyTape));
}

#[test]
fn one_row_legacy_split_is_rejected_before_indexing() {
    let len = 1;
    let train_len = (len as f64 * 0.7) as usize;
    assert_eq!(validate_is_oos_split(len, train_len), Err(PartitionError::EmptyInSample));
}

#[test]
fn nonempty_tape_cannot_have_an_empty_is_partition() {
    assert_eq!(validate_is_oos_split(10, 0), Err(PartitionError::EmptyInSample));
}

#[test]
fn nonempty_tape_cannot_have_an_empty_oos_partition() {
    assert_eq!(validate_is_oos_split(10, 10), Err(PartitionError::EmptyOutOfSample));
}

#[test]
fn split_beyond_the_tape_is_not_silently_clamped() {
    assert_eq!(validate_is_oos_split(10, 11), Err(PartitionError::SplitBeyondTape));
}

#[test]
fn two_rows_are_structurally_valid_not_certified_for_learning() {
    assert_eq!(validate_is_oos_split(2, 1), Ok(1));
}

#[test]
fn accepted_legacy_splits_preserve_both_partitions_and_safe_indices() {
    for total_len in 2..=10_000 {
        let train_len = (total_len as f64 * 0.7) as usize;
        let split = validate_is_oos_split(total_len, train_len).unwrap();
        assert_eq!(split, train_len);
        assert!(split - 1 < total_len);
        assert!(split < total_len);
        assert!(total_len - split > 0);
    }
}

#[test]
fn maximum_lengths_are_checked_without_overflow() {
    assert_eq!(validate_is_oos_split(usize::MAX, usize::MAX - 1), Ok(usize::MAX - 1));
    assert_eq!(validate_is_oos_split(1, usize::MAX), Err(PartitionError::SplitBeyondTape));
}

fn assert_boundary(is_len: usize, oos_len: usize, expected_start: usize, expected_warmup: usize) {
    let in_sample: Vec<usize> = (0..is_len).collect();
    let out_of_sample: Vec<usize> = (is_len..is_len + oos_len).collect();
    let (ticks, warmup_ticks) = prepend_is_context(&in_sample, &out_of_sample, 50_000);

    assert_eq!(warmup_ticks, expected_warmup);
    assert_eq!(&ticks[..warmup_ticks], &in_sample[expected_start..]);
    assert_eq!(&ticks[warmup_ticks..], out_of_sample.as_slice());
    assert_eq!(ticks[warmup_ticks - 1], is_len - 1, "última fila IS");
    assert_eq!(ticks[warmup_ticks], is_len, "primera fila OOS admisible");
    assert_eq!(
        ticks.len() - warmup_ticks,
        oos_len,
        "ninguna fila OOS se omite"
    );
}

#[test]
fn tape_10k_preserves_all_3k_oos_rows() {
    assert_boundary(7_000, 3_000, 0, 7_000);
}

#[test]
fn tape_60k_preserves_all_18k_oos_rows() {
    assert_boundary(42_000, 18_000, 0, 42_000);
}

#[test]
fn exactly_50k_is_keeps_the_existing_boundary() {
    assert_boundary(50_000, 1_000, 0, 50_000);
}

#[test]
fn longer_is_keeps_only_its_last_50k_rows() {
    assert_boundary(70_000, 30_000, 20_000, 50_000);
}

#[test]
fn future_values_and_length_cannot_change_the_warmup_prefix() {
    let in_sample = [10, 11, 12];
    let (short, short_warmup) = prepend_is_context(&in_sample, &[13], 50_000);
    let (changed_future, changed_warmup) = prepend_is_context(&in_sample, &[999, -1, 42], 50_000);

    assert_eq!(short_warmup, 3);
    assert_eq!(changed_warmup, 3);
    assert_eq!(&short[..short_warmup], &in_sample);
    assert_eq!(&changed_future[..changed_warmup], &in_sample);
    assert_eq!(&short[short_warmup..], &[13]);
    assert_eq!(&changed_future[changed_warmup..], &[999, -1, 42]);
}

#[test]
fn empty_is_never_borrows_warmup_rows_from_oos() {
    let (ticks, warmup_ticks) = prepend_is_context(&[], &[1, 2, 3], 50_000);

    assert_eq!(warmup_ticks, 0);
    assert_eq!(ticks, [1, 2, 3]);
}
