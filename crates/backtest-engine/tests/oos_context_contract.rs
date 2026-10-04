//! Contrato ejecutable con `rustc --test`: importa la función de producción
//! directamente, sin compilar el workspace ni construir motores o modelos.

#[path = "../src/oos_context.rs"]
mod oos_context;

use oos_context::prepend_is_context;

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
