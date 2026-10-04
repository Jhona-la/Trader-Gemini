//! Preparación del replay OOS sin modelos, reloj ni acceso a disco.

/// Anteponer al OOS únicamente la cola disponible del IS, hasta el máximo
/// solicitado. Devuelve los ticks y la frontera entre calentamiento y OOS.
/// El orden y el contenido del OOS se conservan íntegros.
pub fn prepend_is_context<T: Clone>(
    in_sample: &[T],
    out_of_sample: &[T],
    max_context_ticks: usize,
) -> (Vec<T>, usize) {
    let start = in_sample.len().saturating_sub(max_context_ticks);
    let mut ticks = in_sample[start..].to_vec();
    // Medir el prefijo antes de añadir el OOS: ninguna fila futura calienta.
    let warmup_ticks = ticks.len();
    ticks.extend_from_slice(out_of_sample);
    (ticks, warmup_ticks)
}
