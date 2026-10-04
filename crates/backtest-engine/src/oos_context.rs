//! Preparación del replay OOS sin modelos, reloj ni acceso a disco.

/// Error estructural de particionado, no juicio sobre suficiencia económica.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionError {
    EmptyTape,
    EmptyInSample,
    EmptyOutOfSample,
    SplitBeyondTape,
}

/// Comprueba la frontera elegida por el caller, sin cambiar su proporción.
/// El éxito garantiza 0 < train_len < total_len: existen ambos segmentos y
/// los índices train_len - 1 y train_len son válidos. No certifica soporte
/// estadístico, orden temporal, calidad del tape ni suficiencia de calentamiento.
pub fn validate_is_oos_split(total_len: usize, train_len: usize) -> Result<usize, PartitionError> {
    if total_len == 0 {
        return Err(PartitionError::EmptyTape);
    }
    if train_len > total_len {
        return Err(PartitionError::SplitBeyondTape);
    }
    if train_len == 0 {
        return Err(PartitionError::EmptyInSample);
    }
    if train_len == total_len {
        return Err(PartitionError::EmptyOutOfSample);
    }
    Ok(train_len)
}

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
