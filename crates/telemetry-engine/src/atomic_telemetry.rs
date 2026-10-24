use crossbeam_queue::ArrayQueue;
use std::fmt;
use std::sync::OnceLock;

pub struct StackBuffer {
    pub buf: [u8; 512],
    pub len: usize,
}

impl fmt::Write for StackBuffer {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let bytes = s.as_bytes();
        let remaining = 512 - self.len;
        let to_copy = std::cmp::min(bytes.len(), remaining);
        self.buf[self.len..self.len + to_copy].copy_from_slice(&bytes[..to_copy]);
        self.len += to_copy;
        Ok(())
    }
}

/// Global Lock-Free Ring Buffer for Binary Telemetry Events
static BINARY_QUEUE: OnceLock<ArrayQueue<([u8; 512], usize)>> = OnceLock::new();

/// RET-51a (censo C-F7b, F155): instala la cola en `celda` si nadie la creó
/// antes. Si ya existía, devuelve su capacidad real en vez de descartar el
/// fallo (un push o un pop previo la crean con 16 384).
fn instalar_cola(
    celda: &OnceLock<ArrayQueue<([u8; 512], usize)>>,
    capacity: usize,
) -> Result<(), usize> {
    // FIX #1447 / D-213: Clamping defensivo de capacidad para no sobrecargar RAM (max 64K)
    let safe_capacity = capacity.clamp(16, 65_536);
    celda
        .set(ArrayQueue::new(safe_capacity))
        .map_err(|_| celda.get().map_or(0, ArrayQueue::capacity))
}

pub(crate) fn init_binary_telemetry(capacity: usize) -> Result<(), usize> {
    instalar_cola(&BINARY_QUEUE, capacity)
}

pub fn push_binary_telemetry(data: &[u8]) {
    let queue = BINARY_QUEUE.get_or_init(|| ArrayQueue::new(16_384));
    let mut fixed_buf = [0u8; 512];
    let len = std::cmp::min(data.len(), 512);
    fixed_buf[..len].copy_from_slice(&data[..len]);
    let _ = queue.force_push((fixed_buf, len)); // O(1) wait-free push with oldest drop on full
}

pub(crate) fn pop_binary_telemetry() -> Option<([u8; 512], usize)> {
    let queue = BINARY_QUEUE.get_or_init(|| ArrayQueue::new(16_384));
    queue.pop()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write;

    #[test]
    fn test_stack_buffer_formatting() {
        let mut buf = StackBuffer {
            buf: [0u8; 512],
            len: 0,
        };
        write!(buf, "PING: {}", 123).unwrap();
        assert_eq!(buf.len, 9);
        assert_eq!(&buf.buf[..9], b"PING: 123");
    }

    #[test]
    fn test_binary_telemetry_push_and_pop() {
        // Ok o Err: otra prueba del proceso pudo crear la cola global antes.
        let _ = init_binary_telemetry(32);
        while pop_binary_telemetry().is_some() {}
        let sample = b"TELEMETRY_SAMPLE_OCTET_STREAM";
        push_binary_telemetry(sample);
        let popped = pop_binary_telemetry();
        assert!(popped.is_some());
        let (buf, len) = popped.unwrap();
        assert_eq!(len, sample.len());
        assert_eq!(&buf[..len], sample);
    }

    /// RET-51a: la primera instalación informa Ok y la segunda devuelve la
    /// capacidad que quedó, sin cambiarla. Sobre un OnceLock local, no el
    /// global, para no chocar con las otras pruebas del proceso.
    #[test]
    fn ret51a_instalar_cola_informa_si_ya_existia() {
        let celda = OnceLock::new();
        assert_eq!(instalar_cola(&celda, 100), Ok(()));
        assert_eq!(celda.get().map(ArrayQueue::capacity), Some(100));
        assert_eq!(instalar_cola(&celda, 5_000), Err(100));
        assert_eq!(celda.get().map(ArrayQueue::capacity), Some(100));

        let pequena = OnceLock::new();
        assert_eq!(instalar_cola(&pequena, 1), Ok(()));
        assert_eq!(pequena.get().map(ArrayQueue::capacity), Some(16));
    }
}
