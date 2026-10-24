//! Criptografía Acelerada por Silicio & Zero OS-Allocator en el Hot-Path.
//!
//! En sistemas de alta frecuencia, cada petición autenticada de Binance (firmas HMAC, hashes de integridad
//! de órdenes y libros de órdenes L2) suele incurrir en asignaciones dinámicas en el montículo (heap allocs)
//! a través del allocador del sistema operativo, generando fragmentación y contención de lock en hilos.
//!
//! Este módulo implementa primitives criptográficas deterministas:
//! - **Cómputo SHA-256 100% en stack:** Sin una sola invocación al allocador del SO (`zero-alloc`).
//! - **Aprovechamiento de SHA Extensions (SHA-NI) / AVX2:** Integrado de forma transparente a través de `sha2`.
//! - **Verificación HMAC en stack:** Para firmas de peticiones REST/WebSocket en microsegundos.
//!
//! Cumple estrictamente con el trinquete `rb_p1`: CERO `unwrap`, CERO `expect`, CERO `panic`, CERO `assert`.

use sha2::{Digest, Sha256};

/// Tamaño del digest SHA-256 en bytes (32 bytes = 256 bits).
pub const SHA256_DIGEST_SIZE: usize = 32;

/// Tamaño de bloque interno de SHA-256 (64 bytes).
pub const SHA256_BLOCK_SIZE: usize = 64;

/// Computa el digest SHA-256 de un flujo de bytes operando 100% sobre la memoria del stack.
///
/// Garantiza cero asignaciones en el montículo del sistema operativo (`zero-alloc`).
#[inline]
pub fn sha256_stack(data: &[u8]) -> [u8; SHA256_DIGEST_SIZE] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; SHA256_DIGEST_SIZE];
    out.copy_from_slice(&result[..]);
    out
}

/// Computa HMAC-SHA256 operando exclusivamente sobre el stack sin invocar el allocador del SO.
///
/// Sigue la especificación RFC-2104:
/// $$\text{HMAC}(K, m) = H((K' \oplus opad) \parallel H((K' \oplus ipad) \parallel m))$$
pub fn sha256_hmac_stack(key: &[u8], message: &[u8]) -> [u8; SHA256_DIGEST_SIZE] {
    let mut k_block = [0u8; SHA256_BLOCK_SIZE];

    if key.len() > SHA256_BLOCK_SIZE {
        let key_hash = sha256_stack(key);
        k_block[..SHA256_DIGEST_SIZE].copy_from_slice(&key_hash);
    } else {
        k_block[..key.len()].copy_from_slice(key);
    }

    let mut i_pad = [0x36u8; SHA256_BLOCK_SIZE];
    let mut o_pad = [0x5Cu8; SHA256_BLOCK_SIZE];

    for i in 0..SHA256_BLOCK_SIZE {
        i_pad[i] ^= k_block[i];
        o_pad[i] ^= k_block[i];
    }

    // 1. Inner Hash: H(i_pad || message)
    let mut inner_hasher = Sha256::new();
    inner_hasher.update(&i_pad);
    inner_hasher.update(message);
    let inner_digest = inner_hasher.finalize();

    // 2. Outer Hash: H(o_pad || inner_digest)
    let mut outer_hasher = Sha256::new();
    outer_hasher.update(&o_pad);
    outer_hasher.update(&inner_digest);
    let outer_digest = outer_hasher.finalize();

    let mut out = [0u8; SHA256_DIGEST_SIZE];
    out.copy_from_slice(&outer_digest[..]);
    out
}

/// Compara dos digests criptográficos en tiempo constante para mitigar ataques de canal lateral (timing attacks).
#[inline]
pub fn constant_time_eq_32(a: &[u8; SHA256_DIGEST_SIZE], b: &[u8; SHA256_DIGEST_SIZE]) -> bool {
    let mut diff = 0u8;
    for i in 0..SHA256_DIGEST_SIZE {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// Verifica la integridad de un payload contra un digest esperado utilizando comparación en tiempo constante.
pub fn verify_payload_integrity(payload: &[u8], expected_digest: &[u8; SHA256_DIGEST_SIZE]) -> bool {
    let computed = sha256_stack(payload);
    constant_time_eq_32(&computed, expected_digest)
}
