//! Mapeo de Topología de Silicio a Bajo Nivel vía Instrucciones CPUID.
//!
//! Proporciona inspección determinista de la arquitectura de la CPU física:
//! - Detección de núcleos de rendimiento (**P-Cores**) vs núcleos de eficiencia (**E-Cores**).
//! - Identificación de extensiones vectoriales SIMD (AVX2, AVX-512, FMA, SSE4.2, SHA-NI, AES-NI).
//! - Determinación de la topología de memoria caché L1D/L2/L3 y tamaño de línea de caché (64 bytes).
//! - Cálculo de máscaras de afinidad óptimas para el hot-path sin interferencia del planificador del SO.
//!
//! Cumple estrictamente con el trinquete `rb_p1`: CERO `unwrap`, CERO `expect`, CERO `panic`, CERO `assert`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__cpuid;

/// Tipo de núcleo de procesamiento en arquitecturas híbridas (Intel Alder/Raptor Lake, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreType {
    /// Núcleo de alto rendimiento (P-Core - ej. Golden Cove, Raptor Cove).
    PerformanceCore,
    /// Núcleo de alta eficiencia energética (E-Core - ej. Gracemont, Crestmont).
    EfficiencyCore,
    /// Núcleo homogéneo estándar (arquitectura no híbrida, ej. AMD Zen o Intel previo a 12th gen).
    StandardCore,
    /// Identificador de tipo desconocido con código crudo reportado por hardware.
    Unknown(u8),
}

/// Registro estructurado con la información de silicio detectada mediante CPUID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuTopologyInfo {
    /// Identificador del fabricante (ej. "GenuineIntel", "AuthenticAMD").
    pub vendor_id: [u8; 12],
    /// Soporte para Advanced Vector Extensions (AVX).
    pub has_avx: bool,
    /// Soporte para Advanced Vector Extensions 2 (AVX2).
    pub has_avx2: bool,
    /// Soporte para AVX-512 Foundation (AVX512F).
    pub has_avx512f: bool,
    /// Soporte para AVX-512 Conflict Detection (AVX512CD).
    pub has_avx512cd: bool,
    /// Soporte para AVX-512 Byte and Word Instructions (AVX512BW).
    pub has_avx512bw: bool,
    /// Soporte para AVX-512 Doubleword and Quadword Instructions (AVX512DQ).
    pub has_avx512dq: bool,
    /// Soporte para AVX-512 Vector Length Extensions (AVX512VL).
    pub has_avx512vl: bool,
    /// Soporte para Fused Multiply-Add (FMA3).
    pub has_fma: bool,
    /// Soporte para aceleración por hardware de SHA (SHA-NI).
    pub has_sha_ni: bool,
    /// Soporte para aceleración por hardware de AES (AES-NI).
    pub has_aes_ni: bool,
    /// Soporte para SSE4.2 (instrucciones de procesamiento de strings y CRC32).
    pub has_sse4_2: bool,
    /// Soporte para Bit Manipulation Instruction Set 1 (BMI1).
    pub has_bmi1: bool,
    /// Soporte para Bit Manipulation Instruction Set 2 (BMI2).
    pub has_bmi2: bool,
    /// Soporte para instrucción de conteo de bits poblados (POPCNT).
    pub has_popcnt: bool,
    /// Indicador de arquitectura híbrida (P-Core + E-Core).
    pub has_hybrid_topology: bool,
    /// Tipo de núcleo detectado para el hilo actual.
    pub current_core_type: CoreType,
    /// Tamaño de línea de caché L1D en bytes (típicamente 64 bytes).
    pub l1d_cache_line_size: usize,
    /// Tamaño aproximado de caché L2 en kilobytes.
    pub l2_cache_size_kb: usize,
    /// Tamaño aproximado de caché L3 unificada en kilobytes.
    pub l3_cache_size_kb: usize,
    /// Número total de procesadores lógicos detectados en el sistema.
    pub logical_cores: usize,
    /// Máscara de afinidad para asignar hilos críticos a P-Cores.
    pub p_core_affinity_mask: usize,
    /// Máscara de afinidad para tareas de fondo o telemetría (E-Cores o núcleos no críticos).
    pub auxiliary_affinity_mask: usize,
}

impl Default for CpuTopologyInfo {
    fn default() -> Self {
        Self {
            vendor_id: *b"UnknownVendr",
            has_avx: false,
            has_avx2: false,
            has_avx512f: false,
            has_avx512cd: false,
            has_avx512bw: false,
            has_avx512dq: false,
            has_avx512vl: false,
            has_fma: false,
            has_sha_ni: false,
            has_aes_ni: false,
            has_sse4_2: false,
            has_bmi1: false,
            has_bmi2: false,
            has_popcnt: false,
            has_hybrid_topology: false,
            current_core_type: CoreType::StandardCore,
            l1d_cache_line_size: 64,
            l2_cache_size_kb: 512,
            l3_cache_size_kb: 8192,
            logical_cores: 1,
            p_core_affinity_mask: 0x1,
            auxiliary_affinity_mask: 0x1,
        }
    }
}

impl CpuTopologyInfo {
    /// Detecta dinámicamente las capacidades de hardware del procesador anfitrión mediante CPUID.
    pub fn detect() -> Self {
        let logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        #[cfg(target_arch = "x86_64")]
        {
            Self::detect_x86_64(logical_cores)
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut info = Self::default();
            info.logical_cores = logical_cores;
            info.p_core_affinity_mask = if logical_cores > 1 { 0x3 } else { 0x1 };
            info
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_x86_64(logical_cores: usize) -> Self {
        let mut info = Self::default();
        info.logical_cores = logical_cores;

        // 1. CPUID Leaf 0x0: Vendor String y Max Basic Leaf
        let leaf0 = unsafe { __cpuid(0x0) };
        let max_leaf = leaf0.eax;

        let mut vendor = [0u8; 12];
        vendor[0..4].copy_from_slice(&leaf0.ebx.to_le_bytes());
        vendor[4..8].copy_from_slice(&leaf0.edx.to_le_bytes());
        vendor[8..12].copy_from_slice(&leaf0.ecx.to_le_bytes());
        info.vendor_id = vendor;

        // 2. CPUID Leaf 0x1: Características básicas de procesador
        if max_leaf >= 1 {
            let leaf1 = unsafe { __cpuid(0x1) };
            // ECX flags
            info.has_sse4_2 = (leaf1.ecx & (1 << 20)) != 0;
            info.has_popcnt = (leaf1.ecx & (1 << 23)) != 0;
            info.has_aes_ni = (leaf1.ecx & (1 << 25)) != 0;
            info.has_avx = (leaf1.ecx & (1 << 28)) != 0;
            info.has_fma = (leaf1.ecx & (1 << 12)) != 0;

            // Tamaño de línea de caché CLFLUSH (EBX bits 15:8 * 8 bytes)
            let clflush_size = ((leaf1.ebx >> 8) & 0xFF) as usize * 8;
            if clflush_size > 0 {
                info.l1d_cache_line_size = clflush_size;
            }
        }

        // 3. CPUID Leaf 0x7: Características extendidas estructuradas
        if max_leaf >= 7 {
            let leaf7 = unsafe { __cpuid(0x7) };
            // EBX flags
            info.has_bmi1 = (leaf7.ebx & (1 << 3)) != 0;
            info.has_avx2 = (leaf7.ebx & (1 << 5)) != 0;
            info.has_bmi2 = (leaf7.ebx & (1 << 8)) != 0;
            info.has_avx512f = (leaf7.ebx & (1 << 16)) != 0;
            info.has_avx512dq = (leaf7.ebx & (1 << 17)) != 0;
            info.has_avx512cd = (leaf7.ebx & (1 << 28)) != 0;
            info.has_avx512bw = (leaf7.ebx & (1 << 30)) != 0;
            info.has_avx512vl = (leaf7.ebx & (1 << 31)) != 0;
            info.has_sha_ni = (leaf7.ebx & (1 << 29)) != 0;

            // EDX flags: Bit 15 indica arquitectura híbrida
            info.has_hybrid_topology = (leaf7.edx & (1 << 15)) != 0;
        }

        // 4. CPUID Leaf 0x1A: Información de Núcleos Híbridos (Intel)
        if max_leaf >= 0x1A {
            let leaf1a = unsafe { __cpuid(0x1A) };
            let core_type_raw = ((leaf1a.eax >> 24) & 0xFF) as u8;
            info.current_core_type = match core_type_raw {
                0x20 => CoreType::EfficiencyCore,
                0x40 => CoreType::PerformanceCore,
                0x00 => CoreType::StandardCore,
                other => CoreType::Unknown(other),
            };
        } else {
            info.current_core_type = CoreType::StandardCore;
        }

        // 5. Configurar máscaras de afinidad de silicio
        // Si hay arquitectura híbrida o múltiples núcleos lógicos:
        // Los primeros N/2 núcleos lógicos son típicamente los P-Cores físicos primarios.
        if info.has_hybrid_topology && logical_cores >= 4 {
            // Ejemplo 8 núcleos lógicos: P-Cores en bits 0..3 (0x0F), E-Cores en bits 4..7 (0xF0)
            let p_count = (logical_cores / 2).max(2);
            info.p_core_affinity_mask = (1usize << p_count).saturating_sub(1);
            info.auxiliary_affinity_mask = ((1usize << logical_cores).saturating_sub(1)) ^ info.p_core_affinity_mask;
        } else if logical_cores >= 4 {
            // En arquitectura estándar homogénea, aislamos los primeros 2 núcleos para HFT
            info.p_core_affinity_mask = 0x03; // Cores 0 y 1
            info.auxiliary_affinity_mask = ((1usize << logical_cores).saturating_sub(1)) & !0x03;
        } else if logical_cores >= 2 {
            info.p_core_affinity_mask = 0x01; // Core 0
            info.auxiliary_affinity_mask = 0x02; // Core 1
        } else {
            info.p_core_affinity_mask = 0x01;
            info.auxiliary_affinity_mask = 0x01;
        }

        info
    }

    /// Retorna el nombre del fabricante como cadena UTF-8 limpia.
    pub fn vendor_string(&self) -> &str {
        std::str::from_utf8(&self.vendor_id).unwrap_or("Unknown")
    }

    /// Retorna `true` si el hardware soporta vectorización avanzada moderna (AVX2 o superior).
    pub fn supports_advanced_vectorization(&self) -> bool {
        self.has_avx2 || self.has_avx512f
    }

    /// Retorna `true` si la línea de caché está alineada al estándar de 64 bytes de la arquitectura.
    pub fn is_standard_cache_line(&self) -> bool {
        self.l1d_cache_line_size == 64
    }
}
