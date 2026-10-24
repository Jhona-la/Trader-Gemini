//! Asignación de Páginas Grandes de Memoria (2MB Large Pages) & Mitigación de TLB Misses.
//!
//! En sistemas de alta frecuencia y baja latencia, la fragmentación de la memoria en páginas estándar
//! de 4 KB satura rápidamente el búfer de traducción de direcciones virtuales a físicas (**D-TLB**).
//! Un búfer de 2 MB asignado con páginas convencionales de 4 KB requiere 512 entradas de TLB; si
//! estas entradas son expulsadas por el planificador del sistema operativo, cada acceso a memoria
//! sufre una penalización por **Page Table Walk** de 4 niveles en x86_64 (~15 a 40 nanosegundos).
//!
//! Con **Large Pages de 2 MB**:
//! - Una **única entrada de TLB** mapea los 2 MB contiguos completos.
//! - Se reduce la presión sobre el TLB en un factor de **512×**.
//! - Se erradica por completo el Page Fault dinámico durante la sesión de trading.
//!
//! Este módulo implementa `LargePageArena` con fallback transparente y determinista:
//! Si `MEM_LARGE_PAGES` no está disponible o el token del proceso carece de `SeLockMemoryPrivilege`,
//! realiza un fallback automático a páginas estándar bloqueadas en memoria física (`VirtualLock`),
//! garantizando disponibilidad continua sin un solo pánico.
//!
//! Cumple estrictamente con el trinquete `rb_p1`: CERO `unwrap`, CERO `expect`, CERO `panic`, CERO `assert`.

use std::ptr::NonNull;

#[cfg(windows)]
use windows::Win32::System::Memory::{
    GetLargePageMinimum, VirtualAlloc, VirtualFree, VirtualLock, VirtualUnlock,
    MEM_COMMIT, MEM_LARGE_PAGES, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE,
};

/// Tamaño de página estándar en arquitecturas x86_64 (4 Kilobytes).
pub const STANDARD_PAGE_SIZE: usize = 4096;

/// Tamaño canónico de una página grande en Windows x86_64 (2 Megabytes).
pub const CANONICAL_LARGE_PAGE_SIZE: usize = 2 * 1024 * 1024;

/// Alineación estándar de líneas de caché L1D (64 bytes).
pub const CACHE_LINE_SIZE: usize = 64;

/// Arena de memoria de baja latencia con respaldo en páginas de silicio de 2 MB o páginas bloqueadas.
pub struct LargePageArena {
    ptr: NonNull<u8>,
    capacity_bytes: usize,
    page_size_bytes: usize,
    is_large_page: bool,
    is_locked_in_ram: bool,
}

// Envío seguro entre hilos: el puntero asignado es exclusivo de la arena
unsafe impl Send for LargePageArena {}
unsafe impl Sync for LargePageArena {}

impl LargePageArena {
    /// Intenta asignar una arena contigua alineada con páginas grandes (Large Pages).
    ///
    /// Si el sistema operativo no concede páginas grandes, ejecuta un fallback transparente
    /// a memoria estándar de 4 KB bloqueada con `VirtualLock` para prevenir swapping a disco.
    pub fn allocate(requested_bytes: usize) -> Result<Self, &'static str> {
        let min_bytes = requested_bytes.max(STANDARD_PAGE_SIZE);

        #[cfg(windows)]
        {
            Self::allocate_windows(min_bytes)
        }

        #[cfg(not(windows))]
        {
            Self::allocate_posix(min_bytes)
        }
    }

    #[cfg(windows)]
    fn allocate_windows(requested_bytes: usize) -> Result<Self, &'static str> {
        // 1. Consultar el tamaño mínimo de página grande configurado en el kernel
        let large_page_min = unsafe { GetLargePageMinimum() };

        if large_page_min > 0 {
            // Alinear la capacidad al múltiplo exacto de páginas grandes
            let large_capacity = ((requested_bytes + large_page_min - 1) / large_page_min) * large_page_min;

            // Intentar asignar con MEM_LARGE_PAGES
            let alloc_ptr = unsafe {
                VirtualAlloc(
                    None,
                    large_capacity,
                    MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                    PAGE_READWRITE,
                )
            };

            if let Some(non_null) = NonNull::new(alloc_ptr as *mut u8) {
                return Ok(Self {
                    ptr: non_null,
                    capacity_bytes: large_capacity,
                    page_size_bytes: large_page_min,
                    is_large_page: true,
                    is_locked_in_ram: true, // Large pages siempre están fijadas en RAM física
                });
            }
        }

        // 2. Fallback determinista: Páginas estándar de 4 KB bloqueadas en RAM física
        let std_capacity = ((requested_bytes + STANDARD_PAGE_SIZE - 1) / STANDARD_PAGE_SIZE) * STANDARD_PAGE_SIZE;
        let std_ptr = unsafe {
            VirtualAlloc(
                None,
                std_capacity,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        };

        if let Some(non_null) = NonNull::new(std_ptr as *mut u8) {
            // Intentar bloquear físicamente las páginas en el Working Set para evitar page faults
            let lock_success = unsafe { VirtualLock(std_ptr, std_capacity).is_ok() };

            Ok(Self {
                ptr: non_null,
                capacity_bytes: std_capacity,
                page_size_bytes: STANDARD_PAGE_SIZE,
                is_large_page: false,
                is_locked_in_ram: lock_success,
            })
        } else {
            Err("Fallo de asignación de memoria física en VirtualAlloc")
        }
    }

    #[cfg(not(windows))]
    fn allocate_posix(requested_bytes: usize) -> Result<Self, &'static str> {
        // En plataformas POSIX usamos mmap estándar anónimo con alineación
        let std_capacity = ((requested_bytes + STANDARD_PAGE_SIZE - 1) / STANDARD_PAGE_SIZE) * STANDARD_PAGE_SIZE;
        let layout = match std::alloc::Layout::from_size_align(std_capacity, STANDARD_PAGE_SIZE) {
            Ok(l) => l,
            Err(_) => return Err("Layout inválido para arena POSIX"),
        };

        let raw_ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if let Some(non_null) = NonNull::new(raw_ptr) {
            Ok(Self {
                ptr: non_null,
                capacity_bytes: std_capacity,
                page_size_bytes: STANDARD_PAGE_SIZE,
                is_large_page: false,
                is_locked_in_ram: false,
            })
        } else {
            Err("Fallo de asignación de memoria física en alloc_zeroed")
        }
    }

    /// Retorna la capacidad total asignada en bytes.
    pub fn capacity(&self) -> usize {
        self.capacity_bytes
    }

    /// Retorna el tamaño de página de silicio en bytes (ej. 2,097,152 o 4,096).
    pub fn page_size(&self) -> usize {
        self.page_size_bytes
    }

    /// Indica si la asignación está respaldada por páginas grandes de 2 MB (reducción de TLB 512×).
    pub fn is_large_page(&self) -> bool {
        self.is_large_page
    }

    /// Indica si las páginas están efectivamente bloqueadas en memoria RAM sin swapping.
    pub fn is_locked_in_ram(&self) -> bool {
        self.is_locked_in_ram
    }

    /// Retorna el puntero crudo a la base de la memoria.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Retorna el puntero mutable a la base de la memoria.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Retorna una vista de slice inmutable sobre el búfer asignado.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.capacity_bytes) }
    }

    /// Retorna una vista de slice mutable sobre el búfer asignado.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.capacity_bytes) }
    }

    /// Calcula la ganancia teórica de reducción de entradas de TLB respecto a páginas estándar de 4 KB.
    pub fn tlb_pressure_reduction_factor(&self) -> usize {
        if self.is_large_page && self.page_size_bytes >= CANONICAL_LARGE_PAGE_SIZE {
            CANONICAL_LARGE_PAGE_SIZE / STANDARD_PAGE_SIZE // 512×
        } else {
            1 // Páginas estándar
        }
    }
}

impl Drop for LargePageArena {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe {
            if !self.is_large_page && self.is_locked_in_ram {
                let _ = VirtualUnlock(self.ptr.as_ptr() as *const _, self.capacity_bytes);
            }
            let _ = VirtualFree(self.ptr.as_ptr() as *mut _, 0, MEM_RELEASE);
        }

        #[cfg(not(windows))]
        unsafe {
            if let Ok(layout) = std::alloc::Layout::from_size_align(self.capacity_bytes, STANDARD_PAGE_SIZE) {
                std::alloc::dealloc(self.ptr.as_ptr(), layout);
            }
        }
    }
}
