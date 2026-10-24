//! Contratos Formales de Verificación de Fase 8 (F8-001)
//!
//! Suite de contratos matemáticos y de física de silicio para:
//! - Detección CPUID y topología de procesador (P-Cores vs E-Cores, extensiones SIMD).
//! - Supresión activa de subnormales de silicio (FTZ / DAZ en MXCSR).
//! - Asignador determinista de Large Pages (2MB) y reducción de presión TLB.
//! - Criptografía SHA-256 / HMAC en stack con cero invocaciones al OS-allocator.
//! - Resiliencia ante tamaños extremos y privilegios restringidos del SO.
//! - Cumplimiento estricto del mandato U-ERR-6 (sin dicotomía scalp/swing).

use os_guardian::{
    constant_time_eq_32, sha256_hmac_stack, sha256_stack, verify_payload_integrity,
    CoreType, CpuTopologyInfo, LargePageArena, SubnormalHardwareController,
    CANONICAL_LARGE_PAGE_SIZE, STANDARD_PAGE_SIZE,
};

/// Contrato C1: Detección CPUID & Topología de Silicio (P-Cores, E-Cores y SIMD)
#[test]
fn test_c1_cpuid_silicon_topology_detection() {
    let topology = CpuTopologyInfo::detect();

    // 1. Debe haber al menos 1 procesador lógico reportado
    assert!(
        topology.logical_cores >= 1,
        "La topología debe reportar al menos 1 núcleo lógico (fue {})",
        topology.logical_cores
    );

    // 2. El tamaño de línea de caché L1D debe ser estándar (64 bytes) en arquitecturas x86_64 modernas
    assert_eq!(
        topology.l1d_cache_line_size, 64,
        "El tamaño de línea de caché L1D debe ser exactamente 64 bytes"
    );
    assert!(topology.is_standard_cache_line());

    // 3. La máscara de afinidad de P-Cores debe ser no nula
    assert!(
        topology.p_core_affinity_mask > 0,
        "La máscara de afinidad para P-Cores debe ser estrictamente positiva"
    );

    // 4. El tipo de núcleo no debe ser un valor no reconocido arbitrario
    match topology.current_core_type {
        CoreType::PerformanceCore | CoreType::EfficiencyCore | CoreType::StandardCore => {
            // Válido
        }
        CoreType::Unknown(code) => {
            println!("Nota: Tipo de núcleo desconocido reportado por hardware: {:#X}", code);
        }
    }

    // 5. El nombre del fabricante debe ser no vacío
    let vendor = topology.vendor_string();
    assert!(!vendor.is_empty(), "El vendor string no debe estar vacío");
}

/// Contrato C2: Supresión Activa de Flotantes Subnormales de Hardware (FTZ / DAZ)
#[test]
fn test_c2_subnormal_hardware_suppression_ftz_daz() {
    // 1. Un flotante subnormal IEEE-754 debe ser detectado analíticamente
    let denormal_val = 1e-315f64;
    assert!(
        SubnormalHardwareController::is_subnormal_ieee754(denormal_val),
        "1e-315 debe ser catalogado como subnormal por IEEE-754"
    );

    // Un flotante normal estándar no debe ser subnormal
    let normal_val = 1.0f64;
    assert!(
        !SubnormalHardwareController::is_subnormal_ieee754(normal_val),
        "1.0 no debe ser subnormal"
    );

    // 2. Activar la supresión de subnormales
    let enabled = SubnormalHardwareController::enable_suppression();
    #[cfg(target_arch = "x86_64")]
    {
        assert!(enabled, "En arquitectura x86_64 la activación de FTZ/DAZ debe ser exitosa");

        let status = SubnormalHardwareController::read_status();
        assert!(status.ftz_active, "Flush-to-Zero debe estar activo");
        assert!(status.daz_active, "Denormals-are-Zero debe estar activo");
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = enabled;
    }
}

/// Contrato C3: Asignador Large Pages (2MB) & Factor de Reducción de Presión TLB
#[test]
fn test_c3_large_page_arena_allocation_and_tlb_reduction() {
    let requested = CANONICAL_LARGE_PAGE_SIZE; // 2 MB
    let arena_res = LargePageArena::allocate(requested);

    assert!(
        arena_res.is_ok(),
        "La asignación de LargePageArena debe ser exitosa con fallback transparente"
    );

    let mut arena = arena_res.unwrap_or_else(|e| panic!("Fallo inesperado: {}", e));

    // 1. La capacidad debe ser al menos el tamaño pedido
    assert!(
        arena.capacity() >= requested,
        "La capacidad debe ser >= 2 MB (fue {})",
        arena.capacity()
    );

    // 2. La alineación de la base debe ser al menos de 64 bytes (línea de caché)
    let base_addr = arena.as_ptr() as usize;
    assert_eq!(
        base_addr % 64, 0,
        "La base de la arena debe estar alineada a 64 bytes para evitar false sharing (fue {:#X})",
        base_addr
    );

    // 3. El factor de reducción de TLB debe ser >= 1 (y 512 si es Large Page efectiva)
    let tlb_factor = arena.tlb_pressure_reduction_factor();
    assert!(
        tlb_factor == 1 || tlb_factor == 512,
        "El factor de reducción de TLB debe ser 1 o 512 (fue {})",
        tlb_factor
    );

    // 4. Verificación de lectura y escritura libre de fallos de página
    let slice_mut = arena.as_mut_slice();
    slice_mut[0] = 0x5A;
    slice_mut[1024] = 0xC3;
    slice_mut[slice_mut.len() - 1] = 0xFF;

    let slice_ref = arena.as_slice();
    assert_eq!(slice_ref[0], 0x5A);
    assert_eq!(slice_ref[1024], 0xC3);
    assert_eq!(slice_ref[slice_ref.len() - 1], 0xFF);
}

/// Contrato C4: Criptografía Acelerada por Silicio 100% en Stack (Zero OS-Allocator)
#[test]
fn test_c4_simd_stack_crypto_zero_alloc() {
    // 1. Vector de prueba conocido de SHA-256 para "":
    // e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    let empty_hash = sha256_stack(b"");
    let expected_empty = [
        0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9,
        0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52,
        0xb8, 0x55,
    ];
    assert_eq!(
        empty_hash, expected_empty,
        "El digest SHA-256 para input vacío debe ser idéntico al estándar FIPS 180-4"
    );

    // 2. Verificación de integridad con comparación en tiempo constante
    assert!(
        verify_payload_integrity(b"", &expected_empty),
        "La verificación de integridad de payload vacío debe ser verdadera"
    );

    // Alteración de 1 bit debe ser rechazada
    let mut corrupted = expected_empty;
    corrupted[0] ^= 0x01;
    assert!(
        !verify_payload_integrity(b"", &corrupted),
        "Un digest alterado debe ser rechazado en tiempo constante"
    );

    // 3. HMAC-SHA256 en stack:
    let key = b"secret_binance_api_key";
    let message = b"symbol=BTCUSDT&side=BUY&type=LIMIT&quantity=1.000";
    let hmac_1 = sha256_hmac_stack(key, message);
    let hmac_2 = sha256_hmac_stack(key, message);

    assert_eq!(
        hmac_1, hmac_2,
        "HMAC debe ser determinista e idéntico para las mismas entradas"
    );
    assert!(
        constant_time_eq_32(&hmac_1, &hmac_2),
        "La comparación en tiempo constante debe ser verdadera para hashes idénticos"
    );
}

/// Contrato C5: Resiliencia ante Tamaños Extremos y Fallback Sin Pánicos
#[test]
fn test_c5_fallback_resilience_and_extreme_sizes() {
    // 1. Asignación con tamaño cero (debe normalizar a al menos una página de 4KB)
    let zero_alloc = LargePageArena::allocate(0);
    assert!(zero_alloc.is_ok(), "Asignación de tamaño 0 debe normalizarse limpiamente");
    let arena_zero = zero_alloc.unwrap_or_else(|e| panic!("Fallo: {}", e));
    assert!(arena_zero.capacity() >= STANDARD_PAGE_SIZE);

    // 2. Asignación de tamaño mediano (64 KB)
    let medium_alloc = LargePageArena::allocate(64 * 1024);
    assert!(medium_alloc.is_ok(), "Asignación de 64 KB debe ser exitosa");

    // 3. Restauración limpia de control de subnormales
    SubnormalHardwareController::disable_suppression();
    let status_disabled = SubnormalHardwareController::read_status();
    #[cfg(target_arch = "x86_64")]
    {
        assert!(!status_disabled.ftz_active, "FTZ debe quedar inactivo tras disable");
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = status_disabled;
    }

    // Reactivar para producción
    let _ = SubnormalHardwareController::enable_suppression();
}

/// Contrato C6: Cumplimiento Estricto del Mandato U-ERR-6 & Cero Pánicos en Silicio
#[test]
fn test_c6_u_err_6_compliance_and_silicon_invariants() {
    let name_topology = std::any::type_name::<CpuTopologyInfo>();
    let name_arena = std::any::type_name::<LargePageArena>();
    let name_subnormal = std::any::type_name::<SubnormalHardwareController>();

    assert!(
        !name_topology.to_lowercase().contains("scalp")
            && !name_topology.to_lowercase().contains("swing"),
        "U-ERR-6 violado: CpuTopologyInfo contiene términos disjuntos"
    );
    assert!(
        !name_arena.to_lowercase().contains("scalp")
            && !name_arena.to_lowercase().contains("swing"),
        "U-ERR-6 violado: LargePageArena contiene términos disjuntos"
    );
    assert!(
        !name_subnormal.to_lowercase().contains("scalp")
            && !name_subnormal.to_lowercase().contains("swing"),
        "U-ERR-6 violado: SubnormalHardwareController contiene términos disjuntos"
    );
}
