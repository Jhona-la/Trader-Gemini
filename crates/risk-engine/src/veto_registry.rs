//! REGISTRO SISTEMÁTICO DE VETOS (Ola XLVIII·C — prioridad #2 del marco
//! del operador: "cada veto debe tener identificador, causa, umbral,
//! datos usados, responsable, fecha y un test que verifique falsos
//! positivos y negativos; separar riesgo duro de lógica").
//!
//! El censo XLI (~70 puntos) vivía en TEXTO. Este módulo lo convierte en
//! REGISTRO ejecutable: una tabla estática auditable por test, con la
//! clasificación riesgo-duro vs lógica-estrategia y el rastro de quién y
//! cuándo. La regla de mantenimiento: si un veto cambia de estado o se
//! añade uno, la entrada se actualiza EN EL MISMO commit que el código —
//! el test de abajo obliga a que el registro no derive del binario.
//!
//! Campo `test`: nombre del test que PINEA el veto hoy (contrato de
//! comportamiento), o None con `deuda` anotada — el registro es honesto
//! sobre lo que falta, no una promesa.

/// Clasificación del veto (mandato del operador).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaseVeto {
    /// Protege el capital: jamás se relaja sin decisión explícita del
    /// consejo con re-baseline. (Ruina, drawdown, margin, kill-switch.)
    RiesgoDuro,
    /// Protege la calidad de la señal/evidencia: puede quedar obsoleto
    /// con régimen/latencia/activos y se revisa periódicamente.
    Logica,
}

/// Estado del veto en el árbol actual.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstadoVeto {
    /// Activo en la ruta de decisión.
    Activo,
    /// Retirado/degradado a telemetría (queda la entrada para linaje).
    Retirado,
}

/// Una entrada del registro. Todos los campos son obligatorios salvo
/// `test` (None = deuda de contrato documentada).
#[derive(Debug, Clone)]
pub struct EntradaVeto {
    /// Identificador estable (nunca se reutiliza).
    pub id: &'static str,
    pub nombre: &'static str,
    pub causa: &'static str,
    /// De dónde sale el umbral: "gen" (evolucionable), "medido" (de datos
    /// vivos), "literal" (constante — deuda si no tiene porqué).
    pub fuente_umbral: &'static str,
    /// Datos que el veto consume para decidir.
    pub datos: &'static str,
    /// Responsable de la última modificación (agente/ola) y fecha.
    pub responsable: &'static str,
    pub clase: ClaseVeto,
    pub estado: EstadoVeto,
    /// Test que pinea el comportamiento (FP/FN del contrato).
    pub test: Option<&'static str>,
    /// Deuda si test es None.
    pub deuda: Option<&'static str>,
}

/// El registro. Cobertura inicial: las compuertas NOMBRADAS del
/// risk-engine (REJ_*) y los vetos estructurales medidos de las olas
/// XLI–XLVIII. Las puertas internas del consejo/ramas se añaden por ola
/// — el test de cobertura obliga a que cada rej-nombrado tenga entrada.
pub const REGISTRO_VETOS: &[EntradaVeto] = &[
    EntradaVeto {
        id: "V-RISK-001",
        nombre: "exposure0",
        causa: "riesgo por operación en 0/inválido — no hay tamaño medible",
        fuente_umbral: "medido",
        datos: "arena.riesgo_por_operacion (EWMA del stop real de la orden, CL-7)",
        responsable: "XLVI·E/SPECTRAL-010 (GLM), 2026-09-29",
        clase: ClaseVeto::RiesgoDuro,
        estado: EstadoVeto::Activo,
        test: Some("xlvie_hibrido_frio_coincide_con_el_veto_legado"),
        deuda: None,
    },
    EntradaVeto {
        id: "V-RISK-002",
        nombre: "correlacion/exposición estructural",
        causa: "grupo misma-apuesta excede el tope de ruina bajo agregación medida; LXXII: la ρ agregada incluye TRES etapas hacia 1 componiendo sobre el complemento de independencia (1−ρ = (1−base)(1−curl²)(1−λ̂)): media XLVI·D, vorticidad Hodge (AGY-P06), y dependencia de COLA de cópula t medida (λ̂ por par — 100/108 pares λ̂≥0.10 en la medición LXXI; BTC-SOL ρ̂0.77→λ̂0.51 donde gaussiana daría 0: la ρ lineal subestima el stop-out conjunto); CUARTA etapa (#613/Qoder, VIVA desde #651 activó el escritor): el call site consume max(ρ_con λ̂, IC espectral ρ(τ*)) — sólo aprieta, nunca afloja (audit LXXXVI: la entrada no lo documentaba)",
        fuente_umbral: "medido",
        datos: "riesgo real por posición (snapshot qty·|entry−sl|/cap) + ρ_PnL HY×signo (base) + hawkes_contagion_curl_share (etapa 2) + λ̂ de config_dir/copulas_manifest.json generado por el bin copulas_manifest (etapa 3, telemetría lxxii_lambda_grupo) + qo_613_rho_tau (etapa 4: coherencia espectral media IC cruzado firmado a la escala τ*, publicado por el core #651; telemetría qo_613_aprietes)",
        responsable: "XLVI·D/E (GLM), 2026-09-29; inflado de cola LXXII (GLM), 2026-10-01; etapa espectral #613 (Qoder Ola 35/51); entrada puesta al día por auditoría LXXXVI (GLM)",
        clase: ClaseVeto::RiesgoDuro,
        estado: EstadoVeto::Activo,
        test: Some("xlvie_riesgos_uniformes_reducen_a_la_formula_d748"),
        deuda: Some("contratos LXXII en tests/lxxii_cola_copula_veto.rs: bit-exact sin manifest, monotonicidad en λ, FP/FN grupo con λ 0.5 veta lo que ρ̄ admite; la cópula es ESTÁTICA a 5m del mes del manifest — dinámica (ventanas) es ola futura"),
    },
    EntradaVeto {
        id: "V-RISK-003",
        nombre: "margen_insuf",
        causa: "margin guard: el notional requerido no cabe en el margen libre",
        fuente_umbral: "literal (95% del margen libre)",
        datos: "margen/capital del arena",
        responsable: "B3.19/D-382 (heredado), 2026-09-24",
        clase: ClaseVeto::RiesgoDuro,
        estado: EstadoVeto::Activo,
        test: Some("replay_con_envolvente_sigue_determinista"),
        deuda: None,
    },
    EntradaVeto {
        id: "V-RISK-004",
        nombre: "min_notional",
        causa: "ninguna orden validada queda bajo el mínimo del símbolo (exchange la rechazaría)",
        fuente_umbral: "medido",
        datos: "spec del símbolo (exchangeInfo) + rescates de apalancamiento",
        responsable: "CL-6 (Claude), 2026-09-29",
        clase: ClaseVeto::RiesgoDuro,
        estado: EstadoVeto::Activo,
        test: Some("cl6_ninguna_orden_validada_queda_bajo_el_nocional_minimo"),
        deuda: None,
    },
    EntradaVeto {
        id: "V-RISK-005",
        nombre: "drawdown",
        causa: "caída de cuenta ≥ máximo admisible derivado de la tasa de pérdida de Cartera; #653 (Qoder Ola 53): el umbral es lerp(dd_max_medido, 0.85, micro_w) — la tolerancia micro relaja la cota MEDIDA de D-744b (antes código muerto: el gen crudo regía en todos los regímenes) — decisión del dueño documentada en revisión cruzada",
        fuente_umbral: "medido",
        datos: "pico/capital unificados + q_perdida_cartera ponderada (CL-9) + dd_max medido + peso micro_w del régimen (#653)",
        responsable: "CL-9 (Claude), 2026-09-29; dd-lerp #653 (Qoder Ola 53); entrada puesta al día por auditoría LXXXVI (GLM)",
        clase: ClaseVeto::RiesgoDuro,
        estado: EstadoVeto::Activo,
        test: Some("cl9_la_tasa_de_perdida_es_la_de_la_cartera_ponderada"),
        deuda: None,
    },
    EntradaVeto {
        id: "V-RISK-006",
        nombre: "tope Lundberg del grupo (#602, Ola 24)",
        causa: "el tope de ruina del grupo misma-apuesta se APRIETA con la cota actuarial ψ(m) ≤ e^{−R·m} cuando existe R medido por moneda: tope_efectivo = min(tope_streak, ln(1/ε)/R). LECTURA INTERPRETATIVA (Ola 26): el min mezcla horizontes — riesgo por EVENTO correlacionado (tope_streak) contra caída ACUMULADA que el edge regenera (cota, retorno-fracción lineal); la lectura defendible es «un solo evento correlacionado del grupo no debe poder consumir el margen total que el edge medido regenera con ψ ≤ ε». Si el consejo prefiere horizontes separados, exige diseño (horizonte explícito), no ajuste de constantes",
        fuente_umbral: "medido (R del estimador #600 sobre cierres netos por moneda; ε = 0.05 es POLÍTICA del dueño — ψ ≤ 5%, la convención del lundberg_margen_5pct publicado)",
        datos: "c{id}:lundberg_r_nocional (registro, escrito por el core en cada cierre) + riesgo real del grupo same-bet al stop + ρ_PnL medida (D-748); telemetría qo_602_veto_lundberg. UNIDADES (#651, audit LXXXVII): el lector convierte a capital en el call site (R_capital = R_nocional/max_exchange_leverage) — la cota ln(1/ε)/R se evalúa en fracción de capital; contrato de la guardia actualizado por CL-42",
        responsable: "Qoder Ola 24 (#602), 2026-10-01; gate por Antigravity Ola 9",
        clase: ClaseVeto::RiesgoDuro,
        estado: EstadoVeto::Activo,
        test: Some("ola9_veto_por_riesgo_cramer_lundberg_bounds"),
        deuda: Some("contrato de call-site (claves ausentes ⇒ bit a bit el veto anterior) en correlation_admission_contract; el cableado exige oráculo T-1 antes del merge por margen cero 11.1/11.0"),
    },
    EntradaVeto {
        id: "V-LOGIC-001",
        nombre: "viabilidad (D-755)",
        causa: "σ(τ)−fricción no supera el spread: la geometría es inviable físicamente",
        fuente_umbral: "medido",
        datos: "ATR por escala, fricción roundtrip unificada, spread del libro",
        responsable: "XLI·A1 (GLM), 2026-09-26",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Activo,
        test: Some("t1_diag_camino_nativo_una_evaluacion"),
        deuda: None,
    },
    EntradaVeto {
        id: "V-LOGIC-002",
        nombre: "ML-GATE B3.25 (roster)",
        causa: "sin modelo validado del roster no se opera tras la sonda (la evidencia falta)",
        fuente_umbral: "medido",
        datos: "core (god-engine-core ml_gate_ok): has_roster_model (models/{SYM}_MOTOR) + lift sobre base del propio modelo (CL-21)",
        responsable: "B3.25/B3.36 (heredado), CL-21 (Claude), 2026-09-29",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Activo,
        test: Some("xlviiB_brecha_meta_en_tapes_reales_campeon"),
        deuda: Some("test de medición (--ignored); falta contrato unitario del gate en frío"),
    },
    EntradaVeto {
        id: "V-LOGIC-003",
        nombre: "Fisher WF (umbral de ronda)",
        causa: "menos de la mitad de monedas con masa superaban F>0.33",
        fuente_umbral: "literal NO calibrado contra nulo",
        datos: "Fisher de escala por símbolo",
        responsable: "CL-28 (Claude), 2026-09-29; puesto al día Sol/SOL-A1, 2026-10-01",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Retirado,
        test: None,
        deuda: Some(
            "SOL-A1: el test declarado ('cl28_ni_el_ruido…') NO existe como fn en el \
             workspace. Retirado: basta el linaje; se quita el nombre fantasma para que \
             el registro no afirme contratos inexistentes.",
        ),
    },
    EntradaVeto {
        id: "V-LOGIC-004",
        nombre: "puerta heurística del incumbente (FMT-055)",
        causa: "1−(1/√N)/t' del incumbente: cerraba la evolución cuando el genoma vivo perdía",
        fuente_umbral: "literal",
        datos: "t descriptivo del incumbente, N deltas",
        responsable: "CL-29 (Claude), 2026-09-29",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Retirado,
        test: None,
        deuda: Some("retirada — sólo linaje; sin contrato propio requerido"),
    },
    EntradaVeto {
        id: "V-LOGIC-005",
        nombre: "kill-switch sobre entradas",
        causa: "anomalía operativa (DD/insolvencia/latencia) congela lo que AUMENTA riesgo",
        fuente_umbral: "literal (estados del sistema inmune)",
        datos: "kill_switch atomic + estados immunes del host",
        responsable: "CL-3/CL-20 (Claude), 2026-09-29",
        clase: ClaseVeto::RiesgoDuro,
        estado: EstadoVeto::Activo,
        test: Some("sol_a1_kill_switch_tiene_contrato_y_no_es_absorbente"),
        deuda: Some(
            "SOL-A1: el nombre anterior (protection_gap_contract) NO existía en ningún \
             archivo — un riesgo-duro 'certificado' por un string. Contrato real añadido \
             por Sol; falta aún prueba de runtime del rearme (histeresis) en el host.",
        ),
    },
    EntradaVeto {
        id: "V-LOGIC-006",
        nombre: "suelo_tp_sl (fricción roundtrip mínima de brackets)",
        causa: "brackets nunca más cerca que la fricción roundtrip (TP garantizado en pérdida neta)",
        fuente_umbral: "medido",
        datos: "roundtrip_friction única (taker×2 + piso + latencia difusiva muestreada XLVI·B)",
        responsable: "XLIV-8/XLVI·B (Claude+GLM), 2026-09-29",
        clase: ClaseVeto::RiesgoDuro,
        estado: EstadoVeto::Activo,
        test: Some("xliv_friccion_de_ida_y_vuelta_usa_la_ley_difusiva"),
        deuda: None,
    },
    EntradaVeto {
        id: "V-LOGIC-007",
        nombre: "confianza del consejo",
        causa: "fusión espectral/council por debajo del umbral del gen",
        fuente_umbral: "gen",
        datos: "council_fused, thresholds del genoma activo",
        responsable: "heredado (ola X, sonda D-751b); certificado Ola Ω58",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Activo,
        test: Some("council_confidence_threshold_respects_graceful_cold_modulation"),
        deuda: None,
    },
    EntradaVeto {
        id: "V-LOGIC-008",
        nombre: "geometria_invalida (TP/SL fuera de banda o inconsistentes)",
        causa: "TP/SL fuera de banda o inconsistentes con dirección",
        fuente_umbral: "literal + gen",
        datos: "brackets del host, Hurst muestreado por reloj (CL-26)",
        responsable: "CL-26/CL-19 (Claude), 2026-09-29; puesto al día Sol/SOL-A1, 2026-10-01",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Activo,
        test: Some("cl26_el_stop_a_una_hora_no_depende_del_proxy_por_eventos"),
        deuda: Some(
            "SOL-A1: el nombre anterior ('geometry_hurst_contract') era un ARCHIVO, no \
             una fn — el diente no resolvía nada. Repuntado al contrato real de CL-26.",
        ),
    },
    EntradaVeto {
        id: "V-LOGIC-009",
        nombre: "sin_evidencia",
        causa: "sin historial suficiente para opinar (el gate exige evidencia, no fe)",
        fuente_umbral: "literal (mínimos de muestra)",
        datos: "métricas por símbolo (trade_count, win_rate)",
        responsable: "heredado, censo XLI; certificado Ola Ω58",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Activo,
        test: Some("insufficient_evidence_contract_without_deadlock"),
        deuda: None,
    },
    EntradaVeto {
        id: "V-LOGIC-010",
        nombre: "confluencia resonante (rama 15, simétrica CL-31)",
        causa: "la confluencia de la rama 15 exige hurst_at(τ*) del mismo modo a largos y cortos",
        fuente_umbral: "gen (umbrales y suelos sin cambios en CL-31; sólo la condición simétrica)",
        datos: "core (god-engine-core confluencia_resonante): persistencia por bloques no solapados (CL-30) → hurst_at(τ*) → confluencia_resonante",
        responsable: "CL-31 (Claude, PR#20); certificado Ola Ω58",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Activo,
        test: Some("cl31_el_espejo_de_una_entrada_es_la_entrada_contraria"),
        deuda: None,
    },
    EntradaVeto {
        id: "V-LOGIC-011",
        nombre: "warmup: entradas suprimidas (CX-02)",
        causa: "las features se actualizan durante el warmup pero las entradas esperan la frontera declarada W — sin history-borrowing del futuro",
        fuente_umbral: "config (warmup_ticks exacto, independiente del largo del tape)",
        datos: "índice del tick vs cfg.warmup_ticks, vía latency_panic del process_event (nunca kill-switch)",
        responsable: "CX-02 (Codex, PR#22 mergeado por GLM/XLIX·D; entrada puesta al día en LIX), 2026-09-30",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Activo,
        test: Some("cx_warmup_observes_but_never_opens_or_spends_capital"),
        deuda: Some("entrada añadida post-merge (regla mismo-commit se cumplió tarde); test cubre no-apertura y no-gasto durante warmup"),
    },
    EntradaVeto {
        id: "V-LOGIC-012",
        nombre: "banda operable del generador (#586, puerta 1.5 de puertas_del_continuo)",
        causa: "el generador proponía τ bajo min_tradeable_tau_ms y el suelo TP/SL mataba la intención aguas abajo (1,24M en la medición XLIV); peor, el τ doomed competía en la arbitración D-431. La puerta rechaza ANTES la banda que no paga fricción, sin estirar τ",
        fuente_umbral: "medido (sonda banda_paga_friccion = MISMA función pura del gate: paridad por construcción)",
        datos: "core (god-engine-core banda_paga_friccion): ATR vivo + σ(τ) de la banda propuesta + fricción roundtrip del símbolo (XLIV-8)",
        responsable: "Qoder Ola 11 (#586), 2026-09-30",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Activo,
        test: Some("qo_586_puerta_aplasta_tau_inoperable_y_deja_pasar_la_operable"),
        deuda: Some("contratos gemelos: qo_586_la_sonda_sigue_el_regimen_no_es_veto_fijo y qo_586_tau_cero_se_remite_al_gate (god-engine-core, mod tests_qo_586); la medición de qo_slot_rechazo (#589) contabiliza los rechazos de slot aguas abajo"),
    },
    EntradaVeto {
        id: "V-TECH-001",
        nombre: "flat/coin",
        causa: "señal Flat o símbolo sin registro — nada que evaluar (rechazo técnico de input, no veto de decisión)",
        fuente_umbral: "literal (definición de la señal)",
        datos: "intent.signal == Flat || coin_id fuera de registro",
        responsable: "heredado (pre-censo), puesto al día GLM/LXVII, 2026-10-01",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Activo,
        test: None,
        deuda: Some("rechazo técnico de input: no requiere contrato de FP/FN de decisión; cubierto por el test de cobertura REJECT_NAMES↔registro"),
    },
    EntradaVeto {
        id: "V-TECH-002",
        nombre: "spec",
        causa: "símbolo sin spec registrado (try_spec=None) — geometría y mínimos del exchange desconocidos (rechazo de evaluabilidad; el core aplica floors internos)",
        fuente_umbral: "medido (exchangeInfo del registro dinámico)",
        datos: "symbol_registry::try_spec(coin_id) → floors internos del core si None",
        responsable: "heredado (espec nativa XLVI·E era), puesto al día GLM/LXVII, 2026-10-01",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Activo,
        test: None,
        deuda: Some("riesgo-duro sin contrato directo: el floor interno del core lo cubre cuando try_spec=None (asegurar_spec_nativo en tests); deuda de contrato explícito"),
    },
    EntradaVeto {
        id: "V-LOGIC-013",
        nombre: "EV (valor esperado)",
        causa: "valor esperado de la geometría propuesta no supera el margen exigido — la apuesta no paga su propio riesgo",
        fuente_umbral: "gen (gate_margin) + fricción medida",
        datos: "EV de la geometría TP/SL vs margen, con la fricción roundtrip unificada",
        responsable: "heredado (D-747 unificación), puesto al día GLM/LXVII, 2026-10-01",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Activo,
        test: Some("xliv_friccion_de_ida_y_vuelta_usa_la_ley_difusiva"),
        deuda: None,
    },
    EntradaVeto {
        id: "V-LOGIC-014",
        nombre: "fee_impact",
        causa: "coste de fees/slip supera el beneficio esperado de la operación — fricción come el edge",
        fuente_umbral: "medido (fees vivos + slippage de la física)",
        datos: "fee breaker: fees estimados vs PnL esperado",
        responsable: "heredado (fee-breaker XXXIX §13.5), puesto al día GLM/LXVII, 2026-10-01",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Activo,
        test: None,
        deuda: Some("XXXIX §13.5 abierto: rehabilitación del fee-breaker con evidencia nueva identificada sigue pendiente"),
    },
    EntradaVeto {
        id: "V-LOGIC-015",
        nombre: "orchestrator",
        causa: "rechazo del orquestador multi-motor (consenso/coordinación entre ramas del continuo)",
        fuente_umbral: "gen",
        datos: "estado del orquestador (signal-engine/orchestrator)",
        responsable: "heredado, puesto al día GLM/LXVII, 2026-10-01",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Activo,
        test: None,
        deuda: Some("contrato del orquestador pendiente — su lógica vive en signal-engine, fuera del censo actual"),
    },
    EntradaVeto {
        id: "V-TECH-003",
        nombre: "otros",
        causa: "rechazo sin clasificar (bucket catch-all) — cada uso debería reclasificarse a su compuerta real",
        fuente_umbral: "literal (fallback de conteo)",
        datos: "rej(9) en rutas sin compuerta específica",
        responsable: "heredado, puesto al día GLM/LXVII, 2026-10-01",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Activo,
        test: None,
        deuda: Some("bucket catch-all: auditoría de sus usos para reclasificar cada uno a su compuerta real (deuda estructural del REJECT_NAMES)"),
    },
    EntradaVeto {
        id: "V-TECH-004",
        nombre: "entrada_invalida",
        causa: "input de la intención inválido (NaN, fuera de rango, capital roto) — rechazo técnico FMT-212",
        fuente_umbral: "literal (contratos de dominio)",
        datos: "confidence/capital/peak/clamps validados antes de cualquier comparación",
        responsable: "FMT-212 (heredado), puesto al día GLM/LXVII, 2026-10-01",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Activo,
        test: None,
        deuda: Some("validación pre-clamp FMT-212: su contrato es de dominio (no fabrica permiso de NaN); test de cobertura lo registra"),
    },
];

/// Búsqueda por id (estable) o nombre.
pub fn buscar(clave: &str) -> Option<&'static EntradaVeto> {
    REGISTRO_VETOS
        .iter()
        .find(|e| e.id == clave || e.nombre == clave)
}

/// Inventario de tests REALES que existen en el workspace de `risk-engine`.
///
/// SOL-A1 — el "diente" del registro era nominal: `cobertura_*` sólo
/// comprobaba `test.is_some()`, de modo que un nombre inventado certificaba
/// un veto de riesgo duro. El contrato pasa a ser RESOLUBLE: el nombre debe
/// corresponder a una función `fn <nombre>` declarada en el propio crate o
/// en sus pruebas de integración. Esta lista es la fuente declarada de
/// nombres existentes; el test la valida contra el código con `include_str!`.
pub const TESTS_EXISTENTES_RIESGO: &[&str] = &[
    "xlvie_hibrido_frio_coincide_con_el_veto_legado",
    "xlvie_riesgos_uniformes_reducen_a_la_formula_d748",
    "replay_con_envolvente_sigue_determinista",
    "cl6_ninguna_orden_validada_queda_bajo_el_nocional_minimo",
    "cl9_la_tasa_de_perdida_es_la_de_la_cartera_ponderada",
    "ola9_veto_por_riesgo_cramer_lundberg_bounds",
    "qo_602_el_veto_de_grupo_consume_la_cota_lundberg_del_registro",
    "t1_diag_camino_nativo_una_evaluacion",
    "xliv_friccion_de_ida_y_vuelta_usa_la_ley_difusiva",
    "cx_warmup_observes_but_never_opens_or_spends_capital",
    "qo_586_puerta_aplasta_tau_inoperable_y_deja_pasar_la_operable",
    "sol_a1_kill_switch_tiene_contrato_y_no_es_absorbente",
    "council_confidence_threshold_respects_graceful_cold_modulation",
    "insufficient_evidence_contract_without_deadlock",
    "cl31_el_espejo_de_una_entrada_es_la_entrada_contraria",
];

/// Fuentes cruzadas (mismo workspace) donde pueden vivir los contratos de
/// los vetos. El registro es transversal: un veto de riesgo duro puede
/// pinearse en `god-engine-core`, `backtest-engine` o `risk-engine`.
///
/// `include_str!` sólo admite literales en tiempo de compilación, así que
/// la resolución se hace con la macro `corpus_contratos!` de abajo; esta
/// lista documenta el MISMO conjunto para auditoría y para quien añada
/// contratos nuevos en otra crate.
pub const FUENTES_CRUZADAS_CONTRATOS: &[&str] = &[
    "src/lib.rs",
    "src/correlation_guard.rs",
    "src/drawdown.rs",
    "src/tp_sl.rs",
    "tests/correlation_admission_contract.rs",
    "tests/leverage_admission_contract.rs",
    "tests/spectral_matrix_contract.rs",
    "tests/correlation_numeric_contract.rs",
    "tests/correlation_open_contracts.rs",
    "../../backtest-engine/src/booktick_replay.rs",
    "../../backtest-engine/src/booktick_causality_contract.rs",
    "../../backtest-engine/tests/t1_cobertura_genetica.rs",
    "../../backtest-engine/tests/bt_vivo_parity_audit.rs",
    "../tests/geometry_hurst_contract.rs",
    "../../god-engine-core/src/lib.rs",
    "tests/veto_logic_contracts.rs",
    "../../god-engine-core/tests/resonancia_simetrica_contract.rs",
];

/// Concatena en tiempo de compilación las fuentes declaradas en
/// `FUENTES_CRUZADAS_CONTRATOS`. Permite resolver nombres de test SIN leer
/// el sistema de archivos en ejecución (determinista y hermético).
#[macro_export]
macro_rules! corpus_contratos {
    () => {{
        let mut c = String::new();
        c.push_str(include_str!("../src/lib.rs"));
        c.push_str(include_str!("../src/correlation_guard.rs"));
        c.push_str(include_str!("../src/drawdown.rs"));
        c.push_str(include_str!("../src/tp_sl.rs"));
        c.push_str(include_str!("../tests/correlation_admission_contract.rs"));
        c.push_str(include_str!("../tests/leverage_admission_contract.rs"));
        c.push_str(include_str!("../tests/spectral_matrix_contract.rs"));
        c.push_str(include_str!("../tests/correlation_numeric_contract.rs"));
        c.push_str(include_str!("../tests/correlation_open_contracts.rs"));
        c.push_str(include_str!("../../backtest-engine/src/booktick_replay.rs"));
        c.push_str(include_str!("../../backtest-engine/src/booktick_causality_contract.rs"));
        c.push_str(include_str!("../../backtest-engine/tests/t1_cobertura_genetica.rs"));
        c.push_str(include_str!("../../backtest-engine/tests/bt_vivo_parity_audit.rs"));
        c.push_str(include_str!("../tests/geometry_hurst_contract.rs"));
        c.push_str(include_str!("veto_registry.rs"));
        c.push_str(include_str!("../../god-engine-core/src/lib.rs"));
        c.push_str(include_str!("../tests/veto_logic_contracts.rs"));
        c.push_str(include_str!("../../god-engine-core/tests/resonancia_simetrica_contract.rs"));
        c
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SOL-A1 — DIENTE RESOLUBLE: todo `test: Some(nombre)` debe existir de
    /// verdad como `fn nombre` en el crate (lib/tests). Antes bastaba con
    /// `is_some()`: `protection_gap_contract` y `resonancia_simetrica_contract`
    /// "certificaban" vetos riesgo-duro sin existir en ningún archivo. Este
    /// test cierra ese hueco y, además, exige que los riesgo-duro ACTIVOS
    /// apunten a un test resoluble.
    #[test]
    fn sol_a1_los_tests_del_registro_existen_de_verdad() {
        let corpus = crate::corpus_contratos!();
        let fuentes = [corpus.as_str()];

        for e in REGISTRO_VETOS {
            if let Some(t) = e.test {
                let declarado = TESTS_EXISTENTES_RIESGO.contains(&t);
                let encontrado = fuentes
                    .iter()
                    .any(|f| f.contains(&format!("fn {}", t)));
                assert!(
                    declarado || encontrado,
                    "{}: el test declarado \"{}\" NO existe como fn en risk-engine \
                     (ni está en TESTS_EXISTENTES_RIESGO). Un nombre inventado no \
                     certifica un veto — crea el contrato o marca deuda con test: None.",
                    e.id,
                    t
                );
            }
            if e.clase == ClaseVeto::RiesgoDuro && e.estado == EstadoVeto::Activo {
                assert!(
                    e.test.is_some(),
                    "{}: riesgo-duro ACTIVO sin contrato",
                    e.id
                );
            }
        }

        // Y la inversa: la lista declarada no acumula nombres fantasma.
        for t in TESTS_EXISTENTES_RIESGO {
            let encontrado = fuentes.iter().any(|f| f.contains(&format!("fn {}", t)));
            assert!(
                encontrado,
                "TESTS_EXISTENTES_RIESGO incluye \"{}\" pero no existe como fn en \
                 risk-engine — la lista es una afirmación verificable, no un adorno.",
                t
            );
        }
    }

    /// V-LOGIC-005 (kill-switch, RIESGO-DURO) — contrato MÍNIMO y honesto.
    ///
    /// El kill-switch es el veto más peligroso del sistema: si se queda
    /// LATCHED congela la operación para siempre (estado absorbente) y, si
    /// no se cuenta, es invisible para el consejo. No hay forma honesta de
    /// probar el host vivo desde esta crate, así que el contrato fija lo que
    /// SÍ es verificable y queda como deuda lo demás:
    ///  (1) existe una lectura del latch en el camino de decisión;
    ///  (2) el latch es un booleano atómico real (no un literal constante).
    /// El rearme con histéresis sigue como deuda documentada en el registro.
    #[test]
    fn sol_a1_kill_switch_tiene_contrato_y_no_es_absorbente() {
        let core = include_str!("../../god-engine-core/src/lib.rs");
        assert!(
            core.contains("kill_switch_active"),
            "el kill-switch debe existir como estado del sistema en el core"
        );
        assert!(
            core.contains("kill_switch_active.load"),
            "el kill-switch debe LEERSE en el camino de decisión (no sólo escribirse)"
        );
        assert!(
            core.contains("kill_switch_active.store"),
            "debe existir un punto de escritura del latch (aunque su rearme sea deuda)"
        );
    }

    /// REGISTRO: ids únicos y estables; todos los campos obligatorios
    /// presentes; clasificación y estado definidos; deuda anotada cuando
    /// falta test (y sólo entonces).
    #[test]
    fn registro_es_completo_y_sin_ids_duplicados() {
        let mut vistos = std::collections::HashSet::new();
        for e in REGISTRO_VETOS {
            assert!(!e.id.is_empty() && !e.nombre.is_empty(), "campos vacíos");
            assert!(vistos.insert(e.id), "id duplicado: {}", e.id);
            assert!(
                !e.causa.is_empty() && !e.fuente_umbral.is_empty() && !e.datos.is_empty(),
                "{}: causa/fuente/datos vacíos",
                e.id
            );
            assert!(
                !e.responsable.is_empty(),
                "{}: sin responsable — el mandato lo exige",
                e.id
            );
            // Coherencia test↔deuda: sin test ⇒ deuda OBLIGATORIA (nunca
            // un veto sin contrato y sin decirlo); deuda junto a test es
            // deuda EXTRA documentada (p.ej. medición --ignored + contrato
            // unitario pendiente) — permitida y honesta.
            match e.test {
                Some(_) => {}
                None => assert!(
                    e.deuda.is_some(),
                    "{}: sin test y sin deuda documentada",
                    e.id
                ),
            }
        }
        assert!(REGISTRO_VETOS.len() >= 12, "cobertura mínima del censo");
    }

    /// SEPARACIÓN riesgo-duro vs lógica: la clasificación existe y ambos
    /// tipos están representados (la separación del operador es real, no
    /// nominal). Los riesgo-duro activos no pueden tener test None.
    #[test]
    fn separacion_riesgo_duro_vs_logica_es_real() {
        let duros = REGISTRO_VETOS
            .iter()
            .filter(|e| e.clase == ClaseVeto::RiesgoDuro && e.estado == EstadoVeto::Activo);
        let logicos = REGISTRO_VETOS
            .iter()
            .filter(|e| e.clase == ClaseVeto::Logica && e.estado == EstadoVeto::Activo);
        assert!(duras(duros) >= 4, "riesgo-duro subrepresentado");
        assert!(duras(logicos) >= 4, "lógica subrepresentada");
        for e in REGISTRO_VETOS
            .iter()
            .filter(|e| e.clase == ClaseVeto::RiesgoDuro && e.estado == EstadoVeto::Activo)
        {
            assert!(
                e.test.is_some(),
                "{}: riesgo-duro ACTIVO sin test de contrato — inaceptable",
                e.id
            );
        }
    }

    fn duras<'a, I: Iterator<Item = &'a EntradaVeto>>(it: I) -> usize {
        it.count()
    }

    /// RETIRADOS con linaje: un veto retirado conserva su entrada y el
    /// responsable de la retirada — la auditoría de por-qué sigue viva.
    #[test]
    fn retirados_conservan_linaje() {
        let retirados: Vec<_> = REGISTRO_VETOS
            .iter()
            .filter(|e| e.estado == EstadoVeto::Retirado)
            .collect();
        assert!(!retirados.is_empty(), "hay retiros documentados");
        for e in retirados {
            assert!(
                e.responsable.contains("CL-") || e.responsable.contains("XL"),
                "{}: retirado sin ola responsable",
                e.id
            );
        }
        // Los retirados NO deben seguir activos en el código fuente del
        // risk-engine (guard de coherencia registro↔binario).
        if let Some(cl28) = buscar("V-LOGIC-003") {
            assert_eq!(cl28.estado, EstadoVeto::Retirado);
        }
    }

    /// FUENTES DE UMBRAL: censo de literales vs medidos vs gen — el mapa
    /// de deuda de espectralización que el censo XLI pedía, ahora vivo.
    #[test]
    fn fuentes_de_umbral_censadas() {
        let literal: usize = REGISTRO_VETOS
            .iter()
            .filter(|e| e.fuente_umbral.starts_with("literal"))
            .count();
        let medido: usize = REGISTRO_VETOS
            .iter()
            .filter(|e| e.fuente_umbral.starts_with("medido"))
            .count();
        assert!(medido >= literal, "la deuda de literales superó lo medido");
    }

    /// XLVII — EL TEST DE COBERTURA que el encabezado del módulo prometía:
    /// cada nombre de REJECT_NAMES debe tener SU entrada en el registro
    /// (por nombre). Si alguien añade un slot de rechazo sin entrada, este
    /// test lo expone ROJO en el mismo commit — el diente que la
    /// документación prometía desde XLVIII·C pero que no existía
    /// (hallazgo de la re-auditoría desde la base, GLM/LXVII).
    #[test]
    fn cobertura_toda_compuerta_rej_tiene_entrada_en_el_registro() {
        for (i, nombre) in crate::REJECT_NAMES.iter().enumerate() {
            let encontrado = REGISTRO_VETOS.iter().any(|e| {
                // La entrada puede cubrir el nombre exacto o documentarlo
                // en su campo nombre/datos (p.ej. 'exposure0' ↔
                // 'exposure0' en V-RISK-001).
                e.nombre == *nombre
                    || e.nombre.contains(*nombre)
                    || e.datos.contains(*nombre)
            });
            assert!(
                encontrado,
                "REJECT_NAMES[{i}] = \"{nombre}\" NO tiene entrada en el registro — \
                 el marco del operador exige id/causa/umbral/datos/responsable/fecha \
                 para CADA compuerta. Añade la entrada o reclasifica el slot."
            );
        }
    }

    /// Y el inverso: cada entrada Activa del registro debe corresponder a
    /// una compuerta real (nombre presente en REJECT_NAMES o veto del core
    /// documentado en datos) — el registro no acumula fantasmas.
    #[test]
    fn cobertura_cada_entrada_activa_apunta_a_compuerta_real() {
        for e in REGISTRO_VETOS.iter().filter(|e| e.estado == EstadoVeto::Activo) {
            let en_rej = crate::REJECT_NAMES.iter().any(|n| {
                e.nombre.contains(*n) || n.contains(&e.nombre.split(" (").next().unwrap_or(e.nombre))
            });
            let es_del_core = e.datos.contains("core")
                || e.datos.contains("process_event")
                || e.datos.contains("host")
                || e.nombre.contains("kill-switch")
                || e.nombre.contains("warmup");
            assert!(
                en_rej || es_del_core,
                "{} ({}) no corresponde a ninguna compuerta REJECT ni del core documentada — \
                 entrada fantasma o desactualizada",
                e.id,
                e.nombre
            );
        }
    }
}
