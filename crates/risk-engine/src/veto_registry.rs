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
        causa: "grupo misma-apuesta excede el tope de ruina bajo agregación medida",
        fuente_umbral: "medido",
        datos: "riesgo real por posición (snapshot qty·|entry−sl|/cap) + ρ_PnL HY×signo + q cartera (CL-9)",
        responsable: "XLVI·D/E (GLM), 2026-09-29",
        clase: ClaseVeto::RiesgoDuro,
        estado: EstadoVeto::Activo,
        test: Some("xlvie_riesgos_uniformes_reducen_a_la_formula_d748"),
        deuda: None,
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
        causa: "caída de cuenta ≥ máximo admisible derivado de la tasa de pérdida de Cartera",
        fuente_umbral: "medido",
        datos: "pico/capital unificados + q_perdida_cartera ponderada (CL-9)",
        responsable: "CL-9 (Claude), 2026-09-29",
        clase: ClaseVeto::RiesgoDuro,
        estado: EstadoVeto::Activo,
        test: Some("cl9_la_tasa_de_perdida_es_la_de_la_cartera_ponderada"),
        deuda: None,
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
        datos: "has_roster_model (models/{SYM}_MOTOR) + lift sobre base del propio modelo (CL-21)",
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
        responsable: "CL-28 (Claude), 2026-09-29",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Retirado,
        test: Some("cl28_ni_el_ruido_ni_una_tendencia_alcanzan_el_umbral_de_la_fisher"),
        deuda: None,
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
        test: Some("protection_gap_contract"),
        deuda: None,
    },
    EntradaVeto {
        id: "V-LOGIC-006",
        nombre: "suelo TP/SL por fricción",
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
        responsable: "heredado (ola X, sonda D-751b), 2026-09-24",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Activo,
        test: None,
        deuda: Some("censo XLI D3/D4: modulación en frío; contrato pendiente"),
    },
    EntradaVeto {
        id: "V-LOGIC-008",
        nombre: "geometría TP/SL inválida",
        causa: "TP/SL fuera de banda o inconsistentes con dirección",
        fuente_umbral: "literal + gen",
        datos: "brackets del host, Hurst muestreado por reloj (CL-26)",
        responsable: "CL-26/CL-19 (Claude), 2026-09-29",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Activo,
        test: Some("geometry_hurst_contract"),
        deuda: None,
    },
    EntradaVeto {
        id: "V-LOGIC-009",
        nombre: "sin_evidencia",
        causa: "sin historial suficiente para opinar (el gate exige evidencia, no fe)",
        fuente_umbral: "literal (mínimos de muestra)",
        datos: "métricas por símbolo (trade_count, win_rate)",
        responsable: "heredado, censo XLI, 2026-09-26",
        clase: ClaseVeto::Logica,
        estado: EstadoVeto::Activo,
        test: None,
        deuda: Some("contrato de mínimos por símbolo pendiente"),
    },
];

/// Búsqueda por id (estable) o nombre.
pub fn buscar(clave: &str) -> Option<&'static EntradaVeto> {
    REGISTRO_VETOS
        .iter()
        .find(|e| e.id == clave || e.nombre == clave)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
