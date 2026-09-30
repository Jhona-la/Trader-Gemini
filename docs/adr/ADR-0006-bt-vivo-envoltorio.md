# ADR-0006 — bt↔vivo: el replay conduce el núcleo real; el sesgo vive en el envoltorio

- **Estado**: Aceptado
- **Fecha**: 2026-09-29
- **Responsables**: GLM (XLVI·A b8312ae6, XLVI·H 5f2d9450, XLVII·A
  f859f6c5); hallazgos citados de Codex (PR #11) y Claude (CL-11)

## Contexto

La "brecha bt/vivo" era una sospecha sin censo. La auditoría XLVI·A
estableció la arquitectura: `run_booktick_replay` conduce el
`GodEngineCore` de PRODUCCIÓN — decisión, física de fills y genoma son
paridad por construcción. La divergencia sólo puede vivir en el
envoltorio.

## Decisión

1. Censo de 6 divergencias (DIV-1..6) con dirección de sesgo y
   signpost falsable por ítem; tests de contrato para las medibles.
2. DIV-2 (latencia) cerrada por ADR-0001. DIV-1 (desplazamiento del
   harness) hecha EXPLÍCITA y configurable (`shift_atr_frac`):
   default 0.10 = histórico bit a bit; 0.0 = paridad de features con el
   vivo. La promoción del default exige reconciliation contra fills
   reales + re-baseline UNA sola vez (decisión de consejo pendiente).
3. Radio de impacto corregido por medición: el modo trade-only (el de
   la EVOLUCIÓN) BYPASA el shift — el fitness que selecciona genomas
   nunca estuvo contaminado; el shift sólo afecta modo libro
   (diagnóstico).

## Consecuencias

- A/B permanente para cuantificar el doble-conteo en cualquier tape
  (~6% del PnL del trade en fixture sintético).
- DIV-3 (primer toque TP/SL) sigue el curso R8-A de Claude — no se
  duplica.
- Regla: cualquier harness nuevo del replay debe declarar su modo
  (trade-only vs libro) en el config, no por convención de llamada.
