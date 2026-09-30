# ADR-0004 — Registro de vetos: el censo es código, no documento

- **Estado**: Aceptado
- **Fecha**: 2026-09-29
- **Responsables**: GLM (XLVIII·C, cff240d7), directiva del operador
  (prioridad #2)

## Contexto

El censo de la Ola XLI (~70 puntos de veto) vivía en texto: imposible
de auditar mecánicamente, sin responsable por entrada, sin linaje de
retiros. El operador exige: identificador, causa, umbral, datos,
responsable, fecha, test, y separación riesgo-duro vs lógica.

## Decisión

1. `risk_engine::veto_registry`: tabla estática con `EntradaVeto`
   (id estable, causa, FUENTE del umbral gen/medido/literal, datos,
   responsable/ola+fecha, clase, estado, test, deuda explícita).
2. Contratos con dientes: riesgo-duro ACTIVO sin test = fallo de suite;
   sin test ⇒ deuda obligatoria; retirados conservan linaje (quién,
   cuándo); medido ≥ literal en fuentes de umbral.
3. Regla de mantenimiento: la entrada se actualiza EN EL MISMO commit
   que el código del veto.

## Consecuencias

- Los retiros de Claude (Fisher-de-ronda CL-28, puerta incumbente
  CL-29) quedaron con linaje en su primer día de vida.
- Nuevos vetos sin entrada = deuda visible en CI; la cobertura crece
  por ola (las puertas del consejo/ramas tienen deuda anotada).
- El mapa de espectralización (cuántos literales quedan) es una query,
  no una re-lectura de informes.
