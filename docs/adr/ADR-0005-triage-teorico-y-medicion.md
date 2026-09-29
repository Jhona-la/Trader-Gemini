# ADR-0005 — Teoría nueva: triage de 4 estados y la medición como portón

- **Estado**: Aceptado
- **Fecha**: 2026-09-29
- **Responsables**: GLM (XLVIII·B/E: fb6412c9, 40c263c2), marco del
  operador ("más teoría ≠ más edge")

## Contexto

El operador entregó un catálogo teórico amplio (Malliavin → transporte
óptimo). El riesgo histórico del repo es la decoración: integrar
ecuaciones por novedad. Ya existían ejemplos en ambas direcciones (Hodge
con falsación vs TE sin consumidor).

## Decisión

1. **Triage de 4 estados** para toda teoría candidata
   (docs/TRIAGE_TEORICO_2026-09-29.md): EXISTE / PARCIAL / CANDIDATO
   (con contrato de transferencia: variable, operador, unidades,
   contorno, identificabilidad, coste, falsación) / ESPECULATIVO
   (registrado sin implementar).
2. **La medición en datos reales es el portón** antes de cablear
   cualquier pieza nueva (doctrina Fisher/D-754 generalizada).
3. Resultado negativo = entregable: se registra la decisión de NO
   cablear con la evidencia.

## Consecuencias

- Casos concretos: firmas de camino (Lyons) nivel-2 IMPLEMENTADAS con
  discretización Stratonovich (línea recta exacta a cualquier densidad;
  Chen y reverso exactos). Transfer entropy IMPLEMENTADA con contrato de
  asimetría… y su medición en tapes reales dio NEGATIVO (~0.001 bits
  ambas direcciones, al piso del sesgo KT) ⇒ NO se cablea a consumidor
  alguno; la vía honesta de refinamiento (simbolización por
  nivel-retorno) queda documentada.
- Lección institucional: la suma discreta de pares ordenados arrastra
  corrección O(δ) dependiente del muestreo — el test de densidades la
  expuso ANTES de producción; media diagonal (Stratonovich) la cierra.
- Candidatos priorizados restantes: cópulas t por par, RG entre escalas,
  Koopman/DMD, Tracy-Widom — cada uno entra por el mismo portón.
