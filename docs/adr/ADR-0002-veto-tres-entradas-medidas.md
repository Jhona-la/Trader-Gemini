# ADR-0002 — Veto estructural: las tres entradas medidas (riesgo real, ρ̄, EWMA)

- **Estado**: Aceptado
- **Fecha**: 2026-09-29
- **Responsables**: GLM (XLVI·D b73bab69 + XLVI·E fef9fd64); insumos de
  Claude: CL-7 (EWMA del stop real), CL-9 (q de cartera)

## Contexto

El veto de exposición estructural same-bet operaba con supuestos en vez
de medidas: correlación perfecta asumida siempre (presupuesto lineal),
y un escalar global de riesgo que cobraba una posición diminuta al 18%
y una grande al 0.01%. La dependencia por par YA se medía
(Hayashi-Yoshida × signo) pero se descartaba tras clasificar.

## Decisión

El veto agrega tres entradas medidas, con continuidad matemática exacta
en los casos límite:

1. **Riesgo real por posición**: qty·|entry−sl|/capital del snapshot
   (XLVI·E). Miembro sin stop utilizable ⇒ proxy tope/8.
2. **ρ̄ medida del grupo** (XLVI·D): media de correlaciones contra la
   candidata; no medidos cuentan 1.0.
3. **Candidata**: la EWMA del stop que la orden lleva (CL-7).

Agregación equicorrelacionada ponderada
`√(Σr² + ρ̄((Σr)²−Σr²))`: riesgos uniformes reducen BIT a BIT a la
fórmula D-748 existente; todos-no-medidos reproducen el lineal legado.
Piso: nunca por debajo del mayor riesgo individual.

## Consecuencias

- Desbloqueo medido: stops reales pequeños dejan de pagar el peor caso
  (3×0.2% + cand 0.5% con ρ̄=0.5: lineal vetaba 1.1%, medido 0.75%
  pasa).
- La precisión del veto depende de CL-7: si la EWMA registra mal, todo
  el grupo se agrega mal — el contrato CL-7 es ahora prerrequisito del
  veto (dependencia declarada).
- Tests de admisión que codificaban el doble-conteo legado se
  actualizaron preservando la doctrina (same-bet cuenta; evidencia
  faltante ≠ independencia).
