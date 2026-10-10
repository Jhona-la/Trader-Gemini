# Ronda 4: discontinuidades pendientes de justificación y contrafactual

Fecha: 2026-10-07, America/Bogota. Revisión estática nueva, sin cambios al motor.

Base inmutable: `adeb8d1b1f8171b14fffa8abc9f54c396e4794dd`.
Auditor: Codex / subagente `system_inventory`.
Estado de ambas fichas: **CONFIRMADO_ESTATICO_NO_REPLAY**.

Estas fichas confirman diferencias entre el contrato declarado y las expresiones
ejecutables. No demuestran pérdidas, oportunidades rentables rechazadas ni que
suavizar los límites mejore resultados. Una política de riesgo puede contener
discontinuidades justificadas. No se exige continuidad de la decisión final de
enviar una orden discreta: se señala la discontinuidad interna de la señal o del
presupuesto de margen respecto de una coordenada continua del mercado.

Los estados históricos «cerrado» de otras rondas no se trasladan a esta revisión.
Ninguna ficha equivale a haber revisado por completo los archivos citados.

## Huellas y alcance

| Archivo | Git blob de la base | Rango inspeccionado |
|---|---|---|
| `crates/god-engine-core/src/lib.rs` | `d5e196182972f1c3eff69093f621ff12e81673f2` | 6550–6691 |
| `crates/risk-engine/src/orchestrator.rs` | `a30856f4e03f1d989bb7c593b522b4017b8f9626` | 132–193, 235–250 |
| `crates/quantum-arena/src/spectral_regime.rs` | `cf8173ed817fef24ca4d0d78d27ae1ee4bed4987` | 86–123 |

Las líneas corresponden exclusivamente a la base. Repetir `git ls-tree` y localizar
los símbolos tras integrar cambios. La futura corrección de no finitos en
`PortfolioOrchestrator` no cierra por sí misma estas fichas: los contraejemplos
usan valores finitos y dentro de dominio. Al escribir esta ficha, el diff local
del orquestador contra la base estaba vacío. **Revalidación tras integración
necesaria**; no se ha observado todavía el resultado final del merge.

## R4-CONT-01: el filtro direccional conserva un corte en alineación táctica cero

- Severidad: **MED**, discrepancia de contrato e integración; impacto económico
  desconocido.
- Estado: **CONFIRMADO_ESTATICO_NO_REPLAY**.
- Responsable propuesto: consejo de integración / zona F3, pendiente de aceptar
  ownership en el documento de coordinación.
- Anclas: `lib.rs:6576–6578`, `lib.rs:6656–6657`.

El comentario en `lib.rs:6561–6562` describe sustitución de un veto binario por
modulación analítica de convicción. Sin embargo, `directional_flow_ok` se forma
mediante:

```rust
let is_momentum_expansion = macro_tide >= -0.05 && tactical_align >= 0.0;
let is_dip_pullback = macro_tide > 0.02 && tactical_align < 0.0 && wave_phase_ok;
let directional_flow_ok = is_momentum_expansion || is_dip_pullback;
```

Después, `!directional_flow_ok` convierte directamente la señal en `Flat` antes de
la modulación de convicción de las líneas 6659–6691.

Contraejemplo estático del bloque, con `epsilon = 1e-9`:

| Entrada | Caso A | Caso B |
|---|---:|---:|
| Señal a la entrada del bloque | Long, confianza 0.60 | Long, confianza 0.60 |
| `macro_tide` | 0.01 | 0.01 |
| `tactical_align` | +epsilon | −epsilon |
| `extreme_entropy`, `extreme_counter_tide`, `extreme_incoherence` | false | false |
| `is_momentum_expansion` | true | false |
| `is_dip_pullback` | false | false |
| Resultado local | conserva Long y entra a modulación | Flat |

`wave_phase_ok` no altera este testigo: `macro_tide > 0.02` es falso en ambos
casos. Para cualquier epsilon positivo arbitrariamente pequeño se mantiene la
diferencia. Esto identifica un corte real, sin atribuirle un resultado financiero.

Antes de cambiar conducta:

1. Revalidar el bloque en el commit integrado y construir un test de contrato
   que invoque la ruta real; la tabla no sustituye un replay.
2. Decidir si el corte es una protección económica intencionada. Si se mantiene,
   documentar su hipótesis, unidad, evidencia y excepción al contrato de suavidad.
3. Si se propone una transición continua, comparar veto actual y alternativa
   en replay determinista y OOS separado, registrando ambos lados, escalas,
   turn-over, costes, drawdown y masa de señales afectadas.
4. Ejecutar el oráculo y controles estadísticos de multiplicidad antes de
   promover la alternativa. Un test de continuidad por sí solo no valida alpha.

## R4-CONT-02: el presupuesto de margen salta al cambiar el signo de coherencia

- Severidad: **MED**, discontinuidad del presupuesto que afecta la admisión;
  impacto económico desconocido.
- Estado: **CONFIRMADO_ESTATICO_NO_REPLAY**.
- Responsable propuesto: consejo de riesgo / zona F4, coordinado con integración.
- Anclas: `orchestrator.rs:155–180`, `orchestrator.rs:239–250` y
  `spectral_regime.rs:112–123`.

Los comentarios en `orchestrator.rs:133–154` describen contracción continua y
simétrica. El cálculo de `crash_max` incluye sólo monedas con
`spectral_coherence < 0.0`; el de `squeeze_max` sólo las que tienen
`spectral_coherence > 0.0`. Después se aplica `0.25 * max(presion_local,
presion_sistemica)` al límite de margen. El salto ocurre en el propio límite,
antes de la decisión discreta legítima de admitir o rechazar una orden.

Contraejemplo estático de `allow_trade`, sin posiciones abiertas:

| Entrada / salida | Caso A | Caso B |
|---|---:|---:|
| Capital | 100 | 100 |
| Margen nuevo solicitado, Long | 80 | 80 |
| `margin_cushion(...)` antes de presión | 0.90 | 0.90 |
| `p_crash` | 0.10 | 0.10 |
| Única moneda con `spectral_crash_flux` | 0.60 | 0.60 |
| Su `spectral_coherence` | +1e-9 | −1e-9 |
| `crash_max` después del filtro | 0 | 0.60 |
| `directional_pressure` | 0.025 | 0.15 |
| Límite de margen en USD | 87.50 | 75.00 |
| Resultado local de admisión | true | false |

Supuestos restantes: estado financiero finito y válido, régimen `Range`, ningún
otro flujo extremo. `p_crash = 0.10` no dispara el veto sistémico de 0.90. El mismo
patrón existe para cortos en el filtro de signo contrario.

`crash_flux` no está obligado a tender a cero cuando la coherencia cruza cero:
su fórmula suma `0.35*accel_fast + 0.30*abs(carrier_tide) +
0.20*coherence_collapse + 0.15*micro_anti`. Por ejemplo,
`accel_fast=1`, `coherence_collapse=0.8`, `micro_anti=0.6` aportan 0.60
independientemente de la contribución de marea, que tiende a cero. Un campo
mixto puede conservar energía positiva al cancelar su coherencia firmada;
por tanto el guard de campo frío no elimina necesariamente este caso.

Antes de cambiar conducta:

1. Revalidar las comparaciones por signo después del merge de saneamiento de
   no finitos y añadir un test en `portfolio_admission_contract.rs` sobre el
   orquestador real, con ambos lados y casos alrededor de cero.
2. Medir frecuencia, persistencia y antigüedad de estos cruces en tape real;
   comprobar si monedas sin posiciones deben modular todo el portafolio.
3. Si se sustituye la clasificación por signo por presión firmada continua,
   justificar escala, saturación y extremos con datos; no inventar una pendiente
   para satisfacer únicamente una prueba local.
4. Comparar política actual y propuesta fuera de muestra con costes y riesgo,
   registrar rechazos contrafactuales y conservar los límites estructurales de
   margen. No autorizar despliegue a partir de esta inspección estática.
