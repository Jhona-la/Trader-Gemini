# R4 — Recuperación mínima del consumidor de clamp_ruin

Fecha: 2026-10-07, America/Bogota. Autor: Codex / quant_foundations.
Base científica contrastada: `adeb8d1b1f8171b14fffa8abc9f54c396e4794dd`.
HEAD del worktree receptor al editar: `42652f70badadb9be66c0fb138b88918c8cc54dc`.
Estado: **candidato local; RED/GREEN, compilación y T-1 pendientes**.

## Procedencia y decisión de recuperación

Se inspeccionó, sin modificarlo, el delta local de
`C:/Users/jhona/.codex/worktrees/ruin-input-contract/Trader Gemini`, base
`f608737803406e5e6f5cf0045755c1fbafb55558`. Conserva cambios no versionados
en `risk-engine/src/lib.rs`, `ruin.rs`, dos tests y dos recibos históricos.

El test C07 procede de
`C:/Users/jhona/.codex/worktrees/codex-review-2026-10-07/Trader Gemini`, HEAD
`a054495c70c54d83190337355aa35376cfae3c15`, archivo
`crates/risk-engine/tests/c07_ruin_composition_contract.rs`.
SHA-256 original:
`46c75f82e2465bb38c54668470c6479040c479e8426bac01c87ca6a285d21752`.

C07 y `ruin_input_admission_contract.rs` contienen los mismos cuatro cuerpos
de prueba y el mismo fixture; sólo diferían la cabecera y los nombres de las
funciones. Se recupera **una** suite, C07, para no duplicar cobertura ni
coste. Su cabecera se actualiza con procedencia y estatus pendiente: la
palabra histórica «rerun» no se utiliza como recibo de ejecución.
Los originales, el helper contract y sus logs permanecen en sus worktrees.

## Qué ya estaba integrado y qué faltaba

En `adeb8d1b`, `clamp_ruin` ya devuelve `+0.0` para fracciones no finitas
(F1-B4), conserva negativos/ceros finitos y contiene una aserción de NaN.
Copiar el `ruin.rs` viejo entero eliminaría esa aserción; el diff original
de ese archivo tampoco aplica limpio sobre la base actual. No se copió ni
se modificó `ruin.rs` en esta recuperación.

El consumidor de `lib.rs:522-529` aún multiplica `kelly_adjusted` por una
convicción que puede ser infinita, transforma el resultado inválido en cero
con el helper y mezcla ese cero con un Kelly estándar positivo. Con peso
micro menor que uno, la mezcla puede reabrir exposición. El guard posterior
de `raw_exposure` no reconoce la invalidez original si la mezcla ya es finita.

Caso de la suite: `min_confidence_btc=-Inf`, intención finita y Kelly finito
positivo. La convicción es `+Inf`; el helper ya integrado devuelve cero;
`lerp(standard,0,w)` es positivo para `w<1`. No se sostiene que todos los
gates posteriores admitan todas las configuraciones: eso lo debe medir C07.

## Delta recuperado

Único cambio productivo, tras re-grep del consumidor:

```rust
let micro_kelly_input = kelly_adjusted * conviccion;
// A zero returned for invalid size must not be blended with the
// positive standard candidate, which would reopen this rejected input.
if !micro_kelly_input.is_finite() {
    return rej(REJ_INVALID_INPUT);
}
let micro_kelly =
    crate::ruin::clamp_ruin(micro_kelly_input, crate::ruin::CONSERVATIVE_Q).max(0.0);
```

Sustituye la multiplicación inline anterior; no cambia ecuación, orden de
operaciones ni límites para productos finitos. El delta original de
`lib.rs` pasó `git apply --check` sobre el receptor; luego se portó sólo
este hunk con `apply_patch`. El rechazo conserva su causa antes de perderla
en un escalar cero. No se cambian umbrales de riesgo ni política de exploración.

## Alcance de C07 y comprobaciones pendientes

La suite llama directamente las APIs actuales `RiskEngine::evaluate_quantum_order`
y `RiskEnvelope::max_leverage` con arena/registro sintéticos. Incluye:

1. Controles saludables Long/Short y capitales 13/25/100, con volumen,
   apalancamiento y targets comprobados.
2. Producto micro no finito: exige Flat, volumen cero positivo, incremento
   exacto del diagnóstico `REJ_INVALID_INPUT` y EWMA de riesgo intacta.
3. Kelly crudo NaN: exige rechazo sin que el mínimo nocional lo rehabilite.
4. Overflow del riesgo mínimo en `max_leverage`: exige `(0,false)` y
   conserva un control de exploración finito.

Los globals del fixture se inicializan una vez y los tests de admisión se
serializan mediante mutex dentro del ejecutable. Se necesitan controles
saludables verdes para descartar que otro veto o un fixture obsoleto oculte
el defecto. La prueba no ejecuta órdenes, red, core vivo ni exchange.

**No se ha ejecutado Rust/Cargo en esta revisión.** Los logs históricos
std-only del helper existen y terminan RED 6/3 y GREEN 9/0 sobre la base
antigua; no son pruebas del consumidor ni del árbol integrado actual.
El coordinador separará test y fix para intentar RED previo al guard y
GREEN posterior, y registrará SHA, comando, salida, entorno y duración.
Antes de publicar: all-targets, contratos afectados, regresión y T-1
aplicable del árbol exacto. Esta nota no sustituye esos recibos.

## Límites y riesgos de compatibilidad

- No recuperamos el archivo `lib.rs` entero antiguo: se conservaron todos
  los cambios posteriores de main y Ω2/F1-B4.
- No se copió el test helper duplicativo ni su fórmula finita congelada;
  permanecen disponibles como evidencia histórica recuperable.
- El guard cubre **producto** no finito. No valida toda la configuración:
  `min_confidence_btc=NaN` puede perderse antes en `.max(0.0)`, y Kelly
  infinito puede saturarse o elegir bootstrap antes de este punto. Esos
  casos no se declaran resueltos.
- No cambia el significado de cero en todos los callers del helper, ni
  certifica `q` o la probabilidad de ruina. Las correcciones teóricas del
  docstring viejo de `ruin.rs` son útiles pero siguen pendientes de un
  port documental separado que preserve código y tests posteriores.
- Cambia el diagnóstico para entradas inválidas que antes terminaban con
  otro rechazo. La suite exige la causa específica; esto es deliberado.
- No se transfiere a main ninguna afirmación de rentabilidad, calibración
  estadística ni seguridad total a partir de este cambio numérico.

Esta subauditoría no hizo commits, push, merge, cambios de procesos ni
limpieza de worktrees. Sólo preparó el hunk, C07 y esta nota bajo ownership
asignado por el coordinador.
