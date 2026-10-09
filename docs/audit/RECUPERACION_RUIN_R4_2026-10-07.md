# R4 — Recuperación mínima del consumidor de clamp_ruin

Fecha: 2026-10-07, America/Bogota. Autor: Codex / quant_foundations.
Base científica contrastada: `adeb8d1b1f8171b14fffa8abc9f54c396e4794dd`.
HEAD del worktree receptor al editar: `42652f70badadb9be66c0fb138b88918c8cc54dc`.
Estado: **guard incorporado a `f3f8696`, con RED causal y GREEN 18/18**.
El oráculo, la publicación y la limpieza tienen recibos separados en
[la integración](../INTEGRACION_RAMAS_2026-10-07.md); no se certifica
validación económica ni toda la configuración de riesgo.

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

**La preparación inicial no ejecutó Rust/Cargo; la reejecución se registra
abajo.** Los logs históricos
std-only del helper existen y terminan RED 6/3 y GREEN 9/0 sobre la base
antigua; no son pruebas del consumidor ni del árbol integrado actual.
El coordinador separó test y fix para reejecutar RED/GREEN como se detalla
a continuación. All-targets, otras regresiones y T-1 tienen sus propios
recibos; esta nota no los sustituye.

## Reejecución RED/GREEN del consumidor

HEAD e índice de integración permanecieron en
`f3f86960190d99aeacdb75b14cdfab0adceaa186` / árbol
`462e1c69c4d08c7e95f22a81893ad23855bb5495`. Para RED se revirtió únicamente
el hunk del guard en el worktree aislado; el árbol de trabajo probado fue
`6d6eb1daf1e27056664f01aa347da208c355de09`, no el árbol corregido de HEAD.
No se creó commit ni ref del código RED.

- **RED:** 1 test ejecutado, 1 fallo esperado, exit 101. La salida contiene
  `nonfinite product admitted` para capitales 25 y 100, ambos Long/Short.
  Eso demuestra reapertura de exposición; no se deduce del mero cambio de
  diagnóstico del caso con capital 13. Duración total 129.430 s, incluido
  rebuild; ejecución del test 0.19 s.
- **Restauración:** `risk-engine/src/lib.rs` recuperó byte por byte SHA-256
  `888b974ecfb3cbf866c084cbdedcf1351049add0ce8ecb3cb1a1707610311e60`.
  El diff posterior estaba vacío; HEAD e índice no cambiaron.
- **GREEN:** C07 4/4, admisión no finita 7/7 y portafolio 7/7: 18/18,
  cero fallos, exit 0. Duración total 16.581 s, incluido rebuild.

Ambos usaron perfil release, `TG_GENOME_ENV=backtest`, un job y caché raíz.
`nightly` correspondía al compilador `1.98.0-nightly (096694416 2026-06-29)`
registrado en los recibos. Comandos exactos:

```text
cargo +nightly test --release -p risk-engine --locked --test c07_ruin_composition_contract c07_t02_nonfinite_micro_product_cannot_reopen_through_lerp -- --exact --nocapture --test-threads=1
cargo +nightly test --release -p risk-engine --locked --test nonfinite_admission_contract --test portfolio_admission_contract --test c07_ruin_composition_contract -- --test-threads=1
```

Hash SHA-256 del log RED:
`76384385b8bb66449bd3663b55c356e3427788f34c96b9f0a16a9f0563fa38d9`.
Hash del log GREEN:
`0096f4c77940c976140c482398820c6dac49d44bcd0aa9116de0363bc683e401`.
Los metadatos durables están en
[RECIBOS_INTEGRACION_R4_2026-10-07.json](RECIBOS_INTEGRACION_R4_2026-10-07.json).
Dos intentos previos se cancelaron durante compilación, antes de ejecutar
tests; no se cuentan como RED ni como resultados del contrato.

La revisión independiente no halló bloqueantes en el fixture: los tests de
admisión comparten mutex, usan arena/motor nuevos y verifican controles
saludables, causa exacta y EWMA intacta. T02 inyecta `-Inf`; no demuestra
overflow causado exclusivamente por operandos finitos. Los límites de
configuración e inferencia siguientes permanecen abiertos.

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

La preparación inicial de esta subauditoría no hizo commits, push, merge, cambios de procesos ni
limpieza de worktrees. Sólo preparó el hunk, C07 y esta nota bajo ownership
asignado por el coordinador.

## Revalidación del 8 de octubre

`7acf36aa` conserva el mismo blob corregido de `risk-engine/src/lib.rs`,
`554e02d060059d9d15247a2194df6d65f0f3ea36`, y la misma suite C07.
La nueva corrida de riesgo registra 54/54 sobre ese candidato: C07 4,
correlación 30, no finitos 7, portafolio 7 y vetos 6. Exit 0, 131.645 s,
incluidos compilación/enlace. No se reetiqueta el RED histórico de f3
como una ejecución sobre el SHA nuevo. Las validaciones posteriores
conservan sus propios árboles y límites en el recibo de integración.
