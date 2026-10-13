# Revisión final del consumidor OU R4 — 2026-10-08

**Dictamen:** aprobado el cambio acotado y verificado el contrato específico del archivo de regresión completo. El RED real demuestra la emisión indebida del consumidor sobre Ω28 y el GREEN demuestra su corrección para la trayectoria del test, sin congelar la actualización causal posterior. No se cierran los dos archivos de fuente, I5 completo, calibración OU ni la hipótesis económica. Esta revisión no ejecutó Cargo: verificó fuente fijada, logs y hashes producidos por integración.

## Corte y diff independiente

- Ω28: fuente `c0ccdf5f85453cc51f291ce7818101be3f561bcc`; fuente+docs `a5c69af3f7a05c8642b803c61a6ff72613d96100`.
- Merge de integración, padre inmediato del fix: `e584d3ef288505281e7cdcda09b8e3978eafdf74`, árbol `4eb4bf4c70d640ef643df8dd30970474ac156140`.
- Fix congelado: `eb32ac9613c847eec75bd8e2591add8ae88234cf`, árbol `10ecee6c773a37d6ac6a0ff7908f3c2c800d8ea0`, fecha de commit **2026-10-08 15:30:11 -0500**.
- Diff exacto padre→fix leído: dos archivos, 75 inserciones; seis líneas en el consumidor y alta del test de **69 líneas**. Sin eliminación, modificación del estimador SDE ni cambio de fórmulas, thresholds o firmas. Comparado también contra Ω28 para estos paths: misma diferencia.

| Path relativo | Blob antes | Blob del fix |
|---|---|---|
| `crates/strategy-core/src/multivariate_coint.rs` | `798b5c5377ba9fe7e64dfedc60c45d1897dfa4e9` | `9d642b5bccef78a77c24589220f41d05e944bb7f` |
| `crates/strategy-core/src/vecm_arbitrage.rs` | `2516f965c456e446e8af4387b517faa156de5cfc` | idéntico |
| `crates/strategy-core/tests/omega28_consumer_clock_contract.rs` | no existía | `f8812c55c5a5c2fbfb1152576c7e5886b5eea320` |

El test completo y sin alteraciones conserva SHA256 `e85e62ed78a1e6d02f9568d7058ae663d5d0dad5908a3f4c37b4fefcec92f8d1`. Coincide con la propuesta revisada, con RED, con GREEN y con el contenido del worktree de integración al verificarlo. El gate se ubica en el bloque SDE antes de `sde.update`: `count > 0 && timestamp_ms <= last_ts_ms` produce `None` sin escribir estado ni evaluar una intención. Si count=0, t=0 conserva su aceptación inicial. Se preserva la exclusividad opt-in de Ω28.

## Recibos reales y trazabilidad

Directorio de recibos leído: `C:/Users/jhona/.codex/worktrees/integration-recovery/Trader Gemini/target/integration-receipts/2026-10-08-refresh/`.

Se revisó `omega28-composition.json`, schema 4, snapshot leído con SHA256 `bcfc930940cd9a90d0dd58f25e586ca6feb1e850a2e9a262ffef9104dbebb058`. Es un recibo vivo que puede ampliarse; este hash identifica el corte leído, cuyo estado aún decía GREEN verificado y staged. La verificación posterior del objeto Git fijado confirmó que el árbol GREEN es exactamente el árbol del commit del fix. No se atribuye un test anterior al commit por coincidencia de nombre: se comparan los árboles y blobs concretos.

Comando común registrado, sin reejecución por este revisor:

`cargo +nightly test --release -p strategy-core --locked --test omega28_consumer_clock_contract -- --nocapture --test-threads=1`

Entorno registrado: `CARGO_TARGET_DIR=C:/Users/jhona/Documents/Proyectos/Trader Gemini/target`, `CARGO_BUILD_JOBS=1`, `TG_GENOME_ENV=backtest`; compilador `rustc 1.98.0-nightly (096694416 2026-06-29)`, host `x86_64-pc-windows-msvc`. Se usó el target compartido de integración, no una compilación hermética o una nueva plataforma.

| Evidencia | RED real | GREEN real |
|---|---|---|
| Log | `39-omega28-consumer-red.log` | `40-omega28-consumer-green.log` |
| Snapshot antes UTC | 2026-10-08T20:29:25.225147+00:00 | 2026-10-08T20:29:39.313244+00:00 |
| Snapshot después UTC | 2026-10-08T20:29:31.474011+00:00 | 2026-10-08T20:29:44.332232+00:00 |
| Árbol antes y después | `bd082fdb3d5a6f92a8cf742d570ea76eccbad4a8` | `10ecee6c773a37d6ac6a0ff7908f3c2c800d8ea0` |
| Fuente consumidor | Ω28, sin gate | gate de seis líneas |
| Resultado | exit 101, 0 pasan / 1 falla | exit 0, 1 pasa / 0 fallan |
| Compilación mostrada en log | 4.99 s | 3.55 s |
| Tiempo total del recibo | 5.419 s | 3.895 s |
| SHA256 del log, verificado | `7e7349d9f05dd3b75d52bf35796f80ae07c991512361ca6476264e36774b5e53` | `a9b62531a37f5f367b567802343b353523aba66d938bb3c049d12e78763226b1` |

RED falla exactamente en `omega28_consumer_clock_contract.rs:43:5`, mensaje `a duplicate observation must not emit a new intent`. Antes de ese punto se ejecutaron exitosamente inicialización t=0, nueve observaciones aceptadas, shock causal décimo que exige Short/Continuous/duración positiva, count=10 y z posterior sobre el umbral. Por ello el fallo no se explica por fixture fría, ausencia de señal, error de importación ni fallo de compilación. GREEN usa el mismo test: no se debilitó la precondición para obtener el resultado.

## Cobertura completa del archivo de regresión

Se leyeron las **69/69 líneas** del blob del fix, que contiene un test; se verificaron sus ramas de aserción contra ambos logs. Aprobación para el ledger: **un archivo verificado, exclusivamente como regresión del contrato temporal del consumidor**.

- Líneas 1–20: importación, motor opt-in y muestra inicial t=0; count identifica inicialización y no un sentinel artificial del reloj.
- Líneas 22–36: madurez y señal Short previas obligatorias; el z posterior excede el threshold. No existe un `if let Some` que vuelva vacua la premisa de emisión.
- Líneas 38–56: snapshot Debug derivado de ambos structs, incluidos momentos privados del SDE; duplicado y retrógrado deben producir None y conservar la representación completa. Esto es evidencia suficiente para esta trayectoria finita. No se presenta como igualdad universal bit a bit, validación de NaN ni contrato de serialización.
- Líneas 58–69: nuevo tiempo estrictamente creciente exige otra señal Short/Continuous/duración positiva, count=11, último valor y timestamp actualizados. Evita que una abstención permanente pase como solución.

La guarda demuestra por inspección el rechazo de cualquier timestamp no creciente del consumidor SDE inicializado, antes de toda actualización. El test ejerce un caso maduro por cada clase (igual y menor), con control causal positivo; no constituye enumeración de todo el dominio público ni de toda combinación de parámetros.

## Separación científica y límites conservados

**Aceptar o rechazar el reloj** es un contrato lógico de incorporación del dato y emisión: un evento duplicado/retrógrado no crea una intención a partir del score anterior. La corrección respeta ese contrato sin cambiar las fórmulas ni el comportamiento ante un tiempo creciente. “Señal válida previa” en el test significa que el motor emite bajo su API y fixture, no que exista alpha, confianza probabilística calibrada ni un trade rentable.

**Calibrar dt irregular** es otro problema: el SDE todavía acumula pares bajo distintos intervalos en una pendiente conjunta y transforma esa pendiente mediante el dt de la actualización actual (`vecm_arbitrage.rs:294–326`). Rechazar dt no positivo no demuestra que ese ajuste sea adecuado a intervalos positivos variables. La regresión nueva usa deltas aceptados de un segundo, por lo que no aporta evidencia de calibración bajo gaps, muestreo irregular o cambios de escala temporal. El lote posterior debe fijar el modelo de observación y unidades; contrastar transiciones por cada dt, sesgo/cobertura del estimador, sensibilidad a gaps y política de memoria con datos sintéticos controlados y evaluación sellada apropiada. Este informe no elige ni valida una nueva fórmula.

**Extensiones físicas o cuánticas** no forman parte del delta ni del recibo. Nombrar OU, Fokker–Planck o un método cuántico no añade una garantía estadística a estos tests. Cualquier propuesta posterior requiere hipótesis falsable, comparación con una referencia clásica, coste computacional y mejora verificable en el protocolo acordado. No se infiere una mejora económica a partir del guard temporal.

Siguen abiertos los dominios públicos alterables, entradas finitas extremas/overflow, parámetros no finitos del SDE, contador máximo, significado económico y calibración de magnitud/confianza/duración, calidad de los pesos de la cesta y producción multiactivo. La búsqueda previa no encontró caller productivo del opt-in; esta revisión no afirma un incidente del sistema vivo ni una cobertura de ejecución real.

Los resultados históricos T1 sobre `e9c6a435` permanecen históricos. El RED/GREEN específico no reemplaza checks o suites posteriores, no certifica T1 en este fix y no valida OOS, retorno neto por 72 horas ni el objetivo aspiracional 100%/72h. Los dos archivos de fuente permanecen abiertos en la auditoría aunque el test de regresión completo sea verificable dentro de este alcance.

Este informe es nuevo y no sustituye ni modifica `REVISION_OMEGA28_OU_2026-10-08.md`, cuyo residual y dictamen describen la fuente anterior al fix.
