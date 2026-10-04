# RA-OOS-F01 — Particionado estructural antes de evolución

Fecha: 2026-10-04. Rama propia `codex/oos-partition-2026-10-04`.
Base: `5a5438310677f1d62539d598238911f9829c5d87`, derivada de RA.
Estado: reparación local comprobada; no publicada ni integrada a main.
No se sustituye el hallazgo histórico RA§24.1: este documento añade su
reproducción, contrato y límites de cierre.

## 1. Fallo, ruta causal y consecuencia

El CLI de evolución sólo rechazaba len=0. Para len=1, la frontera histórica
floor(0,7·len) vale0. Crear slices vacíos es legal, pero los ensayos y el
reporte usan timestamps[train_len−1]. La resta underflow o el índice inválido
interrumpe el proceso sin explicar que no existe muestra IS. El reporte
también accede timestamps[train_len] para OOS, que requiere otro segmento.

El problema no demuestra pérdida de capital, una orden incorrecta ni ventaja
de un genoma. Sí viola el contrato de entrada de una herramienta que evalúa
y puede promover genomas. Fallar después de cargar modelos o consultar FRED
realiza trabajo innecesario y hace más difícil diagnosticar el verdadero error.

## 2. Reparación y demostración del invariante

`validate_is_oos_split(total_len, train_len)` devuelve la frontera sin
modificarla, o uno de cuatro errores estructurales:
EmptyTape, EmptyInSample, EmptyOutOfSample, SplitBeyondTape.

Éxito implica 0 < train_len < total_len. Por tanto:

- train_len−1 existe y está dentro del tape;
- train_len identifica la primera fila OOS dentro del tape;
- total_len−train_len es positivo y no underflow;
- ambas particiones suman exactamente el total.

Las comprobaciones comparan enteros; no multiplican ni suman longitudes.
Se conserva el cálculo 70/30 del caller y su límite de300000ticks. El guard
se coloca antes de NanoForest::load_global, OmniHistory::fetch, ensayos y
GenomeEnvelope::promote. El bloque de carga del modelo se mueve íntegro.
La apertura/mmap y configuración inicial todavía preceden al guard.

Para len≥2 del dominio actual, la misma frontera se conserva. Se acepta
2→1/1 como estructura válida, **no como soporte suficiente para aprender**.
No se introdujo un mínimo de trades, días, significancia o calentamiento.

## 3. Evidencia RED/GREEN realmente ejecutada

Comando std-only:

```powershell
rustc +nightly-2026-06-30 --test crates/backtest-engine/tests/oos_context_contract.rs -o target/oos-partition-evidence/green-contract.exe
./target/oos-partition-evidence/green-contract.exe --test-threads=1
```

RED:9 aprobadas,5 fallidas,0 ignoradas,exit101. GREEN:14/0/0,exit0.
Las mismas14pruebas se usaron en ambas fases.

El RED no fue una ejecución integral del CLI sin modificar: fue una
extracción mecánica del guard anterior en el helper real, que rechazaba
sólo total_len=0 y devolvía la frontera restante. Esa extracción se preserva
en target/oos-partition-evidence/red-production.rs. Los cinco fallos son
aserciones conductuales, no errores de compilación: una fila, IS vacío,
OOS vacío, frontera superior al tape y extremo usize.

Ocho contratos nuevos complementan los seis existentes. Se verifican
9999longitudes válidas de2 a10000, el borde2→1/1, tape vacío, segmentos
vacíos, split fuera de rango y usize::MAX. Los seis contratos anteriores
mantienen íntegro el prefijo IS y todas las filas OOS; no se eliminaron.

`cargo +nightly-2026-06-30 check --workspace --all-targets --locked --offline -j 2`
terminó exit0 en4m28s. Incluye compilación del CLI: **no lo ejecuta**,
no es backtest económico ni una segunda ejecución de las pruebas std-only.
Warnings existentes no se ocultaron; no cargo fix ni formato global.

## 4. Procedencia recuperable

SHA256 de bytes locales:

| Evidencia | SHA256 |
| --- | --- |
| Helper RED | 1E53F1859043CC95C4E728D51EC2FFF5CBD59D2727E85E4477CDFA9CBE81AEBC |
| Helper GREEN | 3C830A1F734AE9C0C8EAC13BCF4A10C94B736F349A61186527E855AE371DF53A |
| Tests idénticos RED/GREEN | 7F7F0041EB2B26D5F6FFA83D88E45C65A97494BDAB7D7F1527D57306D7E7FF85 |
| CLI GREEN | C28D34A748BC89863F33ED5AE90191D883576FD80DB31377AC8C89E4635BF1D4 |
| Log RED | 44F83F0DE212E109896F7B0FEB3F706C6FE5869CDACD083E4E32527D8A227B67 |
| Log GREEN | 7D2F982BF0061A4E3E0813F7E199845FCFAA51A10CC4649B59C891A870B7FF1F |
| Check all-targets | D4F8304B1A8CF5C3806F2C2D774F848A61E7845D93C7B93E759760C3F7F5CAD9 |

Blobs Git de la revisión estática independiente, cotejados22:24:32Z:
helper e0699fe5c547f6fe4cb8403c1e505c73d8a755f1;
tests429c56f8955724541e19b16986f7c1818c448c23;
CLIf1f520ff006cdd1dde17620c6613f2959fbbf3c9.
El revisor encontró0 nuevos bloqueadores, no ejecutó Cargo/rustc y no
certificó el resto del sistema. Parent obtuvo los controles y compilación.

## 5. Lo que continúa abierto

- RA-OOS-F02: período económico del panel OOS frente al contexto IS.
- Calidad/orden/duplicados del tape, truncado de bytes residuales, timestamps
  y soporte estadístico mínimo siguen siendo contratos distintos.
- El CLI conserva la política preexistente de retornar sin código de error
  específico ante datos rechazados. Una automatización no debe confundir
  terminación exitosa del proceso con ejecución de aprendizaje.
- No se ejecutó el binario con fixture de una fila para observar sus efectos
  externos. El orden del preflight se verificó en fuente y el CLI compiló.
- No se corrió T1 completo, entrenamiento, promoción, red, demo ni trading.
  Los recibos T1 anteriores de RA no se atribuyen a este nuevo código.
- Las etiquetas scalp/swing del serializador legacy no se erradican mediante
  este arreglo de índices. Su compatibilidad y consumidores requieren otra
  migración, no reemplazo textual indiscriminado.

## 6. Criterio de integración

Mantener la serie separada de RA/PR28 mientras su CI está en curso.
Antes de integrar: incorporar main vigente, revisar diff contra ambos
padres, compilar all-targets del candidato y ejecutar los contratos en CI.
Revisión cruzada no equivale a aprobación humana ni a evidencia financiera.
No borrar la rama mientras tenga exclusivos o esté ocupada. La validación
de soporte económico deberá expresar ausencia/insuficiencia, no inventar
métricas cero o rentabilidad para segmentos que no las soportan.

## 7. Integración local autorizada con RA74 (2026-10-04)

Esta adenda conserva los cortes anteriores como historia. El operador
autorizó el merge local, no publicación ni integración a main. Padres:

- OOS: be4cacf3a2344a838e4bbc29984e7d940ddb8d16.
- RA: 74be3ed561d155ea4d3e349c53fca46a5b6385a2, que contiene
  main2302278b496aeafbdb71d5516a39933426ea424f.

Se ejecutó `git merge --no-commit 74be3ed561d155ea4d3e349c53fca46a5b6385a2`
con estado previo limpio en el checkout oos-partition. El único conflicto
fue memoria, resuelto mediante apply_patch por unión, sin descartar historia.
La comprobación por subsecuencia exacta preservó1747/1747 líneas no vacías
del padre OOS y1762/1762 del padre RA en su orden original. Los otros
documentos/planes recibidos son idénticos al padre RA.

Se revisaron diffs contra CADA padre. Frente a RA, las únicas diferencias
funcionales son helper OOS, sus tests y src/bin/evolution.rs. Frente a OOS,
veto_registry.rs recibe sólo una ampliación descriptiva de unidades, sin
cambiar consumidor ni umbral. Las tres fuentes OOS permanecen exactamente
idénticas a9a3bb756529002b21f1e85e3571e2ffd9a6fbeac: mismos blobs y SHA256
de§4, sin nueva fórmula, mínimo económico ni política de promoción.

Comandos efectivos de esta integración, desde el checkout OOS:

```powershell
rustc +nightly-2026-06-30 --test crates/backtest-engine/tests/oos_context_contract.rs -o target/oos-integration-74be-20261004/oos-context-contract.exe
./target/oos-integration-74be-20261004/oos-context-contract.exe --test-threads=1
$env:CARGO_TARGET_DIR = 'C:/Users/jhona/Documents/Proyectos/Trader Gemini/target'
cargo +nightly-2026-06-30 check --workspace --all-targets --locked --offline -j2
```

Rustc:1.98.0-nightly (096694416 2026-06-29). Compilación std-only exit0;
ejecución14 aprobadas/0 fallidas/0 ignoradas, exit0. No RED nuevo:§3 conserva
el control negativo original. Check runner24234 exit0,22:46:31Z–22:48:49Z,
137,44s totales; Cargo informa2m15s, perfil dev [unoptimized + debuginfo].
CARGO_PROFILE_DEV_DEBUG y CARGO_PROFILE_DEV_OPT_LEVEL estuvieron unset;
sin override de perfil ni cambios a configuración. Warnings visibles,
sin cargo fix/fmt. Caché DEV compartida, ninguna fuente root/RA editada.

Recibos en target/oos-integration-74be-20261004 (ignorado), SHA256:

| Recibo | SHA256 |
| --- | --- |
| std-test.log | 215200DA2E69A79917B350227A08EC53D2A2B0128F7E214B7243693D14FF826E |
| workspace-check.log | 1870C6D0C95A99A407FA8C70C123B60812263EC6A08952CE63E2D6A8F2D72F04 |
| identity-union.log, previo a esta adenda | 535D0334E4DFE5318B9AC2C68642BF9DB08FF2091E6E4A31E58FB46CB300A99B |
| diff-parent-oos.patch.log, previo a esta adenda | 19B2038EF02B86409BC685CA1323CC146203B23D7CFD9BF59B802830E2476989 |
| diff-parent-ra.patch.log, previo a esta adenda | DC5D43C588223973119ABD2B8413514F28B4EA23CDB22194A1620CADF42AAB44 |

Main57cb8f0f, informado después por el parent como nuevo corte documental,
NO se incorpora ni queda validado por estos recibos. SA se implementa en
otro checkout y tampoco forma parte del merge. No se ejecutaron CLI,
modelos reales, T1, entrenamiento, promociones ni trading; no push/PR,
limpieza o cancelación de procesos. Los límites de§5 siguen abiertos.

## 8. Integración local autorizada con RA f275/main237 (2026-10-04)

Padres del nuevo merge local:

- OOS: 684e47082b57fb962cbd4fad05fac5b998be15ba.
- RA: f275bae39dd81531cfb71b8ddaadb1023f02c6f5, que incorpora
  main23701ecb05486eb13b661d3d50afaadbc9a0015a y el plan RA.

Se ejecutó `git merge --no-commit --no-ff f275bae39dd81531cfb71b8ddaadb1023f02c6f5`
con estado previo limpio, sólo en este checkout OOS. Memoria fue el único
conflicto y se resolvió con apply_patch por unión. No se tomó un padre entero
como sustituto del otro. El verificador reproducible
target/oos-integration-f275-20261004/verify-preservation.ps1 comprueba todos
los documentos que difieren entre padres:6 presentes en OOS y8 en RA.
Cada línea no vacía original se conserva exactamente y en orden; incluye
MEM/COORD, ambos informes RA, planes, barrido y ADR0014. Se comprueba también
el informe OOS anterior. No es una revisión semántica de sus afirmaciones.

Antes de esta adenda: memoria1782/1782 y1816/1816; coordinación3618/3618
y3703/3703, respectivamente. Los documentos recibidos de RA son idénticos
a f275; sólo se agregan recibos propios arriba en MEM/COORD y en este§8.
Los diffs del candidato contra CADA padre se revisan y guardan en target.
Frente a684e no hay delta Rust/Cargo/CI; frente aRA sólo las tres fuentes
OOS constituyen delta funcional. Mismo70/30, sin nuevo mínimo económico,
fórmula, modelo o política de promoción; permanecen los límites de§5.

Para evitar artefactos legacy compartidos se refrescaron únicamente los
mtimes de los tres archivos OOS que difieren de RA. Sus SHA256 se cotejaron
antes/después, idénticos a§4; sus blobs permanecen:

| Fuente | Blob Git |
| --- | --- |
| crates/backtest-engine/src/oos_context.rs | e0699fe5c547f6fe4cb8403c1e505c73d8a755f1 |
| crates/backtest-engine/tests/oos_context_contract.rs | 429c56f8955724541e19b16986f7c1818c448c23 |
| src/bin/evolution.rs | f1f520ff006cdd1dde17620c6613f2959fbbf3c9 |

Comandos efectivos, desde este checkout:

```powershell
rustc +nightly-2026-06-30 --test crates/backtest-engine/tests/oos_context_contract.rs -o target/oos-integration-f275-20261004/oos-context-contract.exe
./target/oos-integration-f275-20261004/oos-context-contract.exe --test-threads=1
$env:CARGO_TARGET_DIR = 'C:/Users/jhona/Documents/Proyectos/Trader Gemini/target'
cargo +nightly-2026-06-30 check --workspace --all-targets --locked --offline -j2
```

Binario std-only recién compilado, no reutilizado:14 aprobadas/0 fallidas/
0 ignoradas, exit0, importa el helper real actual; no los6contratos legacy.
No se repite ni se atribuye como nuevo el RED histórico de§3.
Check completo runner27697: exit0,23:29:17Z–23:30:50Z,92,89s totales;
Cargo informa1m30s y dev [unoptimized + debuginfo]. Overrides de perfil
DEV unset, configuración del proyecto intacta. Warnings visibles; sin
cargo fix/fmt. El runner finalizó normalmente y liberó su lock compartido;
se avisó al parent para su Cargo SA sin iniciar más Cargo propio.

SHA256 de recibos locales en target/oos-integration-f275-20261004:

| Recibo | SHA256 |
| --- | --- |
| source-refresh.log | 847A875F50F0340ADC64B471765DBB79A585522F2A9AD1D1F8B9142081778D4E |
| std-test.log | 8FDD06E670341FEA2263D20F4807616CDE90D666274D53F6030093A597C97BB3 |
| workspace-check.log | 3150A6E5385D2ADF7D3F21C07BF208D1F6D5F0B459690BC4B882519273C97740 |
| preservation-pre-docs.log | F1B39B8A5F1D8AD9F717F999CF3336AB546E1F94105BF8C1DFA6BEE855205FDA |

El corte es exactamente RA f275/main237, no otros avances documentales ni
el nuevo inventario en elaboración por otro worker. Se conserva el barrido
F0/F1 ya versionado, sin confundirlo con esa nueva cobertura. SA no forma
parte de este merge y su parser/pruebas no se ejecutaron aquí. No fetch,
push/PR, CLI/modelos reales, red, entrenamiento, promoción, trading, T1,
limpieza ni cancelaciones. No certificación económica o auditoría total.
