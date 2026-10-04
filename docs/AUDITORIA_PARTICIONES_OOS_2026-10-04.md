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

