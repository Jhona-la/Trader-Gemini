# Hallazgos de riesgo Ω93 — revisión acotada del 2026-10-13

**Base fija:** `3dc88468114205b0257b5e1fd82688a652dede2f`. Revisor: Codex / subagente de riesgo. Alcance: Hodge, predicado LH-02, conexiones productivas y acuse del arreglo de reloj LH-12. No es un cierre del archivo completo, de Fase 2, de calibración económica ni de rentabilidad.

**Propiedad:** Hodge/LH-02 pertenecen a la reserva AGY / Ω93; se entregan contraejemplos y criterios de contrato, sin editar esas fuentes ni sus controles. ROOT es dueño del candidato LH-12 y de planes/memoria comunes. No se ejecutaron Cargo, el motor, entrenamiento, fetch, cambios de refs ni limpieza. Las únicas escrituras de esta revisión son este informe y evidencias ignoradas bajo `logs/si-omega93-hodge-20261013/`.

## 1. Fuentes fijadas y alcance ejecutado

Las anclas de esta tabla corresponden a la base fija, no a números inferidos de otra rama.

| Ruta del repositorio | Blob Git | Anclas utilizadas |
|---|---|---|
| `crates/risk-engine/src/hodge.rs` | `4e4f2155e8abcf52e16ffdd7a33ef7721176ab46` | 84–107 observabilidad y flujo; 125–169 selección K_r; 172–184 fallback r>64 |
| `crates/god-engine-core/src/contagion_publisher.rs` | `fae80967817caeb1e4f8339f5795a060a34aa91c` | 54–59 productor Hawkes; 78–86 cálculo/publicación |
| `crates/risk-engine/src/correlation_guard.rs` | `e51b53de148d83d69be3065d42b7d80a738b6880` | 694–713 lectura y modificación de correlación de riesgo |
| `crates/risk-engine/src/lib.rs` | `35f0317945a17dcc0f6fcaa246132dc784d64822` | 1142–1165 probabilidad y EV; 1210–1229 unidades; 1405–1434 fee y nocional final |
| `crates/risk-engine/tests/cf2_riesgo_solucion_conocida.rs` | `1c1be73cfdc9ff5b6a009707f86729d346d2b35e` | 489–500 cadena censurada; 503–519 par observado equilibrado |
| `crates/god-engine-core/src/lib.rs` | `3c3a500aacc5e72bdd02218702b1c96507f0e747` | 4348 etiqueta neta; 4475–4489 entrenamiento; 8171–8189 consumo |
| `crates/god-engine-core/tests/lg_calibracion_ev_contract.rs` | `23488070551ae5e277d30c9bde839698027f1c4d` | 21–51 fixture económica LG-5, ignorada en línea 25 |
| `crates/god-engine-core/src/calibration.rs` | `b40e35a98fbf4e86ebf15e43c0b67287f2f8874e` | 147 API `calibrate_at_tau` de Ω93 |

Se extrajo `hodge.rs` con `git cat-file blob` mediante copia binaria de stdout. `git hash-object --no-filters logs/si-omega93-hodge-20261013/hodge_exact.rs` produjo exactamente el blob `4e4f2155…`: no se transcribió ni sustituyó el algoritmo. El archivo usa sólo std; el harness lo incluye con `#[path = "hodge_exact.rs"] mod hodge;`, por lo que ejecuta tanto la implementación original como sus seis tests internos. No hace falta imitar dependencias externas.

Compilador observado: `rustc 1.98.0-nightly (096694416 2026-06-29)`, host `x86_64-pc-windows-msvc`, LLVM 22.1.8. Desde el worktree se ejecutó:

```powershell
rustc --edition=2021 --test logs/si-omega93-hodge-20261013/harness.rs -o logs/si-omega93-hodge-20261013/hodge_witness.exe
logs/si-omega93-hodge-20261013/hodge_witness.exe --nocapture --test-threads=1
```

Compilación exit 0. Ejecución exit 101: **10 tests pasaron y 2 contratos semánticos fallaron, intencionalmente RED sobre la fuente intacta**. No se interpreta como suite global fallida ni como una implementación corregida. Los diez verdes son los seis tests originales y cuatro controles de resultados observados. Tiempo reportado por el runner 0,03 s: es duración de esta prueba, no benchmark ni garantía de latencia.

Evidencias: `hodge_exact.rs`, `harness.rs`, `rustc-version.log`, `compile.log`, `metadata.log`, `hashes.json`, `results.log` y ejecutable local. `git check-ignore -v logs/si-omega93-hodge-20261013/harness.rs` confirmó `.gitignore:21:logs/`. Los logs no forman parte del commit del informe; se conservan localmente para revisión.

## 2. Hodge: la topología no puede deducirse de que exista un valor negativo

**Hallazgo de contrato reproducido, con límite explícito.** Para la misma cadena medida A–B–C y el mismo flujo antisimétrico, el resultado depende de si el llamador entrega Hawkes crudo o su antisimetrización:

```text
Hawkes crudo R = [[0,4,0], [0,0,4], [0,0,0]]
F = (R-Rᵀ)/2 = [[0,2,0], [-2,0,2], [0,-2,0]]
```

Ambos producen flujos de arista 2 y 2, divergencia (2,0,−2) y energía 8. En el grafo cadena existe potencial (4,2,0); todo el flujo es gradiente y la energía residual de ciclo es cero. Rust devuelve **0 para R** y **0,66666666666666674 para F**. Test rojo: `semantic_same_path_and_flow_should_survive_antisymmetrization`.

La causa está en `has_negative` (125–138) y `is_complete = has_negative || …` (152): F selecciona K₃, cuya energía proyectada es 8/3. La señal negativa informa sobre orientación; no demuestra que A–C se haya observado. La API sólo recibe una matriz y no recibe la máscara de observación.

Este rojo conserva externamente la topología A–B–C. **No prueba que el productor real entregue F**, pues `contagion_publisher` entrega la matriz Hawkes cruda. Si el contrato intencional de una matriz antisimétrica fuera “todas las parejas observadas, también ceros”, el mismo F significaría otro grafo y el 2/3 sería correcto. El defecto es la imposibilidad de representar ambos significados sin perder información y la insuficiencia de una promesa general de invariancia a antisimetrizar.

**Control que impide un arreglo destructivo:** con `R = [[0,4,5], [0,0,4], [5,0,0]]`, A–C sí fue observado, aunque su flujo neto sea cero. El grafo es K₃ y el 2/3 es correcto. El harness confirmó ese resultado; la base ya tiene `cf2_hodge_par_medido_equilibrado_sigue_en_k3`. Los seis tests originales también pasaron: gradiente denso, ciclo puro, contornos, mezcla monótona, perturbación simétrica y caso multi-activo n=8.

**Decisión para AGY:** explicitar modo/topología o máscara junto al flujo y conservar la diferencia entre par ausente y par observado equilibrado. No basta retirar `has_negative` ni tratar todos los ceros como ausencia: eso rompería el significado denso y el caso equilibrado. La aceptación debe incluir esos controles, invariancia con máscara preservada y entradas censuradas reales. No se propone aquí un nuevo umbral ni cambio de gate.

## 3. Hodge: el fallback r>64 no calcula la proyección del grafo censurado

**Fallo matemático reproducido dentro del dominio que la función acepta.** Matriz 65×65 con `R[i][i+1]=4` para i=0…63 y todo lo demás cero. Es una sola cadena, con 65 nodos activos y 64 aristas: no tiene subespacio de ciclos. Su energía es 64·2²=256, divergencia (2,0,…,0,−2), suma de divergencias al cuadrado 8. La proyección sobre su propio grafo explica toda la energía.

La rama 174–184 usa en cambio `E_grad=8/65`, fórmula de K₆₅, y devuelve:

```text
1 − (8/65)/256 = 1 − 2/(65·64) = 0,99951923076923077
```

Rust reprodujo exactamente ese valor. Test rojo: `semantic_tree65_has_no_cycle_subspace`. El clamp [0,1] limita el rango, pero no convierte la fórmula de K₆₅ en la del árbol.

El hallazgo no demuestra que el roster productivo actual tenga 65 nodos activos. Sí demuestra que la función tiene una rama aceptada para ese tamaño que publica un número con otra semántica, pese al soporte heap para n>64. AGY debe decidir y documentar el dominio soportado y su comportamiento fuera de él, o resolver el grafo general con un presupuesto medido. Devolver ausencia sin estudiar el consumidor tampoco equivale automáticamente a un control seguro: éste usa 0 cuando falta el valor. No se prescribe eliminar controles ni ampliar recursos sin medición.

### 3.1 Controles ejecutados de componentes desconectadas — base 3dc, 2026-10-13

**No se encontró un defecto adicional de anclaje por componente.** El mismo blob exacto `4e4f2155e8abcf52e16ffdd7a33ef7721176ab46` de 3dc recorre cada componente por BFS (194–223) y fija un nodo a tierra dentro de ella (229). Se compiló directamente con `rustc --edition=2021 --test` el nuevo `logs/si-omega93-hodge-20261013/components_harness.rs`, que incluye ese archivo intacto: **compilación exit 0; 10 tests pasan, 0 fallan** (los seis originales y cuatro controles nuevos).

| Grafo Hawkes crudo no negativo | Oráculo de energía | Resultado Rust |
|---|---|---:|
| Dos aristas desconectadas, flujos 2 y 3 | E=4+9=13, residuo=0 | 0 |
| Dos cadenas de tres nodos, flujos 2/2 y 3/3 | E=8+18=26, residuo=0 | 0 |
| Cadena gradiente de energía 8 y ciclo disjunto de energía 3 | curl=3/(8+3) | 0,27272727272727271 |

Agregar un nodo aislado al tercer caso conserva 3/11. Se enumeraron todas las permutaciones de nodos de los tres grafos: 24, 720 y 720, respectivamente; **1.464 permutaciones**, todas conformes al oráculo, con error absoluto máximo observado 0 respecto de los valores f64 esperados. La unión disjunta usa la suma de energías de componentes con soportes de aristas disjuntos; no promedia sus porcentajes de curl.

Evidencia separada: `components_compile.log`, `components_results.log`, `components_metadata.log`, `components_hashes.json` y `hodge_components_witness.exe`, todos en el directorio ignorado anterior. Se conservaron el harness, ejecutable y resultados previos. Estos controles cubren matrices crudas de hasta siete nodos; no corrigen ni invalidan los contraejemplos de máscara antisimétrica y fallback de 65 nodos. Sin Cargo ni cambios de fuentes.
## 4. Conexión productiva y límite de la evidencia

`contagion_publisher.rs:54–59` construye la matriz real con `hawkes_cross::contagion_matrix`; 83–86 llama a Hodge y publica `hawkes_contagion_curl_share`. `correlation_guard.rs:694–701` consume esa misma clave y calcula:

```text
systemic_rho = clamp(base_rho + (1 − base_rho)·curl_share², −1, 1)
```

Luego compone el inflado de cola en 712–713. Por tanto, el comentario del publicador “No veta nada todavía” (81–82) está desactualizado: la señal sí modifica una entrada del control de correlación de riesgo. No se confunde esta clave global con la feature por moneda `hodge_curl_share`.

Esta es evidencia de conexión por código, no una medición de órdenes rechazadas, P&L ni probabilidad de ruina. El harness no ejecutó el publisher, registry, correlation guard ni motor completo. La ambigüedad de antisimetrización no se atribuye al llamador Hawkes actual; el caso 65 no se atribuye al roster real sin datos.

## 5. LH-02: el predicado no certifica que hubo rescate por nocional mínimo

**Contraejemplo algebraico del predicado, sin ejecución de `evaluate_quantum_order`.** En `risk-engine/src/lib.rs:1419–1423`:

```text
rescued = L > 1 && margin <= 1,05·safe_min_notional && max_acceptable_fee_pct >= 0,005
effective_limit = rescued ? max(max_fee_limit, 0,035) : max_fee_limit
```

Tomar margen final USD 5,20, mínimo seguro USD 5,10, L=10, fee roundtrip por nocional f=0,0012 y presupuesto normal de margen 0,01. Para comparar con ese presupuesto, fijar `max_fee_limit=0,01`, es decir el extremo `micro_w_alloc=0` del lerp de 1217–1218.

| Magnitud | Valor | Consecuencia algebraica |
|---|---:|---|
| Nocional incluso a 1x | USD 5,20 | Ya supera USD 5,10; ese mínimo no exige apalancamiento |
| Cota del predicado | 1,05·5,10 = USD 5,355 | El margen 5,20 satisface el predicado |
| Nocional a 10x | USD 52,00 | No demuestra que L=10 surgiera de un rescate |
| Comisión roundtrip | 52·0,0012 = USD 0,0624 | 0,0624/5,20 = 0,012 = 1,2% del margen |
| Presupuesto y límite efectivo | 1% → 3,5% del margen | Este bloque permite 1,2%, aunque excede el presupuesto ordinario |

El factor L de `final_fee_impact = roundtrip_fee * dynamic_leverage` (1405) es dimensionalmente coherente: convierte fee/nocional en fee/margen. `0,005` son **0,5%**, no 0,005% ni USD. El problema demostrado es llamar “rescatada” a una condición que no prueba necesidad ni historia del rescate; no una inconsistencia de esa multiplicación.

La tabla no demuestra que todos esos valores coexistan después del sizing completo ni que una orden real se admita: otros controles pueden rechazarla antes o después. El test existente `veto_logic_contracts.rs:235–252` cubre admisión a capital 50, y `leverage_admission_contract.rs:96–109` cubre presupuesto 0,0001 con resultado Flat; ninguno establece el caso anterior para presupuestos ordinarios, ni el segundo exige un motivo exacto de rechazo.

**Criterio de aceptación para el dueño:** separar la política de tolerancia para posiciones pequeñas de la afirmación “necesitó rescate”, y probar comparativamente una orden con y otra sin necesidad real de rescate, manteniendo el gate terminal de fees y el de nocional. Si la intención es rastrear rescates efectivos, necesita evidencia de la transformación antes/después, no sólo magnitudes finales. La elección de 0,5%, 3,5% o del margen de 5% no queda validada económicamente por esta álgebra; no se inventan sustitutos ni se deducen del objetivo +100%/72h.

## 6. Acuse LH-12 y pendientes económicos distintos

Se revisó de nuevo el candidato ROOT en el worktree `spectral-evidence-1011`, HEAD `04eae49bab4b1747c6896bab6381c427d0c8d8d9`, con la composición Ω93 pendiente de commit. Huellas `git hash-object --no-filters` de los archivos leídos: `calibration.rs` = `d17161634bd73429eeb14a8b920425c321ffcb93`; `estado_aprendido.rs` = `19b16230e481e177301c653439c1d2d2d3270d60`; `platt_evidence_restore_contract.rs` = `c341c5a1376dfcb4dc1d79fb7d9e1e489d366120`. Son huellas del candidato local, no blobs atribuidos a main3dc.

**Acuse del contraejemplo de reloj expulsado:** `evidence_clock_ms` / `restore_evidence_clock_ms` (103–119) conservan un reloj máximo independiente del anillo y no permiten rebobinarlo; al avanzar refit, preservando las fechas explícitas. El snapshot guarda el máximo por moneda y global (estado 240–241 y 261); restore reproduce las filas y luego restaura el máximo (332–349). `validar` rechaza un reloj anterior a cualquier fila retenida (410–421). Así, expulsar la fila que fijó el máximo no rejuvenece la evidencia después de un reinicio.

Los campos `Option` con `serde(default)` mantienen lectura del JSON viejo; sin ellos sólo puede inferirse el máximo retenido, no recuperar un reloj que nunca fue guardado. Los tres contratos nuevos cubren máximo expulsado con cierres tardíos, JSON legado sin campos y reloj inválido. Esta revisión es de código; no ejecutó Cargo ni atribuye una suite de integración verde a ROOT. El método de restore directo conserva la precondición de estado validado; no se convierte por sí mismo en una API de validación integral.

**LG-5 sigue abierto.** El entrenamiento productivo etiqueta `net_trade_pnl > 0` (core 4348) y alimenta por moneda/global (4484–4489); la probabilidad consultada se adjunta al intent (8181–8189). Riesgo utiliza una probabilidad puntual válida en 1142–1146; sólo la rama alternativa usa `win_rate_hierarchical_lcb` en 1154–1162. No es correcto afirmar que todas las decisiones EV ya usan una LCB.

El fixture LG-5 conservado e ignorado describe 15% TP a +2,25R, 40% salidas positivas a +0,02R y 45% SL a −1R. Su media es −0,1045R, aunque la fracción de cierres positivos es 55%; tratar 0,55 como P(TP primero) produciría 0,55·2,25−0,45=+0,7875R. Son cantidades distintas. Es una demostración algebraica del problema de etiqueta, no el resultado ejecutado aquí de Platt. Envejecer correctamente la evidencia y restaurar su reloj no cambia esa semántica de las etiquetas.

**APIs no equivalen a lazos productivos.** En la base3dc, `git grep calibrate_at_tau -- crates` sólo encuentra su definición en calibration.rs:147; el núcleo aún llama `calibrate_at`. `DetectorPerdidaDeVentaja` se encuentra en su implementación y tests de `resultado_orden.rs`, sin consumidores productivos identificados en crates. No se atribuye a esas APIs una mejora operativa de adaptabilidad, velocidad de reacción, EV o rentabilidad aún no cableada y medida.

## 7. Disposición de los hallazgos

| Hallazgo | Evidencia terminada | Pendiente / dueño |
|---|---|---|
| Hodge: topología perdida al antisimetrizar | Rust exacto: cadena 0 → 2/3; dense y par equilibrado preservados | AGY: fijar contrato de máscara/modo y aceptación, antes de cambiar fuentes |
| Hodge: cadena65 usa K_r | Rust exacto: árbol → 0,999519… | AGY: dominio y solver/fallback coherente, con medición de recursos |
| LH-02: condición final no acredita rescate | Álgebra y lectura del predicado; unidades comprobadas | AGY: fixture integral de ambas trayectorias y política explícita |
| LH-12: reloj máximo fuera del anillo | Lectura de arreglo ROOT y tres contratos | ROOT: validación Cargo/composición/CI por su cadena, sin duplicarla aquí |
| LG-5 / probabilidad puntual frente a LCB | Etiqueta, consumidor y fixture existente identificados | Codex + Línea E y coordinación G: objetivo económico coherente y contratos propios |

Ninguno de estos resultados demuestra o garantiza duplicación del capital en 72 horas. Tampoco autoriza a suprimir vetos: cada cambio requiere preservar invariantes numéricos, unidades y comportamiento del consumidor, y evaluar por separado sus consecuencias económicas.
## 8. Adenda 2026-10-13 — revisión independiente de 98aadd2 frente a 3dc

**Nueva base exclusiva de esta adenda:** `98aadd2ef21abae85e68e81f10598c6660ec7c7f`, comparada con `3dc88468114205b0257b5e1fd82688a652dede2f`. No reinterpreta como hallazgos de 3dc los cambios posteriores. Alcance de sólo lectura: tres fuentes de riesgo y dos contratos modificados, contrastados con sus productores/consumidores. Sin Cargo, harness nuevo ni ejecución del motor. Los contraejemplos de esta sección son derivaciones del código, no resultados de una suite ejecutada.

| Fuente en 98aadd2 | Blob |
|---|---|
| `crates/risk-engine/src/lib.rs` | `3842e9c82def1d48d4eed54adde8193868ae3b89` |
| `crates/risk-engine/src/leverage_matrix.rs` | `ae62e9c8ef790afcd28eeeb6c3307b2230e54f44` |
| `crates/risk-engine/src/veto_registry.rs` | `8d19a69ed56af8fb4661579fd302ee394007149e` |
| `crates/risk-engine/tests/cf3_confianza_tamano_contract.rs` | `e22c45cfa138a13ee29af27a298d904809549f7f` |
| `crates/risk-engine/tests/cf3_registro_consejo.rs` | `5dc229b3e63cc98969caa27868a1976eee3c52fd` |
| `crates/metacortex-engine/src/consejo_seniors.rs` | `256d536ff30c1cb05fafa7e0ab8eaf4ade284661` |
| `crates/god-engine-core/src/lib.rs` | `3c3a500aacc5e72bdd02218702b1c96507f0e747` |

`git diff --quiet 3dc… 98aadd2… -- crates/metacortex-engine/src/consejo_seniors.rs crates/god-engine-core/src/lib.rs` dio exit 0: el consejo ejecutable, la etiqueta de entrenamiento y el productor de intención no cambiaron en este diff.

### 8.1 Los contratos no se aflojaron, pero su cobertura es estrecha

En ambos tests se retiró `#[ignore]` y se actualizaron comentarios. No se cambiaron fixtures, tolerancias ni aserciones. El de sizing conserva p=0,9, capital USD 1.000, confianza cruda 0,80/0,95 y tolerancia relativa 1e−6 sobre el nocional. El de registro sólo exige que los campos de texto contengan «consejo» y cada nombre de asiento; no evalúa sus condiciones, estado real, umbral ni clasificación. Activar esos tests es distinto de acreditar cobertura económica o semántica completa.

### 8.2 Sizing: corrección parcial útil, con una dependencia cruda restante

`lib.rs:558–584` usa la probabilidad adjunta cuando es positiva para la convicción micro y el multiplicador de exposición; `leverage_matrix.rs:161–162` la usa para su convicción. Eso elimina tres dependencias directas del score en esas fórmulas y es coherente con el contrato estrecho de igual p bajo diferente escala del score. El preflight de `lib.rs:280–283` sigue exigiendo valores finitos en [0,1]. Todos estos factores son adimensionales; exposición sigue en moneda al multiplicarlos por capital. No se detecta aquí un nuevo error de unidades ni bypass de NaN/Inf en la entrada productiva de riesgo.

**Cierre incompleto de alcance F3.3-08:** `lib.rs:1299–1320` aún usa `intent.confidence` para el techo de apalancamiento de ambos rescates:

```text
x = clamp((confidence − 0,65)/0,10, 0,1)
micro_cap = 5 + 1,5·x²·(3 − 2x)
max_cap = log_lerp(50, micro_cap, micro_w_alloc)
```

Con `micro_w_alloc=1`, confianza 0,70 produce cap 5,75 y confianza 0,75 produce 6,5, aunque p sea idéntica. Ejemplo local: margen USD 1,00, L inicial 4, mínimo USD 5,00, mínimo seguro USD 5,10 y techo genómico ≥6. El candidato `ceil((5/1)·1,02)=6` termina en L=5 o L=6 por `min(max_cap).floor()`. El ajuste posterior al mínimo seguro puede entonces llevar margen/nocional a 1,02/5,10 o 1,00/6,00. Es un contraejemplo algebraico a que hayan desaparecido todas las vías de sizing por score; no acredita admisión final de ambas órdenes ni representa una regresión introducida ahora: la vía quedó sin corregir.

El fixture activado no cubre esa vía: capital 1.000 está fuera del extremo micro y ambas confianzas 0,80/0,95 saturan x=1. Para cerrar el alcance hace falta un contrato de comportamiento sobre rescates con el mismo p, ambos scores por encima de la puerta de admisión y dentro de la zona donde cambia el cap, comprobando margen, L, nocional y razón de rechazo. No se propone eliminar el gate de confianza ni sustituir su umbral sin contrato.

**LG-5 permanece:** el p sigue siendo puntual, aprendido con `net_pnl>0`; esta sustitución no lo transforma en P(TP primero), en esperanza neta ni en LCB. El contraejemplo −0,1045R frente al uso binario +0,7875R de §6 sigue siendo relevante. Tampoco se deriva una fracción óptima de multiplicar Kelly por p y por una convicción afín: el diff preserva una política heurística y no aporta una validación económica de esa forma. En particular, `signal_prob − min_confidence_btc` ahora resta una probabilidad a un umbral de score cuya puerta aún compara la confianza cruda (`lib.rs:1037,1059–1063,1112–1113`). No hay error dimensional, pero ya no representa automáticamente el “exceso sobre la puerta” que afirma el comentario 552–557 sin una transformación de calibración entre ambos espacios.

### 8.3 Registro nuevo: estado, condición y origen no coinciden con el código

**R-98-01 — AuditorInterno catalogado como veto activo aunque no veta.** `veto_registry.rs:464–473` añade `V-CNCL-006` como Activo. `consejo_seniors.rs:928–942` tiene `let is_veto = false` incondicional, precisamente para no duplicar Causal. Cualquier payload válido sirve de contraejemplo a la capacidad de veto atribuida. El test de cobertura arrastra una descripción anterior (`cf3_registro_consejo.rs:25`, `do_calculus_risk>0,92`) y una lista de seis asientos que no revisa el estado ejecutable. Una entrada de linaje puede ser útil; declararla Activo contradice el enum, que define Activo como activo en la ruta y ofrece Retirado para conservar el linaje.

**R-98-02 — Teleonomía: falta el valor absoluto en la causa registrada.** `veto_registry.rs:454` dice utilidad bajo 0,02 con WR<0,35. La condición real en `consejo_seniors.rs:900` es `abs(utility)<0,02 && wr<0,35`. Con utility=−0,2 y wr=0,2, la descripción simple “bajo el piso” es verdadera y el código no veta. Ese utility se obtiene localmente con Hurst=0,5, ml_prob=0,25, ml_model_base=0,5, slippage=0 y riesgo causal=0: el término espectral se anula y queda (−0,5)·0,4=−0,2. No se está proponiendo cambiar la política de utilidad; se pide describirla fielmente. El código advierte además en 892–893 que ese lift no es una esperanza económica calibrada.

**R-98-03 — Orígenes “medidos” no respaldados y drawdown clasificado como lógica.** El contrato del campo `fuente_umbral` (`veto_registry.rs:45–46`) distingue dato vivo de literal. Las nuevas entradas confunden el dato observado con el origen del umbral:

- Cascada: registro 407 afirma P99; la implementación leída usa `CouncilParams::default().cascade_severity_breaker=0,85` (consejo 1011). No se identificó en los productores revisados un estimador P99 que justifique esa atribución.
- Causal: registro 419 dice umbral derivado de VPIN; el productor core lo escribe como literal 0,75 (8264) y el consejo sólo lo clampa a [0,60,0,90] (675). Que VPIN sea el dato comparado no hace adaptativo ese umbral.
- Riesgo: registro 431 dice cota calibrada; consejo 720–721 calcula el literal `1−0,05^(1/3)` y una fórmula de política sobre s. No es una estimación estadística de la cota de ruina con incertidumbre.
- Ejecución: registro 443 describe el slippage observado, mientras el umbral real es `35+65·s` (769). Son magnitudes distintas del contrato de procedencia.
- Teleonomía: registro 455 llama medido al piso; consejo 1032/1036 declara constantes 0,02/0,35. Comparar contra WR observado no estima las constantes.

Además `V-CNCL-003`, veto de drawdown irrevocable, tiene `ClaseVeto::Logica` (434), aunque el enum 20–22 incluye explícitamente drawdown en RiesgoDuro. Esto lo excluye de la exigencia de `test.is_some()` que sólo recorre riesgo-duro activo en 703–712. No es una relajación demostrada del motor, pero sí una clasificación incorrecta con efecto en la verificación del inventario. Etiquetar todo como “medido” también alimenta el contador textual `fuentes_de_umbral_censadas` (745–754) sin acreditar evidencia de medición.

### 8.4 Atribución preservada y decisión de cierre

El diff no modifica los códigos REJ, la entrada `V-LOGIC-007` ni su test, y conserva el comportamiento de `deliberar_traced` y su `vetoed_by` (consejo 1285–1312). No hay evidencia aquí de pérdida nueva de atribución **runtime**. Las deficiencias demostradas están en el nuevo inventario y en declarar completa la independencia del sizing respecto del score.

AGY conserva la reserva de las fuentes. La aceptación de esta ola requiere corregir estado/origen/clase/condición del inventario y distinguir cobertura de nombres de contratos de comportamiento. La invariancia del sizing micro y la semántica económica LG-5 necesitan pruebas separadas. Esta adenda no ejecuta ni declara verdes esas pruebas, no cambia umbrales y no toca controles ni fuentes.