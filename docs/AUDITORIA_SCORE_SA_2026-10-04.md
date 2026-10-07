# El SA premiaba mayor drawdown cuando el crecimiento era negativo

Informe RA-SA-F01 · 2026-10-04 · rama `codex/sa-score-monotonicity-2026-10-04`.
Corte matemático: 22:51Z. Base inspeccionada: `57cb8f0fdaefbef177ff131b66851aa000de621f`.
Documento nuevo; no sustituye los informes raíz ni elimina sus hallazgos.

## 1. Resultado y alcance

Se reproduce y repara localmente una inversión de preferencia en el optimizador
Simulated Annealing (SA) de `src/bin/evolution.rs`: a igual capital terminal
perdedor, el candidato con mayor drawdown obtenía **más** score. Maximizar ese
score podía favorecerlo; no se ha demostrado que haya sido promovido u operado.

La corrección preserva exactamente la multiplicación antigua para scores
positivos. Para scores negativos divide por la misma retención, sin cambiar
el umbral, la exponencial, sus constantes o las penalizaciones posteriores.
Los 13 contratos independientes pasan; la misma suite, antes de la reparación
y tras corregir un error de expectativa del test, daba 8 aprobados/5 fallidos.

En este corte aún faltan el recibo de compilación completa y la revisión cruzada.
No hay ejecución del CLI, entrenamiento, promoción, trading, medición OOS de
rentabilidad ni publicación de esta rama. La frontera OOS es una serie separada,
`codex/oos-partition-2026-10-04`, no una reparación implícita de este parche.
RA/PR28 tampoco está todavía en main. Los recibos posteriores se añaden abajo.

## 2. Qué calcula cada término y dónde actúa

| Magnitud | Definición efectiva | Unidad / significado |
| --- | --- | --- |
| `C0`, `C` | Capital inicial y final del replay IS | Moneda de valoración del replay; no saldo real verificado |
| `growth` | `C/C0` si ambos positivos; respaldo antiguo `1e-6` si no | Factor adimensional de capital |
| `f` | `10000 * max(ln(growth), -20)` | Puntos de utilidad; cero con capital sin cambio |
| `d` | `rep.max_dd` | Fracción de drawdown reportada por el replay |
| `D` | `test_cfg.global_max_drawdown` | Umbral genómico del candidato, no presupuesto independiente fijo |
| `h` | `D/3` | Umbral de inicio de la retención; política heredada |
| `r` | 1 si `d<=h`; si no, `clamp(exp(-20*(d-h)), .01, 1)` | Retención adimensional, no probabilidad estimada |
| score final | Transformación de `f` más castigos por actividad, supervivencia y DD | Criterio de selección, no retorno ni capital |

El score se compara con `current_score` para la aceptación y con `best_score`
para conservar al campeón. En el camino alternativo la probabilidad de aceptar
un deterioro usa `exp((score-current_score)/temp)`. El signo incorrecto de la
retención puede afectar ambas decisiones, no solamente el texto de diagnóstico.

La dependencia auditada es:

```text
bookticks IS → replay/capital/DD/trades → score SA → aceptación/campeón
                                                     ↓
                                    examen OOS separado → gate → promoción
```

La puerta OOS no se suprime ni cambia aquí. Detectar un sesgo de selección IS
no demuestra que esa puerta haya admitido el candidato sesgado.

## 3. RA-SA-F01 — el castigo movía pérdidas hacia cero

**Severidad técnica: alta. Estado: reparación local candidata.**
Responsable del parche: Codex. Consumidor afectado: CLI SA, no CMA-ES.

La operación previa era `f*r` para cualquier signo. Con `f<0` y `0<r<1`,
se cumple `f*r>f`: una penalización mejora la utilidad que el optimizador
maximiza. Además, al aumentar DD, `r` disminuye; la utilidad negativa aumenta
de nuevo hacia cero. Es una contradicción algebraica con el comentario que
prometía destruir la puntuación con mayor drawdown.

Contraejemplo determinista: `C0=13`, `C=11.7`, `D=.30`.
Se mantienen suficientes trades, igual duración y una supervivencia que no
active el castigo de 200000. Ambos drawdowns quedan por debajo de `D`, por
lo que tampoco interviene el castigo absoluto. La pérdida monetaria terminal
es la misma: el término lineal posterior resta aproximadamente 13000 puntos.

| Caso | DD | Retención | Score anterior con pérdida terminal común | Score reparado |
| --- | --- | --- | --- | --- |
| Menor caída | .11 | .8187307530779817 | -13862.618943292175 | -14286.876244256477 |
| Mayor caída | .20 | .13533528323661262 | -13142.589952285081 | -20785.147608079416 |

El mayor DD obtenía **720.028991007094 puntos más** antes del cambio.
Después obtiene menos. Son resultados de la fórmula en un contrato sintético,
no dos resultados históricos de trading. Las trayectorias de capital
13→11.57→11.7 y 13→10.4→11.7 muestran que esas parejas capital/DD son
algebraicamente compatibles; no se reconstruyeron fills que las produzcan.

La penalización monetaria posterior no arregla el defecto: es idéntica en
ambos casos y conserva la inversión. El defecto se había corregido en otro
dominio del CMA-ES, pero esa corrección no estaba conectada al CLI SA.
No se modifica ni se atribuye nuevamente la reparación al CMA-ES.

## 4. Reparación y contrato matemático

Se añade `evolution_engine::score_retention::penalize_signed_score`:

```text
P(f,r) = f*r   si f >= 0
         f/r   si f < 0
dominio: f finito, 0 < r <= 1
```

Para `f>=0`, `P<=f`; reducir `r` nunca aumenta `P`.
Para `f<0`, dividir entre un número no mayor que uno hace la pérdida más
negativa, de modo que `P<=f`. Para `r1<=r2`, se cumple
`P(f,r1)<=P(f,r2)` en ambos signos. Con `r=1` hay identidad.
A retención fija, el orden por score también se conserva a través de cero.
La función es continua en cero, pero no se reclama derivabilidad allí:
las pendientes a izquierda y derecha difieren si `r<1`.

La rama positiva ejecuta la misma multiplicación; los tests cotejan sus bits.
Esto preserva la aritmética positiva, no implica que toda trayectoria de
optimización futura sea idéntica: el SA puede visitar después otra población
al cambiar la aceptación de candidatos negativos. Cero con signo se conserva.

El helper devuelve errores explícitos para score no finito, retención fuera
del dominio y resultado no representable. En el bloque DD del CLI, el error
descarta esa iteración **antes** de aceptación/actualización del campeón.
No se fabrica una buena utilidad finita para reemplazar un desbordamiento.
Estos controles corresponden al helper nuevo; no se afirma que el CLI viejo
ya validara ese dominio ni que ahora todos sus estados numéricos sean válidos.

Este contrato conserva una heurística existente; no prueba que sea la
penalización económicamente óptima. La misma retención tiene ahora un efecto
más severo sobre candidatos perdedores, coherente con la maximización.
No se inventa un nuevo umbral ni se elimina un veto para alcanzar la meta.

## 5. Evidencia reproducible y tratamiento del error propio

Las pruebas importan el archivo real mediante `#[path = "../src/score_retention.rs"]`.
El CLI invoca ese módulo exportado; no hay una implementación verde alternativa
oculta en el test. La revisión del cableado y el check all-targets se registran
separadamente de la ejecución independiente del helper.

Secuencia preservada en `target/sa-score-evidence` (logs locales ignorados):

1. RED inicial: 7 pasan/6 fallan, exit101. Incluye una expectativa **incorrecta
   del propio test**: decimal .10 no coincide exactamente con .30/3 en f64.
2. Se cambia únicamente esa expectativa a la frontera calculada y se añade
   impresión de los scores. RED corregido: 8/5, exit101.
3. Se cambia el helper al contrato signado y se conecta el CLI.
   La misma suite corregida: GREEN13/0/0, exit0.

No se cuentan las cinco aserciones fallidas como cinco bugs independientes:
documentan distintas consecuencias/coberturas de un mismo contrato.
El RED usa una **extracción de la multiplicación anterior dentro de una API
comprobada nueva**, no una ejecución del CLI productivo intacto.

```powershell
rustc +nightly-2026-06-30 --edition 2024 --test crates/evolution-engine/tests/score_retention_contract.rs -o target/sa-score-evidence/score-retention-green.exe
./target/sa-score-evidence/score-retention-green.exe --test-threads=1 --nocapture
```

| Elemento | SHA256 |
| --- | --- |
| Helper RED extraído | 7B182FC8AFD52E0DE2AF5FFE4518898307AF402DA2B1C5C63CF4B795A0C47896 |
| Tests iniciales con expectativa errónea | 7B4EF4F474394DD9B21EB23ACA53329B1E94A8DE15641AB0F9DC7A4D1A972915 |
| Log RED inicial | 9FD3DF31DF0E1EAEEA77C16E61CDFC94E3E4B76566A95F902A9F3AD7F70C575C |
| Tests corregidos, iguales en RED/GREEN | 3862FFB7377A9BCD1DC68805AAB88BC0C8B1E00BD1A392EA12A68FCF4B25283A |
| Log RED corregido | 29E50F6205C00B7F093C1768DD390EEF6E0AB2A7B64524966527C20E144DB654 |
| Helper GREEN | 1E776DF4DEEEE9633719C43B32E2850780E71FE29C3B0502235D29CB1F595457 |
| Log GREEN | 640BC355648F3D380599F95DC3803F5D4A8D23E4F1E52ACBE72ABCB3B770DC0E |

## 6. Residuales abiertos: lo que este cambio no certifica

### SA-R02 — score rotulado como compuesto a tres días

El diagnóstico de iteraciones imprime `best_score/10000` como
`Compound 3D: ...x`. Ni siquiera sin castigos es un factor de capital:
sería un logaritmo de crecimiento, sin exponencial ni normalización por tiempo.
Con castigos, tampoco identifica ese logaritmo. Ejemplo algebraico sin castigos:
duplicar capital da `ln(2)`, no 2. La variable `_compound_rate_3d` calculada
antes usa otra fórmula y no alimenta esa impresión.
Estado: lectura de fuente confirmada, no reparado ni CLI ejecutado.
Cierre requerido: distinguir utilidad de selección, factor medido y proyección
temporal, manteniendo identidad del campeón y duración real de su evaluación.

### SA-R03 — política de utilidad no normalizada ni calibrada

El castigo `10000*(C0-C)` mezcla magnitud monetaria con puntos de logretorno:
escalar ambos capitales diez veces conserva `ln(C/C0)` pero multiplica ese
castigo por diez (13000→130000 en el ejemplo).
Eso prueba dependencia de escala, no que una moneda concreta haya perdido.
Además `D` se muta y se recorta [.05,.30] dentro del candidato: el optimizador
también cambia el inicio del castigo que juzga a ese candidato.
La monotonía corregida mantiene **D fijo**, no asegura invariancia al mutarlo.
Estado: diseño heredado abierto. Hace falta definir y versionar unidades,
restricciones externas al candidato y aceptación OOS antes de recalibrar.
No se sustituyen esos coeficientes por otros arbitrarios en esta ola.

### SA-R04 — identidad finita del reporte completo y campeón admisible

El helper sólo se invoca si `dd>dd_threshold`. Un DD NaN no cumple esa
comparación y no llega a su validación. La normalización anterior del capital
y `max(-20)` tampoco equivalen a un veredicto tipado del reporte completo.
Los centinelas iniciales de score/config y el fallback de duración IS (<.1
días→1 día) subsisten. Rechazar un overflow en el helper no asegura que haya
al menos un campeón evaluado admisible antes de la evaluación final.
Estado: residuales de fuente, no reproducción de promoción de NaN ni nuevo
defecto del helper. Cierre: contrato de reporte finito y campeón opcional
verificado, tiempo observado y fallo explícito si no hay candidato admisible.

### SA-R05 — contrato local no demuestra ventaja espectral ni crecimiento72h

No se ha evaluado el arreglo en varios activos, escalas, distribuciones,
costes/funding, capacidad o trayectorias de capital fuera de muestra.
Tampoco se ejecuta T1 completo aquí: su expresividad genética histórica no
mide rentabilidad y no acredita la selección SA recién cambiada.
Duplicar cada72h es un objetivo, no evidencia.
La medición debe registrar todos los ensayos, separación causal IS/selección/test
y dependencia temporal; una suite verde no satisface esos requisitos.

## 7. Sincronización y teoría al servicio de un contrato falsable

El plan de Qoder `BARRIDO_EXHAUSTIVO_FASES.md` y ADR-0014 se conservan.
SA corresponde al frente F5 de medición/aprendizaje; la coordinación propuesta
reserva el bloque concreto del CLI y el helper a Codex. No se atribuye
aceptación de esta reserva a GLM/Claude/Qoder sin respuesta por ID/hash.

La doctrina de frescura no debe generalizar `None→0` sin revisar al consumidor:
cero puede significar independencia, neutralidad o una medida real. MG05 ya
demuestra que un IC histórico puede sobrevivir con la misma τ sin as-of.
El siguiente contrato necesita validez, par, escala, soporte y edad explícitos,
no sólo recambiar etiquetas o publicar un cero. No se modifica el ADR ni el
productor IC ajeno mientras Qoder trabaja en F1.

El crecimiento logarítmico tiene una base científica en
[Kelly,1956](https://www.kiv.zcu.cz/~vavra/zti/KELLY.PDF), que distingue crecer
a largo plazo de maximizar capital esperado; no garantiza un retorno periódico.
El [Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
aborda selección/multiplicidad y no normalidad, no corrige un score de signo
incorrecto o un test contaminado.
Los [problemas del milenio](https://www.claymath.org/millennium-problems/)
no se incorporan por prestigio: toda teoría propuesta requiere objeto,
supuestos, unidades, implementación comprobable, nulo y evaluación marginal
fuera de muestra. No se infiere ventaja cuántica de nombres o metáforas.

## 8. Gates de cierre

Para este bloque: helper reproducido → CLI compilado → revisión independiente
→ composición con main/RA/OOS, diff por cada padre → contratos integrados
→ publicación autorizada y CI del SHA exacto → revisión cruzada → merge
verificado en main remoto → limpieza sólo de referencias inactivas integradas.

La enumeración de estos gates no afirma que estén cumplidos. La composición
económica y la admisión de genomas siguen siendo frentes distintos; no se
declara auditoría semántica completa de todos los archivos o ausencia de bugs.

## 9. Adenda de revisión: dos regresiones detectadas y cierre local

Esta adenda supera los estados iniciales de §§1,6,8 sin borrarlos: son
el registro de lo que se conocía antes del seguimiento. Primer arreglo
28c0004239b93490f790a03b3c581991e436c984; seguimiento funcional revisado
5e170c39b550e7ed922564dd1fe22debffad512f. La base de esta serie SA es
main57cb8f0fdaefbef177ff131b66851aa000de621f, no main62 ni RA9bada.

### SA-REV01 — rechazar un score también saltaba el calendario

El primer callsite usaba continue al fallar el helper. Ese salto ocurría
antes de temp*=cooling_rate y del recalentamiento heredado. Un candidato
inválido alteraba, por tanto, el calendario de todos los siguientes; no era
sólo una exclusión de ranking. Probe aritmético de revisión:100 permanecería
100 en vez de90 con factor0,9. No se ejecutó CLI ni se midió una operación.

Corrección: el error produce un score NaN de rechazo, que el selector
descarta explícitamente sin RNG ni actualización de estado. El control
alcanza siempre el enfriamiento/recalentamiento original. NaN aquí es sólo
un marcador interno; no se publica como evidencia válida ni se usa para
ordenar. La guardia del callsite prueba su estructura de fuente, no una
ejecución instrumentada de todo el CLI. No se cambian factores o temperatura.

### SA-REV02 — un centinela finito podía ganar sin evaluación

Current/best=-9999999 eran valores ficticios. La corrección para scores
negativos puede producir utilidades finitas por debajo de ese umbral:
por ejemplo base=-200000,r=.01 lleva a-20000000 antes de otros castigos.
El caso completo de revisión ≈-20409999,9999998 tampoco supera el centinela;
su probabilidad de escape puede subfluir a cero. El genoma inicial podía
permanecer como supuesto campeón sin una evaluación que le diera ese score.

SaScoreState guarda current y best como Option<f64>, inicialmente None.
El primer candidato finito constituye ambos estados sin azar, incluso
f64::MIN. Los no finitos no consumen RNG. Con incumbent finito se conservan
comparación y expresión Metropolis originales; aceptar deterioro puede
cambiar current, nunca best. Antes de replay del campeón, Some(best) es
obligatorio; si no existe se retorna sin OOS ni promoción. No se cambia en
esta ola el código de salida0 del retorno heredado: observabilidad de fallo
operacional es una deuda separada, no un éxito de entrenamiento.

### 9.1 Contratos, independencia y reproducción

Suite selección:8 contratos de helper real +1 guardia estructural del CLI.
RED con extracción explícita de centinelas iniciales/primer callsite:5/4,
exit101. GREEN9/0,exit0. Esa extracción no es ejecución intacta del CLI.
La suite retención final conserva13/0. Total22 contratos únicos: las
repeticiones propias e independientes no se suman para inflar cobertura.

| Elemento | SHA256 |
| --- | --- |
| Extracción RED selección | 70DCDE693580F26A664388C80718B14513B8E10DAC725A6D6AE2392A81D0B100 |
| Tests selección, RED/GREEN iguales | 76BE79B8C88F3399564CAA6C1413F4335DEE38B7D3395266A2D3BC0D1E567C9F |
| Log RED selección | 49F6EDBAEAD1842DC27A11A9CE7FC9E81CD9850DD1DE20EB623CD1E336DB6A4B |
| Log GREEN selección inicial | 6C8356F9A413309C8BE859FB0627A1E9118134C986829E432FDC7B48546FF3A8 |
| Log selección final repetido | 251BB88AC26DE80FEDDB328FB5679EC491282021367AF478AF92419F644804FA |
| Helper selección GREEN | 511A2A1DB7CBAFCC69FBBC975DA652549AE54090565E3C378A14D5D39EAB8D2C |
| CLI final | 9019CBADB01F3CB399EA0CB3057D39A7D6596E7176B79509E30A24DCF0D96A1F |
| Exportación evolution-engine/lib | A02011933915B09B061F6E30968425327F183E5D48070ABCAED3EA64EF4DF776 |

Revisión independiente Chandrasekhar, recibo23:03Z: reprodujo22/0 con rustc,
verificó hashes estables de seis fuentes y cerró ambos bloqueadores.
Leyó RED, no lo reejecutó; no ejecutó Cargo, CLI, modelos, delegación o push.
Logs propios del revisor en target/sa-cross-review-20261004-2302 de su
checkout forest-evaluation. No atribuir ese checkout a cambios del bosque.

Check final workspace/alltargets, nightly-2026-06-30,locked,offline,j2,dev:
exit0/124,7735498s. Logtarget/sa-score-evidence/check-selection-final.log,
SHA256BFEB3933A173CF4AD906F34013C22A908F027A34BC9403A38199120E875A15B9.
Tras compilar RA en target compartido se refrescaron nueve fuentes propias
existentes distintas entre cortes, con hashes antes/después iguales, para
reconstruir la identidad SA. No se borró caché ni cambiaron bytes. Los
warnings heredados siguen visibles; no cargo fix ni ocultación de errores.
Workflow añade ejecución fail-fast de ambos contratos standalone y mantiene
sus pasos existentes. Aún no es un recibo de CI remota de esta rama.

## 10. Estado consolidado de fallos y significado de las métricas

| ID | Estado actual local | Qué queda por demostrar |
| --- | --- | --- |
| RA-SA-F01 | reparado para ambos signos,13 contratos | efecto económico marginal neto/multiactivo/OOS |
| SA-REV01/02 | reparados y revisados,9 contratos | integración con RA/OOS y CI del candidato exacto |
| SA-R02 | rótulo reparado; imprime utilidad escalada, NO factor72h | métricas de crecimiento observado con duración/identidad válidas |
| SA-R03 | abierto | utilidad adimensional coherente, umbrales externos y calibración OOS |
| SA-R04 | cerrado sólo centinela/campeón; resto abierto | reporte completo finito, DDNaN, capital/duración observados, salida operacional |
| SA-R05 | sin medición | costes, funding, liquidez/capacidad, supervivencia y crecimiento neto72h |

Se conserva score/10000 para diagnóstico, rotulado explícitamente como
utilidad. No se convierte en factor mediante exp porque los castigos ya
han roto su interpretación como logretorno. Temperatura está en unidades
del score para la aceptación probabilística; no es temperatura física,
evidencia cuántica ni una garantía de escape. Calendario y pesos heredados
requieren sus propias hipótesis y ensayos, no otra constante propuesta sin
evidencia. Primer candidato finito no equivale a reporte íntegro admisible.

## 11. Sincronización, integración y validación de documentación

Plan maestro§15 reserva SA en su worktree, separado de OOS684e+RA74 y del
RA9bada+main62. Qoder F1/IC y Claude/riesgo no se modifican. La coordinación
versionada y el buzón local publican alcance/contratos; no hay acuse externo
nuevo confirmado. Consulta conjunta de publicación SA/OOS pendiente al
recibo: no push ni PR de estas series, no merge remoto ni borrado de ramas.
Las reparaciones locales no pueden marcarse resueltas en producción.

Validación narrativa: fórmulas/signos/unidades y contraejemplo trazados a
fuente; estados iniciales conservados y superados sólo en adendas;22tests
únicos, sin sumar repeticiones; CLI estructural distinto de ejecución;
check distinto de tests; evidencia local distinta de CI y main. JSON añade
un recibo nuevo sin alterar el contenido del corte inicial. Afirmaciones
de rendimiento, omnisciencia, continuo sin discretización o ventaja cuántica
no verificadas no se presentan como resultado. Persisten gates de§8.

## 12. Composición local SA con RA f275/main237, sin integrar OOS-F01

Padres del candidato:SA701fa98e6d9e5788f2c7203c7da1135ad9ff3f77 y
RAf275bae39dd81531cfb71b8ddaadb1023f02c6f5. El merge trae parser,
admisión de riesgo y contexto OOS de RA; NO validate_is_oos_split de la
serie OOSacbc, que sigue aparte. No activar genomas ni modelos por componer.

Sólo hubo conflictos documentales MEM/COORD/plan, resueltos por unión.
Plan15 se reubica entre14y16, sin cambiar sus líneas; ambos padres conservan
su orden relativo. QA no vacías:MEM1583/1816,COORD3539/3703,PLAN122/631.
Los cuatro helpers/tests SA permanecen idénticos al padre SA; helper/test
OOS,parser yorchestrator idénticos al padre RA. Workflow conserva contratos
de ambos padres. CLI sólo suma la corrección RA del prefijo warmup; no cambia
selección SA. HashCLI compuesto:
B76A3EC359873F9E14E598CEFF01D9D5B3585D60341E13D337D220B4D6295BF5.

Pruebas actuales:13retención+9selección+6contextOOS+19parser=47aprobadas,
0fallidas,0ignoradas. Son contratos de esas áreas, no todo el workspace.
Primer intento parser conrustc sinexterns falló al COMPILAR por dependencias
memchr/fast_float/serde_json; error de invocación, no REDconductual ni bug.
Repetición correcta conCargo -p data-pipeline,locked,offline,j2 ejecutó19/0,
incluyendo7tests previos. No se borraron tests ni se cambiaron fuentes para
hacerlo pasar. Tiempo87,5170729s; no sumar repeticiones para cobertura.

Alltargets del candidato exit0/136,8175224s,nightly2026-06-30,dev,locked,
offline,j2. Tras checkOOS ajeno propio terminado, refresco8fuentes SA distintas
con SHAestables; Cargo serializado entre agentes, no runner interrumpido.
Logchecktarget/sa-score-evidence/check-composed-RAf275.log:
83D47D6AA574DC7755C5E5D8FAB99BF710B44946E3328AE115C2D7F29786523F.

| Log en target/sa-score-evidence | SHA256 |
| --- | --- |
| composed-score.log | EF8A513FC78373CEFA51F6503F6E946D79AD3336C182A42BA6D5D623DDC2BC9D |
| composed-selection.log | 6C8356F9A413309C8BE859FB0627A1E9118134C986829E432FDC7B48546FF3A8 |
| composed-oos-context.log | AE282F16E9712BE90577BBC2EE0AEE78733203C9DB76328E672B197347884141 |
| composed-parser-cargo.log | 42ADAFCBA28E355DCC7748FB45FAB6EDA98567F25E71456DAA329D4CEF6E9CFC |

Adendas canónicas del plan y rollback añadidas también en este candidato.
El inventario main237/RA37383 es posterior a f275 y no está incluido aún en
este merge. No usar sus referencias como si fueran archivo local presente.
No push/PR de SA, publicación consultada pendiente; no está en main.
La CI187/0/3 del headRA74be no cubre SA nueva ni este candidato compuesto.
ResidualesSA-R03/R04/R05 y consumidores de métricas siguen pendientes.

## 13. Composición SA con RA f608 y preparación de publicación

El mandato vigente del operador reitera publicar e integrar las ramas del
proyecto. Esta autorización permite continuar SA hacia una PR separada en
el repositorio conocido, conservando CI y revisión antes del merge. Las
anotaciones anteriores de consulta pendiente describen cortes históricos;
no representan el estado de autorización de esta adenda. La integración
de código no activa un genoma ni ejecuta el optimizador.

Padres: d73cca8cdc5191f9c0ccb6b66cebfee583cdd87d y
f608737803406e5e6f5cf0045755c1fbafb55558. El cambio entrante es documental:
plan, inventario y revalidación F2. Los cuatro conflictos se resuelven por
unión. QA comprueba las líneas no vacías y su orden relativo de cada padre:
memoria1834/1860, coordinación3720/3761, maestro658/663 y plan225/275.
Todos los Rust y el workflow permanecen idénticos al padre SA d73; el
prefijo OOS de RA sigue integrado y la validación de partición OOS-F01
continúa en la otra serie. Se conservan los valores anteriores del JSON.

Contratos específicos recompilados con rustc nightly2026-06-30, edition2024:
13 de retención y9 de selección,22 aprobados,0 fallidos,0 ignorados. Los
logs bajo target/sa-score-evidence del checkout principal tienen SHA256:
score-retention-f608.log EF8A513FC78373CEFA51F6503F6E946D79AD3336C182A42BA6D5D623DDC2BC9D;
sa-selection-f608.log 251BB88AC26DE80FEDDB328FB5679EC491282021367AF478AF92419F644804FA.
El total47 de§12 incluye parser/contexto y pertenece a aquella ejecución;
no se infla el conteo sumando repeticiones de los22.

La compilación requirió resolver un problema de procedencia del artefacto:
un primer intento no pudo guardar el log porque faltaba su directorio y no
constituye recibo válido. Después, un check capturado devolvió101 en18.159s:
el binario no encontraba oos_context en backtest_engine, aunque lib.rs:5
lo exportaba y el helper existía. Log check-composed-RAf608.log,
SHA256 B39D9B15463DBB8553172F5028C04FEF9F908448165F915C9761A1D4178372F8.
Es consistente con la interferencia de caché compartida ya registrada en
RA; no se modificó fuente para ocultar el error ni se atribuyó a la fórmula SA.

Se refrescaron mtimes de13 fuentes Rust propias del delta respecto a main,
incluidas las heredadas de RA, verificando bytes invariantes. El check
workspace/all-targets/locked/offline/j2 terminó0 en155.7442739s, perfildev,
misma toolchain; log check-composed-RAf608-refresh.log,
SHA256 7FF59139950E39FD99420B2C5A700AD008853D29A4A2B43694B802F2B810A638.
Quedan ambos resultados; el check exitoso no ejecuta pruebas, y las fuentes
siguen idénticas al SA revisado. Cargo quedó libre antes del check OOS.

La revisión independiente previa de los22 contratos continúa aplicando a
esos mismos bytes. CI remota y revisión de la composición pública quedan
pendientes al momento de este recibo. La PR dependerá de RA28; la llegada
a main y limpieza de la rama se verificarán después de su merge, nunca a
partir de un push. SA-R03/R04/R05, métricas económicas y consumidores de
G72 conservan sus estados abiertos, sin certificación de retorno72h.

## 14. Rectificación de autorización: publicación bloqueada por auto-review

El 2026-10-05 00:39 UTC (2026-10-04, Bogotá) se verifica que la revisión
automática rechazó el comando combinado de commit y push de SA. Consideró
insuficiente el mandato general para publicar el contenido concreto SA en
el repositorio público Jhona-la/Trader-Gemini, porque la consulta específica
SA/OOS aún no tenía respuesta. La interpretación de autorización de §13 es
histórica y queda RECTIFICADA por esta adenda; no es una autorización vigente.

No se ejecutó ninguna parte de ese comando: HEAD sigue d73cca8c y MERGE_HEAD
f6087378. No existe un push ni PR SA de esta composición. Los cuatro archivos
conservan las uniones resueltas en disco pero el índice todavía exige cerrar
el merge. No se considera que compilar equivalga a haber integrado/publicado.

Se solicitó al operador autorización expresa para publicar SA y OOS en dos
PR separadas, con código, pruebas e informes, sin secretos ni datos operativos,
y merge condicionado a CI y revisión cruzada. Hasta recibirla, la publicación
se mantiene bloqueada. La verificación y documentación local continúan; RA28
tiene autorización previa independiente. No se intenta eludir el rechazo por
otro canal. Los resultados técnicos y hashes de §13 permanecen válidos como
recibos locales y no demuestran retorno económico ni operación en producción.
