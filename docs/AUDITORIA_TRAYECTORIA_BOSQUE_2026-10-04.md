# E03/E04 — Coherencia del genoma y trayectoria observada del bosque

## Alcance, versión y estado

Rama `codex/forest-evaluation-2026-10-04`, checkout aislado `forest-evaluation`.
Base del parche: `5ab07f3589a1ab273794bf373b40b9f120a5b438`. El candidato
incorpora main `e3adf74e36aa48c122e83a72d1b7f6c3cbad90bd` sin cambiar la
fuente del bosque ya probada: ese archivo es idéntico en ambas bases antes
del parche. La integración es main→rama propia; no es rama propia→main.

Este informe desarrolla los expedientes existentes RA-E03 y RA-E04, no añade
dos bugs nuevos al inventario. Los controles ejecutados respaldan una reparación
local acotada. No hay publicación, promoción de genomas, entrenamiento, sesión
viva ni certificación económica derivada de este bloque.

Único archivo de producción/pruebas cambiado:
`crates/evolution-engine/src/random_forest.rs`. El workflow añade la ejecución
explícita de la biblioteca del bosque, y las adendas documentales conservan
los recibos anteriores. El resto de los cambios contra la base corresponde
a main incorporado; el diff debe leerse contra cada padre.

## E03: el fenotipo inicial no correspondía al genoma almacenado

### Causa y recorrido del defecto

El constructor aplicaba cada `SuperGenotype` a su arena y guardaba ese genoma
en `genomes`, pero sobrescribía después `latency_penalty_ms` con 25 ms.
Un control cuyo genoma indicaba 50 ms era evaluado inicialmente con 25 ms.
El comentario de «peor caso estadístico» no aportaba una medición ni hacía
que 25 ms representase el valor del genoma. `replant` volvía a aplicar el
genoma sin esa sobrescritura, por lo que también cambiaba el contrato entre
plantación inicial y replantación.

El fallo no consiste en demostrar que 25 ms sea siempre optimista o pesimista;
consiste en evaluar un fenotipo distinto al que se identifica y se conserva
como candidato. Ello impide atribuir limpiamente la puntuación a los parámetros
del artefacto guardado. No se ha medido en este expediente una promoción o
pérdida real atribuible a esa discrepancia.

### Corrección y significado de los controles

Se elimina únicamente la sobrescritura de 25 ms. Se mantiene el modo
HyperRealistic y la proyección existente `apply_to_arena`; no se elige una
nueva latencia ni se altera la mutación. Las pruebas comparan la configuración
aplicada con la proyección del genoma guardado, tanto para el control como
para sus mutantes. La prueba de replantación usa un control de 75 ms y comprueba
que el evento posterior no vuelve a cambiar esa correspondencia.

50 y 75 ms son entradas de contraprueba, no umbrales económicos nuevos. El
control de vector completo verifica la proyección existente, incluidas curvas
de horizonte y sus vistas escalares; no demuestra que cada coordenada sea
independiente, identificable o económicamente útil.

## E04: la recuperación borraba el drawdown máximo de la evaluación

### Causa y distinción entre dos magnitudes diferentes

La cosecha conservaba un pico, pero entregaba al fitness el déficit **actual**
respecto de ese pico: `(peak-capital)/peak`. Después de una caída seguida de
recuperación, ese déficit puede ser cero aunque el universo haya sufrido una
caída histórica significativa. Tampoco se observaba sistemáticamente el capital
al terminar cada broadcast; extremos entre cosechas podían quedar fuera.

El contrato de fitness exige `max_drawdown_pct`, no el déficit al final.
Ambos son adimensionales, pero contestan preguntas distintas: «¿cuánto falta
para recuperar el pico?» frente a «¿cuál fue la peor caída observada desde un
pico previo?». La corrección preserva esa trayectoria dentro de la época del
genoma, sin cambiar la función de aptitud heredada.

### Cálculo conservado, con propósito y dominio explícitos

Para observaciones de capital realizado positivo y finito `C_k`:

```text
P_k = max(P_(k-1), C_k)
d_k = (P_k - C_k) / P_k
D_k = max(D_(k-1), d_k)
```

`P_k` es el máximo capital observado; `d_k` es el déficit instantáneo de
esa observación; `D_k` es el máximo déficit de la época. El máximo histórico
no disminuye cuando mejora el capital. Se calcula en O(1) por universo,
sin guardar toda la trayectoria ni asignar memoria en cada broadcast.

En el contraejemplo de recuperación, el componente heredado de aptitud es
`ln(C_final/C_inicial) - lambda*D²`, con `lambda=4*ln(2)`.
Ese peso expresa una preferencia de riesgo ya adoptada; no es una probabilidad
de ruina, una calibración universal ni una prueba de optimalidad Kelly.
El factor OOS heredado es 1 en estos endpoints de crecimiento positivo.

| Observación del mutante | Capital | Pico observado | Déficit actual | Máximo histórico |
|---|---:|---:|---:|---:|
| Inicial | 13 | 13 | 0 | 0 |
| Caída | 6,5 | 13 | 0,5 | 0,5 |
| Recuperación | 15 | 15 | 0 | 0,5 |

Al final, el cálculo anterior puntuaba aproximadamente 0,1431008436.
Con la caída conservada, la puntuación es aproximadamente −0,5500463369.
Un control estable 13→14 obtiene aproximadamente 0,0741079722 y lo supera.
El oráculo reproduce ese cambio de ranking; no demuestra que ese control
domine una estrategia real ni convierte capital inyectado en fills observados.

### Estado inválido, replantación y continuidad de época

La historia se observa al terminar eventos trade/depth/kline, en cosecha y
también al suspender el procesamiento por presión de memoria. En cosecha,
la misma lectura de capital alimenta la observación y la puntuación. No se
afirma que esa lectura haga atómico capital/posición/contador/orden.

Capital no finito o no positivo, o un pico inválido, marca la historia con
NaN y el contrato de fitness la declara inviable. Una recuperación posterior
no sanea por sí sola esa época. La comparación explícita evita que una
operación `max` o un respaldo numérico oculte esa invalidez.

Re-registrar el mismo genoma o cambiar sólo su número de generación conserva
la historia y la muestra. Sólo `replant` reinicia pico, drawdown y baseline
de operaciones, manteniendo CL-40/CL-40c. El gate heredado de 15 cierres desde
replant sigue intacto. El fitness continúa usando el contador acumulado que
ya usaba: no se certifica aquí la equivalencia estadística entre ambas muestras.

## Oráculos ejecutados, controles negativos e identidad de la evidencia

El worker ejecutó los comandos desde el checkout aislado, con nightly
2026-06-30, lock y resolución offline. La caché DEV fue la del worktree RA;
no se modificó su binario/fixture release T1 ni se inició ningún motor:

```text
cargo +nightly-2026-06-30 test -p evolution-engine --lib random_forest::tests:: --locked --offline -j 2 -- --test-threads=1
cargo +nightly-2026-06-30 test -p evolution-engine --lib --locked --offline -j 2 -- --test-threads=1
```

| Fase | Aprobadas | Fallidas | Ignoradas | Exit | Tiempo de tests |
|---|---:|---:|---:|---:|---:|
| GREEN inicial del bosque | 16 | 0 | 0 | 0 | 1,10 s |
| Biblioteca inicial | 73 | 0 | 0 | 0 | 1,67 s |
| RED: producción anterior + pruebas ampliadas | 7 | 9 | 0 | 101 | 0,98 s |
| GREEN restaurado del bosque | 16 | 0 | 0 | 0 | 0,98 s |
| Biblioteca final | 73 | 0 | 0 | 0 | 1,59 s |

La primera compilación requirió 97m45s. RED compiló correctamente y falló
por aserciones conductuales, no por un error de compilación. Replantación y
las cinco pruebas antiguas ya podían pasar en la producción anterior: RED
no exige que fallen los 16 casos. Hay 11 pruebas nuevas y las cinco funciones
antiguas se conservan byte a byte.

Codex padre leyó el diff completo, los casos añadidos, resultados/aserciones
de los logs y la fuente de fitness, y recalculó las fórmulas. No reejecutó
por segunda vez esos controles del worker. La revisión independiente de la
fuente, de sólo lectura y hash estable, no halló bloqueadores nuevos dentro
del alcance; no ejecutó cargo y no es aprobación humana GitHub.

SHA256 de fuente final:
`E8C7D240A4562A5EAD4450220C2DF07D9B8F5B743EF49B66272878EB179344AD`.
SHA256 del módulo de pruebas final:
`527D991D1BAD59A01181748271059C5784542D5A2DF97C8254D01C15AFD861AA`.

Logs locales ignorados en `target/forest-verification-20261004/`:

| Recibo | SHA256 |
|---|---|
| green-shadow-01.log | 05FCE71EA8FD30FD50AC077A79EF4E43E86976961E280E9ED24375F09362EE80 |
| red-shadow-01.log | 88FE3ADEE9697F2CCD53EB8941CF90D6A8980A013B3EAB42BFADC053F9A96023 |
| restored-green-shadow-01.log | D3B30A6217C6B9721B0C74FCEA77B38061A7B26066493380BFA7A88552A18ACA |
| restored-green-lib-02.log | 58694C9555618D6423395F5BF31C32D3C2474B77CFCB0DAE1F7BDB9066E314A3 |

El worker declara producción baseline exactamente restaurada para RED,
mismas pruebas y retorno a la fuente GREEN por apply_patch. Los logs no
contienen un sello automático de cada fuente y OID: los hashes anteriores
son un cotejo posterior, no una firma criptográfica contemporánea del runner.

## Comprobación del candidato incorporando main

El padre ejecutó `cargo +nightly-2026-06-30 check --workspace --all-targets
--locked --offline -j 2` sobre el candidato 5ab+e3 con el parche intacto:
exit 0, 2m30s, sesión96306. El log local `all-targets-candidate.log` tiene
SHA256 `6A40DEA9531E7FEB6E3EA75D4468CBB93A58790749431083A5903F5E6DC99155`.
Hay warnings heredados; no se ejecutó `cargo fix` ni se cambiaron otras rutas.

El diff contra e3 contiene sólo el parche del bosque y sus adendas de CI,
informe y memoria. Contra 5ab también conserva la coordinación/plan y las
seis cadenas de vetos incorporadas desde main. Se mantiene fuente E8C7D240;
la comprobación no es T1 completo, CI remota ni llegada de esta serie a main.

## Límites y siguientes gates

- El drawdown es del capital realizado **observado**; no incluye MTM, mínimos
  entre observaciones, garantías frente a saltos ni toda trayectoria económica.
- Las pruebas inyectan capital/contadores y ejercitan eventos públicos. No
  ejecutan fills, una promoción real, la cuenta viva ni paridad host↔replay.
- El proxy OOS y la comparación contra un control inviable `-Inf` heredados
  no cambian. Un candidato finito puede superar a ese control bajo la política
  previa; esto no es prueba de superioridad OOS ni aceptación prudente nueva.
- Los vectores públicos mantienen invariantes de forma/longitud por constructor
  y replantación. No se endurece aquí toda manipulación directa de esa API,
  el bosque vacío o un contrato de concurrencia entre esas colecciones.
- Se añade 8 bytes de almacenamiento por universo más la cabecera del Vec,
  y una lectura atómica por universo/broadcast. Es conteo estructural, no
  benchmark de latencia, memoria pico o throughput.
- Los avisos TG_GENOME_ENV/DarkAlpha pertenecen al fixture; no se activan
  modelos o entornos para silenciarlos. No se altera lambda, la muestra mínima,
  el filtro de entorno, la semántica de promoción o el almacén.
- Antes de integración: diff contra cada padre, all-targets del candidato,
  CI que ejecute explícitamente la biblioteca, paridad y T1 pertinentes.
  El T1 RA de fuente496 no cubre esta reparación. Mantener las series separadas
  hasta contrastar el candidato conjunto; no sumar checks como una garantía.

El objetivo de duplicación72h continúa sin una medición neta OOS que lo
acredite. Esta reparación corrige la atribución del fenotipo y la memoria
de riesgo evaluada; no certifica autoevolución universal ni rentabilidad.
