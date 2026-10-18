# Cuatro pares privados de ceros y replay: cierre y siguientes gates

Corte funcional comunicado por ROOT: **18 de octubre de 2026, 16:11 UTC / 11:11 Bogotá**. Core07, Risk01, Kelly02 y Flow03 completaron sus pares causales privados: **4 de 7 pares**. Los ocho recibos mantienen sus 12 invariantes y 8 guardas verdaderas; todos los GREEN terminaron driver/validador 0/0. Este cierre documental no certifica main ni el sistema completo. El control LG5 privado y su ancestría no se publican.

| Par privado | RED PASS/FAIL | GREEN PASS/FAIL | Commit privado GREEN |
|---|---:|---:|---|
| Core07, admisión temporal Core | 3/2 | 5/0 | `084dcaf4fa88733bf62705dd8c2411215f039cea` |
| Risk01, reloj carry y pendiente | 6/7 | 13/0 | `7eeffe1ae481000a5c7bc0a34d368d52f80d5fe2` |
| Kelly02, compresión en este lector | 0/2 | 2/0 | `94fefc5f457effbc426aa7306247ccf3781bd971` |
| Flow03, referencia SS en composición | 1/6 | 7/0 | `25dbbd9aba895803594fd2a24764b9f07cf18bc7` |

Son cuatro ejecuciones GREEN separadas, **no una suite conjunta de 27 pruebas sobre el último árbol**. Al cerrar Flow03 la fuente privada era `25dbbd9aba895803594fd2a24764b9f07cf18bc7`, tree `5a02ac51e6470e4b4fc87852d2b4cefb0158c6ce`, índice raw `9cca59dd8d846b157b1e93f1eeccbdeaa202f2a86fcf4e38fb275d6d1f0e0db0` y stage `03a11f8abae85b579b38e1c6460ebd7ae1cb8b96fcb39cf15246bd3e28e2aef8`. Esos hashes identifican este corte; no conceden turno actual ni equivalencia con un main posterior.

## Qué demuestra cada par

**Core07.** Dos guardas temporales por activo rechazan eventos retrógrados antes de mutar reloj/ledger; preservan igualdad temporal y origen cero. RED falla en Dual y Depth tras controles positivos y rechazo FE comprobados; no alcanza el negativo Trade. GREEN completa los cinco contratos, incluido Depth/Trade y los controles finales de posición, caja, margen y reloj. No se atribuye al RED un assert no alcanzado. [Cierre Core07 y originales](CORE07_CEROS_REPLAY_RED_GREEN_2026-10-18.md).

**Risk01.** El corpus de trece contratos permanece idéntico. Cinco fallos RED tienen localización inequívoca: dos de reloj después de controles positivos y tres de pendiente después de admisión/rechazo. Dos adicionales aparecen en un helper positivo compartido: el log no identifica cuál de sus tres invocaciones falló. La atribución a una evaluación final sigue siendo inferencia aritmética, no stack observado ni severidad nueva. GREEN completa los trece contratos, incluidos TTL inclusivo, funding según reloj de decisión, limpieza de pendientes y consumo único. El delta privado mantiene Kelly máximo 1.25. No prueba fills ni el mecanismo público CL135 de medición al cierre.

**Kelly02.** Ambos RED, largo y corto, llegan al assert final identificado por la etiqueta única «compresión heurística neutral»: 43.75 frente a 34.99999999999999289. Antes han validado los tres mundos de escala 1.00/0.70/1.50 y expansión 2/3. El delta limita a 1.0 el máximo de `(1/scale).clamp(.50,max)` en este lector; GREEN completa ambos contratos. Son nueve retornos sintéticos por lado, entre sonda 5 y mínimo LH04 10, con `lh04_f_riesgo=0`: no son cierres adquiridos, calibración ni LH04 maduro. El par no demuestra no-amplificación en todos los caminos de sizing. El log no imprime un valor absoluto separado de expansión; no se inventa esa medición.

**Flow03.** El corpus conjunto de siete contratos observa seis fallos RED en líneas 45/140/68/103/158/213 y un control pasado de oposición legacy con peso cero. No se atribuyen índices de bucle x/p/ss/k que el log no imprime. Los valores de actividad de fondo observados fueron 0.07997456394865413 y 0.17525854442462205 frente a cero. El delta sólo pasa SS al normalizador y cambia el umbral efectivo a `ss + excitation`; GREEN pasa los siete. Peso cero conserva atenuación por oposición legacy: no significa neutralidad completa. Joint7 publica presión/actividad/peso directamente; no instancia el productor Hidden ni recorre Core E2E. El corpus adaptive-reference de siete pruebas es otro trabajo pendiente; no se suman siete PASS inexistentes.

## Corpus y originales durables

Los borradores históricos 1/7, 2/7 y 3/7 permanecen intactos como sus cortes. Esta entrada actualiza la coordinación a 4/7. Cada manifiesto enumera originales seleccionados, tamaño y SHA-256: logs/recibos, wrappers/cadenas/bindings, un ejecutable histórico retenido por mundo, corpus, parche/runtime antes-después, limpiezas y directiva. No se copia el target/cache completo ni pesos de modelos. La frase histórica «isolated target» no acredita aislamiento: se usó el target canónico compartido.

| Par | SHA-256 corpus idéntico RED/GREEN | SHA-256 MANIFEST |
|---|---|---|
| Core07 | `17e599ed1b37d8051a8644ef698dcd45a28710c634a7051240c8b362e41c6d54` | `15410d8e06efb2d750337769e9a7da5175279a812de8bd76f6e5e8bfc9692b55` |
| Risk01 | `f66246c61d7745ab8bd29d2ccf6cb4dad16868ed1f95cd1f1f30f0143c01b08b` | `dba761b7332df43f5b2f273b763353b4d90e8a00e946011b3f14fc52a1912c5a` |
| Kelly02 | `ca28874650db3680fe8540bdc582f1b38733246bff3b9c79ac6ccad27ad2c765` | `ded74cbae6f5289b0fd6e2a07797ada38bb72b20fe3f038f50d013c5adde1deb` |
| Flow03 | `8e088e4f63447aa80c8b04581aec5876fb8ff49add06e493018608f8ffea61be` | `2eacb7656299bfa7ebc51061d38f82186efdaf755ae48e87e506f747cf1febbc` |

Archivos externos bajo `C:/Users/jhona/.codex/audit-evidence/trader-gemini/`:

- `core07-causal-pair-20261018-bdb20c69-89a6-4654-89f3-75ebeeac8291`.
- `risk01-private-causal-pair-20261018-6ac94fe3-3634-4c44-816a-44a3086dddab`.
- `kelly02-private-causal-pair-20261018-f536fb5d-4262-4b27-83ca-c7c50c21f1d0`.
- `flow03-private-causal-pair-20261018-263e507d-e9a7-4325-8c95-7dd25723c6d4`.

| Par | RED driver SHA-256 | GREEN driver SHA-256 |
|---|---|---|
| Core07 | `621af3e8bea096664bc5ef60e9dee935d4b161b66c2f11f4fb1594162474d0b6` | `bd382b3a9734a93aeb8dc74b6162e8476a4361bd4c581eaa190c5063e78042f5` |
| Risk01 | `6027be41cac5479c26f578a1c5c995fa3d4a179ec968a36ed19513a7e90a923f` | `f04a3b634aa82b9718e1c611389abad236fa362abcf9afb987087184a1e0f71a` |
| Kelly02 | `1c7cbfe0291c1c3c31396ddfaea420b8daa8e76f97529a2a2070b05943349ed1` | `dd571979e0ef0d677be5fe811b239b2a46f92c4781f35219b11c5ac69149c058` |
| Flow03 | `9f376d3b9d0ffe2bdabb707e00e4e8e493be5735607f84c028254d8acb6b2a97` | `55cf1443eb47822ae28e71119196574cabde8ecd0a6084311aa52eaf286c3f10` |

## Recursos y cierre de coordinación

La orden del operador fue trabajar con RAM disponible, aproximadamente 3 GiB. ROOT eligió admisión fija de 2 GiB físicos libres y un job en la copia externa V3 revisada; el V3 original sigue en 4 GiB. No se alteraron cinco protegidos, runner236/wrapperaa8, corpus, 12+8 ni umbral integral T1. Cada ejecución requiere observación real, adquisición atómica del lock canónico y fuente/binding propios. Un fallo de compilación, constructor, stack o guarda no cuenta como RED semántico.

Los GREEN liberaron después de limpieza explícita de su VCTIP propio, con hijo de prueba terminal y comprobación de Job/PID/StartTime/ruta/hash/firma. No fue drenaje espontáneo ni terminación de Cargo/tests ajenos. Liberaciones GREEN: Core07 14:29:47.8911303Z, Risk01 15:35:55.4896704Z, Kelly02 15:42:14.5821541Z y Flow03 15:50:30.9887599Z. Son cortes históricos, no una declaración de recurso libre actual.

Risk01 RED conserva una excepción: el coordinador original retuvo el lock al aparecer otro Cargo en la carrera de salida. La recuperación movió la reserva pero terminó 86 al fallar una postcondición por sufijo Unicode U+013C; ese recibo no se corrige. Una verificación posterior comprobó cuarentena, tres archivos literales y ausencia del lock canónico en ese corte, sin otra mudanza. Es recuperación separada, no liberación original retroactiva. Los endpoints 12+8 no prueban aislamiento continuo, procedencia completa del cache ni todas las escrituras de modelos.

## Secuencia siguiente y reparto

1. **Hidden04:** ROOT comunicó RED numérico real de 16 contratos: 2 PASS / 14 FAIL, driver/validador 101/101 y 12+8 verdaderas. Su coordinador liberó a 16:07:05Z tras limpieza explícita de VCTIP propio. El RED de publicación 2 corre sobre el mismo preimage, todavía sin resultado comunicado. No se ha instalado04; esperar ambos RED clasificados antes de ese delta y después dos GREEN separados con idénticos corpus. Flow03 no acredita su productor.
2. **Replay05:** ambos RED, span 8 y ledger 5, antes de 05; después ambos GREEN. Core07 no sustituye contador/origen replay ni conservación de deltas financieros pequeños.
3. **Reporter06:** RED/GREEN de 22 contratos privados. En el puerto público preservar NoReplay y 21 cuerpos existentes; la unión 23 exige resultado propio.
4. Revisar por separado el hallazgo Hidden positivo cuyo z redondea a cero pese a ratios finales representables. Es análisis RO pendiente de RED/GREEN adicional; conservar los 16 contratos originales, sin declararlo corregido.
5. Ejecutar suites finales, check all-targets y T1 integral auténtico sobre la composición final. El mínimo integral 110 requiere resultados reales; una guarda que acepte algún PASS no lo acredita. Resultados Fable/GLM y check inicial pertenecen a sus fuentes, sin herencia.
6. Adaptar cada puerto autónomo desde main vigente preservando CL135, F-A6 Hawkes recurrente O(1), C-F5b/PR76, CL148/PR77, Ω151–153 y NoReplay. Risk privado no reemplaza el registro público al cierre; Flow no copia la historia retroactiva alpha/cutoff sobre F-A6. Resolver semántica contra cada padre y verificar la nueva composición antes de integrar runtime.

BASE15 mantiene maestro/censo, Quant sus reservas y ROOT los pares/integración. El plan BASE15 registró 4/7 en `a3e3ad84707dcaa8bf60492deb116412185aa04e`. Esta sincronización cambia sólo el cierre, fila ROOT y entradas append-only; preserva filas ajenas, censo y recibos históricos.

Main local `e21ce07ad8782722d95ad8f6d31678eb643efc10` fue observado como ancestro del main remoto `a3e3ad84707dcaa8bf60492deb116412185aa04e` y está abierto en el checkout principal ajeno. Se conserva sin mover ref/índice/archivos. La publicación documental por objetos Git podrá avanzar remoto/origin mediante su rama propia, sin actualizar ese main ocupado. Borrar ramas requiere ancestry, ausencia de exclusivos, liberación/reservas y worktrees actuales; no basta el censo histórico.

**LG5 HIGH01/02 y MED06 siguen abiertos.** Estos cuatro pares no validan toda adquisición/calibración, transferencia de política ni pipeline de ejecución. La duplicación neta cada 72 h sigue sin demostración estadística y económica; no se presenta como consecuencia de estos contratos.

Preparación documental: Git sólo de lectura sin locks y lecturas/copias externas; payloads seleccionados rehasheados, sin ejecutar binarios ni repetir Rust. Ningún resultado privado se transforma en PASS de main.
