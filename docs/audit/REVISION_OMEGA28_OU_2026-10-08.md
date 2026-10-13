# Revisión independiente Ω28 — OU, 2026-10-08

Dictamen estático: aprobación acotada de los dos guards como corrección de los defectos descritos; no cierre completo de I5 ni del dominio OU. La exclusividad del modo opt-in y la preservación del estado SDE ante timestamps no crecientes quedan sustentadas por el flujo. El consumidor aún puede emitir una intención usando el z-score anterior de un tick rechazado. No se ejecutó Cargo, rustc, testigo ni prueba de comportamiento en esta revisión; T1 sobre e9c6a435 no cubre Ω28.

## Corte y evidencia

- Base D1: `a0a5b982df0eb53d3c6d753256ea99c6428cdafd`.
- Fuente Ω28: `c0ccdf5f85453cc51f291ce7818101be3f561bcc`.
- Corte fuente+documentación revisado: `a5c69af3f7a05c8642b803c61a6ff72613d96100`.
- Se leyó el diff completo D1→corte: cuatro archivos, 106 inserciones y cuatro eliminaciones; dos fuentes y `.agents/MEMORIA.md` / `COORDINACION_CODEX_2026-09-28.md`.
- Se leyó el flujo del caller `update_and_evaluate`, constructor/estado y estimador SDE, los dos tests nuevos, el test existente de emisión continua y `tests/basket_state_contract.rs`. Búsqueda de símbolos sobre todos los `*.rs` versionados: definiciones/reexport, tests y fixture aislado; no se encontró instanciación productiva. Esto no excluye consumidores externos de la API pública ni sustituye análisis de toda posible generación dinámica.

| Archivo relativo | Blob D1 | Blob Ω28 |
|---|---|---|
| `crates/strategy-core/src/multivariate_coint.rs` | `3d87089163cf2011e7e9a4efb7d98c88985d4a51` | `798b5c5377ba9fe7e64dfedc60c45d1897dfa4e9` |
| `crates/strategy-core/src/vecm_arbitrage.rs` | `f3a2cc9d344995b7cdf40f7d96324fe2b42a3f4a` | `2516f965c456e446e8af4387b517faa156de5cfc` |

Las líneas siguientes pertenecen al corte Ω28. No se modificaron fuente, refs, HEAD ni índice; este informe se guarda solamente bajo `target/r4-review` ignorado.

## Lo que corrige el delta

1. **OU-R4-01, fallback:** `multivariate_coint.rs:122–154` retorna dentro del bloque `physical_sde: Some`, tanto si emite como si no. Frío, z dentro del umbral o vida media fuera del filtro ya no alcanzan el legacy. El contador y momentos legacy tampoco se actualizan por esa ruta. Toda intención emitida por ese bloque declara `Continuous` y una duración entera positiva, por el clamp con mínimo 500 ms. Es una propiedad del flujo, no evidencia de calibración física, estacionariedad o rentabilidad.
2. **OU-R4-02, estado SDE:** `vecm_arbitrage.rs:270–282` separa inicialización (`count == 0`) de `ts_ms <= last_ts_ms`. Este último retorna antes de cualquier escritura; preserva también parámetros y seis momentos privados, aunque el test nuevo sólo compara tres campos públicos. La rama de inicialización pasa de incrementar count a asignar 1; bajo su precondición count=0 son equivalentes. La resta de timestamps sólo ocurre después de demostrar crecimiento estricto.

## Residual concreto: rechazo del dato no implica abstención del consumidor

**Prioridad de contrato: P2, alcance opt-in/API; incidencia productiva no demostrada.**

`vecm_arbitrage.rs:277–281` devuelve `stationary_zscore(last_value)` ante timestamp duplicado/retrógrado. `multivariate_coint.rs:123–148` no recibe indicación de rechazo: si `count >= 10`, la vida media cumple su filtro y el z anterior supera el umbral, emite otra intención. Ningún campo necesita mutar para que esto ocurra. Los precios finitos de la llamada rechazada no determinan ese z.

Contraejemplo estático por contrato público, **no ejecutado**: construir el motor opt-in y dejar el SDE con `theta=0.1`, `mu=0`, `sigma=0.01`, `last_value=0.1`, `last_ts_ms=1000`, `count=10` (campos públicos). Invocar con precios `[1;4]` y timestamp 1000 o 500 devuelve un z anterior de aproximadamente 4.472 y permite `Short`, con duración aproximada 6931 ms. Este estado se configura mediante campos públicos; no se presenta como trayectoria observada del entrenamiento ni incidente vivo. Más generalmente, para cualquier décima actualización aceptada que emita (en count=10 el retorno usa parámetros posteriores), repetir el timestamp conserva exactamente el mismo z y las condiciones de emisión.

Contrato mínimo propuesto, pendiente de implementación y pruebas: distinguir observación aceptada de rechazada en la API SDE (`Option`/resultado con motivo), y exigir aceptación para emitir en el wrapper. Una guarda temporal previa en el wrapper puede corregir el caso inmediato manteniendo la API actual; debe compartir la política de inicialización y no transformar automáticamente `t=0` en error. Cubrir estado completo y resultado `None` para duplicado/retrógrado tanto frío como maduro, con una actualización causal de control que sí pueda emitir. No se propone cambiar umbrales para esconder esta composición.

## Límites temporales, numéricos y científicos que siguen abiertos

- **T0:** no hay sentinel temporal explícito: la inicialización depende de count, y la primera muestra finita en t=0 se acepta. Después, otro t=0 se rechaza y t=1 ms puede aceptarse. Sin un contrato que reserve cero para timestamp ausente, esto no constituye por sí solo un bug. Decidir si cero representa epoch/origen local o ausencia exige conocer el adaptador; no se encontró caller productivo que lo resuelva. Deben probarse los casos según ese contrato.
- **Entrada no finita:** el wrapper descarta precios no positivos/no finitos, pesos no finitos y spread no finito antes de invocar SDE (`90–115`), preservando ambos estados en esos rechazos. La API SDE directa ante value no finito devuelve z anterior (`267–269`), sin aceptación tipada. Constructores/parámetros públicos y momentos no tienen validación integral: mu no se sanea en el constructor, infinitos positivos en theta/sigma pueden sobrevivir a los mínimos, y productos de valores finitos grandes pueden desbordarse (`295–301`) antes de validar/confirmar el estado. Son límites preexistentes, no regresiones introducidas por los dos guards.
- **Conteo:** `count += 1` en SDE sigue sin suma comprobada, a diferencia del contador legacy. Dominio público extremo no cubierto, no se afirma alcanzabilidad económica.
- **Intervalo mínimo:** con timestamps u64 en milisegundos, después de crecimiento estricto el mínimo delta es 0.001 s. La rama `dt_sec < 1e-4` no es alcanzable por entradas ordinarias de esta firma; no proporciona una política de datos demasiado cercanos.
- **Duración:** el filtro exige vida media finita y <=3600 s; junto al clamp 500–43,200,000 ms, las emisiones ordinarias realmente no superan 3,600,000 ms. El clamp inferior puede hacer la duración mayor que la vida media estimada. `theta=+Inf` produce vida media cero y satisface ese filtro; no se valida theta finita allí. Por tanto “duración positiva calculada desde el estimador y acotada” es verificable; “calibrada”, “100% física” o un certificado de OU continuo no lo son.
- **Tiempo irregular:** el ajuste acumula pares con distintos dt en una sola regresión de pendiente b y transforma esa pendiente usando sólo el dt actual (`294–326`). Los guards no demuestran ajuste correcto bajo tiempos irregulares. La memoria conserva clamp por actualización. Evaluar unidades, transición exacta/likelihood según cada delta y calibración queda para un lote separado; no corresponde arreglarlo alterando thresholds en este merge.

## Alcance de los tests y documentación

- El test nuevo de fallback (`multivariate_coint.rs:533–574`) exige abstención en las dos primeras observaciones. Su comprobación posterior está dentro de `if let Some`, por lo que ese segmento no prueba emisión. Existe un test anterior (`491–530`) que exige una señal continua tras 60 observaciones y un shock, pero no se ejecutó aquí.
- El test nuevo del SDE (`vecm_arbitrage.rs:439–469`) cubre dup/retrógrado en count=1 y luego un tick causal. No prueba wrapper maduro, retorno `None`, todos los campos, t=0 ni parámetros inválidos.
- `basket_state_contract.rs` usa modo legacy: sus verificaciones de no contaminación/overflow no pueden trasladarse al opt-in SDE. El fixture diagnóstico versionado permanece vinculado a su fuente histórica; su reproducción antigua tampoco valida Ω28.
- Las cifras 29 unit + 11 contract y check sin warnings se leyeron en el mensaje documental de Antigravity; no fueron verificadas independientemente en esta revisión. El integrador debe vincular sus recibos al árbol final.
- Redacción de cierre recomendada: “corregidos fallback legacy y mutación SDE ante timestamps no crecientes; permanece pendiente el contrato de abstención del consumidor y la validación numérica/estadística del modo opt-in”. La expresión “rechazo limpio” debe limitarse al estado SDE hasta comprobar el resultado del wrapper. No cerrar I5 completo ni extrapolar estos tests a OOS, calibración de confianza o la hipótesis económica 100%/72h.