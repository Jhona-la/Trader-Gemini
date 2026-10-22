# QRA — recibo por alcance y continuidad

Fecha: 22-Oct-2026, America/Bogota. Responsable: Codex QRA.
Este recibo complementa el inventario inicial; no reclasifica las 2.155 rutas
como auditadas. La rama de integración contiene las aportaciones de Omega217
y MPS sin sustituirlas por la versión de la base inicial.
El ledger `QRA_REVISIONES_2026-10-22.tsv` registra las once rutas de esta ola,
sus blobs base/revisado, alcance parcial, límites y evidencia por SHA.

## Candidato y pruebas

- PR: [#94](https://github.com/Jhona-la/Trader-Gemini/pull/94).
- Head: `55d41e1469963c2f4454d5e03a866988fd664c81`.
- Base: `9f03872da6957ee3e95ae5c2ac21aac37684becf`.
- Merge de prueba: `ece158b2106f6a6ec84f699d2395a166669da380`.
- Árbol: `ef330caf1ecff6976fb2f476f12ca1f76afb0377`.
- `git diff` head→merge de prueba vacío; ambos padres comprobados.
- CI de candidato: unit `39866723713`, replay `39866723776`.
- Candidato: **9/9 checks aprobados**. Unit 7/7; Replay + execution 2/2.
- Integración PR94: main `98e194cb7577df4fa4b37fe9b22304ba4966b07c`,
  padres `3fe8771a` y `55d41e14`, árbol `fde1f05b3dd1439b7155cc0acae8d9f3a03ebe28`.
  Se comprobaron ambos ancestros y el árbol esperado después del merge.
- Main: Unit `39870320003` **7/7 aprobado**; Replay `39870319989`
  **pendiente** al preparar este recibo. La rama de código local/remota ya se retiró; sigue sólo la
  rama documental de recibo hasta registrar ese resultado.

Duraciones observadas del candidato: execution 19m29s, Replay 29m22s,
entrenador 17m14s. El job trasladado ya ejecutó sus contratos desde el inicio;
no se atribuye una reducción causal de tiempo/cómputo a una sola comparación
de runners distintos, ni se confunde esto con latencia de órdenes.

El job [execution-contracts](https://github.com/Jhona-la/Trader-Gemini/actions/runs/39866723776/job/120131029853)
aprobó los once `qra_*` del binario `qra_fixed_filter_contract`: 11 passed,
0 failed, 0 ignored, 0 filtered, a las 23:01:34Z. También aprobó los tres
contratos existentes de `fixed_filter_contract`, incluido el ensayo de
10.000 órdenes sintéticas. No son órdenes reales ni pruebas del wiring.

La coordinación se publicó mediante [PR #95](https://github.com/Jhona-la/Trader-Gemini/pull/95),
main `3fe8771a6940188f455d80a5c440d4c6f439af39`. Sólo añade una fila en
AGENT_SYNC y un Markdown de estado; su rama local/remota ya se retiró.
`git merge-tree` compuso ese main con el head probado: árbol
`fde1f05b3dd1439b7155cc0acae8d9f3a03ebe28`; frente al head sólo cambian esos
dos documentos. No se trasladan resultados a fuentes distintas ni se
reinicia CI por esta publicación documental prevista en paths-ignore.

Historial: `18402aa4b` tuvo Unit aprobado, pero su Replay se canceló al
reanclar sobre la integración de Sol. El log parcial no ejecutó los once
contratos QRA; un “11 passed” de otro contrato no se atribuye al filtro.
`41764cc43` incorporó Sol y se sustituyó por el cambio QRA-03 de distribución
de jobs. Ninguna de esas corridas certifica el SHA final por transferencia.

No hubo ejecución Rust local: el semáforo rechazó el turno por recursos.
Los contratos locales de scripts operaron en directorios temporales. Los
68 contratos únicos están conectados al job Ubuntu y aprobaron allí; el
arreglo del fixture Linux conserva el caso de enumeradores desconocidos.

## Archivos revisados y cambio acotado

| Ruta | Hallazgo o decisión | Evidencia / límite |
|---|---|---|
| `Cargo.toml` | dev/test debug=0 y strip=symbols por mandato actual | inspección de perfiles; sin prometer ahorro medido tras recompilar |
| `.gitignore` | claves PEM/KEY y credentials.json | patrones; no retira archivos ya versionados |
| `crates/execution-engine/src/fixed_filter.rs` | tolerancia creciente redondeaba hacia arriba; pasos fraccionarios >=1; clamp ocultaba rango inválido | proyección entera conservadora sobre valor binario exacto; dominio acotado; sin consumidor vivo |
| `crates/execution-engine/tests/qra_fixed_filter_contract.rs` | once contratos con controles positivos/negativos, formato y oráculos independientes | once aprobados en Rust remoto sobre el candidato `55d41e146`; sin consumidor vivo |
| `scripts/disk_audit.sh` | herramientas/errores podían parecer medición cero; limpieza sin cerrojo | errores explícitos; CLEAN_INCREMENTAL rechaza; no borra target |
| `scripts/higiene_recursos.sh` | borrado legacy de exes sin cerrojo; logs versionados; redondeo por archivo | sólo barrido protegido muta target; logs rastreados se preservan; suma bytes antes de convertir |
| `scripts/barrido_target.sh` | enumeración/rm fallidos podían esconderse o contarse como éxito | fallo cerrado y recibo conservador; retenciones originales |
| `scripts/tests/test_qra_resource_safety.py` | faltaban negativos de los scripts reales | 16 contratos, incluidos borrado parcial, cerrojo, medición y logs versionados |
| `scripts/tests/test_cargo_semaforo.py` | dirname(bash) en Linux aún contiene enumeradores reales | PATH hermético y probe de ausencia; mantiene salida 75 y cerrojo |
| `.github/workflows/unit-suites.yml` | contratos de recursos fuera del CI | job Ubuntu ejecuta cinco archivos Python, 68 casos; instala GNU awk |
| `.github/workflows/replay-contracts.yml` | suite de ejecución empezaba tarde y aún compilaba Polars al cancelar | QRA-03 mueve sólo ese comando a job paralelo con config idéntica; nueve checks exigidos; ahorro por medir |

La revisión del cuantizador no demuestra punto fijo integral: la API f64 ya
perdió el decimal original; 0.3 puede proyectarse a 0.2 en paso 0.1 al
respetar la cota binaria exacta. Nocional sigue en f64. El relevo de ejecución
exige decimal exacto de exchangeInfo y preparación antes de reservar margen.

## Coordinación y hallazgos abiertos

AGY/OMS conserva la implementación de arranque/ejecutor. La revisión estática
QRA-BOOT-01 detectó error de consulta convertido en ausencia de posiciones.
El borrador ajeno observado a las 17:11:35 sigue requiriendo:

1. Diferenciar papel local de Demo Binance al exigir conciliación.
2. Confirmar sólo después de posiciones y órdenes conocidas, adopción completa
   y ausencia de `unresolved`; el wrapper que devuelve ajustes no basta.
3. Validar campos del snapshot sin fabricar ceros; usar sus tipos reales.
4. Revocar al sustituir ejecutor y descartar resultados del ejecutor anterior.
5. Contrato del host con API fallida, Testnet y adopción incompleta; cambiar
   manualmente un booleano sólo prueba el orquestador.

Son hallazgos de un borrador cambiante; buscar de nuevo las anclas. Este recibo
no modifica ese trabajo ni certifica la corrección por su presencia.

Relectura RO de las 18:32: el fallback ya usa campos f64 directamente y
`executor::ActivePosition`; los dos errores de tipos del corte anterior
dejan de ser hallazgos actuales. Su qty firmada se convierte a abs en la
adopción/protección observada, por lo que no se acredita un fallo short por
ese signo. El deserializador normal rechaza NaN/Inf: tampoco se atribuye al
API una omisión NaN sin una entrada reproducible. Persisten el bypass
Testnet, TRUE antes de openOrders/adopción completa y la falta de vínculo
con la generación del ejecutor. Son el contrato negativo prioritario.

Sol `9f03872d` conserva el parser estricto de main y añade qty>0 y el contrato
vivo p/q; elimina el wrapper tensorial que fabricaba ceros. Se integró en
main y se conservan exactamente sus tres rutas Rust en el candidato QRA.
Su cierre documental `9b24655f` se incorporó en main `cfc5ddda`; frente a
`98e194cb` sólo añade dos líneas en AGENT_SYNC. Fuentes idénticas verificadas.
Sol retiró su rama local/remota y worktree: el censo posterior sólo conserva
main en el servidor, además de nuestra rama de recibo cuando se publique.
`codex/verify-pr65-20261022` mantiene cerrojo explícito del dueño; no se declara
abandonada por antigüedad. Reflog actualizado 17:36:23 y recibos modificados
18:04:47: check aprobado, suites WAIT/exit75 por recursos, proceso terminado
pero continuación pendiente y sin liberación explícita. No se retiraron
ramas/worktrees con trabajo activo.

## Recursos y alcance financiero

Snapshot inicial: 67,96 GiB lógicos, target 38,26 y data 23,93; 225.920
archivos sin errores de lectura. Se retiraron 38 gzip no versionados >30 días,
14.900.376 bytes. No se borraron datos/modelos ni artefactos de builds ajenos.
Simulación corregida de higiene: 0 bytes liberados, 407 exes sólo inventariados,
deps/incremental dentro del presupuesto. No hubo motor god_engine iniciado
por QRA ni promoción de Demo a Producción.

La meta +100 %/72 h sigue siendo una hipótesis económica sin certificación
OOS. ROI/PnL/WR y mejora de latencia de esta ola: no medidos. Quedan abiertos
el wiring vivo del cuantizador, la certificación del arranque, la evaluación
causal de modelos, el beneficio neto de fees y las fases posteriores del plan.
