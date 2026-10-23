# QRA — recibo de continuación: fees, tiempo, prefijo y CI

22-Oct-2026, America/Bogota. Doce rutas de código/configuración revisadas en
alcance; no auditoría completa de esos módulos ni del repositorio.
Plan: QRA_FASE2_2026-10-22.md. Ledger: QRA_FASE2_REVISIONES_2026-10-22.tsv.
El censo original sigue separado.

## Composición y verificación

- [PR97](https://github.com/Jhona-la/Trader-Gemini/pull/97).
- Head: `911a4c863436783a3bc1e549a0b68f8ca0ae77a8`.
- Base: `cf87f8cbc6d0fe3330e460f247dd2b38e9d990f2`.
- Merge de prueba: `77daaf4d0d653f3dcff196b3598c02389e4d9c1a`.
- Tree: `620e1b647d381b0ed4bd05f856ab582fe6f65261`.
- Padres comprobados y diff head/merge vacío.
- Candidato **9/9 SUCCESS**: [Unit 39884104253](https://github.com/Jhona-la/Trader-Gemini/actions/runs/39884104253)
  (7/7) y [Replay 39884104244](https://github.com/Jhona-la/Trader-Gemini/actions/runs/39884104244) (2/2).
- PR97 integrado en `2715600c4525afa7777a840b1eaf916b8428b848`, padres cf87/911a,
  tree `620e1b647d381b0ed4bd05f856ab582fe6f65261` igual al probado.
- CI del push real a main: **9/9 SUCCESS**,
  [Unit 39886670203](https://github.com/Jhona-la/Trader-Gemini/actions/runs/39886670203) y
  [Replay 39886670039](https://github.com/Jhona-la/Trader-Gemini/actions/runs/39886670039).
- La rama de código QRA se retiró local/remoto después de verificar ancestría.
- PR100 parte de `2715600c4` y se integra en `0f870d279` con sólo tres Markdown;
  fuentes/buildinputs idénticos a `2715600c4`. No se atribuye una corrida nueva
  al commit documental.

Dieciocho contratos nuevos QRA (10 fees + 3 tiempo + 5 prefijo) y cuatro Sol
rescatados, reparados y conectados. Los 22 contratos objetivo y los 68
casos de recursos se confirmaron en logs del candidato `77daaf4`. Cada suite
se delimitó por su binario/nombres, con cero failed/ignored/filtered. La
repetición en el push real a main también aprobó; no se trasladan resultados
anteriores ni se cuenta un target compilado como contrato ejecutado.

Historial: `7f4/aaad` alcanzó 5/9 antes de Omega218; `eeb/d7` alcanzó 8/9
antes de Sol F1-093. El candidato `0f76` sobre `e646` falló en all-targets:
[E0308](https://github.com/Jhona-la/Trader-Gemini/actions/runs/39881340004/job/120179361324),
`Arc<u32>` frente a entero en el contrato entrante. `acc2` fue sustituido al
entrar Sol F1-093b/`4047`; se preservaron su corrección de clone/desreferencia
del Arc, las mejoras QRA del contrato y las entradas documentales por unión.
La composición `ca878/4047` también fue sustituida; sus resultados parciales
no certifican el candidato actual.

La base actual incorpora GLM 153 y [PR98](https://github.com/Jhona-la/Trader-Gemini/pull/98).
PR98 sólo modificó tres Markdown de coordinación; su rama ya se retiró.
PR100 publica la integración y el estado pendiente del push real; su rama
también se retiró. El turno propuesto se extiende a las 21:35, America/Bogota,
para terminar esa verificación. Ambas actualizaciones son sólo documentales.

## Cambios y límites

QRA-FEE-01 conecta el parser de comisión al ExecutionProvider real: objeto
completo, símbolo exacto, tasas finitas y duplicados reconocidos rechazados.
Admite campos adicionales y preserva cero y tasas finitas con signo. Diez casos;
sin HTTP privado ni operaciones. Sigue f64 y no acredita tarifa económica,
descuentos BNB/RPI o fees por fill. QRA_FEES_REALIDAD_2026-10-22.md traza la
separación entre gate y RealityPhysics: hallazgo abierto del core/host.

QRA-TIME-01 rechaza el primer campo literal T reconocido si no contiene un
u64 completo. Conserva ausencia = 0 y T = 0, whitespace y permutación de
claves; distingue un valor string T de la clave. Tres casos nuevos, incluido
u64 máximo y entradas corruptas. El rechazo ocurre antes de
update_agg_trade/tick/callback. Scanner parcial: no
certifica JSON completo, claves escapadas/duplicadas, ámbitos ni frescura.

QRA-MEM-01 añade `load_binary_ticks_prefix` en dos vehículos que ya evaluaban
sólo los primeros 2 M de ticks. Valida alineación de todo el archivo antes
del límite, usa reserva falible y conserva cabeceras/sanitización. Cinco casos sintéticos
comparan eventos y capacidad, prefijo 0 y cola desalineada. No se ejecutan los
vehículos manuales. API completa/multimoneda aún pueden cargar todo; mmap
mantiene el rango virtual del archivo. La sanitización previa de malos valores
a cero sigue pendiente de otro contrato.

QRA-CI-01 repara compilación y ejecución del contrato Sol: su corrección Arc
publicada se preserva; `join().is_err()` consume el panic intencional dentro
del scope; el tercer caso atraviesa `is_trained` con el flag GBDT realmente
envenenado. Se conservan el esperado 7 y las precondiciones/valores `0.77`/`0.61`.
Cuatro tests seleccionados junto a `wf_numerario`, sin quitar suites previas.
Los cuerpos de producción de los motores evolutivos y de `leer`/`escribir`
no cambian en esta reparación QRA; sus comentarios aclaran que `Copy`/`Arc`
no demuestran invariantes ni consistencia entre varios campos.

Con `panic = "abort"` el primer panic termina el proceso. El contrato está
diseñado para probar recuperación bajo unwinding y se ejecutó en el candidato;
se repitió también en el push real a main. No certifica supervivencia
en release, validez numérica
de todo estado interrumpido ni un modelo entrenado. Fuentes: [scope](https://doc.rust-lang.org/stable/std/thread/fn.scope.html),
[RwLock](https://doc.rust-lang.org/stable/std/sync/struct.RwLock.html) y
[catch_unwind](https://doc.rust-lang.org/stable/std/panic/fn.catch_unwind.html).

## Recursos, coordinación y limpieza

El tape observado tiene 35.676.321 registros binarios de 40 B; `TickEvent`
materializado ocupa 48 B en 64 bits. Vec estimada: 1,595 GiB frente a 91,6 MiB
para 2 M de ticks. No es una medición de working-set ni de ahorro del sistema.
Admisión a las 19:02: 2,31 GiB libres y dos Cargo ajenos. Corte a las 19:57:
3,53 GiB libres de RAM y 339,98 GiB libres en el volumen C:, sin Cargo,
rustc o god_engine observados. Son mediciones previas de esta ola, no del
instante de publicación. No hubo Cargo propio, target privado ni carga de tapes
pesados. Rustfmt de contratos/parser y diff-check PASS; formato global no
certificado, incluidas diferencias previas fuera del cambio en evidence/imports.

El relevo de fees mantiene el corte histórico `7f4`: catorce de sus quince
blobs siguen idénticos en `911a4c86`; sólo cambió el host. El archivo
`online_random_forest.rs` modificado por Sol es distinto de `random_forest.rs`,
incluido en esa tabla. Las rutas MPS nuevas de GLM 153 también quedan fuera
de esos quince blobs; su composición no se certifica con ese cotejo parcial.

Host/OMS sigue con AGY y MPS/DMRG con GLM. Omega218 mejoró adopción y consulta
periódica de posiciones y órdenes con checked; Testnet/flag inicial, revocación
y generación aún requieren contratos del host. Verify PR65 reapareció ACTIVE
al corte de las 20:16 y se preserva. Sol F1-093b quedó integrado; su rama local
sin worktree, con tip `4047` ancestro de main, se retiró con `git branch -d`
a las 20:16.

WT QRA anterior retirado por Git después de varios queued del archivador:
HEAD `c2bdba425` integrado, cero cambios/untracked y sólo pycache ignorado.
Recibo local: `logs/qra_old_worktree_retirement_20261022.json`. No se eliminó
ningún WT ajeno. La rama de código está retirada; rama/worktree del recibo
se retiran tras su merge, con confirmación en el cuerpo del PR y los logs.

No hubo operaciones, promoción de entorno ni entrenamiento operativo con
datos/modelos reales. ROI/PnL/WR,
mejora de latencia y meta +100 %/72 h: no medidos. Fases posteriores abiertas.


La revisión adicional [Omega220](QRA_REVISION_OMEGA220_2026-10-22.md) se ancla
a `461484710` de la rama AGY, no integrada aquí. Conserva ocho blobs históricos y el anexo0bfd, que confirma
corregidos en fuente ownership y el aumento de cantidad. Persisten límites de
grid/fees y otro assert de representación; no se certifica el CI de esa rama.

Corte local adicional 21:03: RAM libre 2,68 GiB, C libre 339,16 GiB; sin
procesos god_engine, cargo o rustc observados. Corte de procesos 21:04:
language_server 1.183,8 MiB de working set, un proceso Chrome 872,6 MiB y
Hermes 816,5 MiB, entre otros agentes/aplicaciones. Son muestras instantáneas;
working sets pueden compartir páginas y memoria privada no equivale a RAM
residente. No se atribuye todo el uso al proyecto ni se diagnostica fuga por
este corte; no se cerraron procesos ajenos. Captura local ignorada:
logs/qra_process_memory_20261022.json. Liberar SSD no implica liberar esa RAM.


## Evidencia del push real y retirada

[Unit 39886670203](https://github.com/Jhona-la/Trader-Gemini/actions/runs/39886670203)
7/7 y [Replay 39886670039](https://github.com/Jhona-la/Trader-Gemini/actions/runs/39886670039)
2/2 SUCCESS, checkout `2715600c4`. Execution job120196332647 y Replay
job120196332286: commission10, qra_aggressor3 dentro parser23, prefix5 y
lock4; recursos68. Cero fallidos/ignorados/filtrados en esos contratos.
La evidencia compacta local conserva candidato y main separados en
logs/qra_fase2_ci_evidence_20261022.json, sin ejecución Rust local.

Desde ese código, PR100 y este recibo sólo modifican documentos; no se
atribuye una corrida nueva a un SHA omitido por paths-ignore. Liberada reserva
QRA. Verificar rama ausente local/remota y WT limpio con HEAD integrado
antes de retirar el checkout; confirmación final en el cuerpo del PR de recibo.
AGY/Omega220 y Verify ACTIVE permanecen reservados a sus dueños.
