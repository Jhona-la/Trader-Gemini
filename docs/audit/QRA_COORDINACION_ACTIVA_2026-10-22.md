# QRA — coordinación visible desde main (validación cerrada)

22-Oct-2026, America/Bogota. Responsable: Codex QRA. Las secciones iniciales
registran la validación; el resultado final está en el cierre de este documento.

## Composición en validación

[PR #94](https://github.com/Jhona-la/Trader-Gemini/pull/94), head
`55d41e1469963c2f4454d5e03a866988fd664c81`, código base
`9f03872da6957ee3e95ae5c2ac21aac37684becf`. Corridas propias:
[Unit](https://github.com/Jhona-la/Trader-Gemini/actions/runs/39866723713) y
[Replay + execution](https://github.com/Jhona-la/Trader-Gemini/actions/runs/39866723776).
Requiere nueve checks: siete Unit/Python y dos jobs Replay/ejecución.
No heredar resultados de candidatos anteriores ni confundir once tests de
otro contrato con los once `qra_fixed_filter_contract`.

Se solicita coordinar el siguiente merge de código con el cierre de esta
composición: dos avances concurrentes de main ya obligaron a reanclar y
repetir compilaciones. Los agentes continúan sus ramas y revisiones. Una
corrección urgente conserva prioridad; no hay cerrojo global de archivos.
Si sólo avanza documentación, comprobar igualdad de fuentes y resolver sus
textos antes de integrar; la política existente omite CI por cambios sólo
documentales. Nunca trasladar ese razonamiento a fuentes distintas.

## Reservas y alcance

QRA escribe fixed_filter y contrato propio, scripts de recursos/contratos,
workflows unit/replay y documentos QRA. El host/OMS pertenece a AGY; QRA no
edita su borrador. Sol F1-092 ya se conserva en la base y sus tres rutas Rust
son idénticas en el candidato. La reserva QRA termina con el recibo de
integración y limpieza de sus ramas/worktree.

Los 68 contratos de recursos pasan; los resultados Rust se atribuyen al SHA
de cada corrida. QRA no ejecuta Cargo local con RAM libre <4 GiB, no abre
targets privados ni elimina artefactos de compilaciones ajenas. El job
paralelo de ejecución conserva comando/configuración/doctests; puede reducir
espera a cambio de más minutos de runner. La ganancia aún no está medida.

## QRA-BOOT-01: relevo abierto al dueño del host

El arranque publicado trataba un error de posiciones como lista vacía. El
borrador Omega218 observado a las 17:11:35 añadió una bandera, pero todavía
requería corregir y probar:

- `is_demo_mode` incluye Testnet vivo y permite omitir la conciliación.
- TRUE debe exigir posiciones y openOrders conocidas y adopción completa;
  `reconcile_arena` descarta `unresolved`, por lo que acabar el loop no basta.
- El fallback aplicaba parse a campos ya f64 y usaba Position sin export
  encontrado; validar el snapshot sin fabricar ceros y con tipos correctos.
- Revocar antes del reemplazo de ejecutor y rechazar resultados antiguos.
- Probar el host/API con fallos, Testnet y adopción incompleta; alternar un
  booleano manualmente sólo prueba el orquestador.

Rebuscar anclas: es revisión estática de un borrador cambiante, no un juicio
sobre una versión futura ni una certificación de su corrección. PR94 no
modifica esas rutas ni acredita operación Demo/Producción o beneficio OOS.

Actualización RO 18:32: el fallback ya usa f64 y `executor::ActivePosition`;
los errores de tipos anteriores no siguen abiertos. El signo de qty se
normaliza en los consumidores observados y el deserializador rechaza NaN/Inf.
Persisten Testnet, confirmación antes de openOrders/adopción y generación del
ejecutor: requieren un contrato del host, no una bandera alternada a mano.

## Cierre de validación y liberación de reserva QRA

PR94 integrada en `98e194cb`; sus corridas de push propias terminaron
**9/9 PASS**: Unit `39870320003` (7/7), Replay `39870319989` (2/2).
Los once QRA Rust, tres contratos previos y 68 pruebas de recursos constan
en los logs de main. PR96 publica el recibo y el ledger por archivo; desde
ese código sólo cambió documentación, sin nueva corrida por paths-ignore.
La reserva de integración QRA queda liberada. Retirar sólo la rama/worktree
propios del recibo; verify PR65 mantiene dueño y cerrojo ACTIVE. Sol ya
retiró su rama y worktree. Host/OMS y las fases posteriores siguen abiertos.
