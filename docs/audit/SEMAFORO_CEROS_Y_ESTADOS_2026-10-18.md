# ROOT — semáforo: desconocido no equivale a cero

Corte 2026-10-18T05:14:49.726146+00:00. Base main `d9332455947e5b983d895822f0daca4a2f62b379`.

`procesos_ajenos` conservaba stdout de enumeradores fallidos: tasklist vacío podía volverse 0 mediante grep; PowerShell podía emitir 0 y fallar antes de tr; pgrep podía emitir 0 con error fatal2/3. Además LG25b admitía el valor desconocido `?` después de agotar paciencia con RAM alta. Son fallos de admisión de recursos, separados de los siete fixes Rust de ceros y replay.

Se exige éxito de PowerShell/tasklist antes de aceptar stdout y se distingue pgrep1+0 (sin coincidencias legítimo) de errores fatales. La paciencia LG25b sólo acepta conteos numéricos. Se preservan TTL, gracia120, RAM, ficha/propiedad, liberación, target canónico, CLI y la excepción ROOT V2/V3. Esto no prueba errores internos de enumeradores que terminen0 ni disponibilidad real.

## Evidencia causal

Pruebas Python/Bash realmente ejecutadas, sólo en raíces temporales con reloj, RAM y comandos simulados; SOLO_DECIDIR=1, sentinel Cargo nunca invocado. Ninguna medición de procesos real, target/lock productivo ni prueba Rust.

| Etapa | Corpus idéntico entre lados | RED | GREEN |
|---|---|---|---|
| Desconocido/tasklist | 13 existentes +10 nuevas |18/23;5fallos esperados,exit1 |23/23,exit0 |
| Estado PS/pgrep sobre candidata anterior |23 anteriores +6 nuevas |26/29;3fallos esperados,exit1 |29/29,exit0 |

Los positivos preservan cero legítimo, pgrep1+0, espera previa, RAM adicional, paciencia numérica y bloqueo por cerrojo ajeno fresco. Los 13 tests existentes no cambian. La segunda suite importa sólo el módulo/fixture, sin heredar la clase TestCase: discover registra29, no duplica los diez tests.

Paquetes externos sellados:

- `root-semaforo-fail-closed-portable-20261018-8bdbe36a-4f2c-4189-ac02-4447cfb607d7`: recibo428a5d7a34cc42c4169a67303e10931665bd2f022ba358382462d950c813811f.
- `root-semaforo-enumerator-status-20261018-5529f466-46ca-41f1-b090-305c71b0cfd2`: recibo57f22d414d1707738efb68eac1fa2a72fda67c9a9fd80cdb95fc2eb91ec59ef2; manifiesto52ae046b7782caae19bc3a5e8c5c44605f5cdf5575feb9e74b013f50d388738e.
- Script final01b2ca6e02cbc5644374c7c70a37c3473963be9a57716e197a1f9a9af6d1588f; tests481150be19e07eb5016da55b390c018495e982cead6332fb496f554b980a1531 y9b2497b5e7971dc79744399d6d2dafe5a4bd1fad9431910ca8f230a9ce685a1a.
- GREEN29 FIRSTstderr7843afb5e1bb9c859513b84c371f681b12daabb5a8e21f732f51547f7fb341b4; Python3.14.4 y Git Bash Windows. No ejecución Linux/CI atribuida.

Comando reproducible: `python -B -m unittest discover -s scripts/tests -p "test_cargo_semaforo*.py" -v`; además bash-n y Python AST0. Los tests se omiten si falta bash.

## Plan y límites

Fable T1 release desde04:53:49Z sigue sin terminal acreditado al preparar esta nota; su turno se respeta. GLM sin cesión acreditada. ROOT conserva fuente8e835/2003, HEADfdc/rawdea17ced/lógicobd400417; check nuevo y siete pares siguen0runs. PR67/69/SI son ancestros de main; PR66 sigue draft. El check original0/86 sigue inválido.

Después del turno real: check protegido0/0+12+8, control sólo local, binding y core-clock-red primero; cada corrección condicionada a RED funcional con control positivo, después GREEN, composición actualizada, suites/fullT1 y publicación acotada. La preparación hipotética main049+7 no es ejecución. HIGH LG5-01/02 (restauración/lados sin muestras) y cohorte LG5-06 siguen bajo auditoría: siete PASS no validan por sí solos la política LG5 completa. BASE P2 auditor4096 permanece externo,12declaraciones/0ejecuciones Rust. No se elimina un veto para imponer una meta de retorno.
