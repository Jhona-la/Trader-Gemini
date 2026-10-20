# F7-L01 — lanzadores de simulación: ejecución única y contexto explícito

Fecha: 2026-10-15, America/Bogota. Codex, chat `01a14167`. Continuación de [F7-T1](ADENDA_RECURSOS_T1_2026-10-15.md), cuya entrega exacta es `94eff4571b65668ac8d2a13c3fbddc73ef9d853a`. ROOT conserva ese corte congelado; este lote se entrega aparte. Base compartida observada: main y remoto `ba445df79fa5f82402cd675af4b150f8a0398dea` (Ω132).

Durante esta ola ROOT reabrió un borde de compatibilidad de F7-T1: su validación rechazaba `02` y `+02`, aunque el parser `usize` del oráculo los acepta. Se corrigió primero en el commit independiente `16b02279c4b0ca79d66f6cac95ac44ddbe473077`, hijo directo de `94eff457`, con sólo la línea de validación y un test nuevo. Los cuatro casos, argumento y entorno para cada formato, dieron RED 4/4 fallidos y GREEN 4/4 aprobados con idéntico harness; conservan el valor exacto y jobs 1. El harness histórico de 11 métodos/25 escenarios no cambió ni se le atribuye esta cobertura posterior. Su recibo separado se archivó en `logs/recursos-t1-formatos-20261015`. Este fix no incluye los wrappers del presente lote.

## Propósito y alcance

La meta sigue siendo contrastar crecimiento neto de equity del 100% cada 72 horas. El plan BASE15 conserva sus 1.863 rutas y su orden de dependencias; el delta F7-T1 hasta Ω132 tiene 9 añadidas y 21 modificadas. Este lote toma dos filas ya existentes de F7, `build_loop.ps1` y `run_sim_5.ps1`, sin crear otra cola ni reiniciar expedientes.

La ejecución de una medición debe tener identidad de proyecto y contexto conocidos. Un fallo de salida no demuestra que no hubo efectos anteriores: en una transformación con estado `F(s)`, reintentar no es inocuo salvo que se haya establecido idempotencia o rollback. No se asume `F(F(s)) = F(s)`. Esta es una condición de reproducibilidad para validación y evolución, no una fórmula que pruebe rentabilidad.

## Evidencia en el corte original

Ambos scripts están rastreados y limpios en Ω132. `build_loop.ps1`, blob `adb752a05e2ab8c31058e71f4d5eda88bc5bddb2`, ejecuta `cargo run --bin multi_coin_simulator -j 1 --target-dir target-sim-3`. `run_sim_5.ps1`, blob `0badf8d62f9c47fffb59a7e6346ad646f00f3ceb`, hace lo mismo en `target-sim-5` y establece `CARGO_INCREMENTAL=0`.

Los dos repiten hasta veinte veces **cualquier** salida no nula, con una pausa de dos segundos. No distinguen fallo de compilación de una simulación ya iniciada. Además, Cargo y las rutas relativas usan el directorio del llamador. Lanzar el script por su ruta absoluta desde otra carpeta puede seleccionar otro manifiesto; no se acredita la identidad del proyecto por el nombre del `.ps1`.

El binario existe: `Cargo.toml:124–126` declara `multi_coin_simulator` y su fuente. El constructor por defecto de Core selecciona `IsolatedSimulation`, y el recorrido revisado no envía órdenes al exchange. Eso no convierte la herramienta en una operación sin efectos persistentes: `src/bin/multi_coin_simulator.rs:321,334` puede guardar `models/DarkAlpha_BTCUSDT.json` en las ramas de modelo corrupto/ausente. Este efecto ya tiene antecedente en Línea G (`docs/audit/linea_g/g7_hallazgos.tsv`, fila src01, familia nan-unknown/parity). No se lo cuenta como una causa nueva ni se modifica ese Rust en este lote.

Por tanto, el mismo reintento automático puede repetir una ejecución con estado parcialmente modificado. El contrato corregido impide esa repetición implícita; no afirma eliminar el efecto de escritura del simulador.

## Contrato de los wrappers corregidos

- Una sola invocación de Cargo, tanto si termina con éxito como si falla; conservar su código nativo de salida. Si Cargo no existe, fallar sin reutilizar un `LASTEXITCODE` anterior.
- Elegir el proyecto de `$PSScriptRoot` mediante `Push-Location -LiteralPath`, y restaurar la ubicación del llamador en `finally`.
- Usar `<proyecto>/target` como `CARGO_TARGET_DIR` sólo si el llamador no definió un valor; conservar un override explícito y restaurar su estado anterior al salir. Desaparecen los `--target-dir target-sim-*` forzados.
- Mantener `cargo run --bin multi_coin_simulator -j 1`, sin cambiar modelo, datos, capital, fees, perfil o criterio financiero. `run_sim_5` conserva `CARGO_INCREMENTAL=0` durante la invocación; restaurar el entorno del llamador al terminar.
- Manejar explícitamente los códigos nativos en PowerShell. Los errores de preparación/resolución de comandos producen salida no nula, con restauración del contexto.

Respetar un override preexistente no autoriza crear targets alternativos prohibidos por la regla del proyecto. Tampoco hay exclusión global entre procesos: el turno pesado permanece coordinado con BASE y ROOT. Esta entrega no ejecuta el simulador ni toca modelos, datos, perfiles, cachés o procesos activos.

Un override relativo conserva su texto y Cargo lo interpreta desde el proyecto seleccionado, ahora `$PSScriptRoot`. Los contratos de override cubren rutas absolutas con espacios; no se afirma conservar una resolución anterior basada en el directorio ajeno del llamador.

## Recursos observados, sin atribución de causalidad

Muestreo ligero a las **16:41:04 de Bogotá**: RAM visible 23,38 GiB, libre 5,55 GiB; disco C libre 220,62 GiB. La lectura anterior de F7-T1 había dado 225,08 GiB libres, con otras compilaciones activas. Esa diferencia no identifica qué proceso escribió los bytes ni demuestra una fuga.

Seguían presentes los PID Cargo 25468 (inicio 16:06:52), 38360 (15:07:33) y 37688 (16:30:40), además de workers rustc. No se observaron instancias de `god_engine.exe`. No se interrumpió ni migró ninguna compilación. No se repitió el recorrido masivo del target principal, cuya medición anterior sigue siendo una cota inferior histórica de 131,107 GiB.

## Auditoría negativa de telemetría

Los revisores descartaron dos atribuciones apresuradas de consumo:

1. `telemetry-server/src/lock_free_logger.rs` sólo tiene sus dos tests como consumidores encontrados; el host usa `macros.rs`. Su canal está acotado entre 1.024 y 500.000 mensajes. Destruir el último emisor permite drenar y terminar: no se demostró una fuga perpetua. Sí faltan una barrera explícita de cierre y un oráculo de escritura en el primer test, que sólo observa creación del archivo; el append no tiene retención al activarlo. Quedan como deuda de API/tests, sin atribuirles el disco del motor vivo.
2. `telemetry-server/src/flight_recorder.rs` no tiene una activación productiva encontrada. Su tamaño cumple `Siguiente = max(actual, 64 × clamp(capacidad, 1, 10.000.000))`. Un archivo creado sólo por esta API queda acotado a 640.000.000 bytes; bajar capacidad conserva el tamaño máximo anterior. Tamaño mapeado no equivale a RAM residente. El archivo externo preexistente mayor y la ausencia de un límite explícito de mmap son límites del constructor, ya relacionados con los expedientes de telemetría. No se borra ni trunca evidencia para corregirlos.

El logger vivo de `macros.rs` ya tiene cola acotada e inicialización LG-11, y el host llama `vaciar_telemetria` al cerrar. Los tres módulos siguen pendientes de revisión completa en su cola; esta lectura no los certifica ni crea identificadores duplicados.

## Validación y sincronización

La regresión ejecuta PowerShell real sobre copias de los scripts en carpetas temporales con espacios. Un `cargo.cmd` situado en un PATH aislado llama únicamente a un recorder Python: captura argumentos, entorno y cwd; nunca compila o ejecuta Rust. Un `Start-Sleep` simulado evita esperas artificiales del baseline, sin alterar el contador de reintentos ni la salida del Cargo simulado. Se comprueban también los casos de fallo y Cargo ausente.

El [recibo del lote](F7_LAUNCHERS_RECIBO_2026-10-15.json) contiene hashes, comandos y contraste RED/GREEN. Se conservan los ensayos fallidos. Un resultado verde de estos wrappers no certifica el binario, sus métricas, la meta económica ni el T-1 genético. No se inicia Cargo real durante el turno reservado de BASE.

Resultado con idéntico harness: **7 métodos y 14 escenarios**. RED sobre los scripts originales: 10 reportes de fallo, cero errores, exit 1 en 66,648 s; los cuatro escenarios de Cargo ausente ya pasaban y se conservan como protección de regresión. GREEN: todos los escenarios aprobados, exit 0 en 42,981 s. No se cuentan los diez reportes como diez causas distintas. La revisión independiente del diff confirmó una única llamada, el alcance de las preferencias de PowerShell y la restauración de contexto.

ROOT y BASE acusaron la reserva de los dos wrappers y sus contratos. Memoria, maestros, coordinación y el árbol BASE continúan reservados por sus dueños; se entrega a ROOT el vínculo y el commit acotado para su siguiente actualización. `314962fe` y `94eff457` aún no eran ancestros de main en la comprobación de esta ola. No hay una nueva eliminación libre acreditada de refs ajenas. El worktree sparse propio se retirará tras archivar y comprobar restauración de sus recibos; la rama se conserva mientras contenga commits no integrados.
