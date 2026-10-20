# Puerto F7 de recursos sobre main — 20 de octubre de 2026

Base de implementación: `597de4af0efae7c64c34b618651c357d406d38c3`. Puerto reanclado sobre `28125c1ae71fb95b25dc25c4c78c1ecddb9d46a7`, incorporando los dos avances documentales de ROOT y BASE sin cambiar los siete pins de fuente/contratos/workflow. ROOT cedió este carril autónomo para preparar, probar, publicar y fusionar el lote de recursos; conserva Core/Risk/Flow/Hidden/Booktick/Reporter y su CONTROL privado. BASE conserva su índice histórico y merge incompleto. El permiso F7 no libera ese índice ni transfiere resultados Rust privados.

## Cambios del protocolo desde el lote original

Los seis commits históricos `314962fe` → `94eff457` → `16b02279` → `ef4cd9c4` → `2a496435` → `59ebb58f` no llegaron por ascendencia a esta base. Sus documentos se incorporan como recibos fechados, con la aclaración editorial de ROOT sobre B15-16 propuesto y V1. Sus resultados históricos no certifican este puerto. La implementación se renueva conservando el hook de Fable y el semáforo introducidos después, en vez de aplicar los archivos antiguos completos.

AGENT_SYNC §4 exige un único target del checkout primario, también desde worktrees; todo Cargo pasa por `scripts/cargo_semaforo.sh`. El lote limita sus cambios a los tres lanzadores, tres contratos Python, CI de esos contratos y documentos. No modifica el semáforo, la configuración Cargo, las APIs financieras ni los archivos de otros agentes.

## Comportamiento que se verifica

- Los dos lanzadores PowerShell invocan una vez el semáforo con `run --bin multi_coin_simulator -j 1`, desde su propio proyecto. Restauran ubicación, PATH y variables que cambian, y conservan el código de salida del hijo. Un fallo del simulador no lo vuelve a arrancar veinte veces.
- MSYS Bash se resuelve a su ejecutable real, evitando el lanzador WSL del PATH. El semáforo decide el target compartido. Los scripts no crean target-sim ni un target por worktree.
- T1 usa jobs e hilos independientes, ambos uno por defecto, con formatos positivos 02/+02 admitidos y sintaxis válida de Cargo para jobs conservada. El hook publicado del semáforo se conserva sin adquirirlo dos veces. Targets alternativos, hooks ambiguos y RUSTFLAGS de entorno no obtienen un build por fallback.
- Se conserva el comando, perfil, corpus y criterio del oráculo T1. El sello/touch heredado sigue siendo un mecanismo previo a la adquisición: este lote no acredita su atomicidad ni frescura frente a otros builds. RB-H2 conserva ese frente y no se instala por proximidad su rama pendiente.
- La configuración y los modelos/tapes de simulación no cambian. Las mediciones que dependan de modelos/datos del checkout primario siguen debiendo ejecutarse allí; no se cambia silenciosamente el proyecto seleccionado para buscarlos.

## Validación propia

En curso con fuentes de main597 (RED) y el candidato. Los primeros ensayos se conservan: T1 necesitó corregir el PATH de MSYS en el fixture y una comparación de alias de ruta en el script; PowerShell reveló que el fixture debe fijar el PATH después del arranque de pwsh, que añade su propio directorio. PS inicial: RED 20 FAIL/2 TimeoutExpired y candidato 21 FAIL/1 TimeoutExpired, ambos 10 métodos; no cuentan como GREEN. Los timeouts de 30 s aparecieron también sin reintentos, por lo que su causa no se atribuye a los veinte intentos del baseline. T1 RED final con harness 5478e180: 17 métodos, 33 fallos de aserción, cero errores, exit1 en 434,682 s; GREEN final: 17/17 métodos, exit0 en 326,753 s, fuente 2a20b383 y harness 5478e180 idénticos antes/después; formatos: un método/cuatro casos aprobado, exit0 en 50,536 s. PowerShell final con harness b068e510: RED 10 métodos/22 fallos de aserción/cero errores en 126,913 s; GREEN 10/10 en 85,372 s, inputs estables. Total GREEN local: 28 métodos, sin skips. [Recibo y hashes](RECIBO_PUERTO_F7_2026-10-20.json). Los resultados finales y hashes se incorporan al recibo de este puerto. PowerShell y Bash son reales; Cargo es un capturador aislado, sin compilación ni ejecución de Rust. El semáforo se copia a un sandbox con raíz temporal, procesos/RAM de fixture y barrido deshabilitado; no se ejecuta sobre el target vivo. Todos los ensayos se serializan.

CI incorpora un job Windows específico de contratos Python. Los gates Rust existentes se conservan y tienen que terminar sobre la composición publicada antes de fusionar. No se heredan los verdes de main ni los privados de ROOT.

## GitHub: estado actualizado

PR64, PR67, PR68 y PR69 ya están fusionados. Los tres fixtures de expiración que fallaban el 15 de octubre se repararon independientemente en `65428a01` (LG-24); no se vuelve a aplicar 7c sólo porque su SHA esté ausente. La avería posterior de ast_mutator en d709 se corrigió en e0ae80f3 y entró con PR86/ef1ce50. Los seis jobs Unit y Replay de ef1ce50 aprobaron. El tip597 es documental y no tiene ejecución propia, conforme a paths-ignore.

PR65 y PR66 siguen abiertos; el CONTROL ROOT no es un candidato público validado. La simetría de liquidez oculta sigue pendiente en main y permanece en el carril ROOT. El diagnóstico del 15 de octubre se conserva como corte histórico.

Auditoría RO actual: dos PR abiertos, 65/66, ambos draft reservados y sin checks/runs actuales; mergeabilidad API desconocida, no fallo de merge demostrado. Los últimos doce runs fallidos revisados se agrupan en aserción ast_mutator 0.62 (2), F1-027/028 (6), contrato IOC (1), voto BTC F1-098 (2) y E0046 de on_bracket_close ausente en una rama Sol (1). La referencia ef1 verde comprueba explícitamente AST/F1/IOC/F1-098 y all-targets; no valida F7 ni A0.

PowerShell autentica el CLI con su APPDATA, con lectura exit0 y permiso push; errores Node exit4 por APPDATA y timeouts TLS/transporte se distinguen de fallos de código. No se encontró recibo de rechazo de commit/push que permita atribuirlo a auth o al hook. El pre-commit compartido sólo ejecuta diff --cached --check. No hay protección/rulesets activos en main en este corte: aplicaremos manualmente los ocho checks del candidato antes de fusionar. Recibo RO: logs/f7-20261020/github-triage-ro-20261020.json, SHA256 190c39f7b2ccfc68a397e95d055571d5a27780f44ffb224673458a514692d7ca. No se accede al contenido de credenciales ni se modifican políticas.

## Recursos y cierre

Al reanudar se observaron aproximadamente 0,49 GiB libres de 23,38 GiB físicos y cero Cargo/rustc/god_engine. Este lote no abre Cargo local ni pide cerrar aplicaciones. Los builds de CI son remotos. ROOT ocupa A0 Hidden en un runner GitHub hospedado. Confirmado ese aislamiento, amplió el acuerdo: F7 y SI pueden usar runners remotos distintos sin Cargo local; la integración en main permanece serializada por head/base verificados. Los logs se conservan bajo logs/f7-20261020, y el worktree propio se retirará sólo después de preservar evidencia y comprobar la ascendencia en el servidor.
