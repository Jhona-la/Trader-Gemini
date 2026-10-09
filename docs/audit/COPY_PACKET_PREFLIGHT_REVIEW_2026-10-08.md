# Dictamen acotado del helper de publicación documental

Fecha: 2026-10-08. Revisor independiente: system_inventory.
Leído completo `target/r4-review/copy_packet_to_integration.py`, 83 líneas,
SHA256 `6ae5be065f7da8382297f3304a4b8f5bb24b3d7d4abae2a71c6dded82b8e7f84`.
**Aprobación estática acotada del helper corregido; todavía no procede ejecutarlo.**
No se ejecutó ni importó el helper; no se copió packet, no hubo Cargo/rustc,
ediciones en integración ni mutaciones Git. Sólo lecturas, hashes y simulación
de concatenación de bytes en memoria. Este dictamen no acredita T1 ni publicación.

El recibo real usa `schema: 3`, `completed_behavioral_suites` con `label` directo
en cada fila, candidato `e9c6a435eb1780da7f2c6e29103f3bd7d227d94c` y árbol
`10b8a427dfbdd5b203595b039b2dad50c69d6f68`. Sigue `status: in_progress`,
`pending_suites: ["36-t1"]`, conteos 736/0/1 y sin fila T1 completada.
Por lectura de los guards, el helper abortaría antes de la primera escritura.
El recibo definitivo deberá aportar la fila 36-t1 con exit0, HEAD/post_HEAD
e9, candidate_tree/post_tree 10b8 y resultado exacto 1/0/0, elevando el total
a 737/0/1. El helper ya exige esos campos y esquema3, además de árboles
committed/index de candidato y ausencia de cambios tracked/staged.

Se corrigieron durante la revisión el detector de placeholders (ahora
`@[A-Z0-9_]+@`, que sí detecta el marcador T1_STATE), la identidad de T1 y la protección
de destinos sin seguimiento. El único untracked sobrescribible es nuestra
copia previa de `docs/audit/PROTOCOLO_OOS_R4_2026-10-07.md`, exclusivamente
si conserva SHA256 `60bcff3fc19948b36a4a9fba64763167c8fd0e9706e22c88eb200731f32c1a9a`.
Su contenido actual coincide con ese hash. Otro archivo preexistente sin
seguimiento dentro de los destinos aborta antes del bucle de escritura.

El manifest observado tiene **26 archivos**, todos con hashes correctos,
sin archivos omitidos; SHA256 del manifest
`95f5862835eb22d9e5bd437ebb134a8837e7767e9c42c7d0438c082067266c27`.
Los destinos ordinarios están bajo docs o los prefijos explícitos de scripts
R4; su ruta resuelta queda dentro del worktree integration-recovery. Los dos
destinos especiales son MEMORIA y COORD: ambos se resuelven dentro de ese
worktree y no son symlinks en este snapshot. No se toca otra rama/worktree.
`INTEGRACION_RAMAS_2026-10-07.md` y `RECIBOS_INTEGRACION_R4_2026-10-07.json`
no son destinos del packet; el segundo sólo se lee como requisito.

Las notas empiezan con `##` y se insertan después de la primera línea `#`
del documento. En la simulación, retirar únicamente esa inserción recuperó
**todos los bytes originales**, incluido cuerpo y saltos de línea:

| Documento | Bytes originales | SHA256 original | Recuperación exacta |
|---|---:|---|---|
| `.agents/MEMORIA.md` | 254315 | `552fd92d3165551a4dc00067d76ecd72896a2b5c98add23ea9f5d26a0884c870` | Sí |
| `COORDINACION_CODEX_2026-09-28.md` | 414788 | `7b11bc8be5a094f763dfdad862908f6a5389a78a359a2203cedcd00e865700f6` | Sí |

Precisión del otro manifiesto, el de equivalencia de fuentes: se recalcularon
sus entradas Git con lecturas `ls-tree` sin ejecutar su generador. Son
**532 entradas de inputs por árbol, 2 cambiadas excluidas y 530 idénticas**;
no 528. El digest recalculado de las 530 entradas coincide con el recibo:
`b6cae69ba3e3d973f01fe711c8d3f61106cedf6cfaa5c6f8fa74a340ca4d7d7e`.
No debe confundirse esta identidad con una nueva ejecución de 530 pruebas.

Los únicos comandos Git del helper son lectura de HEAD/árbol/diffs y
`ls-files --error-unmatch`; no contiene add, commit, checkout, reset, clean,
fetch, push ni borrados de refs. Conserva un recibo local ignored después
de copiar. Límite: la copia de varios archivos no es transaccional y sus
guards son asserts; ejecutar con Python normal, sin `-O`, y mantener packet
y destinos sin escritores concurrentes durante la operación. Las notas
especiales se releen tras validar el manifest. Regenerar/revalidar el manifest
cuando root cierre templates y resultados; estos hashes describen este snapshot.
