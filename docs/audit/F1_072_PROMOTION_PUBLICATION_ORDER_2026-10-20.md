# F1-072 — orden de publicación de promociones (20 octubre 2026)

Estado: implementación aislada con revisión estática, **pendiente de ejecución Rust/CI**. ROOT asignó a SI exclusivamente genome_store.rs y sus contratos; no se modifican configuración, espectro o Core de AGY. Base inicial: 4bcf2f47748d54968a0eec0427c2d5f386c187ea; reanclada a fd956f9abf47e935ba84f76659b0ae1e631bce07 para preservar Ω179/F1-075. Los blobs F1-072 públicos no cambian entre estas bases y28125c1ae71fb95b25dc25c4c78c1ecddb9d46a7. La publicación de esta entrega no acredita integración ni GREEN. No hay cesión de un lease de Cargo.

## Fallo observado, sin reinterpretar el histórico

Unit39421522643, Spectrum118623374360, ejecutó merge6fc584f073fede319fbb7ffa2fa6cf8c045f912c (d2bf352c sobre28125c1a). El log terminal SHA256403f07d4db7a1c48810cbbbf5c82e6d2d79a6781c65186b16577f428fc8e81e1 registra en1497–1517 el fallo de f1_072_genome_store_concurrent_promote_contract.rs:95: activo5, esperado8. Los ocho retornos Ok, generaciones únicas1..8 y ocho historiales parseables ya habían pasado las aserciones anteriores. No demuestra pérdida de historia.

Contraorden finito compatible con el código anterior: gen5 reclama/escribe su historial y se retrasa antes de publicar;6..8 publican y dejan active8;5 renombra después y deja active5. O_EXCL asegura nombres únicos; rename asegura reemplazo íntegro. Ninguno serializa la asignación y publicación de generaciones. Se conserva el criterio active==máxima generación; no se reintenta hasta obtener verde.

Dedup: F1-072, MED, registrado como CERRADO Ω176 en f1_hallazgos.tsv:73. Se conserva ese histórico3/3, pero no acredita esta ejecución ni resuelve el hueco de orden observado. El delta LG22 de PR87 modifica únicamente resultado_orden.rs; el almacén y este test conservan los blobs públicos b012d792… y cc8597ce… de la base. No se atribuye este fallo al cálculo LG22.

## Cambio aislado

promote() adquiere un cerrojo OS exclusivo sobre active.json.promote.lock, archivo persistente hermano de active.json, antes de leer parent. Conserva el handle hasta terminar historia, active y espejo legacy. O_EXCL, fsync y reemplazo atómico permanecen.

El archivo del cerrojo no se elimina: recrearlo durante una liberación permitiría bloquear objetos distintos bajo el mismo nombre. Cada llamada abre un handle independiente; cierre/Drop libera el bloqueo incluso por retorno de error o pánico. El sistema operativo libera handles al terminar el proceso. No hay PID stale que limpiar. File::lock está estable desdeRust1.89; la base fija nightly2026-06-30. Véase la [documentación oficial de File::lock](https://doc.rust-lang.org/std/fs/struct.File.html#method.lock).

El bloqueo coordina publicadores que usan este embudo y puede ser consultivo según plataforma; no protege escrituras ajenas que lo ignoren. No se reutiliza recursivamente un handle bloqueado. rollback carga antes de llamar promote; el bloque de promoción no llama promote ni load_active. Los lectores siguen usando active mediante reemplazo atómico y no necesitan bloquear el archivo sibling. Esto no convierte historia+active+espejo en una transacción durable ante pérdida de energía.

## Contratos y validación pendiente

Los tres tests existentes, incluida la aserción95==8, se conservan byte exactos (SHA25637229f7fde6b644434b7e7ac85ab304057be9cc0369d7426639897c9503857ca). El contrato multiproceso separado usa ocho hijos de current_exe, helper ignorado invocado mediante --exact/--ignored, raíz temporal y entorno explícitos por Command. No muta las variables de entorno del proceso padre. Exige retornos exitosos, generaciones1..8 únicas, padres consecutivos, historia completa/parseable y active==max==8 con igualdad de contenido, además de igualdad del espejo legacy; tiene tiempo acotado y limpieza de hijos propios. El listado normal contendrá un contrato padre y un helper ignorado: sólo el resultado real de CI permitirá afirmar que el padre ejecutó sus ocho hijos.

Comprobaciones estáticas de esta composición y revisión independiente; cero Cargo/rustc local. El candidato inicial f49/4bc obtuvo Unit39425087681 SUCCESS (incluidos originales3 y multiproceso1) y completó el paso all-targets de Replay39425087622. No se transfiere ese resultado a la nueva composición fd956f9a: su Rust/CI propio sigue pendiente. La siguiente validación es CI en runners GitHub-hosted separados, incluyendo los tres contratos originales, el nuevo multiproceso y check de todos los targets. El fallo original se conserva; no se relaja ni se reintenta hasta obtener verde.

## Límite de la composición

El orden del activo persistido no certifica el orden de hot-swap después del retorno. online_daemon.rs:1993 aplica el envelope retornado. El host god_engine.rs:5438–5441 relee sólo si coincide la generación y, si no, conserva el retornado. Un publicador puede demorarse después de liberar el cerrojo y aplicar memoria más tarde. Esta entrega no cambia ni valida ese acoplamiento, calibración, selección evolutiva, fills o crecimiento económico.
