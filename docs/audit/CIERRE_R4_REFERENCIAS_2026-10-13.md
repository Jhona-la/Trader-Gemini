# Cierre R4 de T1, integración y referencias — 2026-10-13

Main remoto, main local y origin/main quedaron verificados en f59f19b8a6eaa78b0ff31260846e9447ece66528, árbol 2f72e3df4ab85f45fc822b6a945df0a48d69d6a8, antes de este último cierre documental. El árbol contiene 1.825 rutas. El SHA que incorpora este documento se verifica posteriormente mediante el recibo externo terminal, sin incluir su propio hash dentro del commit.

T1 terminó en f625: 18/144 = 12,5 % de dimensiones sensibles, mínimo 0,110 intacto. Los nueve recibos finales registran 713 aprobadas, cero fallidas, siete ignoradas y una filtrada; no son tests únicos ni una auditoría de todos los archivos. La compilación de todos los targets en release pasó. El [agregado publicado](RECIBO_VALIDACION_R4_2026-10-13.json), la [adenda](ADENDA_CIERRE_R4_2026-10-13.md) y el [primer recibo de publicación](RECIBO_PUBLICACION_R4_2026-10-13.json) conservan sus respectivos cortes.

Los ceros no aportan capital de evidencia; timestamps duplicados o antiguos se rechazan antes de cualquier escritura y ambos cierres de un par deben avanzar. Los contratos prueban abstención con ceros, falta de inflación por replay y composición con el consumidor real de riesgo. Las políticas y umbrales no se relajaron.

Referencias efectivamente cerradas:

- codex/root-spectral-zero-replay-2026-10-11: detach en f59f19b8 y eliminación local mediante branch -d; nunca estuvo publicada como rama remota.
- codex/integration-recovery-2026-10-07: detach en eb32ac96 y eliminación local mediante branch -d; sus 229 archivos de recibos, 35.054.953 bytes, conservan los hashes originales.
- codex/platt-age-evidence-1013: el propietario SI comprobó la integración y eliminó su ref local y remota con lease sobre 92792116 a las 08:52:51 UTC. Su worktree y archivos se conservaron.

La lectura fresca de refs confirma que el remoto contiene sólo main. Quedan reservas activas AGY F7, G8 y Sol, además de C-F5, H3, Qoder y las ramas locales cf4/cl84 con commits exclusivos. No se eliminan por ausencia de worktree o por semejanza de contenido. AGY F7 conserva cambios locales sin publicar.

La verificación independiente SI de f59 confirmó ascendencia 927/f625, 664 inputs compilables/637 Rust idénticos a T1, modelos físicos iguales en presencia, contenido, tamaño y mtime, y 24 candidatos de genoma sin cambios. Recibo original logs/lh12-final-main-proof.json SHA a64bb42f765157eb732178cb6bf405d060bbc1432cc426dc9319748a1e2227d4, copiado binariamente a target/r4-zero-replay/oct13/SI_FINAL_MAIN_PROOF_f59.json. Incluye límites sobre activación real, entradas no declaradas y observación transitoria. La revisión adicional declaró 104 includes literales y 24 manifests sin consumo del delta documental; no se usa este conteo como auditoría semántica integral.

Recibos de cierre externos al índice, conservados en target/r4-zero-replay:

- oct13/FINAL_MAIN_PUBLICATION_EXTERNAL_PROOF.json: SHA 62375a2a115a29c6593b0b69fa5a218150df83277bfdc797d6e1ef1cbb93c75a, verificación main f59 y actualización local por CAS.
- oct13/ROOT_TWO_REFS_CLEANUP_EXECUTED.json: SHA e63f20e2cb7408feb16171689712826e8d044ca67c2969be3f6b06943cd4ea1b, cierres propios y preservación de evidencia.
- 25-post-detach-final.compiled-equivalence.json: SHA f8edd621270677e5e264a047d4c0d3dd2e4eb7be9be06eb652544652321909c9, equivalencia de los 664 inputs tras detach.

Una consulta Graphify del revisor SI actualizó sólo graphify-out/cache/last_query_stamp en el worktree R4. Los 18 bytes quedaron preservados sin staging ni restauración; su SHA 61f1cde1c7079a8943a0368e3767bb7f58b24991685612117f861b52b092b901 es igual antes y después del detach. Es un derivado de consulta local, fuera de los inputs compilables y de los cambios publicados.

La copia externa sellada previa contiene 1.030 archivos/286.020.220 bytes, manifest SHA 41301930675f4689a870e782ba597606377e8592046bffb519e879ea9b9c5c34; los recibos posteriores entran en una nueva copia con GUID y manifiesto propios. No constituye un entorno completamente reproducible.

El CI observado de f59 sigue pendiente. No se heredan resultados de 88, 98 ni otros padres. El último commit sólo documental y la rama temporal codex/root-cierre-r4-2026-10-13 se verifican y cierran después de su push FF. Su recibo terminal se guarda con las pruebas externas para evitar un ciclo de hash.

El ledger de 1.816 rutas permanece íntegramente pendiente de revisión de archivo completo, con tres rutas de hunks verificados. Las rutas posteriores, incluido este documento, siguen pendientes en R8/R9. Próximo trabajo: R4-O01/O02/EPOCHS y SI-O02, filtración/nulo condicional, familia efectiva consultada, frescura del capital de evidencia y mapping a cartera/PnL. Siete ignorados y la meta económica ×2/72h continúan abiertos; sensibilidad genética no demuestra rentabilidad ni consistencia.
