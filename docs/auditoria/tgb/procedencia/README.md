# Procedencia de la generacion de cintas REAL (TGB-G)

Aqui viven, por hash del original, los metadatos pequenos de la cadena que acredita QUIEN genero una cinta REAL.
Nada de cintas, entradas de mercado ni fuentes: solo texto `clave=valor` (una linea `\n` por par, sin espacios
alrededor, sin `\r`, sin comentarios). Lo verifica `tick_tape::procedencia::admitir_original`.

- `X.bin.procedencia` (junto a la cinta): recibo `formato=tg-procedencia-2` con `version_analizador`, `ruta_logica`,
  `formato_cinta`, `sha256_original`, `bytes_original`, `registros_esperados`, `registros_analizados`, `cobertura`,
  `recetas_compatibles`, `estado_firma`, `primer_testigo_relativa`, `procedencia_acreditada`,
  `evidencia_productor_sha256` y `analisis=intervalos_exhaustivos`. Cuentas, estado, recetas y testigo deben ser los
  recalculados sobre el original entero.
- `<sha256>.evidencia` (en esta carpeta): evidencia del productor `formato=tg-evidencia-productor-1` con `productor`,
  `commit_productor`, `sha256_fuentes_productor`, `sha256_ejecutable`, `receta`, `constante_r_bits`, `origen_receta`,
  `comando`, una o mas `entrada_sha256`, `sha256_salida`, `bytes_salida`, `sha256_registro_salida` y
  `transformaciones=ninguna`.
- Indice revisado `formato=tg-indice-procedencia-1`: `revisado_por=` y una linea
  `evidencia=<sha256>;<productor>;<commit>;<receta>` por evidencia reconocida. Se pasa con `--generation-index` y su
  SHA-256 con `--generation-index-sha256`; ese SHA debe salir del registro revisado por coordinador/integrador, nunca
  del recibo.

ESTADO (17 nov 2026): no hay indice revisado ni recibos. Ningun productor emite todavia la evidencia, asi que
`train_forest` no admite ninguna cinta y `dataset_admission` marca todas `training_eligible=false`. Crear el indice
es una decision del coordinador; el recibo lateral de H (`tg-procedencia-1`) ya no acredita nada.
