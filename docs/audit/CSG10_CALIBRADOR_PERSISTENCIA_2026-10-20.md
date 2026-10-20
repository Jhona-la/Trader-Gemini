# CSG-10: identidad temporal entre calibración y persistencia

## Problema y alcance

En `81edb20ed96be7558e55102af861ef5a9802fbf6`, `PlattCalibrator::update_at`
conserva el último reloj válido cuando llega una fecha inverosímil. Sin embargo,
`estado_aprendido::alimentar` anota la fecha original. Después de un cierre en
`1_800_000_000_000` ms y otro fechado en `1_800_000_000_000_000`, el modelo
conserva el primer reloj y el registro contiene el segundo. `validar` rechaza
ese estado y `guardar` no lo publica. El contrato previo de CSG-10 sólo ejercía
el calibrador directamente.

Este seguimiento pertenece a CSG-10. No resuelve G022, que trata del transporte
de magnitudes de pagos para estimar esperanza matemática.

## Corrección propuesta

Un único resolvedor determina la fecha efectiva de cada calibrador. El productor
persistente aprende y registra esa misma fecha; una fecha tardía válida conserva
su edad, y una corrupta usa el ancla propia de cada destino. El calibrador por
moneda y el global pueden tener anclas distintas.

Sin ancla, la fecha efectiva de una muestra corrupta es `None`. El formato
persistente vigente exige un `u64`, por lo que ese destino omite tanto el
aprendizaje como el registro de esa muestra. No se inventa una fecha cero.
La API directa de Platt conserva su comportamiento de observaciones sin fecha.

La restauración recupera el reloj máximo guardado antes de reproducir las filas
retenidas. Ese máximo puede haber salido de la ventana y estar a más de 365 días
de una fila tardía; recuperarlo no equivale a recibir un evento futuro nuevo.
Las guardas públicas contra fechas corruptas permanecen activas. La reproducción
usa la misma función de ingesta que el productor.

Para snapshots legacy sin reloj guardado utilizable, se infiere primero el
máximo de las fechas retenidas válidas. Sus diferencias de más de 365 días no
se interpretan como eventos futuros recién recibidos. Las filas ya persistidas
con fechas inverosímiles se omiten del modelo y del registro durante la
restauración: su fecha efectiva original no se puede reconstruir. Esta migración
no imputa esas filas a la fecha de otra operación. El productor en ejecución
conserva su política de anclar una fecha corrupta al último reloj válido.

## Verificación y límites

El candidato añade contratos sobre el Core real, snapshot, `validar`, escritura,
lectura y restauración, con controles de relojes, filas, recuentos y ajuste.
La comprobación nativa del candidato está pendiente. No se transfiere el CI de
otro árbol ni el resultado de una reproducción Python a estos contratos.

No se modifica el formato del archivo. Un archivo antiguo que ya falla
`validar` sigue requiriendo tratamiento separado; esta corrección no acredita
su recuperación ni demuestra una ventaja financiera.
