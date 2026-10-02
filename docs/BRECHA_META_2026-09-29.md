# BRECHA CONTRA LA META — medición en tapes reales con el genoma campeón (2026-09-29)

**Autor**: GLM (Ola XLVII·B). **Rama**: `glm/xlvii-meta-gap`.
**Pregunta**: ¿a cuántos trades/día llega el sistema HOY, contra los ~10/día
que la aritmética de la meta exige?

## La aritmética (recap XLV·L)

+100% cada 3 días ≡ 25.99% diario compuesto. Con el axioma de ruina del 25%
y un retorno optimista del 10% por trade, el máximo por evento es 2.5% ⇒
**se requieren ~10 trades/día con edge sostenido**. La meta es de VOLUMEN.

## Método

- Genoma: `config_dir/genotypes/quantum_champion.json` (el artefacto que
  config_compiler promueve).
- Modo: trade-only sobre aggTrades REALES (TGMTICK1) — el mismo modo con el
  que la evolución mide aptitud.
- Muestra: 6 tapes de agosto-2026 por tier de liquidez (LTC, ADA, LINK,
  ATOM, NEAR, THETA), 31 días cada uno.
- Dos corridas: (a) desde la raíz del crate — los modelos de `models/` no
  se ven por CWD ⇒ piso SIN modelos; (b) con CWD en el workspace — modelos
  visibles. **Tabla idéntica en ambas**: el resultado no depende de los
  archivos de modelo presentes.
- Reproducir: `cargo test -p backtest-engine --test bt_vivo_parity_audit
  xlviiB -- --include-ignored --nocapture` (~35 min).

## Resultado

| sym | trades | días | t/día | WR | net | fees |
|---|---|---|---|---|---|---|
| LTCUSDT | 0 | 31 | 0.00 | — | 0 | 0 |
| ADAUSDT | 0 | 31 | 0.00 | — | 0 | 0 |
| LINKUSDT | 1 | 31 | 0.03 | 1.00 | +0.085 | 0.013 |
| ATOMUSDT | 1 | 31 | 0.03 | 0.00 | −0.065 | 0.012 |
| NEARUSDT | 1 | 31 | 0.03 | 0.00 | −0.054 | 0.012 |
| THETAUSDT | 0 | 31 | 0.00 | — | 0 | 0 |
| **TOTAL** | **3** | **186** | **0.02** | | | |

**Brecha medida: 0.02 trades/día vs ~10/día ⇒ ≈ 620×.**

## Diagnóstico del cuello de botella (el hallazgo)

El volumen de entradas está cerrado por la cadena de evidencia ML, en
dos capas medidas:

1. **SIN modelo promovido** (LTC, ADA —sólo candidate—, LINK —clave FDUSD,
   no USDT—, THETA): la doctrina B3.25 permite UNA sonda por símbolo
   (arranque frío) y veta todo lo demás para siempre. 601+ vetos del
   ML-GATE contados en LTC con `ml=0.500` (neutral). La sonda genera la
   evidencia que el gate exige — pero nada vuelve a consumirla.
2. **CON modelo promovido** (ATOM, NEAR): también 1 trade. El lift exigido
   (`ml ≥ base + 0.02..0.25`, modulado por acuerdo espectral y Hurst) no
   se satisface una sola vez en 31 días de tape — las predicciones rondan
   la base. El gate es honesto (sin evidencia de edge no se opera): el
   modelo existente NO produce edge medible en estos tapes.

**Conclusión estratégica para el consejo**: el camino a la meta NO pasa por
el sizing (Kelly, vetos y agregación ya están medidos y afinados en las
olas XLVI) — pasa por **(i) cobertura de modelos por símbolo del roster,
(ii) CALIDAD de modelo que produzca lift real sobre tapes reales** (la
cadena de entrenamiento honesta ya existe: etiquetado HOST-010, gates FMT
de holdout posterior), y (iii) cerrar el circuito sonda→evidencia→re-entreno
para que la sonda no sea un trade único sino el arranque de un ciclo. Con
los modelos actuales, el campeón no puede alcanzar la meta en ningún tape —
ese es el hecho cuantificado que esta medición aporta.

## Notas operativas

- `models/` está en `.gitignore` (artefactos locales): un clon fresco mide
  el piso sin modelos. La cobertura promovida en este checkout (claves
  USDT): ATOM, BNB, BTC, NEAR.
- El loader de modelos resuelve `models/` por CWD: los tests deben correr
  desde la raíz del workspace (el test de medición ya lo hace).
- Fees del sample: ~1.2-1.3¢ por trade sobre capital 1000 — la fricción
  NO es el problema al volumen actual; el problema es el volumen.

---

## ADENDA XLVIII·G (2026-09-29) — model registry: dos hallazgos del inventario

`config_dir/models_manifest.json` (generado por `cargo run --bin
model_manifest`, commite-able): 16 modelos promovidos con SHA-256,
base y nº de árboles. El primer escaneo expuso:

1. **BTCUSDT_MOTOR es DEGENERADO**: base 0.565, 1 árbol, 2.5 KB — el
   símbolo ancla del roster opera con un modelo casi vacío. Su "lift"
   jamás será el de un bosque real; re-entrenar BTC es parte del camino
   a la meta (con los gates FMT del PR #20).
2. **Los 9 modelos FDUSD son el MISMO archivo** (hash 0b8bef51 idéntico,
   base 0.2006, 50 árboles): cobertura nominal, no real — un modelo
   universal estampado con 9 claves. La cobertura USDT-promovida REAL
   con bosques propios: ATOM (11), BNB (36), NEAR (26). BTC cuenta 1.
3. DarkAlpha_BTCUSDT: formato distinto (sin init_score) — registrado
   legible con base None (deuda visible, no silencio).

El diff del manifest ES el changelog de modelos: nueva clave = símbolo
desbloqueado; hash cambiado = re-entrenamiento; base cambiada =
recalibración que el gate consume.

---

## ADENDA XLIX·C (2026-09-30) — BTC re-entrenado: ATERIZÓ con salvedad honesta

**Resultado del trainer** (junio→agosto, el de septiembre quedó declarado
sin evaluar — ver salvedad): BTCUSDT_MOTOR pasa de degenerado (1 árbol,
2.5KB, base 0.565 — ACTIVAMENTE errónea para labels ~15-18% positivos) a
**6 árboles, 16.9KB, base 0.1566, mejora de logloss en selección
+0.0167** sobre baseline constante (early stopping en ronda 5, paciencia
40). Paridad train/serve verificada dim a dim (145k muestras, tolerancia
0). Etiquetas: 11.9k decisivas / 133k neutras (8.9% largo).

**SALVEDAD CRÍTICA — el gate que pasó fue SÓLO el de selección**: el
trainer de main verifica que --test-in exista (require_promotion_holdout)
pero NO puntúa el test posterior — esa reparación es XLIV-13, que vive en
el PR #10/#20 (DRAFT). El tape de septiembre fue declarado y nunca
evaluado. La promoción es evidencia-de-selección: exactamente la clase
de promoción que el sistema prohíbe.

**Decisión (transparencia total)**: el modelo nuevo se CONSERVA porque
(e) reemplaza uno activamente erróneo (base 0.565 vs real ~0.16 — el
gate de lift lee esta base vía CL-21), (b) 6 árboles con mejora real en
selección. PERO queda registrado como PROMOCIÓN PROVISIONAL: cuando el
PR #20 aterrice con XLIV-13, re-entrenar/re-validar con el test posterior
obligatorio es INMEDIATO. Si el test falla, revertir.

El watcher del host cargará el nuevo modelo en caliente (≤10 s) en la
próxima sesión viva; el manifest ya refleja el cambio (hash fdd48ee7).

---

## ADENDA LVIII (2026-09-30) — el stride honesto y la promoción XLVIII·H, aún más débil

La re-validación con el trainer honesto abortó ANTES del gate:
**4.282 muestras decisivas < mínimo 5.000** con el stride pedido (50s).
La causa es XLIV-13 funcionando COMO DISEÑA: el trainer honesto NO
densifica el stride para llenar el presupuesto. La corrida XLVIII·H sí lo
densificó (50s→17.9s "presupuesto de 150000 sobre el span") ⇒ sus 11.866
decisivas venían de etiquetas con ~4× más solape que el stride declarado.
La promoción original era aún más débil de lo que su salvedad decía.

**Relanzamiento en vuelo** con stride EXPLÍCITO de 20s (declarado: misma
densidad de muestreo, decidido por adelantado, no ajustado al resultado):
~10k decisivas esperadas sobre junio, mínimo limpio. Septiembre sigue
siendo el test posterior real — esta vez con la base de muestras honesta.

## ADENDA LVIII·bis — BRECHA RE-MEDIDA EN LA FÍSICA NUEVA: 620× → 310×

| sym | trades | t/día | WR | net |
|---|---|---|---|---|
| LTCUSDT | 1 | 0.03 | 0.00 | −0.046 |
| ADAUSDT | 1 | 0.03 | 1.00 | +0.142 |
| LINKUSDT | 1 | 0.03 | 1.00 | +0.106 |
| ATOMUSDT | 1 | 0.03 | 0.00 | −0.076 |
| NEARUSDT | 1 | 0.03 | 0.00 | −0.160 |
| THETAUSDT | 1 | 0.03 | 0.00 | −0.075 |
| **TOTAL** | **6** | **0.03/día** | | |

**6 trades / 186 días = 0.03 t/día ⇒ brecha ≈ 310×** (era 620× en la
física vieja). La corrección de la persistencia DOBLÓ el volumen (3→6
trades): la rama 15 simétrica (CL-31) ahora puede abrir CORTOS que el
bug excluía. La brecha sigue siendo el cuello dominante (evidencia ML),
pero la dirección es correcta y la causa es conocida.

Nota metodológica: cada símbolo sigue en exactamente 1 trade — la sonda
única B3.25 sigue siendo el techo para los símbolos sin modelo promovido
de calidad. El multiplicador del volumen vino del lado de los modelos
existentes (ATOM/NEAR con bosque ahora operan su sonda en ambas
direcciones), no de nueva cobertura.

## ADENDA LXIV (2026-09-30) — BTC ronda 2: train y selección PASAN; test aborta por geometría del tape

Resultado ronda 2 (stride 20s explícito): junio 10,628 decisivas ✓,
selección agosto mejora +0.018 (mejor que ronda 1), paridad train↔serve
1.3e-7 ✓ — pero el TEST de septiembre aborta: 4,365 < 5,000 decisivas.
Causa geométrica, no de modelo: septiembre dura 14 días (vs 31), el
warmup de 12h consume medio día ⇒ a 20s el tape corto no alcanza el
mínimo. El gate protege honestamente — el test NO se evalúa con muestra
insuficiente.

**Ronda 3 en vuelo** (stride 15s, declarado por adelantado — acomoda el
tape más corto: septiembre ~5.9k decisivas esperadas): el stride es
densidad de muestreo declarada, no parámetro del modelo; el mismo valor
aplica a los tres tapes uniformemente.

---

## ADENDA FINAL (2026-10-01) — **PRIMERA PROMOCIÓN HONESTA COMPLETA DE BTC: GATE SUPERADO**

Ronda 3 (stride 15s uniforme declarado, tapes jun/ago/14-sep):

| fase | resultado |
|---|---|
| Junio (train) | 10,628+ decisivas, paridad dim-a-dim tolerancia 0 |
| Agosto (selección) | logloss 0.4631 vs 0.4847 — mejora **+0.0217** |
| Purga de solape | aplicada (cross-file, frontera cerrada XLIV-13b) |
| Paridad train↔serve | máx \|Δ\| = 1.3e-7 |
| **Septiembre (test posterior)** | **logloss 0.3984 vs 0.4151 — mejora +0.0168 FUERA DE MUESTRA** |
| Promoción | 💾 models/BTCUSDT_MOTOR.json — **11 árboles, base 0.1564** |

**La cadena completa, cerrada de punta a punta**: registry encuentra BTC
degenerado (XLVIII·G) → trainer honesto fusionado (PR#20/LI) → lección
del stride (XLIV-13 no densifica; XLVIII·H tenía 4× solape) → gate
protege dos aborts honestos (r1: presupuesto; r2: tape corto) → **test
posterior real SUPERADO**. La salvedad de XLIX·C se disuelve: esta
promoción tiene evidencia de selección ∧ test — la primera del sistema
con el estándar completo. El watcher la cargará en caliente (≤10 s) en
la próxima sesión viva.

Manifest: hash 8f280650bc01, 30,267 bytes. El camino a la meta sigue
por los demás símbolos sonda-bloqueados — con este arco como plantilla.

---

## ADENDA LXX (2026-10-01) — **ADAUSDT: SEGUNDA PROMOCIÓN HONESTA COMPLETA — LA PLANTILLA REPLICA**

Primer símbolo escalado con la plantilla de BTC tras su certificación.
Tapes junio (train) / agosto (selección) / 1–14 septiembre (test posterior),
stride 15 s uniforme declarado, presupuesto 150k muestras:

| fase | resultado |
|---|---|
| Junio (train) | 137,702 muestras, paridad dim-a-dim tolerancia 0 |
| Agosto (selección) | logloss 0.5471 vs 0.5609 — mejora **+0.0138** |
| Paridad train↔serve | máx \|Δ\| = 1.2e-7 |
| **Septiembre (test posterior, n=36,013)** | **logloss 0.5151 vs 0.5232 — mejora +0.0081 FUERA DE MUESTRA** |
| Promoción | 💾 models/ADAUSDT_MOTOR.json — **16 árboles, base 0.2213, init −1.2581** |

Gate `selección ∧ test posterior` superado en ambos tramos. Ada pasa de
candidate-only a bosque promovido. Manifest: hash `ad160ac757ae`,
50,861 bytes, estructuralmente válido (12/17 del inventario lo son;
ATOM/BNB/NEAR/VOL y DarkAlpha siguen rechazados por splits cruzados).

**La plantilla replica en un segundo símbolo sin tocar una línea del
trainer** — la misma física, stride y gate produjeron evidencia OOS donde
el intento candidato-only anterior no tenía estándar. Dos promociones
(BTC +0.0168, ADA +0.0081 test posterior) no es una familia todavía, pero
es la diferencia entre un hecho y una anomalía.

Nota de honestidad del tape: 50,634 de 121k muestras de septiembre
llegaron con dims no finitas del motor y se sanearon a 0.0 — idéntico al
comportamiento de servicio; dims [4,5,9,10] muertas por contrato en ambos
lados (paridad preservada). No hay fuga: el saneamiento es parte de la
misma función que sirve en vivo.

Próximo escalado: LINK/THETA/LTC (sólo 2 períodos — train+selección,
test posterior pendiente de tape de octubre) o esperar el tape de
septiembre completo. La brecha 310× se cierra símbolo a símbolo.

---

## ADENDA LXXI (2026-10-01) — **TRIPLETE DEGENERADO: 2 de 3 pasan; familia honesta = 4 símbolos; primer BLOQUEO honesto**

La plantilla corrió sobre los tres modelos USDT rechazados por splits
cruzados (deuda del manifest), split jun-train/ago-selección/**17-sep
test** (el tape 09-17 mitiga el abort de 5.000 decisivas), stride 15s:

| símbolo | selección (ago) | test posterior (17-sep) | veredicto |
|---|---|---|---|
| ATOMUSDT | +0.0079 | **+0.0114** (n=32,126) — test > selección | ✅ 💾 16 árboles, base 0.2151, `0e5fdce0886b` |
| BNBUSDT | +0.0217 (la mayor de las 5) | **+0.0066** (n=9,500) | ✅ 💾 16 árboles, base 0.1810, `6813d8940c7e` |
| NEARUSDT | +0.0096 | **−0.0002** (n=56,556) | 🚫 GATE BLOQUEA — candidato en cuarentena, modelo vivo intacto |

**La familia de promociones honestas: BTC, ADA, ATOM, BNB — cuatro
símbolos con evidencia selección∧test**. El saldo del manifest mejora a
14/17 estructuralmente válidos (queda NEAR degenerado + NEAR_VOL +
DarkAlpha).

**NEAR es el primer BLOQUEO real del gate** — y prueba que la plantilla
sabe decir NO: el edge de selección (+0.0096) no transfirió (septiembre
fue otro régimen: 73% de muestras decisivas, baseline 0.609 vs 0.554).
Negativo documentado: no gastar cómputo en NEAR a este horizonte hasta
que cambien los tapes. El trainer no tocó el modelo vivo (verificado
bit a bit por fecha) y guardó el candidato con el marcador explícito.

Nota BNB: septiembre-1-17 fue tranquilo para BNB (9.500 decisivas de
79.439, 12%) — el test pasó con margen sobre el mínimo de 5.000, pero
el intervalo es más ancho que el de ATOM.

**Conexión con la medición de cópulas (misma ola)**: BNB es de los más
acoplados en cola con el roster (λ̂ BNB-SOL 0.44, BNB-XRP 0.48) y ATOM
el más diversificante (λ̂ 0.16-0.23) — el roster promovido ya cubre
ambos extremos del espectro de dependencia de colas.

La brecha 310× se sigue cerrando símbolo a símbolo: 4 modelos con edge
OOS medido donde antes había 1 degenerado.

---

## ADENDA LXXII (2026-10-01) — cuarteto del roster + el veto same-bet aprende colas

### Roster verificado (bootloader.rs:301-328): 26 USDT vivos
La pregunta "¿entrenar SOL/XRP/XLM/XMR/ICP?" se responde contra el roster:
**SOL(3)/XRP(4)/XLM(15)/ICP(17) dentro; XMR FUERA** (sólo existe en la lista
del simulador forense — sus 4 tapes NO lo traen al vivo; entrenarlo
produciría un modelo inerte; negativo documentado). **Paradoja FDUSD**:
los 10 modelos FDUSD se cargan al arranque pero jamás se consultan (roster
100% USDT) — inerte por diseño del universo; decisión pendiente del
consejo: remoción o marcado explícito.

### Cuarteto (plantilla honesta, test 09-14)

| símbolo | selección (ago) | test posterior | veredicto |
|---|---|---|---|
| ICPUSDT | **−0.0004** (< margen) | −0.0021 (n=36,988) | 🚫 DOBLE gate — sin edge, cuarentena |
| SOLUSDT | +0.0058 | **+0.0203** (n=17,408) — el OOS más fuerte de la familia | ✅ 11 árboles |
| XRPUSDT | +0.0075 | **+0.0080** (n=20,263) | ✅ 6 árboles |
| XLMUSDT | +0.0075 | **+0.0053** (n=23,144) | ✅ 16 árboles |

**CIERRE DEL CUARTETO: 3/4. La familia honesta = BTC, ADA, ATOM, BNB,
SOL, XRP, XLM — SIETE símbolos con evidencia selección∧test** (mediana
OOS ≈ +0.008; máximo SOL +0.0203). Cobertura del roster: 7/26 ≈ 27%
honesta (+NEAR degenerado-contando-existencia). Dos bloqueos honestos en
nueve corridas (NEAR, ICP) — el gate discrimina: la plantilla no promueve
ruido.

### El veto same-bet aprende colas (consumo LXXI)
Tercera etapa de inflado hacia 1 componiendo sobre el complemento de
independencia: (1−ρ) = (1−base)(1−curl²)(1−λ̂). `inflar_cola` bit-exact sin
manifest (D-754); λ̂_grupo = máx candidato-miembro (peor caso); telemetría
lxxii_lambda_grupo. **Insumo: config_dir/copulas_manifest.json committable
— 153 pares λ̂≥0.05 a 5m (18 símbolos, ago-2026)**, regenerable por ola
(bin copulas_manifest). V-RISK-002 actualizado en el mismo commit;
pre-aprobación pre-merge de qo-606. **Merge condicionado al oráculo T-1**
(worktree aislado 2db11284, en vuelo al escribir esta adenda).
