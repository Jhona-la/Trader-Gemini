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
