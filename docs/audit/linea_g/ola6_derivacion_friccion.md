# Derivación del techo de fricción (ola 6 de la Línea G, lente H2)

Texto del revisor tal cual, seguido del comentario del verificador. Ver el plan de la Línea G §14.

## Derivación

1) Datos. Genoma base (genome.rs:759-762): taker 0,05 %, suelo 1 pb, latencia 13,59 ms, max_fee_pct 1 %. Fricción f = roundtrip_friction = 2·0,0005 + 2·(0,0001 + ATR·√(13,59/60 000)) ≈ 0,00133, con un ATR de 1 min ≈ 0,3-0,45 % (tp_sl.rs:279-282). Orden mínima 5,0 + 0,10 = 5,10 USD; redondeada al lote fino queda en ≈ 5,1-5,7. Geometría de referencia SL 0,5 % y RR 2,25 ⇒ f_R = f/SL = 0,266.

2) Cómo cobra hoy el sistema la fricción: tres veces, y sólo una con p.
(a) Geometría: SL ≥ f/0,65 ⇔ f_R ≤ 0,65, y RR ≥ 1,5 + 2,5·f_R (tp_sl.rs:366-379; genome.rs:2350 y 2450).
(b) EV fuera de sonda (lib.rs:1176-1219): p·RR − (1 − p) > m·f_R, con p puntual y m = 1,1 en estándar o 1,25 en micro ⇒ p > 0,398 / 0,410. En sonda sólo se exige RR > f_R.
(c) fee_impact: f·L ≤ lerp(1 %; 3,5 %; w) ⇔ f_R ≤ techo(w)/(L·SL). Aquí L = mín(⌈5,1/M⌉, 30, ⌊log_lerp(50; 6,5; w)⌋) tras el rescate y M es el margen Kelly. Esta tercera condición no contiene ni p ni RR.

3) Qué mide (c). f·L = f·N/M. Con N = N_min (rescate) es la comisión fija de la orden (≈ 0,0068 USD) dividida por el margen contable. Para un N dado, L sólo reparte el nocional entre margen y préstamo (capital_regime.rs:202-210). No cambia ni la comisión en USD, ni el riesgo al stop, ni f_R, ni la fricción por unidad de capital f·N/C. Sólo tendría sentido si M = C: entonces f·L = f·N/C sería el arrastre de cada operación sobre el capital (es el origen visible en _max_safe_leverage, lib.rs:1246-1250, que no tiene lector). El sistema nunca pone M = C.

4) Derivación. Sea r = N·SL/C la fracción de riesgo, con ganancia (TP − f)·N y pérdida (SL + f)·N. Crecimiento log por operación:
g(r) = p·ln(1 + r(RR − f_R)) + (1 − p)·ln(1 − r(1 + f_R)).
g'(0) = p(RR − f_R) − (1 − p)(1 + f_R) = p·RR − (1 − p) − f_R.
La fricción entra con coeficiente exactamente 1, porque f_R·[p + (1 − p)] = f_R. Con p desconocida se usa su cota inferior. La orden es viable si y sólo si
p_LCB·RR − (1 − p_LCB) − f_R > 0 ⇔ p_LCB > p* = (1 + f_R)/(1 + RR) = (SL + f)/(TP + SL) ⇔ f < p_LCB·(TP + SL) − SL.
Ése es el techo de fricción: en unidades de R, sin L, sin margen y sin capital. El tamaño que se sigue de lo mismo es r* = p_LCB/(1 + f_R) − (1 − p_LCB)/(RR − f_R) (con ½ por política) y N = r·C/(SL + f).

5) Números.
- p* = 1,266/3,25 = 0,390.
- Peldaño a = 0,9, d = 0,2 %: con la p_TP del oráculo (0,657), g'(0) = +0,87 R y el techo es f < 0,005·(0,657·3,25 − 1) = 57 pb.
- Con el acierto neto realizado y su cota (58/89, Beta(58,5; 31,5), p_LCB ≈ 0,568; sólo ilustrativo, porque por LG-5 no es P(TP primero)): f_R,máx = 0,846 ⇒ f < 42 pb.
- f = 13,3 pb pasa con holgura en todos los capitales. ½·r* = ½·(0,568/1,266 − 0,432/1,984) = 0,115.

El veto actual en esos mismos capitales, con conf 0,98 y M en frío = 0,98·0,00845·rn·(1 + 0,42w)·C, expresado como techo implícito en f_R = techo(w)/(L·SL) frente al f_R real de 0,266:
- 13 USD: L ≤ 6, techo 3,5 % ⇒ 1,17. Pasa.
- 26 USD: L = 15, techo 2,41 % ⇒ 0,32. Pasa.
- 35 USD: L = 17-22, techo 1,53 % ⇒ 0,18-0,14. Veta.
- 50 USD: L = 13-17, techo 1 % ⇒ 0,15-0,12. Veta.
- 65 USD: L = 10-13 ⇒ 0,20-0,15. Veta.
- 100 USD: con τ 30 s L = 7 ⇒ 0,29, pasa; con L = 8-9 ⇒ 0,25-0,22, veta.
- 215 USD: L = 3-4 ⇒ 0,67-0,50. Pasa.
El techo derivado (0,85 con p_LCB; 1,14 con el oráculo) es el mismo en los siete capitales. El actual oscila entre 0,12 y 1,17 según el capital para la misma orden.

6) Veredicto. Derivar el techo desde la geometría da exactamente la condición del EV neto (b). A un techo que dependa de L no le queda nada que medir salvo el reparto entre margen y préstamo. Por tanto debe RETIRARSE: lib.rs:1222-1230 y 1417-1420, con el literal 0,035. La puerta EV debe corregirse: p_LCB en lugar de la p puntual, y coeficiente 1 en lugar de m. Los literales 1,05/0,20/1,80 no tienen origen, y su margen (m − 1)·f_R = 0,03-0,07 R es entre 4 y 10 veces menor que la incertidumbre de p a n = 89 (0,27 R). Si se quiere acotar L, que sea por la liquidación: 1/L − mmr_spec > SL + deslizamiento del stop, todo sacado del spec y de la orden; con SL 0,5 % y L ≤ 30, 1/30 = 3,3 % ≫ SL, así que no muerde. La viabilidad de la orden mínima ya la juzga D-750 (orden_viable, lib.rs:1279-1286). Por aritmética, retirar el veto cierra el valle del T-3: a 35-65 USD nada más rechaza hasta :1426. Lo que no cierra es el estado de exploración con ¼·clamp_min (H2-02), que con D1 tampoco debe dimensionar. Con fee_impact vigente, el ½-Kelly en espacio de riesgo quedaría topado en r ≤ 4 % fuera de micro (H2-04).

7) Rama de la Línea C (origin/claude/project-thread-ilf0a8). No cambia nada de esto: kelly_cold (:381-392), m (:1175-1180), max_fee_limit (:1217-1218), max_lev_cap (:1296), el rescate (:1310-1314) y fee_impact (:1405-1407) quedan idénticos. Lo que añade es REJ_ESTRATEGIA_FALSADA (Ville, H0 = +0,30 R), pasa R-15 a sombra y cambia el freno de drawdown. Además CL-108 (45981888) persiste trade_count, PF y kelly_fraction entre reinicios. Tras el merge, LH-02 sigue en pie: el valle deja de reaparecer en cada reinicio, pero queda de forma persistente en monedas nuevas o con pf_lcb ≤ 0,5.

## Comentario del verificador

La derivación es correcta en lo esencial y sus números se reproducen; tiene un error de conversión al nocional.

Correcto:
- g'(0) = p·RR − (1 − p) − f_R: la fricción entra con coeficiente 1.
- p* = (1 + f_R)/(1 + RR) = (SL + f)/(TP + SL), y el techo f < p·(TP + SL) − SL.
- r* = p/(1 + f_R) − q/(RR − f_R) maximiza g con r = N·SL/C.
- Unidades coherentes: f_R y techo/(L·SL) son adimensionales (en R).

Números verificados, con f = 13,3 pb, conf 0,98, clamp_min = 0,0338, min_conf 0,5599 y rn = (30 s/τ)^0,1986 (curva de SL entre 0,177 % y 0,75 % a 12 h):
- p* = 0,390; +0,87 R; 57 pb; p_LCB = 0,568 (Beta(58,5; 31,5), sd 0,05); f_R,máx = 0,846, es decir 42 pb; ½·r* = 0,115.
- Techos implícitos: 1,17 / 0,32 / 0,18-0,14 / 0,15-0,12 / 0,20-0,15 / 0,29 (0,25-0,22) / 0,67-0,50.
- Constantes de (2)(a): 0,65 y WORST_TOLERATED_WR = 0,40, que dan 1,5 + 2,5·f_R.
- _max_safe_leverage no tiene lector.
- Rama de la Línea C (:381-392, :1175-1180, :1217-1218, :1296, :1310-1314, :1405-1407), CL-108 (trade_count, gross_*, kelly_fraction) y Ville con H0 = +0,30 R.

Error:
- Escribe «N = r·C/(SL + f)», pero con su propia definición r = N·SL/C se obtiene N = r·C/SL. La alternativa coherente es usar la pérdida ℓ = r·(1 + f_R) = p − q/b_net con N = ℓ·C/(SL + f).
- Tal como está, infradimensiona por 1/(1 + f_R) ≈ 0,79.
- Se propaga a H2-04: 912 → 1 150 USD; 237 → 299 USD; r ≤ 4,3 % → 3,4 % (4,0 % → 3,15 % con L ≤ 7); en micro, 0,63 % → 0,50 %.
- Se propaga también a la propuesta de H2-08: el «63 %» sólo cuadra con N = r·C/SL.
- Ninguna conclusión cambia.

Matices:
1. «El 0,035 no salva 13-26 USD» es falso para 20 y 26 USD: f·L = 1,06 % y 2,0 % superan el 1 %. Sólo con w = 1 deja de morder.
2. «p puntual» vale en el camino nativo porque el núcleo siempre adjunta la p de Platt. La rama con cota jerárquica (lib.rs:1166-1174) existe pero no se usa ahí.
3. «Nada más rechaza hasta :1426» deja fuera orchestrator.allow_trade (:1434). El apoyo es empírico: la intervención con 3,5 % a 100 USD da 89 cierres. No está medido a 35-65 USD.
4. A 100 USD el umbral es rn < 0,8798, es decir τ ≳ 57 s; no sólo τ = 30 s.
5. f_R = 0,266 usa el SL de referencia (0,5 %), no el SL que fija el motor para cada τ: es ilustrativo.
6. Los dos bordes son sensibles a f: el inferior está en ≈ 27,5 USD y 27 USD pasa por sólo 0,03 pp.

La lista de LH «confirmados sin matiz» venía vacía ([]): no hay nada que añadir.
