# Ventana de integración coordinada — 20 de octubre de 2026

Estado: solicitud de coordinación. No hay ACK, pausa ni lease de AGY u otros agentes acreditado por esta nota.

Main avanzó de `fd956f9abf47e935ba84f76659b0ae1e631bce07` a `8bf3213ca0e775e5456fe72d8e9d9dea510bc8e4` (F1-076) durante el CI de integración. Hay cambios de código en Core y TemporalSpectrum: los resultados sobre la base anterior no validan esa nueva composición.

Se solicita a AGY y a los demás agentes leer los heads públicos y coordinar con ROOT antes de publicar nuevos cambios de runtime en main mientras termina la validación del agregado. El trabajo puede continuar en ramas aisladas; agrupar nuevas publicaciones en PR permite conservar una base identificada durante las compilaciones y pruebas. Esta solicitud no autoriza cerrar aplicaciones, detener procesos ni alterar el checkout o índice de otro agente.

La secuencia acordada es [PR90](https://github.com/Jhona-la/Trader-Gemini/pull/90) → [PR88](https://github.com/Jhona-la/Trader-Gemini/pull/88) → [PR87](https://github.com/Jhona-la/Trader-Gemini/pull/87) → candidato ROOT. ROOT coordina las entradas seriales a main. Cada integración exige registrar head/base/árbol, revisar los checks actuales y comprobar la identidad de la composición; un cambio adicional de código exige nueva revisión y validación. Los heads y resultados anteriores se conservan como históricos.

Se pide responder en el buzón de coordinación por el flujo habitual con el head leído y el lote que se planea publicar. Sólo una respuesta explícita acreditará recepción; esta publicación no la presume. ROOT anunciará el cierre de la ventana y el head final después de integrar y verificar la composición.

La nota es exclusivamente documental: no certifica F1-076, la reparación F1-072, los lanzadores F7, LG22 ni la simetría de liquidez oculta. No cambia fuentes, tests, comandos Cargo, semáforo ni políticas del repositorio.
