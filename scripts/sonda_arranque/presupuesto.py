"""Presupuesto de pesos de la sonda de arranque C3 (MEDIDASV4-P2), ventana deslizante con reloj inyectado.

Regla (especificacion v4 §3 P2): antes de cada envio, peso reservado en los ultimos 60 000 ms + reserva
del siguiente envio <= 0,5*B, con B = 2400/min (docs/BINANCE_API_MAP.md:116). Sin w atribuible, la
reserva es la cota documentada del endpoint, 10 (docs/BINANCE_API_MAP.md:27), que NO es w medido.
Una reserva no se libera antes de que salga de la ventana (ni por timeout ni por un contador incierto);
un delta medido solo puede subirla. Ademas, un envio y una espera: el siguiente envio se espacia
60 000*w/tope ms del anterior (250 ms con w = 5), asi que no hay rafagas aunque sobre cupo.
"""


class Presupuesto:
    def __init__(self, b_por_minuto=2400, fraccion=0.5, ventana_ms=60000, cota_reserva=10, espaciar=True):
        self.b = b_por_minuto
        self.tope = int(b_por_minuto * fraccion)
        self.ventana_ms = int(ventana_ms)
        self.cota_reserva = int(cota_reserva)
        self.espaciar = espaciar
        self.reservas = []  # [t_ms, w_reservado, motivo]

    def reserva_para(self, w_atribuible):
        """Reserva del siguiente envio: el w atribuible si existe; si no, la cota documentada."""
        if w_atribuible is None or w_atribuible <= 0:
            return self.cota_reserva
        return int(w_atribuible)

    def usado(self, t_ms):
        """Peso reservado vigente en t: reservas con t - ventana < u <= t."""
        return sum(w for (u, w, _) in self.reservas if t_ms - self.ventana_ms < u <= t_ms)

    def _espaciado(self, t_ms):
        if not self.espaciar or not self.reservas:
            return 0
        u, w, _ = self.reservas[-1]
        listo = u + (self.ventana_ms * w + self.tope - 1) // self.tope
        return max(0, listo - t_ms)

    def espera_ms(self, t_ms, w_atribuible):
        """Milisegundos a esperar desde t para que el siguiente envio quepa (0 si cabe ya)."""
        w = self.reserva_para(w_atribuible)
        if w > self.tope:
            raise ValueError(f"una reserva de {w} nunca cabe en el tope {self.tope}")
        dt = self._espaciado(t_ms)
        while self.usado(t_ms + dt) + w > self.tope:
            vencen = sorted(u + self.ventana_ms for (u, _, _) in self.reservas if u + self.ventana_ms > t_ms + dt)
            dt = vencen[0] - t_ms
        return dt

    def reservar(self, t_ms, w_atribuible, motivo=""):
        """Anota la reserva del envio hecho en t. Devuelve su indice para resolverla luego."""
        self.reservas.append([int(t_ms), self.reserva_para(w_atribuible), motivo])
        return len(self.reservas) - 1

    def resolver(self, indice, w_medido=None, motivo=""):
        """Tras la respuesta: un delta medido mayor sube la reserva; nunca la baja ni la libera."""
        fila = self.reservas[indice]
        if w_medido is not None and w_medido > fila[1]:
            fila[1] = int(w_medido)
        fila[2] = motivo
        return fila[1]

    def plan(self, peticiones, w_reserva=None):
        """Duracion minima compatible con el presupuesto para n peticiones (para imprimir antes de empezar)."""
        w = self.reserva_para(w_reserva)
        por_ventana = self.tope // w
        ms = 0 if peticiones <= 0 else (peticiones - 1) * ((self.ventana_ms * w + self.tope - 1) // self.tope)
        return {"peticiones": peticiones, "w_reserva": w, "tope_por_ventana": self.tope,
                "peticiones_por_ventana": por_ventana, "duracion_minima_ms": ms,
                "peso_agregado_reservado": peticiones * w,
                "nota": "el agregado no se compara con el tope por minuto; cuenta la ventana deslizante"}
