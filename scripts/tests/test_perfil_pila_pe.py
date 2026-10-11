"""Tests del desenrollador de pilas x64 de Windows (CL-63)."""
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import perfil_pila_pe as perfil  # noqa: E402

BASE = 0x1_4000_0000
RET = BASE + 0x9000  # dirección de retorno en el llamador


def codigo(desplazamiento, op, info=0):
    return bytes([desplazamiento, op | (info << 4)])


def unwind_info(prologo, codigos, registro_marco=0, desplazamiento_marco=0):
    cab = bytes([1, prologo, len(codigos), registro_marco | (desplazamiento_marco << 4)])
    return cab + b"".join(codigos)


class ModuloFalso:
    """Un módulo con funciones (inicio, fin, rva del UNWIND_INFO) en memoria."""

    def __init__(self, funcs, memoria):
        self.base, self.tamano = BASE, 0x10000
        self.funcs, self.memoria = funcs, memoria

    def contiene(self, va):
        return self.base <= va < self.base + self.tamano

    def funcion(self, rva):
        return next((f for f in self.funcs if f[0] <= rva < f[1]), None)

    def leer_rva(self, rva, n):
        for inicio, datos in self.memoria.items():
            if inicio <= rva < inicio + len(datos):
                return datos[rva - inicio:rva - inicio + n]
        raise KeyError(hex(rva))


def pila(rsp, *qwords):
    return perfil.Pila(rsp, b"".join(struct.pack("<Q", q) for q in qwords))


# push rbp (fin en 1); push rbx (fin en 2); sub rsp, 0x28 (fin en 6).
CLASICA = unwind_info(6, [codigo(6, perfil.ALLOC_SMALL, 4), codigo(2, perfil.PUSH_NONVOL, 3),
                          codigo(1, perfil.PUSH_NONVOL, 5)])
# push rbp (1); sub rsp, 0x40 (5); lea rbp, [rsp+0x20] (10): marco rbp, desplazamiento 2·16.
CON_MARCO = unwind_info(10, [codigo(10, perfil.SET_FPREG), codigo(5, perfil.ALLOC_SMALL, 7),
                             codigo(1, perfil.PUSH_NONVOL, 5)], registro_marco=5, desplazamiento_marco=2)
# push rbp (1); mov rbp, rsp (4); sub rsp, 0x40 (8): el marco se fija ANTES de reservar.
MARCO_TEMPRANO = unwind_info(8, [codigo(8, perfil.ALLOC_SMALL, 7), codigo(4, perfil.SET_FPREG),
                                 codigo(1, perfil.PUSH_NONVOL, 5)], registro_marco=5)


class DesenrolladoTests(unittest.TestCase):
    def setUp(self):
        self.mod = ModuloFalso([(0x1000, 0x1100, 0x2000), (0x3000, 0x3100, 0x2100), (0x4000, 0x4100, 0x2200)],
                               {0x2000: CLASICA, 0x2100: CON_MARCO, 0x2200: MARCO_TEMPRANO})

    def test_cuerpo_de_una_funcion_con_prologo_clasico(self):
        rsp = 0x7000_0000
        p = pila(rsp, *([0] * 5), 0xB0B, 0xBEEF, RET)  # 0x28 de locales, rbx, rbp, retorno
        r = {"rip": BASE + 0x1050, "rsp": rsp, "rbx": 1, "rbp": 2}
        perfil.un_marco(self.mod, r, p)
        self.assertEqual((r["rip"], r["rsp"], r["rbx"], r["rbp"]), (RET, rsp + 0x28 + 24, 0xB0B, 0xBEEF))

    def test_dentro_del_prologo_solo_deshace_lo_ejecutado(self):
        rsp = 0x7000_0000
        r = {"rip": BASE + 0x1002, "rsp": rsp, "rbx": 1, "rbp": 2}  # tras los dos push
        perfil.un_marco(self.mod, r, pila(rsp, 0xB0B, 0xBEEF, RET))
        self.assertEqual((r["rip"], r["rsp"], r["rbx"], r["rbp"]), (RET, rsp + 16 + 8, 0xB0B, 0xBEEF))
        r = {"rip": BASE + 0x1001, "rsp": rsp, "rbx": 1, "rbp": 2}  # tras push rbp
        perfil.un_marco(self.mod, r, pila(rsp, 0xBEEF, RET))
        self.assertEqual((r["rip"], r["rsp"], r["rbx"], r["rbp"]), (RET, rsp + 16, 1, 0xBEEF))

    def test_con_registro_de_marco_ignora_el_alloca_dinamico(self):
        # Tras el prólogo, un alloca de 0x100 bajó RSP; rbp = RSP_prologo + 0x20.
        rsp_prologo = 0x7000_0000
        rbp = rsp_prologo + 0x20
        rsp = rsp_prologo - 0x100
        p = pila(rsp, *([0] * (0x100 // 8)), *([0] * 8), 0xBEEF, RET)
        r = {"rip": BASE + 0x3050, "rsp": rsp, "rbp": rbp}
        perfil.un_marco(self.mod, r, p)
        self.assertEqual((r["rip"], r["rsp"], r["rbp"]), (RET, rsp_prologo + 0x40 + 16, 0xBEEF))

    def test_marco_fijado_antes_de_reservar(self):
        # rbp = RSP tras push rbp; la reserva de 0x40 viene después y no cuenta.
        rbp = 0x7000_0000
        rsp = rbp - 0x40 - 0x80
        p = pila(rsp, *([0] * ((0x40 + 0x80) // 8)), 0xBEEF, RET)
        r = {"rip": BASE + 0x4050, "rsp": rsp, "rbp": rbp}
        perfil.un_marco(self.mod, r, p)
        self.assertEqual((r["rip"], r["rsp"], r["rbp"]), (RET, rbp + 16, 0xBEEF))

    def test_funcion_hoja_sin_pdata(self):
        rsp = 0x7000_0000
        r = {"rip": BASE + 0x5000, "rsp": rsp}
        perfil.un_marco(self.mod, r, pila(rsp, RET))
        self.assertEqual((r["rip"], r["rsp"]), (RET, rsp + 8))

    def test_desenrollar_para_fuera_de_los_modulos(self):
        rsp = 0x7000_0000
        p = pila(rsp, *([0] * 5), 0xB0B, 0xBEEF, 0x1234)  # el llamador no es de ningún módulo
        marcos = perfil.desenrollar([self.mod], {"rip": BASE + 0x1050, "rsp": rsp, "rbx": 0, "rbp": 0}, p)
        self.assertEqual(marcos, [BASE + 0x1050, 0x1234])

    def test_ranuras_por_codigo(self):
        self.assertEqual(perfil.ranuras(perfil.ALLOC_LARGE, 0), 2)
        self.assertEqual(perfil.ranuras(perfil.ALLOC_LARGE, 1), 3)
        self.assertEqual(perfil.ranuras(perfil.SAVE_NONVOL, 3), 2)
        self.assertEqual(perfil.ranuras(perfil.SAVE_XMM128_FAR, 6), 3)
        self.assertEqual(perfil.ranuras(perfil.PUSH_NONVOL, 3), 1)


if __name__ == "__main__":
    unittest.main()
