#!/usr/bin/env python3
"""Perfil por muestreo de un ejecutable PE x64 que corre bajo wine (CL-63).

Bajo wine no hay `perf`, y ni winedbg ni gdb desenrollan las pilas de un
binario de Rust para x86_64-pc-windows-gnu: el código no trae CFI DWARF,
sólo la información de desenrollado de Windows (.pdata/.xdata).
`scripts/muestrear_pila_wine.sh` toma muestras con gdb (registros y 64 KiB
de pila desde RSP); este script las desenrolla con la .pdata/.xdata de cada
módulo PE mapeado (el exe y las DLL de wine) y simboliza las direcciones del
exe con `addr2line -i`, que también da las funciones inlineadas.

Uso: perfil_pila_pe.py MAPS CARPETA EXE [--addr2line PROG]
  MAPS     copia de /proc/<pid>/maps del proceso de wine
  CARPETA  muestras regs_N.txt y pila_N.bin
  EXE      nombre (o sufijo de ruta) del ejecutable principal

Imprime el perfil de hoja (función más interna, contando el inlining) y el
inclusivo (muestras en cuya pila aparece cada función), y escribe
CARPETA/pilas.txt con una pila por línea. El exe debe compilarse con
CARGO_PROFILE_RELEASE_DEBUG=line-tables-only y STRIP=none.
"""
import argparse
import bisect
import collections
import glob
import os
import re
import struct
import subprocess

REGS = {0: "rax", 1: "rcx", 2: "rdx", 3: "rbx", 4: "rsp", 5: "rbp", 6: "rsi", 7: "rdi",
        8: "r8", 9: "r9", 10: "r10", 11: "r11", 12: "r12", 13: "r13", 14: "r14", 15: "r15"}

# Códigos de desenrollado de x64 (UNWIND_CODE.UnwindOp).
PUSH_NONVOL, ALLOC_LARGE, ALLOC_SMALL, SET_FPREG = 0, 1, 2, 3
SAVE_NONVOL, SAVE_NONVOL_FAR, EPILOG, SAVE_XMM128, SAVE_XMM128_FAR, PUSH_MACHFRAME = 4, 5, 6, 8, 9, 10
UNW_FLAG_CHAININFO = 0x4


class Modulo:
    """Imagen PE mapeada en `base`: secciones y tabla .pdata ordenada."""

    def __init__(self, ruta, base, datos=None):
        self.ruta, self.base = ruta, base
        if datos is None:
            with open(ruta, "rb") as f:
                datos = f.read()
        self.datos = datos
        pe = struct.unpack_from("<I", datos, 0x3C)[0]
        nsec = struct.unpack_from("<H", datos, pe + 6)[0]
        opt = struct.unpack_from("<H", datos, pe + 20)[0]
        self.tamano = struct.unpack_from("<I", datos, pe + 24 + 56)[0]
        exc_rva, exc_tam = struct.unpack_from("<II", datos, pe + 24 + 112 + 3 * 8)
        self.secciones = []
        for i in range(nsec):
            vsize, va, rawsize, rawptr = struct.unpack_from("<IIII", datos, pe + 24 + opt + 40 * i + 8)
            self.secciones.append((va, max(vsize, rawsize), rawptr))
        funcs = []
        for i in range(exc_tam // 12):
            b, e, u = struct.unpack_from("<III", self.leer_rva(exc_rva + 12 * i, 12), 0)
            if b:
                funcs.append((b, e, u))
        self.funcs = sorted(funcs)
        self.inicios = [f[0] for f in self.funcs]

    def contiene(self, va):
        return self.base <= va < self.base + self.tamano

    def leer_rva(self, rva, n):
        for va, tam, raw in self.secciones:
            if va <= rva < va + tam:
                o = raw + (rva - va)
                return self.datos[o:o + n]
        raise KeyError(hex(rva))

    def funcion(self, rva):
        i = bisect.bisect_right(self.inicios, rva) - 1
        if i >= 0 and self.funcs[i][0] <= rva < self.funcs[i][1]:
            return self.funcs[i]
        return None


class Pila:
    """Copia de la pila desde `rsp`."""

    def __init__(self, rsp, datos):
        self.rsp, self.datos = rsp, datos

    def q(self, addr):
        o = addr - self.rsp
        if o < 0 or o + 8 > len(self.datos):
            raise IndexError(hex(addr))
        return struct.unpack_from("<Q", self.datos, o)[0]


def ranuras(op, info):
    """Ranuras de 16 bits que ocupa un código de desenrollado."""
    if op == ALLOC_LARGE:
        return 2 if info == 0 else 3
    if op in (SAVE_NONVOL, SAVE_XMM128, EPILOG):
        return 2
    if op in (SAVE_NONVOL_FAR, SAVE_XMM128_FAR):
        return 3
    return 1


def un_marco(mod, r, pila):
    """Desenrolla un marco: actualiza `r` (rip, rsp y no volátiles) en sitio."""
    rva = r["rip"] - mod.base
    f = mod.funcion(rva)
    if f is None:  # función hoja: sin prólogo, la dirección de retorno está en [rsp]
        r["rip"] = pila.q(r["rsp"])
        r["rsp"] += 8
        return
    desplazamiento = rva - f[0]
    uinfo, primero = f[2], True
    while True:
        cab = mod.leer_rva(uinfo, 4)
        flags, prologo, n, marco = cab[0] >> 3, cab[1], cab[2], cab[3]
        bruto = mod.leer_rva(uinfo + 4, 2 * n)
        codigos = [(bruto[2 * i], bruto[2 * i + 1] & 0xF, bruto[2 * i + 1] >> 4,
                    struct.unpack_from("<H", bruto, 2 * i)[0]) for i in range(n)]
        en_prologo = primero and desplazamiento < prologo
        # Como RtlVirtualUnwind: SET_FPREG fija RSP al valor del registro de
        # marco ANTES de restaurar nada, menos su desplazamiento.
        marco_rsp = r[REGS[marco & 0xF]] - 16 * (marco >> 4) if marco & 0xF else None
        i = 0
        while i < n:
            co, op, info, _ = codigos[i]
            k = ranuras(op, info)
            if en_prologo and co > desplazamiento:  # aún no ejecutado
                i += k
                continue
            if op == PUSH_NONVOL:
                r[REGS[info]] = pila.q(r["rsp"])
                r["rsp"] += 8
            elif op == ALLOC_LARGE:
                r["rsp"] += codigos[i + 1][3] * 8 if info == 0 else \
                    codigos[i + 1][3] | (codigos[i + 2][3] << 16)
            elif op == ALLOC_SMALL:
                r["rsp"] += info * 8 + 8
            elif op == SET_FPREG and marco_rsp is not None:
                r["rsp"] = marco_rsp
            elif op == SAVE_NONVOL:
                r[REGS[info]] = pila.q(r["rsp"] + codigos[i + 1][3] * 8)
            elif op == SAVE_NONVOL_FAR:
                r[REGS[info]] = pila.q(r["rsp"] + (codigos[i + 1][3] | (codigos[i + 2][3] << 16)))
            elif op == PUSH_MACHFRAME:
                base = r["rsp"] + (8 if info else 0)
                r["rip"], r["rsp"] = pila.q(base), pila.q(base + 24)
                return
            i += k
        if flags & UNW_FLAG_CHAININFO:
            _, _, uinfo = struct.unpack_from("<III", mod.leer_rva(uinfo + 4 + 2 * ((n + 1) & ~1), 12), 0)
            primero = False
            continue
        break
    r["rip"] = pila.q(r["rsp"])
    r["rsp"] += 8


def desenrollar(modulos, regs, pila, max_marcos=96):
    """Direcciones de la pila, de la hoja hacia afuera."""
    marcos, r = [], dict(regs)
    for _ in range(max_marcos):
        marcos.append(r["rip"])
        mod = next((m for m in modulos if m.contiene(r["rip"])), None)
        if mod is None:
            break
        try:
            un_marco(mod, r, pila)
        except (IndexError, KeyError):
            break
        if r["rip"] == 0:
            break
    return marcos


def leer_regs(ruta):
    regs = {}
    with open(ruta, errors="replace") as f:
        for linea in f:
            m = re.match(r"^(r\w+)\s+0x([0-9a-f]+)", linea)
            if m:
                regs[m.group(1)] = int(m.group(2), 16)
    return regs


def simbolizar(addr2line, exe, direcciones):
    """{dirección: [función más interna, …, la que la contiene]} con -i."""
    if not direcciones:
        return {}
    salida = subprocess.run([addr2line, "-a", "-f", "-C", "-i", "-e", exe],
                            input="\n".join(hex(a) for a in direcciones) + "\n",
                            capture_output=True, text=True, check=True).stdout
    nombres, actual, lineas = {}, None, []
    for linea in salida.splitlines() + ["0x0"]:
        if re.match(r"^0x[0-9a-f]+$", linea):
            if actual is not None:
                nombres[actual] = lineas[0::2] or ["??"]
            actual, lineas = int(linea, 16), []
        else:
            lineas.append(linea.strip())
    return nombres


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("maps")
    ap.add_argument("carpeta")
    ap.add_argument("exe")
    ap.add_argument("--addr2line", default="x86_64-w64-mingw32-addr2line")
    args = ap.parse_args()
    modulos = []
    with open(args.maps) as f:
        for linea in f:
            p = linea.split()
            if len(p) >= 6 and p[2] == "00000000" and re.search(r"\.(dll|exe)$", p[5], re.I):
                modulos.append(Modulo(p[5], int(p[0].split("-")[0], 16)))
    principal = next(m for m in modulos if m.ruta.endswith(args.exe))
    pilas = []
    for regs_txt in sorted(glob.glob(os.path.join(args.carpeta, "regs_*.txt"))):
        n = regs_txt.rsplit("_", 1)[1].split(".")[0]
        pbin = os.path.join(args.carpeta, "pila_%s.bin" % n)
        regs = leer_regs(regs_txt)
        if "rip" in regs and os.path.exists(pbin):
            with open(pbin, "rb") as f:
                pilas.append(desenrollar(modulos, regs, Pila(regs["rsp"], f.read())))
    pilas = [p for p in pilas if p]
    # La hoja se simboliza tal cual; una dirección de retorno, por su llamada
    # (dirección − 1), que es la línea que la contiene.
    hojas = sorted({p[0] for p in pilas if principal.contiene(p[0])})
    retornos = sorted({a for p in pilas for a in p[1:] if principal.contiene(a)})
    de_hoja = simbolizar(args.addr2line, principal.ruta, hojas)
    de_llamada = {a + 1: v for a, v in
                  simbolizar(args.addr2line, principal.ruta, [a - 1 for a in retornos]).items()}

    def de(a, es_hoja):
        if principal.contiene(a):
            return (de_hoja if es_hoja else de_llamada).get(a, ["?%x" % a])
        m = next((m for m in modulos if m.contiene(a)), None)
        return ["[%s]" % (os.path.basename(m.ruta) if m else "?")]

    hoja, inclusivo = collections.Counter(), collections.Counter()
    for p in pilas:
        hoja[de(p[0], True)[0]] += 1
        vistos = set()
        for i, a in enumerate(p):
            for nm in de(a, i == 0):
                if nm not in vistos:
                    vistos.add(nm)
                    inclusivo[nm] += 1
    total = len(pilas)
    print("muestras %d" % total)
    print("== hoja (función más interna, contando el inlining)")
    for nm, c in hoja.most_common(40):
        print("%5.1f%% %4d  %s" % (100.0 * c / total, c, nm[:200]))
    print("== inclusivo (muestras en cuya pila aparece)")
    for nm, c in inclusivo.most_common(150):
        print("%5.1f%% %4d  %s" % (100.0 * c / total, c, nm[:200]))
    with open(os.path.join(args.carpeta, "pilas.txt"), "w") as f:
        for p in pilas:
            f.write(" <- ".join(" / ".join(de(a, i == 0)) for i, a in enumerate(p)) + "\n")


if __name__ == "__main__":
    main()
