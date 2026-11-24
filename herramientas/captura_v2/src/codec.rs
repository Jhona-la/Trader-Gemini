//! Codec sin perdida TGLL (Trader Gemini Lossless) para las horas JSONL de la captura.
//!
//! Idea: cada linea se parte en un "esqueleto" (el texto con cada tramo de digitos ASCII
//! sustituido por el byte 0x00) y una lista de enteros. El esqueleto se guarda una sola vez
//! por archivo (diccionario); cada entero se guarda como diferencia contra el ultimo entero
//! visto en su mismo contexto (flujo de la linea + texto que lo precede), en varint zigzag,
//! con los enteros de cada contexto contiguos dentro del bloque. El resultado pasa por zstd.
//! La decodificacion reproduce los bytes exactos (ceros a la izquierda incluidos); la
//! verificacion compara SHA-256 y longitud del original y de lo decodificado.

use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    io::{self, BufRead, Read, Write},
};

pub const MAGIC: &[u8; 8] = b"TGLL0001";
const PLACEHOLDER: u8 = 0x00;
/// u64::MAX tiene 20 digitos; con 19 todo tramo cabe en u64 sin desbordar.
const MAX_DIGITS: usize = 19;
/// Bytes de texto previo que identifican el campo de un entero (p. ej. `\"E\":` o `\",\"`).
const CONTEXT_LITERAL: usize = 12;
const START_MARK: u8 = 0x01;
/// Tope de una seccion al decodificar: evita reservar memoria absurda si el archivo esta dañado.
const MAX_SECTION: usize = 1 << 30;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub zstd_level: i32,
    pub block_raw_bytes: usize,
    pub window_log: u32,
}

pub struct Report {
    pub lines: u64,
    pub raw_bytes: u64,
    pub raw_sha256: String,
    pub skeletons: u64,
    pub contexts: u64,
    pub raw_fallback_lines: u64,
}

fn put_varint(out: &mut Vec<u8>, mut v: u128) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Unica validacion de varints para secciones y para el flujo descomprimido.
/// Rechaza bits que no caben en u128 y representaciones no minimas antes de
/// desplazar, de forma que una conversion posterior nunca vea un valor truncado.
fn parse_varint(mut next_byte: impl FnMut() -> io::Result<u8>) -> io::Result<u128> {
    let mut v: u128 = 0;
    let mut shift = 0;
    loop {
        let byte = next_byte()?;
        let payload = u128::from(byte & 0x7f);
        if payload > (u128::MAX >> shift) {
            return Err(bad("varint fuera de rango"));
        }
        v |= payload << shift;
        if byte & 0x80 == 0 {
            if shift != 0 && payload == 0 {
                return Err(bad("varint no canonico"));
            }
            return Ok(v);
        }
        shift += 7;
        if shift >= u128::BITS {
            return Err(bad("varint demasiado largo"));
        }
    }
}

fn get_varint(buf: &[u8], pos: &mut usize) -> io::Result<u128> {
    parse_varint(|| {
        let byte = *buf.get(*pos).ok_or_else(|| bad("varint truncado"))?;
        *pos += 1;
        Ok(byte)
    })
}

fn get_usize(buf: &[u8], pos: &mut usize) -> io::Result<usize> {
    usize::try_from(get_varint(buf, pos)?).map_err(|_| bad("longitud fuera de rango"))
}

fn take<'a>(buf: &'a [u8], pos: &mut usize, len: usize) -> io::Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| bad("longitud desborda"))?;
    let s = buf.get(*pos..end).ok_or_else(|| bad("seccion truncada"))?;
    *pos = end;
    Ok(s)
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("TGLL: {msg}"))
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

fn unzigzag(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

/// Clase de la linea: el valor del campo de envolvente `"stream":"..."` (o `"kind":"..."`)
/// leido del esqueleto, para que codificador y decodificador la calculen igual.
fn class_key(skel: &[u8]) -> Vec<u8> {
    for key in [&b"\"stream\":\""[..], &b"\"kind\":\""[..]] {
        if let Some(at) = find(skel, key) {
            let start = at + key.len();
            let end = skel[start..]
                .iter()
                .position(|&b| b == b'"')
                .map_or(skel.len(), |e| start + e);
            let mut k = key.to_vec();
            k.extend_from_slice(&skel[start..end]);
            return k;
        }
    }
    Vec::new()
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Contextos de los enteros de un esqueleto, en orden.
fn skeleton_contexts(
    skel: &[u8],
    classes: &mut HashMap<Vec<u8>, u32>,
    contexts: &mut HashMap<(u32, Vec<u8>), u32>,
    prev: &mut Vec<u64>,
) -> Vec<u32> {
    let next_class = classes.len() as u32;
    let class = *classes.entry(class_key(skel)).or_insert(next_class);
    let mut out = Vec::new();
    let mut start = 0;
    for (i, &b) in skel.iter().enumerate() {
        if b != PLACEHOLDER {
            continue;
        }
        let mut lit = Vec::with_capacity(CONTEXT_LITERAL + 1);
        if i - start <= CONTEXT_LITERAL {
            lit.push(START_MARK);
            lit.extend_from_slice(&skel[start..i]);
        } else {
            lit.extend_from_slice(&skel[i - CONTEXT_LITERAL..i]);
        }
        let next = contexts.len() as u32;
        let id = *contexts.entry((class, lit)).or_insert(next);
        if id as usize == prev.len() {
            prev.push(0);
        }
        out.push(id);
        start = i + 1;
    }
    out
}

/// Parte una linea en esqueleto y enteros (valor, ceros a la izquierda). false = linea que
/// no se puede representar (contiene 0x00 o un tramo de mas de 19 digitos): va en crudo.
fn tokenize(line: &[u8], skel: &mut Vec<u8>, nums: &mut Vec<(u64, u32)>) -> bool {
    skel.clear();
    nums.clear();
    let mut i = 0;
    while i < line.len() {
        let b = line[i];
        if b == PLACEHOLDER {
            return false;
        }
        if b.is_ascii_digit() {
            let s = i;
            while i < line.len() && line[i].is_ascii_digit() {
                i += 1;
            }
            let run = &line[s..i];
            if run.len() > MAX_DIGITS {
                return false;
            }
            let lz = run
                .iter()
                .take(run.len() - 1)
                .take_while(|&&c| c == b'0')
                .count() as u32;
            let mut v: u64 = 0;
            for &c in &run[lz as usize..] {
                v = v * 10 + u64::from(c - b'0');
            }
            skel.push(PLACEHOLDER);
            nums.push((v, lz));
        } else {
            skel.push(b);
            i += 1;
        }
    }
    true
}

struct Block {
    lines: u64,
    raw_bytes: usize,
    ids: Vec<u8>,
    raw: Vec<u8>,
    new_skel_count: u64,
    new_skel: Vec<u8>,
    ctx: Vec<Vec<u8>>,
}

impl Block {
    fn new() -> Self {
        Self {
            lines: 0,
            raw_bytes: 0,
            ids: Vec::new(),
            raw: Vec::new(),
            new_skel_count: 0,
            new_skel: Vec::new(),
            ctx: Vec::new(),
        }
    }

    fn write_to(&mut self, out: &mut impl Write) -> io::Result<()> {
        let mut head = Vec::new();
        head.push(1u8);
        put_varint(&mut head, u128::from(self.lines));
        put_varint(&mut head, self.ids.len() as u128);
        out.write_all(&head)?;
        out.write_all(&self.ids)?;
        head.clear();
        put_varint(&mut head, self.raw.len() as u128);
        out.write_all(&head)?;
        out.write_all(&self.raw)?;
        head.clear();
        put_varint(&mut head, u128::from(self.new_skel_count));
        put_varint(&mut head, self.new_skel.len() as u128);
        out.write_all(&head)?;
        out.write_all(&self.new_skel)?;
        let used: Vec<usize> = (0..self.ctx.len())
            .filter(|&i| !self.ctx[i].is_empty())
            .collect();
        head.clear();
        put_varint(&mut head, used.len() as u128);
        out.write_all(&head)?;
        for i in used {
            head.clear();
            put_varint(&mut head, i as u128);
            put_varint(&mut head, self.ctx[i].len() as u128);
            out.write_all(&head)?;
            out.write_all(&self.ctx[i])?;
            self.ctx[i].clear();
        }
        self.lines = 0;
        self.raw_bytes = 0;
        self.ids.clear();
        self.raw.clear();
        self.new_skel_count = 0;
        self.new_skel.clear();
        Ok(())
    }
}

/// Codifica todo `input` (JSONL crudo) en `output`. Devuelve conteos y SHA-256 del crudo.
pub fn encode(input: impl Read, output: impl Write, p: Params) -> io::Result<Report> {
    let mut input = io::BufReader::with_capacity(1 << 20, input);
    let mut output = output;
    output.write_all(MAGIC)?;
    let mut z = zstd::stream::Encoder::new(output, p.zstd_level)?;
    z.long_distance_matching(true)?;
    z.window_log(p.window_log)?;
    z.include_checksum(true)?;
    let mut hasher = Sha256::new();
    let mut skel_ids: HashMap<Vec<u8>, u32> = HashMap::new();
    let mut skel_ctx: Vec<Vec<u32>> = Vec::new();
    let mut classes = HashMap::new();
    let mut contexts = HashMap::new();
    let mut prev: Vec<u64> = Vec::new();
    let mut block = Block::new();
    let mut line = Vec::with_capacity(1 << 16);
    let mut skel = Vec::with_capacity(1 << 16);
    let mut nums = Vec::with_capacity(4096);
    let (mut lines, mut raw_bytes, mut fallback) = (0u64, 0u64, 0u64);
    let mut final_newline = true;
    loop {
        line.clear();
        let n = input.read_until(b'\n', &mut line)?;
        if n == 0 {
            break;
        }
        hasher.update(&line);
        raw_bytes += n as u64;
        lines += 1;
        block.lines += 1;
        block.raw_bytes += n;
        let body = if line.last() == Some(&b'\n') {
            &line[..n - 1]
        } else {
            final_newline = false;
            &line[..]
        };
        if !tokenize(body, &mut skel, &mut nums) {
            fallback += 1;
            put_varint(&mut block.ids, 0);
            put_varint(&mut block.raw, body.len() as u128);
            block.raw.extend_from_slice(body);
        } else {
            let id = match skel_ids.get(&skel) {
                Some(&id) => id,
                None => {
                    let id = skel_ctx.len() as u32;
                    let ctxs = skeleton_contexts(&skel, &mut classes, &mut contexts, &mut prev);
                    skel_ctx.push(ctxs);
                    skel_ids.insert(skel.clone(), id);
                    block.new_skel_count += 1;
                    put_varint(&mut block.new_skel, skel.len() as u128);
                    block.new_skel.extend_from_slice(&skel);
                    id
                }
            };
            put_varint(&mut block.ids, u128::from(id) + 1);
            let ctxs = &skel_ctx[id as usize];
            if block.ctx.len() < prev.len() {
                block.ctx.resize_with(prev.len(), Vec::new);
            }
            for (&(v, lz), &c) in nums.iter().zip(ctxs.iter()) {
                let c = c as usize;
                let d = zigzag(v.wrapping_sub(prev[c]) as i64);
                prev[c] = v;
                let buf = &mut block.ctx[c];
                if lz == 0 {
                    put_varint(buf, u128::from(d) << 1);
                } else {
                    put_varint(buf, (u128::from(d) << 1) | 1);
                    put_varint(buf, u128::from(lz));
                }
            }
        }
        if block.raw_bytes >= p.block_raw_bytes {
            block.write_to(&mut z)?;
        }
    }
    if block.lines > 0 {
        block.write_to(&mut z)?;
    }
    let sha = hasher.finalize();
    let mut tail = vec![0u8];
    put_varint(&mut tail, u128::from(lines));
    put_varint(&mut tail, u128::from(raw_bytes));
    tail.extend_from_slice(&sha);
    tail.push(u8::from(final_newline));
    z.write_all(&tail)?;
    let mut out = z.finish()?;
    out.flush()?;
    Ok(Report {
        lines,
        raw_bytes,
        raw_sha256: format!("{sha:x}"),
        skeletons: skel_ctx.len() as u64,
        contexts: prev.len() as u64,
        raw_fallback_lines: fallback,
    })
}

fn read_exact_vec(r: &mut impl Read, len: usize) -> io::Result<Vec<u8>> {
    if len > MAX_SECTION {
        return Err(bad("seccion mayor que el maximo"));
    }
    let mut v = vec![0u8; len];
    r.read_exact(&mut v)?;
    Ok(v)
}

fn read_varint(r: &mut impl Read) -> io::Result<u128> {
    parse_varint(|| {
        let mut b = [0u8];
        r.read_exact(&mut b)?;
        Ok(b[0])
    })
}

fn read_len(r: &mut impl Read) -> io::Result<usize> {
    usize::try_from(read_varint(r)?).map_err(|_| bad("longitud fuera de rango"))
}

/// Limites del visitante. `capture` fija bloque, ventana y buffers de la captura;
/// el llamante aporta sus presupuestos medidos de linea y diccionario.
/// El cargo de diccionario no equivale al RSS del proceso ni al allocator.
#[derive(Clone, Copy, Debug)]
pub struct VisitLimits {
    /// Suma de payloads de las secciones ids, raw, esqueletos nuevos y contextos
    /// del bloque, una vez descomprimido zstd. Excluye cabeceras y varints de
    /// longitud externos. No mide el JSONL crudo de `Params.block_raw_bytes`,
    /// ni garantiza que dichas secciones sean menores que ese crudo.
    pub max_block_bytes: usize,
    /// Bytes por linea reconstruida, incluido LF cuando existe.
    pub max_line_bytes: usize,
    /// Cargo logico acumulado del estado global, incluso entre bloques.
    pub max_dictionary_bytes: usize,
    pub io_buffer_bytes: usize,
    pub zstd_window_log_max: u32,
}

impl VisitLimits {
    /// Presupuesto canonico de secciones: valor de CODEC_BLOCK_RAW, src/main.rs:92.
    /// El mismo numero tiene distinta unidad logica en codificador y visitante.
    pub const CAPTURE_BLOCK_BYTES: usize = 64 << 20;
    /// CODEC_WINDOW_LOG, src/main.rs:96; ventana maxima de 128 MiB.
    pub const CAPTURE_WINDOW_LOG_MAX: u32 = 27;
    /// Capacidad ya usada por los buffers del codec en encode y decode.
    pub const CAPTURE_IO_BUFFER_BYTES: usize = 1 << 20;

    /// Los limites de linea y diccionario siguen siendo decisiones explicitas
    /// del llamante. Un bloque valido con secciones mayores que el presupuesto
    /// produce error; no se amplia silenciosamente ni se equipara al crudo.
    pub const fn capture(max_line_bytes: usize, max_dictionary_bytes: usize) -> Self {
        Self {
            max_block_bytes: Self::CAPTURE_BLOCK_BYTES,
            max_line_bytes,
            max_dictionary_bytes,
            io_buffer_bytes: Self::CAPTURE_IO_BUFFER_BYTES,
            zstd_window_log_max: Self::CAPTURE_WINDOW_LOG_MAX,
        }
    }
}

fn bounded_sum(current: usize, extra: usize, limit: usize, msg: &str) -> io::Result<usize> {
    let next = current.checked_add(extra).ok_or_else(|| bad(msg))?;
    if next > limit {
        return Err(bad(msg));
    }
    Ok(next)
}

fn extend_line(out: &mut Vec<u8>, bytes: &[u8], limit: usize) -> io::Result<()> {
    bounded_sum(
        out.len(),
        bytes.len(),
        limit,
        "linea mayor que el presupuesto",
    )?;
    out.extend_from_slice(bytes);
    Ok(())
}

fn read_count(r: &mut impl Read) -> io::Result<u64> {
    u64::try_from(read_varint(r)?).map_err(|_| bad("conteo fuera de rango"))
}

fn read_block_section(r: &mut impl Read, bytes: &mut usize, limit: usize) -> io::Result<Vec<u8>> {
    let len = read_len(r)?;
    *bytes = bounded_sum(*bytes, len, limit, "bloque mayor que el presupuesto")?;
    read_exact_vec(r, len)
}

/// Se cobra antes de copiar o ampliar los diccionarios. Cada esqueleto paga
/// tambien una posible clase y cada placeholder un posible contexto nuevo.
/// Los size_of cubren registros; las tablas y capacidades tienen sobrecoste
/// del allocator. Al cobrar cada registro, tambien queda acotado su numero.
fn dictionary_charge(s: &[u8]) -> io::Result<usize> {
    use std::mem::size_of;
    let per_context = (CONTEXT_LITERAL + 1)
        .checked_add(size_of::<((u32, Vec<u8>), u32)>())
        .and_then(|n| n.checked_add(size_of::<u32>()))
        .and_then(|n| n.checked_add(size_of::<u64>()))
        .ok_or_else(|| bad("presupuesto de diccionario desborda"))?;
    let slots = s.iter().filter(|&&b| b == PLACEHOLDER).count();
    let charge = s
        .len()
        .checked_mul(2)
        .and_then(|n| n.checked_add(size_of::<(Vec<u8>, Vec<u32>)>()))
        .and_then(|n| n.checked_add(size_of::<(Vec<u8>, u32)>()))
        .and_then(|n| {
            slots
                .checked_mul(per_context)
                .and_then(|c| n.checked_add(c))
        })
        .ok_or_else(|| bad("presupuesto de diccionario desborda"))?;
    Ok(charge)
}

fn emit_line(
    line: &[u8],
    hasher: &mut Sha256,
    raw_bytes: &mut u64,
    visit: &mut impl FnMut(&[u8]) -> io::Result<()>,
) -> io::Result<()> {
    *raw_bytes = raw_bytes
        .checked_add(line.len() as u64)
        .ok_or_else(|| bad("conteo de bytes desborda"))?;
    hasher.update(line);
    visit(line)
}

/// Recorre TODOS los bloques y entrega cada linea exacta, incluido su LF si
/// existia. Conserva una sola linea pendiente para resolver el LF de la ultima.
/// El slice solo vive durante la llamada. El callback puede filtrar devolviendo
/// Ok(()), pero no evita reconstruir ni verificar los bytes descartados.
/// Las filas son provisionales: solo Ok(Report) certifica la cola y EOF zstd.
/// Si el callback falla, se propaga su error y no se certifica el archivo.
pub fn visit_lines(
    input: impl Read,
    limits: VisitLimits,
    mut visit: impl FnMut(&[u8]) -> io::Result<()>,
) -> io::Result<Report> {
    if limits.io_buffer_bytes == 0 {
        return Err(bad("buffer de lectura vacio"));
    }
    if limits.zstd_window_log_max == 0 {
        return Err(bad("ventana zstd sin presupuesto"));
    }
    let mut input = io::BufReader::with_capacity(limits.io_buffer_bytes, input);
    let mut magic = [0u8; 8];
    input.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(bad("cabecera desconocida"));
    }
    let mut dec = zstd::stream::Decoder::with_buffer(input)?;
    dec.window_log_max(limits.zstd_window_log_max)?;
    let mut z = io::BufReader::with_capacity(limits.io_buffer_bytes, dec);
    let mut hasher = Sha256::new();
    let mut skels: Vec<(Vec<u8>, Vec<u32>)> = Vec::new();
    let mut classes = HashMap::new();
    let mut contexts = HashMap::new();
    let mut prev: Vec<u64> = Vec::new();
    let mut dictionary_bytes = 0usize;
    let (mut lines, mut raw_bytes, mut fallback) = (0u64, 0u64, 0u64);
    let mut out = Vec::new();
    let mut digits = [0u8; 20];
    let mut pending_line = false;
    loop {
        let mut tag = [0u8];
        z.read_exact(&mut tag)?;
        if tag[0] == 0 {
            break;
        }
        if tag[0] != 1 {
            return Err(bad("etiqueta de bloque desconocida"));
        }
        let nlines = read_count(&mut z)?;
        let mut block_bytes = 0usize;
        let ids = read_block_section(&mut z, &mut block_bytes, limits.max_block_bytes)?;
        if nlines > ids.len() as u64 {
            return Err(bad("bloque con menos ids que lineas"));
        }
        let raw = read_block_section(&mut z, &mut block_bytes, limits.max_block_bytes)?;
        let count = read_len(&mut z)?;
        let new_skel = read_block_section(&mut z, &mut block_bytes, limits.max_block_bytes)?;
        if count > new_skel.len() {
            return Err(bad("bloque con menos esqueletos que los declarados"));
        }
        let mut pos = 0;
        for _ in 0..count {
            let l = get_usize(&new_skel, &mut pos)?;
            let s = take(&new_skel, &mut pos, l)?;
            if s.contains(&b'\n') {
                return Err(bad("LF dentro del esqueleto"));
            }
            dictionary_bytes = bounded_sum(
                dictionary_bytes,
                dictionary_charge(s)?,
                limits.max_dictionary_bytes,
                "diccionario mayor que el presupuesto",
            )?;
            let c = skeleton_contexts(s, &mut classes, &mut contexts, &mut prev);
            skels.push((s.to_vec(), c));
        }
        if pos != new_skel.len() {
            return Err(bad("bloque con esqueletos sobrantes"));
        }
        // Ya se copiaron los esqueletos necesarios al diccionario global.
        drop(new_skel);
        let used = read_len(&mut z)?;
        if used > prev.len() {
            return Err(bad("demasiados contextos en el bloque"));
        }
        let mut ctx: HashMap<usize, (Vec<u8>, usize)> = HashMap::new();
        for _ in 0..used {
            let id = read_len(&mut z)?;
            if id >= prev.len() || ctx.contains_key(&id) {
                return Err(bad("contexto inexistente o duplicado"));
            }
            let data = read_block_section(&mut z, &mut block_bytes, limits.max_block_bytes)?;
            if data.is_empty() {
                return Err(bad("contexto sin datos"));
            }
            ctx.insert(id, (data, 0));
        }
        let (mut ip, mut rp) = (0usize, 0usize);
        for _ in 0..nlines {
            if pending_line {
                extend_line(&mut out, b"\n", limits.max_line_bytes)?;
                emit_line(&out, &mut hasher, &mut raw_bytes, &mut visit)?;
            }
            out.clear();
            let id = get_usize(&ids, &mut ip)?;
            if id == 0 {
                let l = get_usize(&raw, &mut rp)?;
                let body = take(&raw, &mut rp, l)?;
                if body.contains(&b'\n') {
                    return Err(bad("LF dentro del cuerpo raw"));
                }
                extend_line(&mut out, body, limits.max_line_bytes)?;
                fallback = fallback
                    .checked_add(1)
                    .ok_or_else(|| bad("conteo desborda"))?;
            } else {
                let (s, c) = skels
                    .get(id - 1)
                    .ok_or_else(|| bad("esqueleto inexistente"))?;
                let mut k = 0;
                for &b in s {
                    if b != PLACEHOLDER {
                        extend_line(&mut out, &[b], limits.max_line_bytes)?;
                        continue;
                    }
                    let cid = *c.get(k).ok_or_else(|| bad("contexto inexistente"))? as usize;
                    k += 1;
                    let (buf, cur) = ctx.get_mut(&cid).ok_or_else(|| bad("contexto sin datos"))?;
                    let w = get_varint(buf, cur)?;
                    let d = u64::try_from(w >> 1).map_err(|_| bad("delta fuera de rango"))?;
                    let lz = if w & 1 == 1 { get_usize(buf, cur)? } else { 0 };
                    let v = prev[cid].wrapping_add(unzigzag(d) as u64);
                    prev[cid] = v;
                    let end = bounded_sum(
                        out.len(),
                        lz,
                        limits.max_line_bytes,
                        "linea mayor que el presupuesto",
                    )?;
                    out.resize(end, b'0');
                    extend_line(&mut out, fmt_u64(v, &mut digits), limits.max_line_bytes)?;
                }
            }
            pending_line = true;
            lines = lines
                .checked_add(1)
                .ok_or_else(|| bad("conteo de lineas desborda"))?;
        }
        if ip != ids.len() || rp != raw.len() || ctx.values().any(|(b, c)| *c != b.len()) {
            return Err(bad("bloque con datos sobrantes"));
        }
    }
    let total = read_count(&mut z)?;
    let total_bytes = read_count(&mut z)?;
    let sha = read_exact_vec(&mut z, 32)?;
    let mut fnl = [0u8];
    z.read_exact(&mut fnl)?;
    if fnl[0] > 1 {
        return Err(bad("marca de salto final invalida"));
    }
    if pending_line {
        if fnl[0] == 1 {
            extend_line(&mut out, b"\n", limits.max_line_bytes)?;
        } else if out.is_empty() {
            return Err(bad("ultima linea vacia sin LF"));
        }
        emit_line(&out, &mut hasher, &mut raw_bytes, &mut visit)?;
    }
    // Fuerza a zstd a consumir fin de trama/checksum, incluso si la cola TGLL
    // ya cabia en el buffer. No se certifican tramas truncadas ni datos extra.
    let mut extra = [0u8];
    if z.read(&mut extra)? != 0 {
        return Err(bad("datos despues de la cola"));
    }
    let got = hasher.finalize();
    if total != lines || total_bytes != raw_bytes || got.as_slice() != sha.as_slice() {
        return Err(bad("la cola no coincide con lo decodificado"));
    }
    Ok(Report {
        lines,
        raw_bytes,
        raw_sha256: format!("{got:x}"),
        skeletons: skels.len() as u64,
        contexts: prev.len() as u64,
        raw_fallback_lines: fallback,
    })
}

/// Entrada por ruta para el entrenamiento: rechaza .partial antes de abrirlo.
/// La variante Read no conoce nombres y no puede comprobar este requisito.
pub fn visit_file(
    path: impl AsRef<std::path::Path>,
    limits: VisitLimits,
    visit: impl FnMut(&[u8]) -> io::Result<()>,
) -> io::Result<Report> {
    let path = path.as_ref();
    if path.file_name().is_some_and(|name| {
        name.to_string_lossy()
            .to_ascii_lowercase()
            .ends_with(".partial")
    }) {
        return Err(bad("archivo .partial rechazado"));
    }
    visit_lines(std::fs::File::open(path)?, limits, visit)
}

/// Decodifica el JSONL exacto mediante el mismo recorrido y las mismas
/// comprobaciones que visit_lines. Conserva los limites historicos de decode;
/// el entrenamiento usa visit_file con presupuestos explicitos y medidos.
pub fn decode(input: impl Read, mut output: impl Write) -> io::Result<Report> {
    let report = visit_lines(
        input,
        VisitLimits {
            max_block_bytes: usize::MAX,
            max_line_bytes: usize::MAX,
            max_dictionary_bytes: usize::MAX,
            io_buffer_bytes: 1 << 20,
            zstd_window_log_max: 31,
        },
        |line| output.write_all(line),
    )?;
    output.flush()?;
    Ok(report)
}

fn fmt_u64(mut v: u64, buf: &mut [u8; 20]) -> &[u8] {
    let mut i = buf.len();
    loop {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            return &buf[i..];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: Params = Params {
        zstd_level: 3,
        block_raw_bytes: 64,
        window_log: 20,
    };

    fn roundtrip(data: &[u8]) {
        let mut enc = Vec::new();
        let r = encode(data, &mut enc, P).unwrap();
        let mut dec = Vec::new();
        let d = decode(&enc[..], &mut dec).unwrap();
        assert_eq!(dec, data, "ida y vuelta");
        assert_eq!(r.raw_sha256, d.raw_sha256);
        assert_eq!(r.lines, d.lines);
        let mut rows = Vec::new();
        let visited = visit_lines(&enc[..], test_limits(), |line| {
            rows.push(line.to_vec());
            Ok(())
        })
        .unwrap();
        assert_eq!(rows.concat(), data);
        assert_eq!(
            rows,
            data.split_inclusive(|&b| b == b'\n')
                .map(|line| line.to_vec())
                .collect::<Vec<_>>()
        );
        assert_report_eq(&r, &visited);
        assert_report_eq(&d, &visited);
    }

    #[test]
    fn ida_y_vuelta_exacta_en_bordes() {
        roundtrip(b"");
        roundtrip(b"\n");
        roundtrip(b"\n\n\n");
        roundtrip(b"sin salto final");
        roundtrip(b"raw \x00 sin salto final");
        roundtrip(b"0\n00\n000\n0045\n10\n");
        roundtrip(b"9999999999999999999 18446744073709551615 123456789012345678901234\n");
        roundtrip(b"bytes \x00 nulos 12\n");
        roundtrip("acentos \u{f1} 12,5 \u{20ac} 007\n".as_bytes());
        let row = r#"{"connection":1,"depth_generation":null,"ingest_seq":7,"kind":"frame","payload":"{\"stream\":\"btcusdt@depth@100ms\",\"data\":{\"e\":\"depthUpdate\",\"E\":1794690000123,\"b\":[[\"0.00100\",\"12.300\"]]}}","recv_mono_ns":5,"recv_utc_ms":1794690000125,"route":"public","seq":3,"stream":"btcusdt@depth@100ms","symbol":"BTCUSDT"}"#;
        let mut many = Vec::new();
        for i in 0..50u64 {
            many.extend_from_slice(
                row.replace("0.00100", &format!("0.0{:04}", 100 - i))
                    .replace("1794690000125", &(1794690000125 - i * 7).to_string())
                    .as_bytes(),
            );
            many.push(b'\n');
        }
        roundtrip(&many);
        many.pop();
        roundtrip(&many);
    }

    #[test]
    fn decodificar_detecta_bytes_alterados() {
        let data = b"{\"stream\":\"a\",\"x\":1}\n{\"stream\":\"a\",\"x\":2}\n".repeat(100);
        let mut enc = Vec::new();
        encode(&data[..], &mut enc, P).unwrap();
        let mid = enc.len() / 2;
        enc[mid] ^= 0x55;
        assert!(decode(&enc[..], io::sink()).is_err());
    }

    fn test_limits() -> VisitLimits {
        VisitLimits {
            max_block_bytes: MAX_SECTION,
            max_line_bytes: MAX_SECTION,
            max_dictionary_bytes: MAX_SECTION,
            io_buffer_bytes: P.block_raw_bytes,
            zstd_window_log_max: P.window_log,
        }
    }

    fn assert_report_eq(a: &Report, b: &Report) {
        assert_eq!(a.lines, b.lines);
        assert_eq!(a.raw_bytes, b.raw_bytes);
        assert_eq!(a.raw_sha256, b.raw_sha256);
        assert_eq!(a.skeletons, b.skeletons);
        assert_eq!(a.contexts, b.contexts);
        assert_eq!(a.raw_fallback_lines, b.raw_fallback_lines);
    }

    fn encoded(data: &[u8]) -> Vec<u8> {
        let mut enc = Vec::new();
        encode(data, &mut enc, P).unwrap();
        enc
    }

    // Solo los fixtures de test materializan el payload para modificar una
    // seccion concreta y recomprimirla con una trama zstd valida.
    fn payload(enc: &[u8]) -> Vec<u8> {
        let mut data = Vec::new();
        zstd::stream::Decoder::new(&enc[MAGIC.len()..])
            .unwrap()
            .read_to_end(&mut data)
            .unwrap();
        data
    }

    fn repack(data: &[u8]) -> Vec<u8> {
        let mut enc = MAGIC.to_vec();
        let mut z = zstd::stream::Encoder::new(&mut enc, P.zstd_level).unwrap();
        z.window_log(P.window_log).unwrap();
        z.include_checksum(true).unwrap();
        z.write_all(data).unwrap();
        z.finish().unwrap();
        enc
    }

    fn tail_at(data: &[u8]) -> usize {
        let mut pos = 0usize;
        loop {
            let tag = data[pos];
            pos += 1;
            if tag == 0 {
                return pos;
            }
            assert_eq!(tag, 1);
            get_varint(data, &mut pos).unwrap(); // lineas
            for _ in 0..2 {
                // ids y raw
                let len = get_usize(data, &mut pos).unwrap();
                take(data, &mut pos, len).unwrap();
            }
            get_varint(data, &mut pos).unwrap(); // esqueletos nuevos
            let len = get_usize(data, &mut pos).unwrap();
            take(data, &mut pos, len).unwrap();
            let used = get_usize(data, &mut pos).unwrap();
            for _ in 0..used {
                get_varint(data, &mut pos).unwrap();
                let len = get_usize(data, &mut pos).unwrap();
                take(data, &mut pos, len).unwrap();
            }
        }
    }

    fn assert_both_reject(enc: &[u8], message: &str) {
        let visitor_error = visit_lines(enc, test_limits(), |_| Ok(())).err().unwrap();
        let decode_error = decode(enc, io::sink()).err().unwrap();
        assert_eq!(visitor_error.kind(), decode_error.kind());
        assert_eq!(visitor_error.to_string(), decode_error.to_string());
        assert!(
            visitor_error.to_string().contains(message),
            "{visitor_error}"
        );
    }

    #[test]
    fn lector_ida_y_vuelta_por_fila_y_bloques() {
        let rows: &[&[u8]] = &[
            b"\n",
            b"\r\n",
            b"{\"symbol\":\"BTCUSDT\",\"n\":001}\n",
            b"binario \x00 12\n",
            "acentos \u{f1} 007\n".as_bytes(),
            b"ultima 12",
        ];
        let data = rows.concat();
        let mut enc = Vec::new();
        let p = Params {
            block_raw_bytes: 1,
            ..P
        }; // fuerza fronteras de bloque
        encode(&data[..], &mut enc, p).unwrap();
        let mut got = Vec::new();
        let report = visit_lines(&enc[..], test_limits(), |line| {
            got.push(line.to_vec());
            Ok(())
        })
        .unwrap();
        assert_eq!(got, rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>());
        let mut decoded = Vec::new();
        let legacy = decode(&enc[..], &mut decoded).unwrap();
        assert_eq!(decoded, data);
        assert_report_eq(&report, &legacy);
    }

    #[test]
    fn lector_rechaza_todos_los_prefijos_truncados() {
        for data in [&b""[..], &b"primera 001\nsegunda 002\n"[..]] {
            let enc = encoded(data);
            for end in 0..enc.len() {
                assert!(
                    visit_lines(&enc[..end], test_limits(), |_| Ok(())).is_err(),
                    "prefijo aceptado: {end}/{}",
                    enc.len()
                );
                assert!(decode(&enc[..end], io::sink()).is_err());
            }
        }
    }

    #[test]
    fn lector_rechaza_sha_distinto_con_zstd_valido() {
        let mut data = payload(&encoded(b"una linea 12\n"));
        let mut pos = tail_at(&data);
        get_varint(&data, &mut pos).unwrap();
        get_varint(&data, &mut pos).unwrap();
        data[pos] ^= 1;
        assert_both_reject(&repack(&data), "la cola no coincide");
    }

    #[test]
    fn lector_rechaza_conteos_distintos_con_zstd_valido() {
        let original = payload(&encoded(b"una linea 12\n"));
        let lines_at = tail_at(&original);
        let mut bytes_at = lines_at;
        get_varint(&original, &mut bytes_at).unwrap();
        for pos in [lines_at, bytes_at] {
            let mut data = original.clone();
            data[pos] ^= 1;
            assert_both_reject(&repack(&data), "la cola no coincide");
        }
    }

    #[test]
    fn lector_rechaza_bloque_alterado_con_zstd_valido() {
        let mut data = payload(&encoded(b"una linea 12\n"));
        let mut pos = 1; // tras la etiqueta de bloque
        get_varint(&data, &mut pos).unwrap();
        let ids_len = get_usize(&data, &mut pos).unwrap();
        assert_eq!(ids_len, 1);
        data[pos] = 0; // fuerza fallback, pero el bloque no tiene datos raw
        assert_both_reject(&repack(&data), "varint truncado");
    }

    #[test]
    fn lector_filtrar_no_omite_el_hash() {
        let data = b"{\"symbol\":\"BTCUSDT\",\"n\":1}\n{\"symbol\":\"ETHUSDT\",\"n\":2}\n";
        let enc = encoded(data);
        let mut kept = Vec::new();
        let report = visit_lines(&enc[..], test_limits(), |line| {
            if find(line, b"BTCUSDT").is_some() {
                kept.extend_from_slice(line);
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(kept, b"{\"symbol\":\"BTCUSDT\",\"n\":1}\n");
        assert_eq!(report.raw_bytes, data.len() as u64);
        assert_eq!(report.raw_sha256, format!("{:x}", Sha256::digest(data)));
        let mut altered = payload(&enc);
        let at = find(&altered, b"ETHUSDT").unwrap();
        altered[at] = b'X';
        assert_both_reject(&repack(&altered), "la cola no coincide");
    }

    #[test]
    fn lector_rechaza_partial_antes_de_abrir() {
        for name in [
            "no_existe_lector.tgll.partial",
            "no_existe_lector.tgll.PARTIAL",
        ] {
            let error = visit_file(name, test_limits(), |_| panic!("no visitar partial"))
                .err()
                .unwrap();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
            assert!(error.to_string().contains("archivo .partial rechazado"));
        }
    }

    #[test]
    fn lector_propaga_error_del_visitante() {
        let enc = encoded(b"primera\nsegunda\n");
        let mut calls = 0;
        let error = visit_lines(&enc[..], test_limits(), |_| {
            calls += 1;
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "fallo del destino",
            ))
        })
        .err()
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(error.to_string(), "fallo del destino");
    }

    #[test]
    fn lector_limita_bloque_y_linea() {
        let raw = b"abc\n";
        let enc = encoded(raw);
        let mut limits = test_limits();
        limits.max_block_bytes = 0;
        let error = visit_lines(&enc[..], limits, |_| Ok(())).err().unwrap();
        assert!(
            error
                .to_string()
                .contains("bloque mayor que el presupuesto")
        );
        let mut limits = test_limits();
        limits.max_line_bytes = raw.len() - 1;
        let error = visit_lines(&enc[..], limits, |_| Ok(())).err().unwrap();
        assert!(error.to_string().contains("linea mayor que el presupuesto"));
    }

    #[test]
    fn lector_exige_presupuesto_de_buffer_y_ventana() {
        let mut limits = test_limits();
        limits.io_buffer_bytes = 0;
        let error = visit_lines(io::empty(), limits, |_| Ok(())).err().unwrap();
        assert!(error.to_string().contains("buffer de lectura vacio"));
        let mut limits = test_limits();
        limits.zstd_window_log_max = 0;
        let error = visit_lines(io::empty(), limits, |_| Ok(())).err().unwrap();
        assert!(error.to_string().contains("ventana zstd sin presupuesto"));
    }

    #[test]
    fn lector_limita_diccionario_acumulado_entre_bloques() {
        let first = b"primera 1\n";
        let mut skel = Vec::new();
        let mut nums = Vec::new();
        assert!(tokenize(&first[..first.len() - 1], &mut skel, &mut nums));
        let mut limits = test_limits();
        limits.max_dictionary_bytes = dictionary_charge(&skel).unwrap();
        let data = [first.as_slice(), b"segunda 2\n".as_slice()].concat();
        let mut enc = Vec::new();
        encode(
            &data[..],
            &mut enc,
            Params {
                block_raw_bytes: 1,
                ..P
            },
        )
        .unwrap();
        let error = visit_lines(&enc[..], limits, |_| Ok(())).err().unwrap();
        assert!(
            error
                .to_string()
                .contains("diccionario mayor que el presupuesto")
        );
    }

    #[test]
    fn lector_rechaza_datos_tras_cola_y_marca_final_invalida() {
        let original = payload(&encoded(b"abc\n"));
        let mut extra = original.clone();
        extra.push(0);
        assert_both_reject(&repack(&extra), "datos despues de la cola");
        let mut invalid = original;
        *invalid.last_mut().unwrap() = 2;
        assert_both_reject(&repack(&invalid), "marca de salto final invalida");
    }

    fn assert_varint_rejected(bytes: &[u8], message: &str) {
        let from_slice = get_varint(bytes, &mut 0).unwrap_err();
        let from_read = read_varint(&mut &bytes[..]).unwrap_err();
        assert_eq!(from_slice.kind(), io::ErrorKind::InvalidData);
        assert_eq!(from_slice.kind(), from_read.kind());
        assert_eq!(from_slice.to_string(), from_read.to_string());
        assert!(from_slice.to_string().contains(message), "{from_slice}");
    }

    // Representacion matematica de value + 2^128: los bits bajos coinciden,
    // pero el grupo final contiene un bit que nunca cabe en u128.
    fn overflowing_varint(mut value: u128) -> Vec<u8> {
        let groups = u128::BITS.div_ceil(7);
        let mut bytes = Vec::new();
        for group in 0..groups {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if group + 1 < groups {
                byte |= 0x80;
            } else {
                byte |= 1u8 << (u128::BITS % 7);
            }
            bytes.push(byte);
        }
        assert_eq!(value, 0);
        bytes
    }

    #[test]
    fn varints_aceptan_bordes_canonicos_hasta_u128_max() {
        let mut values = vec![0, u128::MAX];
        for bit in 0..u128::BITS {
            values.push(1u128 << bit);
            values.push((1u128 << bit) - 1);
        }
        for value in values {
            let mut bytes = Vec::new();
            put_varint(&mut bytes, value);
            let mut pos = 0;
            assert_eq!(get_varint(&bytes, &mut pos).unwrap(), value);
            assert_eq!(pos, bytes.len());
            let mut reader = io::Cursor::new(&bytes);
            assert_eq!(read_varint(&mut reader).unwrap(), value);
            assert_eq!(reader.position(), bytes.len() as u64);
        }
    }

    #[test]
    fn varints_rechazan_bits_altos_fuera_de_u128() {
        for value in [0, 1, u128::from(u64::MAX)] {
            assert_varint_rejected(&overflowing_varint(value), "varint fuera de rango");
        }
    }

    #[test]
    fn varints_rechazan_codificaciones_demasiado_largas() {
        let mut bytes = vec![0x80; u128::BITS.div_ceil(7) as usize];
        bytes.push(0);
        assert_varint_rejected(&bytes, "varint demasiado largo");
        // No hace falta leer otro byte para saber que el ultimo grupo continua.
        bytes.pop();
        assert_varint_rejected(&bytes, "varint demasiado largo");
    }

    #[test]
    fn varints_rechazan_relleno_no_canonico_y_truncado() {
        for bytes in [&b"\x80\x00"[..], &b"\x81\x00"[..]] {
            assert_varint_rejected(bytes, "varint no canonico");
        }
        for bytes in [&b""[..], &b"\x80"[..]] {
            let from_slice = get_varint(bytes, &mut 0).unwrap_err();
            assert!(from_slice.to_string().contains("varint truncado"));
            assert_eq!(
                read_varint(&mut &bytes[..]).unwrap_err().kind(),
                io::ErrorKind::UnexpectedEof
            );
        }
    }

    fn serialize_block(mut block: Block) -> Vec<u8> {
        let mut data = Vec::new();
        block.write_to(&mut data).unwrap();
        data
    }

    fn raw_block(bodies: &[&[u8]]) -> Vec<u8> {
        let mut block = Block::new();
        block.lines = bodies.len() as u64;
        for body in bodies {
            put_varint(&mut block.ids, 0);
            put_varint(&mut block.raw, body.len() as u128);
            block.raw.extend_from_slice(body);
        }
        serialize_block(block)
    }

    fn literal_block(bodies: &[&[u8]]) -> Vec<u8> {
        let mut block = Block::new();
        block.lines = bodies.len() as u64;
        block.new_skel_count = bodies.len() as u64;
        for (id, body) in bodies.iter().enumerate() {
            put_varint(&mut block.ids, id as u128 + 1);
            put_varint(&mut block.new_skel, body.len() as u128);
            block.new_skel.extend_from_slice(body);
        }
        serialize_block(block)
    }

    // Conserva exactamente conteos y SHA de encode(original). La marca final
    // puede cambiar para construir la particion falsa de la revision N1.
    fn fixture_with_blocks(original: &[u8], blocks: &[Vec<u8>], final_lf: bool) -> Vec<u8> {
        let original_payload = payload(&encoded(original));
        let mut data = blocks.concat();
        data.extend_from_slice(&original_payload[tail_at(&original_payload) - 1..]);
        *data.last_mut().unwrap() = u8::from(final_lf);
        repack(&data)
    }

    #[test]
    fn lector_rechaza_particion_raw_con_lf_y_cola_original() {
        let blocks = [raw_block(&[b"a\nb", b""])];
        let enc = fixture_with_blocks(b"a\nb\n", &blocks, false);
        let mut callbacks = 0;
        let error = visit_lines(&enc[..], test_limits(), |_| {
            callbacks += 1;
            Ok(())
        })
        .err()
        .unwrap();
        assert_eq!(callbacks, 0);
        assert!(error.to_string().contains("LF dentro del cuerpo raw"));
        assert_both_reject(&enc, "LF dentro del cuerpo raw");
    }

    #[test]
    fn lector_rechaza_particion_de_esqueleto_con_lf_y_cola_original() {
        let blocks = [literal_block(&[b"a\nb", b""])];
        let enc = fixture_with_blocks(b"a\nb\n", &blocks, false);
        assert_both_reject(&enc, "LF dentro del esqueleto");
    }

    #[test]
    fn lector_rechaza_ultima_fila_vacia_sin_lf_con_sha_valido() {
        for block in [raw_block(&[b""]), literal_block(&[b""])] {
            let mut data = block;
            data.push(0); // cola
            put_varint(&mut data, 1); // una fila declarada, pero ningun byte
            put_varint(&mut data, 0);
            data.extend_from_slice(&Sha256::digest(b""));
            data.push(0); // sin LF final
            let enc = repack(&data);
            let error = visit_lines(&enc[..], test_limits(), |_| {
                panic!("no se debe emitir una fila vacia sin terminador")
            })
            .err()
            .unwrap();
            assert!(error.to_string().contains("ultima linea vacia sin LF"));
            assert_both_reject(&enc, "ultima linea vacia sin LF");
        }
    }

    #[test]
    fn lector_rechaza_varints_desbordados_en_conteos_y_longitudes() {
        let original = payload(&encoded(b"a\x00\n"));
        let mut pos = 1; // tras la etiqueta de bloque
        let count_at = pos;
        get_varint(&original, &mut pos).unwrap();
        let ids_len_at = pos;
        let ids_len = get_usize(&original, &mut pos).unwrap();
        take(&original, &mut pos, ids_len).unwrap();
        let raw_len_at = pos;
        let total_at = tail_at(&original);
        let mut bytes_at = total_at;
        get_varint(&original, &mut bytes_at).unwrap();
        for start in [count_at, ids_len_at, raw_len_at, total_at, bytes_at] {
            let mut end = start;
            let value = get_varint(&original, &mut end).unwrap();
            let mut data = original.clone();
            data.splice(start..end, overflowing_varint(value));
            assert_both_reject(&repack(&data), "varint fuera de rango");
        }
    }

    #[test]
    fn lector_rechaza_varints_desbordados_en_ids_raw_y_esqueletos() {
        let body = b"a";
        let mut blocks = Vec::new();
        let mut id = Block::new();
        id.lines = 1;
        id.ids = overflowing_varint(0);
        put_varint(&mut id.raw, body.len() as u128);
        id.raw.extend_from_slice(body);
        blocks.push(serialize_block(id));

        let mut raw = Block::new();
        raw.lines = 1;
        put_varint(&mut raw.ids, 0);
        raw.raw = overflowing_varint(body.len() as u128);
        raw.raw.extend_from_slice(body);
        blocks.push(serialize_block(raw));

        let mut skel = Block::new();
        skel.lines = 1;
        put_varint(&mut skel.ids, 1);
        skel.new_skel_count = 1;
        skel.new_skel = overflowing_varint(body.len() as u128);
        skel.new_skel.extend_from_slice(body);
        blocks.push(serialize_block(skel));
        for block in blocks {
            let enc = fixture_with_blocks(b"a\n", &[block], true);
            assert_both_reject(&enc, "varint fuera de rango");
        }
    }

    #[test]
    fn lector_rechaza_varint_desbordado_en_delta_de_contexto() {
        let mut block = Block::new();
        block.lines = 1;
        put_varint(&mut block.ids, 1);
        block.new_skel_count = 1;
        let skel = b"x\x00";
        put_varint(&mut block.new_skel, skel.len() as u128);
        block.new_skel.extend_from_slice(skel);
        block.ctx.push(overflowing_varint(0));
        let enc = fixture_with_blocks(b"x0\n", &[serialize_block(block)], true);
        assert_both_reject(&enc, "varint fuera de rango");
    }

    #[test]
    fn lector_rechaza_bloques_desordenados_con_zstd_y_cola_original() {
        // Raw independientes: al permutarlos no se invalidan referencias ni
        // deltas. Solo cambia el orden de bytes frente al SHA de la cola.
        let first = b"primera\x00\n";
        let second = b"segunda\x00\n";
        let original = [first.as_slice(), second.as_slice()].concat();
        let blocks = [
            raw_block(&[&first[..first.len() - 1]]),
            raw_block(&[&second[..second.len() - 1]]),
        ];
        assert_ne!(blocks[0], blocks[1]);
        let valid = fixture_with_blocks(&original, &blocks, true);
        let mut decoded = Vec::new();
        decode(&valid[..], &mut decoded).unwrap();
        assert_eq!(decoded, original);

        let permuted = [blocks[1].clone(), blocks[0].clone()];
        let enc = fixture_with_blocks(&original, &permuted, true);
        let mut provisional = Vec::new();
        let error = visit_lines(&enc[..], test_limits(), |line| {
            provisional.extend_from_slice(line);
            Ok(())
        })
        .err()
        .unwrap();
        assert_eq!(provisional, [second.as_slice(), first.as_slice()].concat());
        assert_ne!(provisional, original);
        assert!(error.to_string().contains("la cola no coincide"));
        assert_both_reject(&enc, "la cola no coincide");
    }

    #[test]
    fn lector_constructor_usa_limites_canonicos_de_captura() {
        let measured = test_limits();
        let limits = VisitLimits::capture(measured.max_line_bytes, measured.max_dictionary_bytes);
        assert_eq!(limits.max_block_bytes, 64 << 20); // src/main.rs:92
        assert_eq!(limits.zstd_window_log_max, 27); // src/main.rs:96
        assert_eq!(1usize << limits.zstd_window_log_max, 128 << 20);
        assert_eq!(limits.io_buffer_bytes, 1 << 20); // encode/decode
        assert_eq!(limits.max_line_bytes, measured.max_line_bytes);
        assert_eq!(limits.max_dictionary_bytes, measured.max_dictionary_bytes);
    }

    #[test]
    fn lector_limite_de_bloque_mide_secciones_decodificadas() {
        let original = b"\n";
        let enc = encoded(original);
        let data = payload(&enc);
        let mut pos = 1;
        get_varint(&data, &mut pos).unwrap(); // lineas
        let mut section_bytes = 0;
        for _ in 0..2 {
            // ids y raw
            let len = get_usize(&data, &mut pos).unwrap();
            section_bytes += len;
            take(&data, &mut pos, len).unwrap();
        }
        get_varint(&data, &mut pos).unwrap(); // esqueletos nuevos
        let len = get_usize(&data, &mut pos).unwrap();
        section_bytes += len;
        take(&data, &mut pos, len).unwrap();
        let used = get_usize(&data, &mut pos).unwrap();
        for _ in 0..used {
            get_varint(&data, &mut pos).unwrap();
            let len = get_usize(&data, &mut pos).unwrap();
            section_bytes += len;
            take(&data, &mut pos, len).unwrap();
        }
        assert!(section_bytes > original.len()); // no es una cota del crudo
        let mut limits = test_limits();
        limits.max_block_bytes = section_bytes;
        let mut rows = Vec::new();
        visit_lines(&enc[..], limits, |line| {
            rows.extend_from_slice(line);
            Ok(())
        })
        .unwrap();
        assert_eq!(rows, original);
        limits.max_block_bytes = section_bytes - 1;
        let error = visit_lines(&enc[..], limits, |_| Ok(())).err().unwrap();
        assert!(
            error
                .to_string()
                .contains("bloque mayor que el presupuesto")
        );
    }
}
