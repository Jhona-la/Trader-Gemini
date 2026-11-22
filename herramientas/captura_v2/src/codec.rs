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

fn get_varint(buf: &[u8], pos: &mut usize) -> io::Result<u128> {
    let mut v: u128 = 0;
    let mut shift = 0;
    loop {
        let byte = *buf.get(*pos).ok_or_else(|| bad("varint truncado"))?;
        *pos += 1;
        if shift >= 128 {
            return Err(bad("varint demasiado largo"));
        }
        v |= u128::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(v);
        }
        shift += 7;
    }
}

fn get_usize(buf: &[u8], pos: &mut usize) -> io::Result<usize> {
    usize::try_from(get_varint(buf, pos)?).map_err(|_| bad("longitud fuera de rango"))
}

fn take<'a>(buf: &'a [u8], pos: &mut usize, len: usize) -> io::Result<&'a [u8]> {
    let end = pos.checked_add(len).ok_or_else(|| bad("longitud desborda"))?;
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
    let mut v: u128 = 0;
    let mut shift = 0;
    loop {
        let mut b = [0u8];
        r.read_exact(&mut b)?;
        if shift >= 128 {
            return Err(bad("varint demasiado largo"));
        }
        v |= u128::from(b[0] & 0x7f) << shift;
        if b[0] & 0x80 == 0 {
            return Ok(v);
        }
        shift += 7;
    }
}

fn read_len(r: &mut impl Read) -> io::Result<usize> {
    usize::try_from(read_varint(r)?).map_err(|_| bad("longitud fuera de rango"))
}

/// Decodifica un archivo TGLL y escribe el JSONL exacto en `output`. Comprueba al final que
/// lineas, bytes y SHA-256 coinciden con la cola del archivo.
pub fn decode(input: impl Read, mut output: impl Write) -> io::Result<Report> {
    let mut input = io::BufReader::with_capacity(1 << 20, input);
    let mut magic = [0u8; 8];
    input.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(bad("cabecera desconocida"));
    }
    let mut dec = zstd::stream::Decoder::with_buffer(input)?;
    dec.window_log_max(31)?;
    let mut z = io::BufReader::with_capacity(1 << 20, dec);
    let mut hasher = Sha256::new();
    let mut skels: Vec<(Vec<u8>, Vec<u32>)> = Vec::new();
    let mut classes = HashMap::new();
    let mut contexts = HashMap::new();
    let mut prev: Vec<u64> = Vec::new();
    let (mut lines, mut raw_bytes, mut fallback) = (0u64, 0u64, 0u64);
    let mut out = Vec::with_capacity(1 << 16);
    let mut digits = [0u8; 20];
    let mut pending_newline = false;
    loop {
        let mut tag = [0u8];
        z.read_exact(&mut tag)?;
        if tag[0] == 0 {
            break;
        }
        if tag[0] != 1 {
            return Err(bad("etiqueta de bloque desconocida"));
        }
        let nlines = read_varint(&mut z)? as u64;
        let len = read_len(&mut z)?;
        let ids = read_exact_vec(&mut z, len)?;
        let len = read_len(&mut z)?;
        let raw = read_exact_vec(&mut z, len)?;
        let count = read_varint(&mut z)? as u64;
        let len = read_len(&mut z)?;
        let new_skel = read_exact_vec(&mut z, len)?;
        let mut pos = 0;
        for _ in 0..count {
            let l = get_usize(&new_skel, &mut pos)?;
            let s = take(&new_skel, &mut pos, l)?.to_vec();
            let c = skeleton_contexts(&s, &mut classes, &mut contexts, &mut prev);
            skels.push((s, c));
        }
        let used = read_len(&mut z)?;
        let mut ctx: HashMap<usize, (Vec<u8>, usize)> = HashMap::with_capacity(used);
        for _ in 0..used {
            let id = read_len(&mut z)?;
            let len = read_len(&mut z)?;
            ctx.insert(id, (read_exact_vec(&mut z, len)?, 0));
        }
        let (mut ip, mut rp) = (0usize, 0usize);
        for _ in 0..nlines {
            out.clear();
            if pending_newline {
                out.push(b'\n');
            }
            let id = get_usize(&ids, &mut ip)?;
            if id == 0 {
                let l = get_usize(&raw, &mut rp)?;
                out.extend_from_slice(take(&raw, &mut rp, l)?);
                fallback += 1;
            } else {
                let (s, c) = skels
                    .get(id - 1)
                    .ok_or_else(|| bad("esqueleto inexistente"))?;
                let mut k = 0;
                for &b in s {
                    if b != PLACEHOLDER {
                        out.push(b);
                        continue;
                    }
                    let cid = *c.get(k).ok_or_else(|| bad("contexto inexistente"))? as usize;
                    k += 1;
                    let (buf, cur) = ctx
                        .get_mut(&cid)
                        .ok_or_else(|| bad("contexto sin datos"))?;
                    let w = get_varint(buf, cur)?;
                    let d = u64::try_from(w >> 1).map_err(|_| bad("delta fuera de rango"))?;
                    let lz = if w & 1 == 1 {
                        get_usize(buf, cur)?
                    } else {
                        0
                    };
                    let v = prev[cid].wrapping_add(unzigzag(d) as u64);
                    prev[cid] = v;
                    out.extend(std::iter::repeat_n(b'0', lz));
                    out.extend_from_slice(fmt_u64(v, &mut digits));
                }
            }
            pending_newline = true;
            lines += 1;
            hasher.update(&out);
            raw_bytes += out.len() as u64;
            output.write_all(&out)?;
        }
        if ip != ids.len() || rp != raw.len() || ctx.values().any(|(b, c)| *c != b.len()) {
            return Err(bad("bloque con datos sobrantes"));
        }
    }
    let total = read_varint(&mut z)? as u64;
    let total_bytes = read_varint(&mut z)? as u64;
    let sha = read_exact_vec(&mut z, 32)?;
    let mut fnl = [0u8];
    z.read_exact(&mut fnl)?;
    if pending_newline && fnl[0] == 1 {
        hasher.update(b"\n");
        raw_bytes += 1;
        output.write_all(b"\n")?;
    }
    output.flush()?;
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
    }

    #[test]
    fn ida_y_vuelta_exacta_en_bordes() {
        roundtrip(b"");
        roundtrip(b"\n");
        roundtrip(b"\n\n\n");
        roundtrip(b"sin salto final");
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
}
