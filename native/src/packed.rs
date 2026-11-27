use std::borrow::Cow;
use std::fmt::Display;

use flate2::{Compress, Compression, Decompress, FlushCompress, FlushDecompress, Status};

pub const RAW: u8 = 0;
pub const PACKED: u8 = 1;
const PACKED_FROM: usize = 64;
const LEVEL: u32 = 6;
const ROOM: usize = 64;
const NO_PACKING: &str = "a list does not pack";
const NO_UNPACKING: &str = "a packed list does not unpack";
const AFTER_END: &str = "a packed list holds bytes after its end";

fn taken(total: u64) -> usize {
    usize::try_from(total).unwrap_or(usize::MAX)
}

fn told<T, E: Display>(found: Result<T, E>, refusal: &str) -> Result<T, String> {
    found.map_err(|error| format!("{refusal}: {error}"))
}

pub fn deflated(body: &[u8], lead: Option<u8>) -> Result<Vec<u8>, String> {
    let mut packing = Compress::new(Compression::new(LEVEL), true);
    let mut packed = Vec::with_capacity(body.len() / 2 + ROOM);
    packed.extend(lead);
    loop {
        let rest = &body[taken(packing.total_in())..];
        let step = packing.compress_vec(rest, &mut packed, FlushCompress::Finish);
        let status = told(step, NO_PACKING)?;
        if status == Status::StreamEnd {
            return Ok(packed);
        }
        packed.reserve(packed.capacity().max(ROOM));
    }
}

pub fn wrapped(body: &[u8]) -> Result<Vec<u8>, String> {
    if body.len() >= PACKED_FROM {
        return deflated(body, Some(PACKED));
    }
    let mut raw = Vec::with_capacity(body.len() + 1);
    raw.push(RAW);
    raw.extend_from_slice(body);
    Ok(raw)
}

fn too_large(most: usize) -> String {
    format!("a packed list unpacks to more than {most} bytes, or ends early")
}

fn inflated(data: &[u8], most: usize) -> Result<Vec<u8>, String> {
    let mut unpacking = Decompress::new(true);
    let mut found = Vec::with_capacity((data.len() * 4).max(ROOM).min(most + 1));
    loop {
        let (read, held) = (taken(unpacking.total_in()), found.len());
        let step = unpacking.decompress_vec(&data[read..], &mut found, FlushDecompress::None);
        let status = told(step, NO_UNPACKING)?;
        if found.len() > most {
            return Err(too_large(most));
        }
        if status == Status::StreamEnd && taken(unpacking.total_in()) != data.len() {
            return Err(AFTER_END.to_owned());
        }
        if status == Status::StreamEnd {
            return Ok(found);
        }
        let stalled = taken(unpacking.total_in()) == read && found.len() == held;
        if stalled && found.len() < found.capacity() {
            return Err(too_large(most));
        }
        if found.len() == found.capacity() {
            found.reserve_exact(found.capacity().min(most + 1 - found.len()));
        }
    }
}

pub fn inflated_into(data: &[u8], found: &mut [u8]) -> Result<(), String> {
    let mut unpacking = Decompress::new(true);
    let step = unpacking.decompress(data, found, FlushDecompress::Finish);
    let status = told(step, NO_UNPACKING)?;
    if status != Status::StreamEnd || taken(unpacking.total_out()) != found.len() {
        return Err(format!("a packed list unpacks to other than {} bytes", found.len()));
    }
    if taken(unpacking.total_in()) != data.len() {
        return Err(AFTER_END.to_owned());
    }
    Ok(())
}

pub fn unwrapped(data: &[u8], most: usize) -> Result<Cow<'_, [u8]>, String> {
    match data.split_first() {
        Some((&RAW, body)) if body.len() <= most => Ok(Cow::Borrowed(body)),
        Some((&RAW, _)) => Err(format!(
            "a list holds more than {most} bytes: more than its counts allow"
        )),
        Some((&PACKED, body)) => inflated(body, most).map(Cow::Owned),
        _ => Err("a list starts with an unknown flag byte".to_owned()),
    }
}
