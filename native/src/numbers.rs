const MORE: u8 = 0x80;
const LOW: u8 = 0x7F;
const BITS: u32 = 7;
const MOST_BYTES: usize = 9;

pub fn written(out: &mut Vec<u8>, number: i64) -> Result<(), String> {
    let mut left = u64::try_from(number).map_err(|_| format!("{number} is below zero: no number a list keeps"))?;
    while left > u64::from(LOW) {
        out.push(left.to_le_bytes()[0] & LOW | MORE);
        left >>= BITS;
    }
    out.push(left.to_le_bytes()[0]);
    Ok(())
}

pub fn read(data: &[u8], found: &mut Vec<i64>) -> Result<(), String> {
    if data.last().is_some_and(|&byte| byte >= MORE) {
        return Err("a number runs past the end of its list".to_owned());
    }
    let (mut number, mut size) = (0u64, 0usize);
    for &byte in data {
        if size == MOST_BYTES {
            return Err("a number takes more than nine bytes".to_owned());
        }
        number |= u64::from(byte & LOW) << (BITS * u32::try_from(size).unwrap_or(u32::MAX));
        size += 1;
        if byte < MORE {
            found.push(i64::try_from(number).unwrap_or(i64::MAX));
            (number, size) = (0, 0);
        }
    }
    Ok(())
}

pub fn narrow(number: usize) -> Result<u32, String> {
    u32::try_from(number).map_err(|_| "a segment holds more than it numbers".to_owned())
}

pub fn little_endian(numbers: &[i64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size_of_val(numbers));
    for number in numbers {
        bytes.extend_from_slice(&number.to_le_bytes());
    }
    bytes
}
