use crate::hamming::{apart, gathered, scanned};
use crate::tests::support::refused;

fn spread(each: u64) -> u64 {
    each.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn bytes_of(words: &[u64]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn distances_of(out: &[u8]) -> Vec<u16> {
    out.as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

#[test]
fn each_code_is_scanned_for_the_bits_it_differs_in_from_the_query_and_counted_by_distance() -> Result<(), String> {
    let query = [spread(1), spread(2)];
    let first: Vec<u64> = (0..40_000u64)
        .flat_map(|each| [spread(each), !spread(each + 7)])
        .collect();
    let second: Vec<u64> = (0..3u64).flat_map(|each| [spread(each), spread(each + 1)]).collect();
    let held = [bytes_of(&first), Vec::new(), bytes_of(&second)];
    let codes: Vec<(&[u8], usize)> = held.iter().map(|each| (each.as_slice(), 16)).collect();
    let mut out = vec![0u8; 2 * 40_003];
    let counted = scanned(&codes, &bytes_of(&query), &mut out)?;
    let wanted: Vec<u16> = first
        .as_chunks::<2>()
        .0
        .iter()
        .chain(second.as_chunks::<2>().0)
        .map(|row| u16::try_from(apart(row.iter().copied(), &query)).unwrap_or(u16::MAX))
        .collect();
    assert_eq!(distances_of(&out), wanted);
    let mut counts = vec![0u64; 129];
    for each in &wanted {
        counts[usize::from(*each)] += 1;
    }
    assert_eq!(counted, counts);
    assert_eq!(scanned(&[], &bytes_of(&query), &mut [])?, vec![0u64; 129]);
    assert_eq!(scanned(&[(&[], 0)], &bytes_of(&query), &mut [])?, vec![0u64; 129]);
    Ok(())
}

#[test]
fn a_scan_refuses_a_query_or_codes_of_part_words_and_room_for_other_than_each_code() {
    let query = bytes_of(&[1, 2]);
    let codes = bytes_of(&[3, 4, 5, 6]);
    for (held, asked, room) in [
        ((&codes[..], 16), &query[..7], 4),
        ((&codes[..], 16), &query[..0], 4),
        ((&codes[..24], 16), &query[..], 4),
        ((&codes[..], 8), &query[..], 4),
    ] {
        assert!(refused(scanned(&[held], asked, &mut vec![0u8; room]), "whole words"));
    }
    for room in [2, 6] {
        assert!(refused(
            scanned(&[(&codes, 16)], &query, &mut vec![0u8; room]),
            "two bytes"
        ));
    }
}

#[test]
fn rows_are_gathered_by_their_numbers_and_one_the_codes_do_not_hold_is_refused() -> Result<(), String> {
    let codes: Vec<u8> = (0..12u8).collect();
    assert_eq!(
        gathered(&codes, 4, &[2, 0, 2])?,
        [8, 9, 10, 11, 0, 1, 2, 3, 8, 9, 10, 11]
    );
    assert_eq!(gathered(&codes, 4, &[])?, Vec::<u8>::new());
    for at in [[3], [-1]] {
        assert!(refused(gathered(&codes, 4, &at), "do not hold"));
    }
    assert!(refused(gathered(&[], 0, &[0]), "do not hold"));
    Ok(())
}
