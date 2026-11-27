use crate::numbers::{little_endian, narrow, read, written};
use crate::packed::{PACKED, RAW, deflated, unwrapped, wrapped};
use crate::tests::support::refused;

const MOST_INFLATED: usize = 64 << 20;

#[test]
fn numbers_come_back_from_their_bytes_a_byte_for_every_seven_bits() -> Result<(), String> {
    let numbers = [0, 127, 128, (1 << 14) - 1, 1 << 14, i64::MAX];
    let mut coded = Vec::new();
    for number in numbers {
        written(&mut coded, number)?;
    }
    assert_eq!(coded.len(), 1 + 1 + 2 + 2 + 3 + 9);
    let mut back = Vec::new();
    read(&coded, &mut back)?;
    assert_eq!(back, numbers);
    Ok(())
}

#[test]
fn a_number_below_zero_cut_off_or_of_more_than_nine_bytes_is_refused() {
    assert!(refused(written(&mut Vec::new(), -1), "below zero"));
    assert!(refused(read(&[0x80], &mut Vec::new()), "runs past the end"));
    let long: Vec<u8> = [0x81; 9].into_iter().chain([1]).collect();
    assert!(refused(read(&long, &mut Vec::new()), "more than nine bytes"));
}

#[test]
fn numbers_are_kept_little_endian_and_a_count_past_32_bits_is_refused() {
    assert_eq!(
        little_endian(&[1, 258]),
        [1, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(narrow(7), Ok(7));
    assert!(refused(narrow(usize::MAX), "more than it numbers"));
}

#[test]
fn a_short_list_is_kept_as_it_is_and_a_long_one_packed() -> Result<(), String> {
    let (short, long) = (wrapped(&[7; 63])?, wrapped(&[7; 64])?);
    assert_eq!((short.first(), short.len()), (Some(&RAW), 64));
    assert_eq!(long.first(), Some(&PACKED));
    assert!(long.len() < 30);
    assert_eq!(&*unwrapped(&short)?, [7; 63]);
    assert_eq!(&*unwrapped(&long)?, [7; 64]);
    Ok(())
}

#[test]
fn bytes_that_hardly_pack_come_back_as_they_were() -> Result<(), String> {
    let mut state = 7u64;
    let noise: Vec<u8> = (0..4096)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state.to_le_bytes()[7]
        })
        .collect();
    let packed = wrapped(&noise)?;
    assert!(packed.len() > noise.len() / 2 + 64);
    assert_eq!(&*unwrapped(&packed)?, noise);
    Ok(())
}

#[test]
fn bytes_that_are_no_list_are_refused_with_what_is_wrong() -> Result<(), String> {
    assert!(refused(unwrapped(&[]), "unknown flag byte"));
    assert!(refused(unwrapped(&[2, 1]), "unknown flag byte"));
    assert!(refused(unwrapped(&[PACKED, 9, 9, 9]), "does not unpack"));
    let packed = wrapped(&[7; 1000])?;
    assert!(refused(unwrapped(&packed[..packed.len() - 4]), "ends early"));
    Ok(())
}

#[test]
fn a_list_that_unpacks_to_more_than_any_list_holds_is_refused() -> Result<(), String> {
    let most = deflated(&vec![0; MOST_INFLATED], Some(PACKED))?;
    assert_eq!(unwrapped(&most)?.len(), MOST_INFLATED);
    let more = deflated(&vec![0; MOST_INFLATED + 1], Some(PACKED))?;
    assert!(refused(unwrapped(&more), "unpacks to more than"));
    Ok(())
}

#[test]
fn a_number_written_longer_than_it_is_is_refused() {
    assert!(refused(read(&[0x81, 0x00], &mut Vec::new()), "longer than it is"));
    assert!(refused(read(&[0x80, 0x80, 0x00], &mut Vec::new()), "longer than it is"));
    let mut found = Vec::new();
    assert_eq!(read(&[0x00, 0x81, 0x01], &mut found), Ok(()));
    assert_eq!(found, [0, 129]);
}

#[test]
fn bytes_after_the_end_of_a_packed_list_are_refused() -> Result<(), String> {
    let mut packed = wrapped(&[7; 1000])?;
    assert_eq!(unwrapped(&packed)?.len(), 1000);
    packed.extend_from_slice(b"more");
    assert!(refused(unwrapped(&packed), "bytes after its end"));
    Ok(())
}
