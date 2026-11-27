use crate::clusters::{majority, nearest};
use crate::tests::support::refused;

#[test]
fn each_code_is_given_its_nearest_centroid_and_the_first_of_those_equally_near() -> Result<(), String> {
    let centroids = [0b1111, 0, 0b1100, 0, 0b1111, u64::MAX];
    let held = [0b1110, 0, 0b1100, 1, 0b0011, 0, 0, u64::MAX];
    assert_eq!(nearest(&held, &centroids, 2)?, [0, 1, 0, 2]);
    assert_eq!(nearest(&held[..0], &centroids, 2)?, Vec::<i64>::new());
    let spread = |each: u64| each.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let many: Vec<u64> = (0..20_000u64).flat_map(|each| [spread(each), !spread(each)]).collect();
    let found = nearest(&many, &[0, u64::MAX, u64::MAX, 0], 2)?;
    let wanted: Vec<i64> = (0..20_000u64)
        .map(|each| i64::from(spread(each).count_ones() > 32))
        .collect();
    assert_eq!(found, wanted);
    for (rows, middles, width) in [
        (&held[..3], &centroids[..], 2),
        (&held[..], &centroids[..5], 2),
        (&held[..], &[][..], 2),
    ] {
        assert!(refused(
            nearest(rows, middles, width),
            "codes and centroids of whole words"
        ));
    }
    assert!(refused(
        nearest(&held, &centroids, 0),
        "codes and centroids of whole words"
    ));
    Ok(())
}

#[test]
fn a_centroid_takes_each_bit_most_of_its_codes_hold_and_one_without_codes_stays() -> Result<(), String> {
    let held = [0b0111, 0b0011, 0b0001, 0b1000, 0b1100];
    let made = majority(&held, &[0, 0, 0, 2, 2], &[9, 7, 5], 1)?;
    assert_eq!(made, [0b0011, 7, 0b1000]);
    let many: Vec<u64> = (0..30_000u64).map(|each| if each % 3 == 0 { 1 } else { 3 }).collect();
    let owners = vec![1i64; many.len()];
    assert_eq!(majority(&many, &owners, &[5, 5], 1)?, [5, 3]);
    assert!(refused(
        majority(&held, &[0, 0, 0, 2], &[9, 7, 5], 1),
        "a centroid for each code"
    ));
    assert!(refused(
        majority(&held, &[0, 0, 0, 2, 3], &[9, 7, 5], 1),
        "a centroid for each code"
    ));
    assert!(refused(
        majority(&held, &[0, 0, 0, 2, -1], &[9, 7, 5], 1),
        "a centroid for each code"
    ));
    assert!(refused(
        majority(&held, &[0; 5], &[9, 7, 5], 2),
        "codes and centroids of whole words"
    ));
    Ok(())
}
