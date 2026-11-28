use crate::packed::deflated;
use crate::ranges::{Bounds, ranges_all, ranges_read};
use crate::tests::support::{held_ranges, ranges_body, refused};

fn bounds(low: i64, high: i64, chunks: usize) -> Bounds {
    Bounds { low, high, chunks }
}

#[test]
fn ranges_hold_where_their_postings_start_and_end_and_their_best_scores() -> Result<(), String> {
    let held = held_ranges(&[3, 4, 9], &[2, 1, 4], &[7, 8, 255])?;
    assert_eq!(
        (held.ids(), held.ends(), held.best()),
        (&[3, 4, 9][..], &[2, 3, 7][..], &[7, 8, 255][..])
    );
    assert_eq!(
        (held.start(0), held.start(1), held.start(2), held.total()),
        (0, 2, 3, 7)
    );
    Ok(())
}

#[test]
fn ranges_of_many_lists_are_read_together_as_one_by_one() -> Result<(), String> {
    let rows: Vec<Vec<u8>> = (0..40)
        .map(|at| ranges_body(&[at, at + 2], &[1, 3], &[1, 2]))
        .collect::<Result<_, _>>()?;
    let told: Vec<Bounds> = (0..40).map(|at| bounds(at, at + 2, 4)).collect();
    let found = ranges_all(&rows, &told)?;
    assert_eq!(
        found.iter().map(|each| each.ids()[1]).collect::<Vec<i64>>(),
        (2..42).collect::<Vec<i64>>()
    );
    let few = ranges_all(&rows[..2], &told[..2])?;
    assert_eq!(few.len(), 2);
    assert!(refused(
        ranges_all(&rows[..2], &told[..1]),
        "a segment and a count for each"
    ));
    Ok(())
}

#[test]
fn ranges_outside_their_segment_or_of_another_count_than_their_list_are_refused() -> Result<(), String> {
    let row = ranges_body(&[4, 5, 9], &[2, 1, 2], &[9, 9, 9])?;
    for (low, high) in [(5, 9), (4, 8), (9, 4), (i64::MIN, i64::MAX), (0, i64::MAX)] {
        assert!(refused(ranges_read(&row, &bounds(low, high, 5)), "outside its segment"));
    }
    assert!(refused(
        ranges_read(&row, &bounds(4, 9, 6)),
        "hold 5 passages, not its 6"
    ));
    let last = bounds(i64::MAX - 1, i64::MAX, 2);
    let far = ranges_body(&[i64::MAX - 1, i64::MAX], &[1, 1], &[1, 1])?;
    assert_eq!(ranges_read(&far, &last)?.total(), 2);
    let mut past = (i64::MAX - 1).to_le_bytes().to_vec();
    past.extend_from_slice(&2u32.to_le_bytes());
    past.extend_from_slice(&5u32.to_le_bytes());
    past.extend_from_slice(&[0, 0, 1, 1]);
    assert!(refused(
        ranges_read(&deflated(&past, None)?, &last),
        "outside its segment"
    ));
    Ok(())
}

#[test]
fn damaged_ranges_are_refused_with_what_is_wrong() -> Result<(), String> {
    let told = bounds(0, 8, 0);
    assert!(refused(ranges_read(b"junk", &told), "does not unpack"));
    for body in [&[0u8; 5][..], &[0; 12], &[0; 14]] {
        assert!(refused(
            ranges_read(&deflated(body, None)?, &told),
            "other than their head says"
        ));
    }
    let mut disordered = 0i64.to_le_bytes().to_vec();
    disordered.extend_from_slice(&2u32.to_le_bytes());
    disordered.extend_from_slice(&[0; 8]);
    assert!(refused(
        ranges_read(&deflated(&disordered, None)?, &told),
        "ids out of order"
    ));
    let held = ranges_body(&[0, 1, 2], &[2, 1, 2], &[9, 9, 9])?;
    assert!(refused(
        ranges_read(&held, &bounds(0, 1, 5)),
        "unpacks to more than 20 bytes"
    ));
    let mut longer = held.clone();
    longer.push(0);
    assert!(refused(ranges_read(&longer, &bounds(0, 2, 5)), "bytes after its end"));
    Ok(())
}
