use crate::lists::{Lists, block, list};
use crate::postings::{Kept, Postings, Shape, blocks_of, gathered, joined, owned_blocks, ranged};
use crate::tests::support::{AVERAGE, held_ranges, refused, small};

fn of(places: &[i64]) -> Result<Postings, String> {
    let mut held = Postings::default();
    for &place in places {
        held.push(place, 3, [1, 4].into_iter())?;
    }
    Ok(held)
}

#[test]
fn a_shape_whose_blocks_ranges_or_lengths_cannot_be_written_is_refused() -> Result<(), String> {
    for (range_shift, block, long, lengths_shift) in [
        (9, 2, 3, 3),
        (2, 0, 3, 3),
        (2, 4097, 3, 3),
        (2, 2, 0, 3),
        (2, 2, 4097, 3),
        (2, 2, 3, 0),
        (2, 2, 3, 21),
    ] {
        assert!(refused(
            Shape::new(range_shift, block, long, lengths_shift),
            "1 to 4,096 entries"
        ));
    }
    let shape = Shape::new(8, 4096, 1, 20)?;
    assert_eq!(
        (shape.range_shift(), shape.block(), shape.long(), shape.lengths_shift()),
        (8, 4096, 1, 20)
    );
    Ok(())
}

#[test]
fn a_list_shorter_than_a_long_one_is_kept_whole_and_a_long_one_in_blocks_under_its_ranges() -> Result<(), String> {
    let short = joined("word", &[&of(&[4, 9])?], AVERAGE, small()?)?;
    assert_eq!((&*short.term, short.chunks), ("word", 2));
    assert!(matches!(short.held, Kept::Short(_)));
    let long = joined("word", &[&of(&[4, 9])?, &of(&[10, 11, 40])?], AVERAGE, small()?)?;
    let Kept::Long { ranges, blocks } = long.held else {
        return Err("a list of five passages is long in a shape of three".to_owned());
    };
    assert_eq!(long.chunks, 5);
    let ids: Result<Vec<Vec<i64>>, String> = blocks
        .iter()
        .map(|coded| block(&coded.0).map(|found| found.ids))
        .collect();
    assert_eq!(ids?, [vec![4, 9], vec![10, 11], vec![40]]);
    assert_ne!(ranges.len(), 0);
    Ok(())
}

#[test]
fn postings_are_gathered_across_the_lists_they_lie_in() -> Result<(), String> {
    let (one, other) = (of(&[4, 9])?, of(&[10, 11, 40])?);
    let parts = [&one, &Postings::default(), &other];
    assert_eq!(block(&gathered(&parts, 1, 4).coded()?.0)?.ids, [9, 10, 11]);
    assert_eq!(gathered(&parts, 5, 9).chunks(), 0);
    let coded = blocks_of(&parts, 5, 4)?;
    assert_eq!(block(&coded[0].0)?.ids, [4, 9, 10, 11]);
    assert_eq!(block(&coded[1].0)?.ids, [40]);
    assert_eq!(blocks_of(&parts, 0, 4)?.len(), 0);
    Ok(())
}

#[test]
fn ranges_too_far_apart_too_full_or_of_no_posting_are_refused() -> Result<(), String> {
    let far = joined("word", &[&of(&[0, 1, 1 << 34])?], AVERAGE, small()?);
    assert!(refused(far, "2**32 apart: more than ranges hold"));
    assert!(refused(ranged(&[]), "holds no range"));
    let (mut found, full) = (Vec::new(), of(&(0..257).collect::<Vec<i64>>())?);
    full.ranged(&mut found, AVERAGE, 9);
    assert!(refused(ranged(&found), "more postings than its count keeps"));
    Ok(())
}

fn coded(ids: &[i64]) -> Result<Vec<u8>, String> {
    let lists = Lists {
        ids,
        counts: &vec![1; ids.len()],
        positions: &vec![0; ids.len()],
        offsets: &(0..=ids.len()).collect::<Vec<usize>>(),
    };
    Ok(list(&lists, 0, ids.len())?.0)
}

#[test]
fn the_blocks_of_a_list_are_read_together_and_one_outside_its_ranges_or_of_another_size_is_refused()
-> Result<(), String> {
    let rows = [coded(&[0, 1])?, coded(&[5, 9])?, coded(&[10])?];
    let (ranges, shape) = (held_ranges(&[0, 1, 2], &[2, 1, 2], &[9, 9, 9])?, small()?);
    let found = owned_blocks(&rows, &[0, 1, 2], &ranges, shape)?;
    let ids: Vec<&[i64]> = found.iter().map(|each| &each.ids[..]).collect();
    assert_eq!(ids, [&[0, 1][..], &[5, 9][..], &[10][..]]);
    assert_eq!(owned_blocks(&rows[2..], &[2], &ranges, shape)?[0].counts, [1]);
    assert!(owned_blocks(&rows[..0], &[], &ranges, shape)?.is_empty());
    assert!(refused(
        owned_blocks(&rows[1..2], &[0], &ranges, shape),
        "outside its ranges"
    ));
    assert!(refused(
        owned_blocks(&rows[..1], &[1], &ranges, shape),
        "outside its ranges"
    ));
    for (numbers, placed) in [([0], 2), ([3], 0), ([usize::MAX], 0)] {
        let saying = format!("a block holds 1 postings where its ranges place {placed}");
        assert!(refused(owned_blocks(&rows[2..], &numbers, &ranges, shape), &saying));
    }
    assert!(refused(
        owned_blocks(&rows, &[0, 1], &ranges, shape),
        "a block for each number"
    ));
    assert!(refused(
        owned_blocks(&[vec![9u8]], &[0], &ranges, shape),
        "unknown flag byte"
    ));
    Ok(())
}
