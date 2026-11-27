use crate::lists::{Lists, block, blocks, gaps, list, placed, places};
use crate::tests::support::refused;

const RAW: u8 = 0;

fn headed(first: i64, held: u32, rest: &[u8]) -> Vec<u8> {
    let mut data = vec![RAW];
    data.extend_from_slice(&first.to_le_bytes());
    data.extend_from_slice(&held.to_le_bytes());
    data.extend_from_slice(rest);
    data
}

#[test]
fn a_list_comes_back_with_its_ids_counts_and_positions_whole_or_in_part() -> Result<(), String> {
    let lists = Lists {
        ids: &[5, 9, 300],
        counts: &[1, 2, 1],
        positions: &[0, 1, 7, 3],
        offsets: &[0, 1, 3, 4],
    };
    let whole = list(&lists, 0, 3)?;
    let found = block(&whole.0)?;
    assert_eq!((found.ids, &found.counts), (vec![5, 9, 300], &vec![1, 2, 1]));
    let mut positions = gaps(&whole.1)?;
    placed(&mut positions, &found.counts)?;
    assert_eq!(positions, [0, 1, 7, 3]);
    let part = list(&lists, 1, 3)?;
    assert_eq!(block(&part.0)?.ids, [9, 300]);
    assert_eq!(gaps(&part.1)?, [1, 6, 3]);
    Ok(())
}

#[test]
fn a_list_of_no_id_or_of_ids_too_far_apart_or_out_of_order_is_refused() {
    for (ids, saying) in [
        (vec![0, 1 << 32], "2**32 apart"),
        (vec![9, 3], "2**32 apart"),
        (vec![5, 5], "2**32 apart"),
    ] {
        let lists = Lists {
            ids: &ids,
            counts: &[1, 1],
            positions: &[0, 0],
            offsets: &[0, 1, 2],
        };
        assert!(refused(list(&lists, 0, 2), saying));
        assert!(refused(list(&lists, 1, 1), "holds no id"));
        assert!(refused(list(&lists, 0, 3), "holds no id"));
    }
}

#[test]
fn a_damaged_block_is_refused_with_what_is_wrong() {
    let damaged = [
        (vec![RAW, 0, 0, 0, 0, 0], "shorter than its head"),
        (headed(1, 3, &[]), "fewer ids than its head says"),
        (headed(1, 0, &[]), "fewer ids than its head says"),
        (headed(1, 1, &[0x80]), "runs past the end"),
        (headed(1, 1, &[0]), "a count below one"),
        (headed(1, 1, &[1, 1]), "counts other than its ids"),
        (headed(1, 2, &[0, 0, 0, 0, 1, 1]), "ids out of order"),
    ];
    for (data, saying) in damaged {
        assert!(refused(block(&data), saying), "{saying}");
    }
}

#[test]
fn positions_other_than_their_counts_or_past_any_passage_are_refused() {
    assert!(refused(placed(&mut [1, 2], &[3]), "other than its counts"));
    assert!(refused(placed(&mut [1, 2, 3], &[2]), "other than its counts"));
    assert!(refused(placed(&mut [1], &[-1]), "other than its counts"));
    assert!(refused(placed(&mut [1 << 32], &[1]), "past any passage"));
    assert!(refused(placed(&mut [1 << 31, 1 << 31], &[2]), "past any passage"));
    assert_eq!(placed(&mut [(1 << 32) - 1], &[1]), Ok(()));
}

#[test]
fn many_lists_decode_together_as_each_alone() -> Result<(), String> {
    let (mut coded, mut placing) = (Vec::new(), Vec::new());
    for id in 0..300i64 {
        let lists = Lists {
            ids: &[id, id + 1000],
            counts: &[1, 2],
            positions: &[id, 4, 9],
            offsets: &[0, 1, 3],
        };
        let (postings, positions) = list(&lists, 0, 2)?;
        coded.push(postings);
        placing.push(positions);
    }
    let found = blocks(&coded)?;
    assert_eq!(found.bounds, (0..=600).step_by(2).collect::<Vec<usize>>());
    assert_eq!(found.ids[598..], [299, 1299]);
    let positions = places(&placing, &found.counts)?;
    assert_eq!(positions[897..], [299, 4, 9]);
    assert_eq!(blocks(&coded[..2])?.ids, [0, 1000, 1, 1001]);
    assert_eq!(places(&placing[..2], &[1, 2, 1, 2])?, [0, 4, 9, 1, 4, 9]);
    assert_eq!(blocks::<Vec<u8>>(&[])?.bounds, [0]);
    Ok(())
}
