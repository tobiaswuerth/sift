use crate::lists::{Lists, Occurring, Sought, block, blocks, list, occurring, placed, positioned};
use crate::packed::wrapped;
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
    assert_eq!(positioned(&whole.1, &found.counts)?, [0, 1, 7, 3]);
    let part = list(&lists, 1, 3)?;
    assert_eq!(block(&part.0)?.ids, [9, 300]);
    assert_eq!(positioned(&part.1, &[2, 1])?, [1, 7, 3]);
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
    let mut coded = Vec::new();
    for id in 0..300i64 {
        let lists = Lists {
            ids: &[id, id + 1000],
            counts: &[1, 2],
            positions: &[id, 4, 9],
            offsets: &[0, 1, 3],
        };
        coded.push(list(&lists, 0, 2)?.0);
    }
    let found = blocks(&coded)?;
    assert_eq!(found.bounds, (0..=600).step_by(2).collect::<Vec<usize>>());
    assert_eq!(
        (&found.ids[598..], &found.counts[598..]),
        (&[299, 1299][..], &[1, 2][..])
    );
    assert_eq!(blocks(&coded[..2])?.ids, [0, 1000, 1, 1001]);
    assert_eq!(blocks::<Vec<u8>>(&[])?.bounds, [0]);
    Ok(())
}

#[test]
fn a_block_whose_ids_fall_below_zero_or_run_past_the_last_place_is_refused() {
    assert!(refused(block(&headed(-5, 1, &[1])), "an id below zero"));
    assert!(refused(
        block(&headed(i64::MAX, 2, &[1, 0, 0, 0, 1, 1])),
        "ids past the last place"
    ));
    assert_eq!(
        block(&headed(i64::MAX, 1, &[1])).map(|found| found.ids),
        Ok(vec![i64::MAX])
    );
    assert_eq!(block(&headed(0, 1, &[1])).map(|found| found.ids), Ok(vec![0]));
}

#[test]
fn a_position_held_twice_by_a_passage_is_refused_read_or_written() {
    assert!(refused(placed(&mut [3, 0], &[2]), "a position twice"));
    assert!(refused(placed(&mut [0, 2, 0], &[3]), "a position twice"));
    assert_eq!(placed(&mut [0, 1, 0, 0], &[2, 1, 1]), Ok(()));
    let lists = Lists {
        ids: &[5],
        counts: &[2],
        positions: &[4, 4],
        offsets: &[0, 2],
    };
    assert!(refused(list(&lists, 0, 1), "a position twice"));
}

fn told(found: &Occurring) -> (&[i64], &[i64], &[i64]) {
    (&found.ids, &found.counts, &found.positions)
}

#[test]
fn the_positions_of_the_passages_sought_come_back_from_many_blocks_at_once_as_from_each_alone() -> Result<(), String> {
    let (mut rows, mut ids, mut counts, mut bounds) = (Vec::new(), Vec::new(), Vec::new(), vec![0usize]);
    for block in 0..300i64 {
        let lists = Lists {
            ids: &[block * 10, block * 10 + 3, block * 10 + 7],
            counts: &[1, 2, 1],
            positions: &[block, 4, 9, 2],
            offsets: &[0, 1, 3, 4],
        };
        rows.push(list(&lists, 0, 3)?.1);
        ids.extend_from_slice(lists.ids);
        counts.extend_from_slice(lists.counts);
        bounds.push(ids.len());
    }
    let sought = |wanted| Sought {
        bounds: &bounds,
        ids: &ids,
        counts: &counts,
        wanted,
    };
    let found = occurring(&rows, &sought(&[3, 7, 7, 8, 2990, 2993, 5000]))?;
    assert_eq!(
        told(&found),
        (&[3, 7, 2990, 2993][..], &[2, 1, 1, 2][..], &[4, 9, 2, 299, 4, 9][..])
    );
    assert_eq!(told(&occurring(&rows, &sought(&[]))?), (&[][..], &[][..], &[][..]));
    let alone = Sought {
        bounds: &[0, 3],
        ids: &ids[3..6],
        counts: &counts[3..6],
        wanted: &[0, 10, 17, 20],
    };
    let found = occurring(&rows[1..2], &alone)?;
    assert_eq!(told(&found), (&[10, 17][..], &[1, 1][..], &[1, 2][..]));
    let none = Sought {
        bounds: &[0],
        ids: &[],
        counts: &[],
        wanted: &[5],
    };
    assert_eq!(told(&occurring::<Vec<u8>>(&[], &none)?), (&[][..], &[][..], &[][..]));
    Ok(())
}

#[test]
fn passages_told_other_than_block_by_block_or_with_other_positions_than_a_block_holds_are_refused() -> Result<(), String>
{
    let lists = Lists {
        ids: &[5, 9],
        counts: &[1, 2],
        positions: &[0, 1, 7],
        offsets: &[0, 1, 3],
    };
    let rows = [list(&lists, 0, 2)?.1];
    let uneven: [(&[usize], &[i64], &[i64]); 6] = [
        (&[0], &[5, 9], &[1, 2]),
        (&[1, 2], &[5, 9], &[1, 2]),
        (&[0, 1], &[5, 9], &[1, 2]),
        (&[0, 2], &[5, 9], &[1]),
        (&[0, 3], &[5, 9], &[1, 2]),
        (&[], &[], &[]),
    ];
    for (bounds, ids, counts) in uneven {
        let sought = Sought {
            bounds,
            ids,
            counts,
            wanted: &[5],
        };
        assert!(refused(occurring(&rows, &sought), "other than block by block"));
    }
    let two = [rows[0].clone(), rows[0].clone()];
    let falling = Sought {
        bounds: &[0, 2, 1],
        ids: &[5],
        counts: &[1],
        wanted: &[5],
    };
    assert!(refused(occurring(&two, &falling), "other than block by block"));
    let miscounted = Sought {
        bounds: &[0, 2],
        ids: &[5, 9],
        counts: &[1, 3],
        wanted: &[5],
    };
    assert!(refused(
        occurring(&rows, &miscounted),
        "positions other than its counts"
    ));
    assert!(refused(
        occurring(
            &[vec![7u8]],
            &Sought {
                bounds: &[0, 0],
                ids: &[],
                counts: &[],
                wanted: &[]
            }
        ),
        "unknown flag byte"
    ));
    Ok(())
}

#[test]
fn the_largest_block_a_list_may_hold_is_read_and_one_counting_or_holding_more_is_refused() -> Result<(), String> {
    let ids: Vec<i64> = (0..4096).map(|each| each * 3).collect();
    let counts = vec![1024i64; 4096];
    let own: Vec<i64> = (0..1024).map(|each| (1 << 28) + each).collect();
    let positions: Vec<i64> = (0..4096).flat_map(|_| own.iter().copied()).collect();
    let offsets: Vec<usize> = (0..=4096).map(|each| each * 1024).collect();
    let lists = Lists {
        ids: &ids,
        counts: &counts,
        positions: &positions,
        offsets: &offsets,
    };
    let coded = list(&lists, 0, 4096)?;
    let found = block(&coded.0)?;
    assert_eq!((found.ids.len(), found.counts[4095]), (4096, 1024));
    assert_eq!(positioned(&coded.1, &found.counts)?, positions);
    assert!(refused(block(&headed(1, 1, &[0x81, 0x08])), "a count above 1,024"));
    let longer = [RAW, 1, 1, 1, 1, 1, 1];
    assert!(refused(positioned(&longer, &[1]), "holds more than 5 bytes"));
    let far = wrapped(&vec![1u8; 1 << 20])?;
    assert!(refused(positioned(&far, &[2, 2]), "unpacks to more than 20 bytes"));
    assert!(refused(positioned(&far, &[-1]), "unpacks to more than 0 bytes"));
    assert_eq!(positioned(&far, &vec![1; 1 << 20])?.len(), 1 << 20);
    Ok(())
}
