use std::collections::BTreeMap;
use std::sync::Arc;

use crate::lists::Block;
use crate::reading::{LengthsTable, Reached, Segment, length_blocks};
use crate::scoring::Node;
use crate::search::searched;
use crate::segment::lengths_written;
use crate::tests::searching::{Built, GivenLeaf, Posted, built, run, scored, segments};
use crate::tests::support::refused;

const A: &[Posted] = &[(1, 1), (5, 3), (6, 2), (9, 1), (14, 1)];
const A_WHOLE: &[Posted] = &[(40, 2), (47, 1)];
const DEAD: [i64; 2] = [6, 47];
const C: &[Posted] = &[(9, 2), (50, 1)];

fn every(from: i64, to: i64, step: usize) -> Vec<Posted> {
    (from..to).step_by(step).map(|place| (place, 1)).collect()
}

fn two() -> Vec<Segment> {
    segments(&[(0, 31, &DEAD[..1]), (32, 63, &DEAD[1..])])
}

fn given<'a>(b_first: &'a [Posted], b_second: &'a [Posted]) -> [GivenLeaf<'a>; 3] {
    [
        GivenLeaf {
            whole: vec![(A_WHOLE, Some(1))],
            ranged: vec![(0, A)],
            bounding: 2.0,
        },
        GivenLeaf {
            whole: Vec::new(),
            ranged: vec![(0, b_first), (1, b_second)],
            bounding: 0.5,
        },
        GivenLeaf {
            whole: vec![(C, None)],
            ranged: Vec::new(),
            bounding: 1.0,
        },
    ]
}

fn brute(lists: &[(f64, Vec<Posted>)], keep: impl Fn(&[bool]) -> bool, k: usize) -> Vec<(f64, i64)> {
    let mut held: BTreeMap<i64, Vec<Option<i64>>> = BTreeMap::new();
    for (at, (_, postings)) in lists.iter().enumerate() {
        for &(place, count) in postings {
            held.entry(place).or_insert_with(|| vec![None; lists.len()])[at] = Some(count);
        }
    }
    let mut found: Vec<(f64, i64)> = Vec::new();
    for (place, counts) in held {
        let present: Vec<bool> = counts.iter().map(Option::is_some).collect();
        if DEAD.contains(&place) || !keep(&present) {
            continue;
        }
        let each = lists
            .iter()
            .zip(&counts)
            .map(|((bounding, _), count)| count.map_or(0.0, |count| scored(count, *bounding)));
        found.push((each.fold(0.0, |total, score| total + score), place));
    }
    found.sort_by(|one, other| other.0.total_cmp(&one.0).then(one.1.cmp(&other.1)));
    found.truncate(k);
    found
}

fn lists(b: &[Posted]) -> Vec<(f64, Vec<Posted>)> {
    let a: Vec<Posted> = A.iter().chain(A_WHOLE).copied().collect();
    vec![(2.0, a), (0.5, b.to_vec()), (1.0, C.to_vec())]
}

fn once(asked: &[impl Ord + Clone]) -> bool {
    let mut held = asked.to_vec();
    held.sort();
    held.dedup();
    held.len() == asked.len()
}

#[test]
fn a_search_of_any_word_finds_the_best_passages_and_asks_for_each_block_and_length_once() -> Result<(), String> {
    let (first, second) = (every(0, 32, 1), every(32, 64, 2));
    for k in [1, 3, 40] {
        let made = built(
            two(),
            &given(&first, &second),
            vec![Node::Leaf(0), Node::Leaf(1), Node::Leaf(2), Node::Or(vec![0, 1, 2])],
            k,
        )?;
        let (cut, fake) = run(made)?;
        let b: Vec<Posted> = first.iter().chain(&second).copied().collect();
        assert_eq!(
            (cut, fake.best.clone()),
            (false, brute(&lists(&b), |held| held.iter().any(|&one| one), k))
        );
        assert!(once(&fake.asked_blocks) && once(&fake.asked_lengths));
    }
    Ok(())
}

#[test]
fn a_search_of_every_word_or_of_words_kept_and_dropped_finds_what_holds_them() -> Result<(), String> {
    let (first, second) = (every(0, 32, 3), every(32, 64, 2));
    let b: Vec<Posted> = first.iter().chain(&second).copied().collect();
    let all = vec![Node::Leaf(0), Node::Leaf(1), Node::Leaf(2), Node::And(vec![0, 1, 2])];
    let (_, fake) = run(built(two(), &given(&first, &second), all, 5)?)?;
    assert_eq!(fake.best, brute(&lists(&b), |held| held.iter().all(|&one| one), 5));
    let pair = vec![Node::Leaf(0), Node::Leaf(1), Node::And(vec![0, 1])];
    let (_, fake) = run(built(two(), &given(&first, &second)[..2], pair, 5)?)?;
    assert_eq!(fake.best, brute(&lists(&b)[..2], |held| held.iter().all(|&one| one), 5));
    let dropped = vec![
        Node::Leaf(0),
        Node::Leaf(2),
        Node::Or(vec![0, 1]),
        Node::Leaf(1),
        Node::Not([2, 3]),
    ];
    let (_, fake) = run(built(two(), &given(&first, &second), dropped, 40)?)?;
    let kept = |held: &[bool]| (held[0] || held[2]) && !held[1];
    let want: Vec<(f64, i64)> = brute(
        &[lists(&b)[0].clone(), (0.0, b.clone()), lists(&b)[2].clone()],
        kept,
        40,
    );
    assert_eq!(fake.best, want);
    let (_, fake) = run(built(two(), &given(&first, &second)[..1], vec![Node::Leaf(0)], 2)?)?;
    assert_eq!(fake.best, brute(&lists(&b)[..1], |held| held[0], 2));
    Ok(())
}

#[test]
fn a_search_out_of_time_stops_after_a_batch_once_it_holds_a_passage() -> Result<(), String> {
    let (first, second) = (every(0, 32, 1), every(32, 64, 2));
    let nodes = vec![Node::Leaf(0), Node::Leaf(1), Node::Leaf(2), Node::Or(vec![0, 1, 2])];
    let mut made = built(two(), &given(&first, &second), nodes, 3)?;
    made.fake.expiring = true;
    let (cut, fake) = run(made)?;
    assert_eq!((cut, fake.offers), (true, 1));
    Ok(())
}

fn refusing(made: Built, saying: &str) -> bool {
    refused(run(made), saying)
}

fn one_word(segments: Vec<Segment>, whole: &[Posted]) -> Result<Built, String> {
    let leaf = [GivenLeaf {
        whole: vec![(whole, Some(0))],
        ranged: vec![(0, A)],
        bounding: 1.0,
    }];
    built(segments, &leaf, vec![Node::Leaf(0)], 3)
}

#[test]
fn a_search_whose_reads_fail_or_give_other_than_it_asked_for_is_refused() -> Result<(), String> {
    let lone = || segments(&[(0, 63, &[])]);
    for name in ["blocks", "lengths", "offered"] {
        let mut made = one_word(lone(), &[(40, 1)])?;
        made.fake.failing = name;
        assert!(refusing(made, &format!("{name} failed")));
    }
    let mut made = built(
        two(),
        &given(&every(0, 32, 1), &every(32, 64, 2)),
        vec![Node::Leaf(0), Node::Leaf(1), Node::Leaf(2), Node::Or(vec![0, 1, 2])],
        1,
    )?;
    made.fake.failing = "expired";
    assert!(refusing(made, "expired failed"));
    for name in ["blocks", "lengths"] {
        let mut made = one_word(lone(), &[(40, 1)])?;
        made.fake.fewer = name;
        assert!(refusing(made, "other than it asked for"));
    }
    let mut made = one_word(lone(), &[(40, 1)])?;
    made.fake.lengths.insert((0, 0), Arc::new(vec![1; 3]));
    assert!(refusing(made, "other than it asked for"));
    let mut made = one_word(lone(), &[(40, 1)])?;
    made.fake.blocks[0][0] = Arc::new(Block {
        ids: vec![1],
        counts: vec![1],
    });
    assert!(refusing(made, "fewer postings than its ranges place"));
    assert!(refusing(
        one_word(segments(&[(0, 15, &[]), (20, 63, &[])]), &[(17, 1)])?,
        "outside its segments"
    ));
    assert!(refusing(
        one_word(segments(&[(4, 63, &[])]), &[(40, 1)])?,
        "outside its segments"
    ));
    let mut made = one_word(lone(), &[(40, 1)])?;
    made.plan.store.average = 0.0;
    assert!(refusing(made, "passages of no length"));
    let mut made = one_word(lone(), &[(40, 1)])?;
    made.plan.store.parts[0].segment = 3;
    assert!(refusing(made, "does not hold"));
    let mut made = one_word(lone(), &[(40, 1)])?;
    made.plan.batches = (0, 4);
    assert!(refusing(made, "does not hold"));
    let made = one_word(lone(), &[(40, 1)])?;
    let (plan, mut fake) = (made.plan, made.fake);
    let reached = Reached {
        threshold: 99.0,
        held: 3,
    };
    let found = searched(plan, reached, &mut fake).map_err(|failure| failure.to_string())?;
    assert_eq!((found, fake.offers, fake.asked_blocks.len()), (false, 0, 0));
    Ok(())
}

#[test]
fn a_search_of_a_word_in_no_passage_finds_nothing() -> Result<(), String> {
    let leaf = [GivenLeaf {
        whole: vec![(&[][..], Some(0))],
        ranged: Vec::new(),
        bounding: 1.0,
    }];
    let (cut, fake) = run(built(segments(&[(0, 63, &[])]), &leaf, vec![Node::Leaf(0)], 3)?)?;
    assert_eq!((cut, fake.best.len(), fake.asked_lengths.len()), (false, 0, 0));
    Ok(())
}

#[test]
fn a_search_given_the_lengths_tables_an_earlier_one_filled_reads_no_lengths_again() -> Result<(), String> {
    let (first, second) = (every(0, 32, 1), every(32, 64, 2));
    let nodes = || vec![Node::Leaf(0), Node::Leaf(1), Node::Leaf(2), Node::Or(vec![0, 1, 2])];
    let made = built(two(), &given(&first, &second), nodes(), 5)?;
    let tables: Vec<Arc<LengthsTable>> = made
        .plan
        .store
        .segments
        .iter()
        .map(|each| Arc::clone(&each.lengths))
        .collect();
    let (_, before) = run(made)?;
    let mut again = built(two(), &given(&first, &second), nodes(), 5)?;
    for (segment, table) in again.plan.store.segments.iter_mut().zip(&tables) {
        segment.lengths = Arc::clone(table);
    }
    let (_, after) = run(again)?;
    let held: usize = tables.iter().map(|table| table.held_bytes()).sum();
    assert_eq!((after.best, after.asked_lengths.len()), (before.best, 0));
    assert_eq!(held, before.asked_lengths.len() * 4 * 8);
    Ok(())
}

#[test]
fn rows_of_lengths_come_back_as_a_block_of_lengths_each() -> Result<(), String> {
    let ids: Vec<i64> = (0..40).collect();
    let lengths: Vec<u32> = (1..=40).collect();
    let rows: Vec<Vec<u8>> = lengths_written(&ids, &lengths, 3)?
        .into_iter()
        .map(|(_, row)| row)
        .collect();
    let found = length_blocks(&rows, 3)?;
    let told: Vec<u32> = found.iter().flat_map(|block| block.iter().copied()).collect();
    assert_eq!((found.len(), told), (5, lengths));
    assert!(refused(length_blocks(&[vec![1u8, 2, 3]], 3), "does not unpack"));
    Ok(())
}
