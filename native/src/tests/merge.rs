use std::sync::Arc;

use crate::merge::{Chunks, Fed, Limits, Merger};
use crate::packs::entries;
use crate::postings::Shape;
use crate::tests::support::{
    AVERAGE, Feed, built, fed_whole, merged, refused, shown, small, sources, terms, wide, words,
};

const EVERYTHING: usize = 1_000_000_000;
const WHOLE: Limits = Limits {
    postings: EVERYTHING,
    bytes: EVERYTHING,
    tokens: EVERYTHING,
};
const PIECES: Limits = Limits {
    postings: 1,
    bytes: 1,
    tokens: EVERYTHING,
};
const DENSE: Limits = Limits {
    postings: EVERYTHING,
    bytes: EVERYTHING,
    tokens: 1,
};
const TEXTS: [(i64, &str); 9] = [
    (5, "alpha beta beta"),
    (8, "beta gamma"),
    (11, "alpha"),
    (14, "beta delta beta alpha"),
    (17, "gamma gamma beta"),
    (20, "beta"),
    (23, "alpha beta epsilon"),
    (26, "beta zeta"),
    (29, "alpha beta"),
];

fn chunks(texts: &[(i64, &str)], dead: &[i64]) -> Result<Chunks, String> {
    let live: Vec<&(i64, &str)> = texts.iter().filter(|each| !dead.contains(&each.0)).collect();
    let lengths: Result<Vec<usize>, String> = live.iter().map(|each| words(each.1).map(|held| held.len())).collect();
    Ok(Chunks {
        live: live.iter().map(|each| each.0).collect(),
        lengths: lengths?
            .into_iter()
            .map(|length| u32::try_from(length).unwrap_or(u32::MAX))
            .collect(),
        dead: dead.to_vec(),
        average: AVERAGE,
    })
}

fn given(texts: &[(i64, &str)], cuts: &[usize], shape: Shape) -> Result<Vec<Feed>, String> {
    let parts = cuts
        .windows(2)
        .map(|pair| Ok(sources(&built(&texts[pair[0]..pair[1]], shape)?, shape)));
    parts.collect()
}

#[test]
fn segments_merge_into_the_segment_their_live_chunks_alone_build() -> Result<(), String> {
    let dead = [8, 20];
    let live: Vec<(i64, &str)> = TEXTS.iter().filter(|each| !dead.contains(&each.0)).copied().collect();
    for shape in [small()?, wide()?] {
        let want = built(&live, shape)?;
        for limits in [WHOLE, PIECES, DENSE] {
            for at_once in [1, 2, 64] {
                let fed = given(&TEXTS, &[0, 4, 5, 9], shape)?;
                let mut merger = Merger::new(chunks(&TEXTS, &dead)?, shape, &limits, 2 * fed.len())?;
                let made = merged(&mut merger, fed, at_once)?;
                assert_eq!(shown(&made.filed)?, shown(&want.filed)?);
            }
        }
    }
    Ok(())
}

#[test]
fn a_term_only_dead_chunks_held_is_gone_after_a_merge() -> Result<(), String> {
    for limits in [WHOLE, PIECES] {
        let fed = given(&TEXTS, &[0, 9], wide()?)?;
        let mut merger = Merger::new(chunks(&TEXTS, &[14, 23, 26])?, wide()?, &limits, 2 * fed.len())?;
        let made = merged(&mut merger, fed, 64)?;
        assert_eq!(terms(&made.filed)?, ["alpha", "beta", "gamma"]);
    }
    Ok(())
}

#[test]
fn a_merge_takes_terms_together_up_to_its_postings_or_bytes_and_a_term_past_either_alone() -> Result<(), String> {
    let texts = [(0, "a b"), (1, "a b"), (2, "a b"), (3, "a c"), (4, "a c"), (5, "a c")];
    let first = built(&texts[..3], wide()?)?;
    let sizes: Vec<usize> = entries(&first.filed.packs[0].lists, Some(&first.filed.packs[0].places))?
        .iter()
        .map(|each| each.postings.len() + each.positions.len())
        .collect();
    let size = sizes[0];
    assert_eq!(sizes, [size, size]);
    let steps = |postings: usize, bytes: usize| -> Result<usize, String> {
        let limits = Limits {
            postings,
            bytes,
            tokens: EVERYTHING,
        };
        let fed = given(&texts, &[0, 3, 6], wide()?)?;
        let mut merger = Merger::new(chunks(&texts, &[])?, wide()?, &limits, 2 * fed.len())?;
        fed_whole(&mut merger, fed)?;
        let mut taken = 1;
        while !merger.stepped()?.done {
            taken += 1;
        }
        Ok(taken)
    };
    assert_eq!(steps(EVERYTHING, 1)?, 4);
    assert_eq!(steps(4, EVERYTHING)?, 2);
    assert_eq!(steps(EVERYTHING, 3 * size)?, 2);
    assert_eq!(steps(EVERYTHING, 3 * size + 1)?, 1);
    assert_eq!(steps(EVERYTHING, EVERYTHING)?, 1);
    Ok(())
}

#[test]
fn a_merge_of_chunks_the_store_does_not_hold_or_of_damaged_lists_is_refused() -> Result<(), String> {
    let texts = [(0, "a"), (1, "a")];
    for limits in [WHOLE, PIECES] {
        let unknown = Chunks {
            live: vec![1],
            lengths: vec![1],
            dead: Vec::new(),
            average: AVERAGE,
        };
        let fed = given(&texts, &[0, 2], wide()?)?;
        let mut merger = Merger::new(unknown, wide()?, &limits, 2 * fed.len())?;
        assert!(refused(
            merged(&mut merger, fed, 64).map(|_| ()),
            "chunks the store does not"
        ));
        let mut merger = Merger::new(chunks(&texts, &[])?, wide()?, &limits, 2)?;
        let damaged = Fed {
            term: Arc::from("a"),
            chunks: 2,
            postings: vec![0],
            positions: vec![0],
        };
        let alone = vec![Feed {
            packs: Vec::new(),
            blocks: vec![damaged],
        }];
        assert!(refused(
            merged(&mut merger, alone, 64).map(|_| ()),
            "shorter than its head"
        ));
        let mut merger = Merger::new(chunks(&texts, &[])?, wide()?, &limits, 2)?;
        assert!(refused(merger.fed_packs(0, &[(vec![9], vec![])]), "a pack of lists"));
    }
    let uneven = Chunks {
        live: vec![1, 2],
        lengths: vec![1],
        dead: Vec::new(),
        average: AVERAGE,
    };
    assert!(refused(
        Merger::new(uneven, wide()?, &WHOLE, 1).map(|_| ()),
        "each chunk of a merge has a length"
    ));
    Ok(())
}

#[test]
fn a_term_of_more_rows_than_a_merge_takes_at_once_merges_as_whole_as_in_pieces() -> Result<(), String> {
    let texts: Vec<(i64, &str)> = (0..7000).map(|place| (place, "w")).collect();
    let want = built(&texts, small()?)?;
    for limits in [WHOLE, PIECES] {
        let fed = given(&texts, &[0, 3000, 7000], small()?)?;
        let mut merger = Merger::new(chunks(&texts, &[])?, small()?, &limits, 2 * fed.len())?;
        let made = merged(&mut merger, fed, 4096)?;
        assert_eq!(shown(&made.filed)?, shown(&want.filed)?);
    }
    Ok(())
}

#[test]
fn a_merge_takes_rows_of_the_sources_it_was_made_for_alone() -> Result<(), String> {
    let mut merger = Merger::new(chunks(&TEXTS, &[])?, wide()?, &WHOLE, 2)?;
    assert_eq!(merger.fed(2, Vec::new()), None);
    assert_eq!(merger.ended(2), None);
    assert_eq!(merger.fed_packs::<Vec<u8>>(2, &[])?, None);
    assert_eq!(merger.ended(1), Some(false));
    assert_eq!(merger.fed(0, Vec::new()), Some(true));
    Ok(())
}
