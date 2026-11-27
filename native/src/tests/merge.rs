use std::sync::Arc;

use crate::merge::{Chunks, Fed, Limits, Merger};
use crate::postings::Shape;
use crate::tests::support::{AVERAGE, built, merged, refused, shown, small, sources, wide, words};

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

fn given(texts: &[(i64, &str)], cuts: &[usize], shape: Shape) -> Result<Vec<Vec<Fed>>, String> {
    let mut found = Vec::new();
    for pair in cuts.windows(2) {
        found.extend(sources(&built(&texts[pair[0]..pair[1]], shape)?, shape));
    }
    Ok(found)
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
                let mut merger = Merger::new(chunks(&TEXTS, &dead)?, shape, &limits, fed.len())?;
                let made = merged(&mut merger, fed, at_once)?;
                assert_eq!(shown(&made.rows, &made.pieces), shown(&want.rows, &[]));
            }
        }
    }
    Ok(())
}

#[test]
fn a_term_only_dead_chunks_held_is_gone_after_a_merge() -> Result<(), String> {
    for limits in [WHOLE, PIECES] {
        let fed = given(&TEXTS, &[0, 9], wide()?)?;
        let mut merger = Merger::new(chunks(&TEXTS, &[14, 23, 26])?, wide()?, &limits, fed.len())?;
        let made = merged(&mut merger, fed, 64)?;
        let terms: Vec<&str> = made.rows.iter().map(|row| &*row.term).collect();
        assert_eq!(terms, ["alpha", "beta", "gamma"]);
    }
    Ok(())
}

#[test]
fn a_merge_takes_terms_together_up_to_its_postings_or_bytes_and_a_term_past_either_alone() -> Result<(), String> {
    let texts = [(0, "a b"), (1, "a b"), (2, "a b"), (3, "a c"), (4, "a c"), (5, "a c")];
    let size = given(&texts, &[0, 3, 6], wide()?)?[0][0].size().bytes;
    let taken = |postings: usize, bytes: usize| -> Result<Vec<Vec<String>>, String> {
        let limits = Limits {
            postings,
            bytes,
            tokens: EVERYTHING,
        };
        let fed = given(&texts, &[0, 3, 6], wide()?)?;
        let mut merger = Merger::new(chunks(&texts, &[])?, wide()?, &limits, fed.len())?;
        for (source, rows) in fed.into_iter().enumerate() {
            merger.fed(source, rows);
            merger.fed(source, Vec::new());
        }
        let made = merged(&mut merger, Vec::new(), 1)?;
        Ok(made.steps.into_iter().filter(|terms| !terms.is_empty()).collect())
    };
    assert_eq!(taken(EVERYTHING, 1)?, [["a"], ["b"], ["c"]]);
    assert_eq!(taken(4, EVERYTHING)?, [vec!["a"], vec!["b", "c"]]);
    assert_eq!(taken(EVERYTHING, 3 * size)?, [vec!["a", "b"], vec!["c"]]);
    assert_eq!(taken(EVERYTHING, EVERYTHING)?, [["a", "b", "c"]]);
    Ok(())
}

#[test]
fn a_merge_of_chunks_the_store_does_not_hold_or_of_damaged_lists_is_refused() -> Result<(), String> {
    let texts = [(0, "a"), (1, "a")];
    let unknown = Chunks {
        live: vec![1],
        lengths: vec![1],
        dead: Vec::new(),
        average: AVERAGE,
    };
    for limits in [WHOLE, PIECES] {
        let fed = given(&texts, &[0, 2], wide()?)?;
        let mut merger = Merger::new(unknown_like(&unknown), wide()?, &limits, fed.len())?;
        assert!(refused(
            merged(&mut merger, fed, 64).map(|_| ()),
            "chunks the store does not"
        ));
        let mut merger = Merger::new(chunks(&texts, &[])?, wide()?, &limits, 1)?;
        let damaged = Fed {
            term: Arc::from("a"),
            chunks: 2,
            postings: vec![0],
            positions: vec![0],
        };
        assert!(refused(
            merged(&mut merger, vec![vec![damaged]], 64).map(|_| ()),
            "shorter than its head"
        ));
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

fn unknown_like(held: &Chunks) -> Chunks {
    Chunks {
        live: held.live.clone(),
        lengths: held.lengths.clone(),
        dead: held.dead.clone(),
        average: held.average,
    }
}

#[test]
fn a_term_of_more_rows_than_a_merge_takes_at_once_merges_as_whole_as_in_pieces() -> Result<(), String> {
    let texts: Vec<(i64, &str)> = (0..7000).map(|place| (place, "w")).collect();
    let want = built(&texts, small()?)?;
    for limits in [WHOLE, PIECES] {
        let fed = given(&texts, &[0, 3000, 7000], small()?)?;
        let mut merger = Merger::new(chunks(&texts, &[])?, small()?, &limits, fed.len())?;
        let made = merged(&mut merger, fed, 4096)?;
        assert_eq!(shown(&made.rows, &made.pieces), shown(&want.rows, &[]));
    }
    Ok(())
}

#[test]
fn a_merge_takes_rows_of_the_sources_it_was_made_for_alone() -> Result<(), String> {
    let mut merger = Merger::new(chunks(&TEXTS, &[])?, wide()?, &WHOLE, 2)?;
    assert_eq!(merger.fed(2, Vec::new()), None);
    assert_eq!(merger.fed(1, Vec::new()), Some(false));
    Ok(())
}
