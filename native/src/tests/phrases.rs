use crate::phrases::{Word, phrased};
use crate::tests::support::refused;

fn word<'a>(ids: &'a [i64], counts: &'a [i64], positions: &'a [i64]) -> Word<'a> {
    Word { ids, counts, positions }
}

#[test]
fn a_phrase_is_counted_in_each_candidate_where_its_words_follow_one_another() -> Result<(), String> {
    let first = word(&[10, 20, 30], &[2, 1, 1], &[0, 5, 3, 7]);
    let second = word(&[10, 30], &[2, 1], &[1, 6, 9]);
    assert_eq!(phrased(&[10, 20, 30], &[first, second])?, [2, 0, 0]);
    let again = word(&[10, 20], &[3, 1], &[0, 2, 4, 0]);
    let between = word(&[20, 10], &[1, 2], &[1, 1, 3]);
    let told = [word(again.ids, again.counts, again.positions), between, again];
    assert_eq!(phrased(&[10, 20, 30], &told)?, [2, 0, 0]);
    let alone = word(&[30, 10, 15], &[1, 2, 1], &[4, 1, 2, 9]);
    assert_eq!(phrased(&[5, 10, 30], &[alone])?, [0, 2, 1]);
    assert_eq!(phrased(&[], &[word(&[7], &[1], &[0])])?, Vec::<i64>::new());
    Ok(())
}

#[test]
fn a_candidate_told_twice_counts_once_and_a_position_past_the_last_follows_none() -> Result<(), String> {
    let first = word(&[10], &[2], &[3, i64::MAX]);
    let second = word(&[10], &[1], &[4]);
    assert_eq!(phrased(&[10, 10, 20], &[first, second])?, [1, 0, 0]);
    Ok(())
}

#[test]
fn many_candidates_are_counted_as_each_alone() -> Result<(), String> {
    let candidates: Vec<i64> = (0..5000).map(|each| each * 3).collect();
    let ones = vec![1i64; candidates.len()];
    let starts: Vec<i64> = (0..5000).collect();
    let next: Vec<i64> = (0..5000).map(|each| each + 1 + each % 2).collect();
    let found = phrased(
        &candidates,
        &[word(&candidates, &ones, &starts), word(&candidates, &ones, &next)],
    )?;
    let wanted: Vec<i64> = (0..5000).map(|each| 1 - each % 2).collect();
    assert_eq!(found, wanted);
    Ok(())
}

#[test]
fn words_told_with_other_positions_than_their_counts_or_no_word_at_all_are_refused() {
    let uneven = [
        word(&[10], &[2], &[1]),
        word(&[10], &[1], &[1, 2]),
        word(&[10], &[-1], &[1]),
        word(&[10, 20], &[1], &[1]),
    ];
    for told in uneven {
        assert!(refused(phrased(&[10], &[told]), "other positions than their counts"));
    }
    assert!(refused(phrased(&[10], &[]), "a phrase of no word"));
}
