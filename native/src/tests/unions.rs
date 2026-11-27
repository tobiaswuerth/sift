use crate::tests::support::refused;
use crate::unions::summed;

#[test]
fn the_passages_of_many_lists_come_back_once_each_in_order_with_their_counts_summed() -> Result<(), String> {
    let found = summed(&[7, 3, 9, 3, 7, 3], &[1, 2, 4, 8, 16, 32])?;
    assert_eq!(found, (vec![3, 7, 9], vec![42, 17, 4]));
    assert_eq!(summed(&[], &[])?, (vec![], vec![]));
    assert_eq!(summed(&[5, 5], &[i64::MAX, 1])?, (vec![5], vec![i64::MAX]));
    assert!(refused(summed(&[1, 2], &[1]), "a count for each"));
    Ok(())
}

#[test]
fn many_passages_are_summed_as_few() -> Result<(), String> {
    let ids: Vec<i64> = (0..100_000).map(|each| (each * 7919) % 30_011).collect();
    let counts: Vec<i64> = (0..100_000).map(|each| each % 5 + 1).collect();
    let (found, sums) = summed(&ids, &counts)?;
    let mut wanted = vec![0i64; 30_011];
    for (&id, &count) in ids.iter().zip(&counts) {
        wanted[usize::try_from(id).map_err(|error| error.to_string())?] += count;
    }
    assert_eq!(found, (0..30_011).collect::<Vec<i64>>());
    assert_eq!(sums, wanted);
    Ok(())
}
