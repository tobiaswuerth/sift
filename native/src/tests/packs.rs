use crate::packs::{Pack, Packer, counted, entries};
use crate::tests::support::refused;

const PACK_BYTES: usize = 4000;

type Entered = (String, usize, Vec<u8>, Vec<u8>);

fn told(pack: &Pack, placed: bool) -> Result<Vec<Entered>, String> {
    let found = entries(&pack.lists, placed.then_some(&pack.places[..]), None)?;
    let shown = found.iter().map(|each| {
        let term = each.term.to_owned();
        (term, each.chunks, each.postings.to_vec(), each.positions.to_vec())
    });
    Ok(shown.collect())
}

fn held(pack: &Pack) -> usize {
    pack.first.len() + pack.lists.len() + pack.places.len()
}

#[test]
fn lists_packed_together_come_back_each_under_its_term_with_or_without_their_positions() -> Result<(), String> {
    let mut packer = Packer::default();
    assert!(packer.push("alpha", 2, Some((&[0, 7, 7], &[0, 9])))?.is_none());
    assert!(packer.push("beta", 5000, None)?.is_none());
    assert!(packer.push("m\u{fc}ller", 1, Some((&[1, 3], &[1, 4, 4, 4])))?.is_none());
    let Some(pack) = packer.finished()? else {
        return Err("a packer that took lists hands back a pack".to_owned());
    };
    assert_eq!(&*pack.first, "alpha");
    assert_eq!(
        told(&pack, true)?,
        [
            ("alpha".to_owned(), 2, vec![0, 7, 7], vec![0, 9]),
            ("beta".to_owned(), 5000, vec![], vec![]),
            ("m\u{fc}ller".to_owned(), 1, vec![1, 3], vec![1, 4, 4, 4]),
        ]
    );
    let unplaced = told(&pack, false)?;
    assert_eq!(unplaced[2], ("m\u{fc}ller".to_owned(), 1, vec![1, 3], vec![]));
    assert!(packer.finished()?.is_none());
    Ok(())
}

#[test]
fn a_pack_is_closed_before_it_would_hold_more_than_a_page_and_the_next_begins_with_the_next_term() -> Result<(), String>
{
    let (mut packer, mut packs) = (Packer::default(), Vec::new());
    let terms: Vec<String> = (0..600).map(|number| format!("term{number:04}")).collect();
    for term in &terms {
        packs.extend(packer.push(term, 3, Some((&[7; 40], &[9; 20])))?);
    }
    let closed = packs.len();
    packs.extend(packer.finished()?);
    assert!(closed > 5 && packs.len() == closed + 1);
    let mut back = Vec::new();
    for (number, pack) in packs.iter().enumerate() {
        assert!(held(pack) <= PACK_BYTES, "{}", held(pack));
        assert!(number == closed || held(pack) > PACK_BYTES - 100, "{}", held(pack));
        assert_eq!(Some(&*pack.first), told(pack, true)?.first().map(|each| &*each.0));
        back.extend(told(pack, true)?.into_iter().map(|each| each.0));
    }
    assert_eq!(back, terms);
    Ok(())
}

#[test]
fn a_list_larger_than_a_page_is_a_pack_of_its_own() -> Result<(), String> {
    let mut packer = Packer::default();
    assert!(packer.push("a", 1, Some((&[1; 10], &[2; 10])))?.is_none());
    let before = packer.push("b", 9, Some((&[3; PACK_BYTES], &[4; 10])))?;
    let large = packer.push("c", 1, Some((&[5; 10], &[6; 10])))?;
    let (Some(before), Some(large), Some(after)) = (before, large, packer.finished()?) else {
        return Err("a large list closes the pack before it and the one it is in".to_owned());
    };
    assert_eq!((&*before.first, &*large.first, &*after.first), ("a", "b", "c"));
    assert_eq!((told(&large, true)?.len(), held(&large) > PACK_BYTES), (1, true));
    Ok(())
}

#[test]
fn a_damaged_pack_is_refused() -> Result<(), String> {
    let mut packer = Packer::default();
    packer.push("alpha", 2, Some((&[0, 7, 7], &[0, 9])))?;
    packer.push("beta", 1, Some((&[0, 1], &[0, 2])))?;
    let Some(pack) = packer.finished()? else {
        return Err("a packer that took lists hands back a pack".to_owned());
    };
    for cut in 0..pack.lists.len() {
        let found = entries(&pack.lists[..cut], Some(&pack.places), None);
        assert!(refused(found, "a pack of lists"), "{cut}");
    }
    assert!(refused(
        entries(&pack.lists, Some(&pack.places[..3]), None),
        "a pack of lists"
    ));
    for (changed, more) in [(5, 0xFF), (4, 0x89)] {
        let mut named = pack.lists.clone();
        named[changed] = more;
        assert!(refused(entries(&named, None, None), "a pack of lists"));
    }
    let mut endless = pack.lists[..4].to_vec();
    endless.extend([0x80; 12]);
    assert!(refused(entries(&endless, None, None), "a pack of lists"));
    let (mut longer, mut placed) = (pack.lists.clone(), pack.places.clone());
    longer.push(0);
    placed.push(0);
    assert!(refused(entries(&longer, Some(&pack.places), None), "a pack of lists"));
    assert!(refused(entries(&pack.lists, Some(&placed), None), "a pack of lists"));
    Ok(())
}

#[test]
fn the_terms_of_a_pack_within_bounds_are_told_with_their_passages_and_a_pack_cut_short_is_refused() -> Result<(), String>
{
    let mut packer = Packer::default();
    packer.push("alpha", 2, Some((&[0, 7, 7], &[0, 9])))?;
    packer.push("beta", 5000, None)?;
    packer.push("betz", 4, Some((&[1], &[2])))?;
    packer.push("m\u{fc}ller", 1, Some((&[1, 3], &[1, 4, 4, 4])))?;
    let Some(pack) = packer.finished()? else {
        return Err("a packer that took lists hands back a pack".to_owned());
    };
    let every = [("alpha", 2), ("beta", 5000), ("betz", 4), ("m\u{fc}ller", 1)];
    assert_eq!(counted(&pack.lists, None, "", None)?, every);
    assert_eq!(counted(&pack.lists, None, "bet", Some("beu"))?, every[1..3]);
    assert_eq!(counted(&pack.lists, None, "beta", Some("betz"))?, every[1..2]);
    assert_eq!(counted(&pack.lists, None, "m", Some("n"))?, every[3..]);
    assert_eq!(counted(&pack.lists, None, "n", None)?, every[..0]);
    assert_eq!(counted(&pack.lists, None, "a", Some("alpha"))?, every[..0]);
    for cut in 0..pack.lists.len() {
        assert!(
            refused(counted(&pack.lists[..cut], None, "", None), "a pack of lists"),
            "{cut}"
        );
    }
    let mut longer = pack.lists.clone();
    longer.push(0);
    assert!(refused(counted(&longer, None, "", None), "a pack of lists"));
    Ok(())
}

type Crafted<'a> = (&'a str, u8, &'a [u8], &'a [u8]);

fn made(held: &[Crafted]) -> (Vec<u8>, Vec<u8>) {
    let short = |size: usize| u8::try_from(size).unwrap_or(u8::MAX);
    let count = u32::from(short(held.len()));
    let (mut lists, mut places) = (count.to_le_bytes().to_vec(), Vec::new());
    for (term, chunks, postings, positions) in held {
        lists.push(short(term.len()));
        lists.extend_from_slice(term.as_bytes());
        lists.extend([*chunks, short(postings.len()), short(positions.len())]);
    }
    for (_, _, postings, positions) in held {
        lists.extend_from_slice(postings);
        places.extend_from_slice(positions);
    }
    (lists, places)
}

#[test]
fn a_pack_whose_terms_do_not_rise_or_that_holds_no_list_is_refused() -> Result<(), String> {
    let rising = made(&[("a", 1, &[1], &[2]), ("b", 1, &[3], &[4])]);
    assert_eq!(counted(&rising.0, None, "", None)?, [("a", 1), ("b", 1)]);
    assert_eq!(entries(&rising.0, Some(&rising.1), None)?.len(), 2);
    for terms in [["b", "a"], ["a", "a"]] {
        let (lists, places) = made(&[(terms[0], 1, &[1], &[2]), (terms[1], 1, &[3], &[4])]);
        assert!(refused(entries(&lists, Some(&places), None), "a pack of lists"));
        assert!(refused(counted(&lists, None, "", None), "a pack of lists"));
    }
    assert!(refused(counted(&[0, 0, 0, 0], None, "", None), "a pack of lists"));
    assert!(refused(entries(&[0, 0, 0, 0], None, None), "a pack of lists"));
    assert_eq!(counted(&[1, 0, 0, 0, 1, b'a', 1, 0, 0], None, "", None)?, [("a", 1)]);
    assert!(refused(
        counted(&rising.0, Some("b"), "", None),
        "filed under another term"
    ));
    assert!(refused(entries(&rising.0, None, Some("")), "filed under another term"));
    assert_eq!(counted(&rising.0, Some("a"), "b", None)?, [("b", 1)]);
    let longer_than_it_is = [1, 0, 0, 0, 0x81, 0, b'a', 1, 0, 0];
    assert!(refused(counted(&longer_than_it_is, None, "", None), "a pack of lists"));
    Ok(())
}

#[test]
fn a_packer_takes_terms_in_their_order_alone_across_its_packs() -> Result<(), String> {
    let mut packer = Packer::default();
    packer.push("b", 1, Some((&[1], &[2])))?;
    for term in ["a", "b"] {
        assert!(refused(packer.push(term, 1, None), "in the order of their terms"));
    }
    assert!(packer.finished()?.is_some());
    assert!(refused(packer.push("b", 1, None), "in the order of their terms"));
    assert!(packer.push("c", 1, None)?.is_none());
    Ok(())
}
