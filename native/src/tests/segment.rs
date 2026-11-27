use crate::packed::{PACKED, unwrapped};
use crate::postings::Kept;
use crate::segment::{Held, Passages, lengths_written};
use crate::tests::support::{AVERAGE, built, decoded, refused, row, small, table, wide, words};
use crate::words::Table;

const MOST_TIMES: usize = 1024;

fn lengths(packed: &[u8]) -> Result<Vec<u32>, String> {
    let wrapped: Vec<u8> = [PACKED].into_iter().chain(packed.iter().copied()).collect();
    let raw = unwrapped(&wrapped)?;
    Ok(raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|each| u32::from_le_bytes(*each))
        .collect())
}

#[test]
fn a_table_of_words_holds_every_point_and_a_text_is_cut_and_folded_by_it() -> Result<(), String> {
    assert!(refused(Table::new(&[0; 7]).map(|_| ()), "four bytes for each"));
    assert_eq!(words("Don't stop, M\u{fc}ller!")?, ["dont", "stop", "m\u{fc}ller"]);
    assert_eq!(words(" ,; ")?, Vec::<String>::new());
    Ok(())
}

#[test]
fn a_segment_holds_where_each_term_occurs_and_how_long_each_passage_is() -> Result<(), String> {
    let made = built(&[(7, "one two two"), (9, "two three")], wide()?)?;
    let terms: Vec<&str> = made.rows.iter().map(|row| &*row.term).collect();
    assert_eq!(terms, ["one", "three", "two"]);
    assert_eq!(decoded(row(&made, "two")?)?, (vec![7, 9], vec![2, 1], vec![2, 3, 1]));
    assert_eq!(row(&made, "two")?.chunks, 2);
    let [(block, packed)] = &made.lengths[..] else {
        return Err("passages 7 and 9 lie in one block of lengths".to_owned());
    };
    assert_eq!((*block, &lengths(packed)?[7..10]), (0, &[3, 0, 2][..]));
    Ok(())
}

#[test]
fn a_term_of_many_passages_is_kept_in_blocks() -> Result<(), String> {
    let made = built(&[(1, "w"), (2, "w"), (5, "w x"), (6, "w"), (9, "w")], small()?)?;
    assert!(matches!(row(&made, "x")?.held, Kept::Short(_)));
    let Kept::Long { blocks, .. } = &row(&made, "w")?.held else {
        return Err("a term of five passages is long in a shape of three".to_owned());
    };
    assert_eq!(blocks.len(), 3);
    assert_eq!(decoded(row(&made, "w")?)?.0, [1, 2, 5, 6, 9]);
    Ok(())
}

#[test]
fn a_segment_takes_the_words_of_title_and_fields_around_those_of_the_text() -> Result<(), String> {
    let mut held = Held::new(usize::MAX, 9);
    let passages = Passages {
        places: &[3, 4],
        of: &[0, 0],
        fields: &[("Title words cut here", "meta")],
        texts: &["body", "more body"],
    };
    assert_eq!(held.taken(table()?, &passages)?, 2);
    assert_eq!(
        (held.chunks(), held.first(), held.last(), held.tokens()),
        (2, Some(3), Some(4), 7)
    );
    let made = held.written(AVERAGE, wide()?)?;
    assert_eq!(decoded(row(&made, "title")?)?, (vec![3, 4], vec![1, 1], vec![0, 0]));
    assert_eq!(decoded(row(&made, "body")?)?, (vec![3, 4], vec![1, 1], vec![2, 3]));
    assert_eq!(decoded(row(&made, "meta")?)?, (vec![3, 4], vec![1, 1], vec![4, 5]));
    assert!(row(&made, "words").is_err());
    Ok(())
}

#[test]
fn a_segment_takes_passages_until_it_holds_the_tokens_asked_and_a_word_no_more_often_than_a_list_holds()
-> Result<(), String> {
    let mut held = Held::new(3, usize::MAX);
    let often = vec!["x"; MOST_TIMES + 6].join(" ");
    let passages = Passages {
        places: &[1, 2, 3],
        of: &[0, 0, 0],
        fields: &[("", "")],
        texts: &[&often, "y", "z"],
    };
    assert_eq!(held.taken(table()?, &passages)?, 1);
    assert_eq!(held.taken(table()?, &passages)?, 0);
    let made = held.written(AVERAGE, wide()?)?;
    assert_eq!(decoded(row(&made, "x")?)?.1, [1024]);
    Ok(())
}

#[test]
fn a_segment_refuses_what_it_was_not_given_whole() -> Result<(), String> {
    let mut held = Held::new(usize::MAX, usize::MAX);
    let mut passages = Passages {
        places: &[1],
        of: &[1],
        fields: &[("", "")],
        texts: &["a"],
    };
    assert!(refused(
        held.taken(table()?, &passages),
        "a document that was not given"
    ));
    passages.of = &[];
    assert!(refused(
        held.taken(table()?, &passages),
        "a document that was not given"
    ));
    passages.texts = &["a", "b"];
    assert!(refused(held.taken(table()?, &passages), "a place and a text"));
    assert!(refused(
        held.written(AVERAGE, wide()?).map(|_| ()),
        "holds a chunk at least"
    ));
    Ok(())
}

#[test]
fn lengths_are_kept_by_id_in_blocks_of_consecutive_ids() -> Result<(), String> {
    let found = lengths_written(&[1, 6, 9, -1], &[4, 258, 7, 5], 3)?;
    let blocks: Vec<i64> = found.iter().map(|each| each.0).collect();
    assert_eq!(blocks, [0, 1, -1]);
    assert_eq!(lengths(&found[0].1)?, [0, 4, 0, 0, 0, 0, 258, 0]);
    assert_eq!(lengths(&found[1].1)?, [0, 7, 0, 0, 0, 0, 0, 0]);
    assert_eq!(lengths(&found[2].1)?, [0, 0, 0, 0, 0, 0, 0, 5]);
    assert!(refused(lengths_written(&[1], &[], 3), "each chunk has a length"));
    assert!(refused(lengths_written(&[1], &[1], 21), "each chunk has a length"));
    Ok(())
}
