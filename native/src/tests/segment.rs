use crate::packed::{PACKED, unwrapped};
use crate::segment::{Held, Passages, lengths_read, lengths_written};
use crate::tests::support::{
    AVERAGE, built, counts, decoded, long, refused, shown, small, table, taken, terms, wide, words,
};
use crate::words::Table;

const MOST_TIMES: usize = 1024;
const PACK_BYTES: usize = 4000;

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
    assert_eq!(terms(&made.filed)?, ["one", "three", "two"]);
    assert_eq!(
        counts(&made.filed)?,
        [("one".to_owned(), 1), ("three".to_owned(), 1), ("two".to_owned(), 2)]
    );
    assert_eq!(
        decoded(&made.filed, "two")?,
        (2, (vec![7, 9], vec![2, 1], vec![2, 3, 1]))
    );
    let [(block, packed)] = &made.lengths[..] else {
        return Err("passages 7 and 9 lie in one block of lengths".to_owned());
    };
    assert_eq!((*block, &lengths(packed)?[7..10]), (0, &[3, 0, 2][..]));
    Ok(())
}

#[test]
fn a_term_of_many_passages_is_kept_in_blocks_under_its_ranges() -> Result<(), String> {
    let made = built(&[(1, "w"), (2, "w"), (5, "w x"), (6, "w"), (9, "w")], small()?)?;
    assert_eq!((long(&made.filed, "x"), long(&made.filed, "w")), (0, 3));
    assert_eq!(
        made.filed
            .ranged
            .iter()
            .map(|each| (&*each.term, each.chunks))
            .collect::<Vec<_>>(),
        [("w", 5)]
    );
    assert_eq!(decoded(&made.filed, "w")?.1.0, [1, 2, 5, 6, 9]);
    assert_eq!(decoded(&made.filed, "x")?, (1, (vec![5], vec![1], vec![2])));
    Ok(())
}

#[test]
fn the_short_lists_of_a_segment_lie_many_to_a_pack_none_larger_than_a_page() -> Result<(), String> {
    let body = |place: i64| {
        (0..50)
            .map(|word| format!("w{place}x{word}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let texts: Vec<(i64, String)> = (0..40).map(|place| (place, body(place))).collect();
    let borrowed: Vec<(i64, &str)> = texts.iter().map(|each| (each.0, &*each.1)).collect();
    let made = built(&borrowed, wide()?)?;
    let held: Vec<usize> = made
        .filed
        .packs
        .iter()
        .map(|pack| pack.first.len() + pack.lists.len() + pack.places.len())
        .collect();
    assert_eq!(terms(&made.filed)?.len(), 2000);
    assert!(held.iter().all(|&each| each <= PACK_BYTES));
    assert!(held.len() < 2000 / 50 && held.iter().sum::<usize>() > (held.len() - 1) * (PACK_BYTES - 100));
    Ok(())
}

fn varied(count: i64) -> Vec<(i64, String)> {
    let words = |place: i64| (0..place % 17 + 3).map(move |word| format!("w{}", (place * 31 + word * word) % 97));
    (0..count)
        .map(|place| (place, words(place).collect::<Vec<_>>().join(" ")))
        .collect()
}

#[test]
fn a_segment_fed_its_passages_in_any_portions_holds_the_same() -> Result<(), String> {
    let texts = varied(500);
    let borrowed: Vec<(i64, &str)> = texts.iter().map(|each| (each.0, &*each.1)).collect();
    let whole = built(&borrowed, small()?)?;
    for portion in [1, 7, 64, 200] {
        let mut held = Held::new(usize::MAX, usize::MAX);
        for part in borrowed.chunks(portion) {
            assert_eq!(taken(&mut held, part)?, part.len());
        }
        let made = held.written(AVERAGE, small()?)?;
        assert_eq!(shown(&made.filed)?, shown(&whole.filed)?);
    }
    Ok(())
}

#[test]
fn a_segment_given_more_passages_than_its_tokens_allow_takes_those_before_it_was_full() -> Result<(), String> {
    let (texts, most) = (varied(300), 1000);
    let borrowed: Vec<(i64, &str)> = texts.iter().map(|each| (each.0, &*each.1)).collect();
    let (mut streamed, mut fitting) = (0, 0);
    for (place, _) in &borrowed {
        if streamed < most {
            streamed += usize::try_from(place % 17 + 3).unwrap_or(0) + 2;
            fitting += 1;
        }
    }
    let mut held = Held::new(most, usize::MAX);
    assert_eq!(taken(&mut held, &borrowed)?, fitting);
    assert!(0 < fitting && fitting < borrowed.len());
    let (made, want) = (held.written(AVERAGE, small()?)?, built(&borrowed[..fitting], small()?)?);
    assert_eq!(shown(&made.filed)?, shown(&want.filed)?);
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
    assert_eq!(decoded(&made.filed, "title")?.1, (vec![3, 4], vec![1, 1], vec![0, 0]));
    assert_eq!(decoded(&made.filed, "body")?.1, (vec![3, 4], vec![1, 1], vec![2, 3]));
    assert_eq!(decoded(&made.filed, "meta")?.1, (vec![3, 4], vec![1, 1], vec![4, 5]));
    assert!(decoded(&made.filed, "words").is_err());
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
    assert_eq!(decoded(&made.filed, "x")?.1.1, [1024]);
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

#[test]
fn the_lengths_of_many_blocks_are_read_at_once_and_a_block_of_another_size_is_refused() -> Result<(), String> {
    let ids: Vec<i64> = (0..800).collect();
    let kept: Vec<u32> = (0..800).map(|each| each * 3 + 1).collect();
    let written = lengths_written(&ids, &kept, 3)?;
    let rows: Vec<&[u8]> = written.iter().map(|each| &each.1[..]).collect();
    let read = lengths_read(&rows, 3)?;
    let told: Vec<u32> = read
        .as_chunks::<4>()
        .0
        .iter()
        .map(|each| u32::from_le_bytes(*each))
        .collect();
    assert_eq!((rows.len(), told), (100, kept));
    assert_eq!(lengths_read(&rows[98..], 3)?, read[98 * 32..]);
    assert_eq!(lengths_read(&rows[..0], 3)?, Vec::<u8>::new());
    for (shift, some) in [(4, &rows[..]), (2, &rows[..]), (4, &rows[..1]), (2, &rows[..1])] {
        assert!(refused(lengths_read(some, shift), "unpacks to other than"));
    }
    let cut = &rows[0][..rows[0].len() - 4];
    assert!(refused(lengths_read(&[cut], 3), "unpacks to other than 32 bytes"));
    let longer: Vec<u8> = rows[0].iter().copied().chain([0]).collect();
    assert!(refused(lengths_read(&[longer], 3), "bytes after its end"));
    assert!(refused(lengths_read(&[b"junk"], 3), "does not unpack"));
    assert!(refused(lengths_read(&rows, 21), "blocks of at most 2**20"));
    Ok(())
}

#[test]
fn a_segment_takes_passages_at_rising_places_alone_and_none_of_a_batch_it_refuses() -> Result<(), String> {
    let mut held = Held::new(usize::MAX, usize::MAX);
    assert_eq!(taken(&mut held, &[(3, "one"), (5, "two")])?, 2);
    for batch in [[(9, "x"), (9, "y")], [(9, "x"), (4, "y")], [(5, "x"), (9, "y")]] {
        assert!(refused(taken(&mut held, &batch), "at rising places"));
        assert_eq!((held.chunks(), held.last()), (2, Some(5)));
    }
    let passages = Passages {
        places: &[7, 8],
        of: &[0, 5],
        fields: &[("", "")],
        texts: &["a", "b"],
    };
    assert!(refused(
        held.taken(table()?, &passages),
        "a document that was not given"
    ));
    assert_eq!((held.chunks(), held.tokens()), (2, 2));
    assert_eq!(taken(&mut held, &[(7, "three")])?, 1);
    Ok(())
}
