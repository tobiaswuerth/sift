use crate::postings::TOP;
use crate::scoring::{Expression, Node, may_reach, unquantized, unreached_alone};
use crate::tests::support::refused;

fn leaves(count: usize) -> Vec<Node> {
    (0..count).map(Node::Leaf).collect()
}

fn expression(mut nodes: Vec<Node>, root: Node, count: usize) -> Result<Expression, String> {
    nodes.push(root);
    Expression::new(nodes, count)
}

#[test]
fn a_query_whose_parts_are_not_each_under_one_other_or_leaves_it_holds_is_refused() {
    let told = "not each under one other";
    assert!(refused(Expression::new(Vec::new(), 1), told));
    assert!(refused(Expression::new(vec![Node::Leaf(1)], 1), told));
    assert!(refused(Expression::new(vec![Node::Or(Vec::new())], 1), told));
    assert!(refused(
        Expression::new(vec![Node::Leaf(0), Node::Or(vec![1])], 1),
        told
    ));
    assert!(refused(
        Expression::new(vec![Node::Leaf(0), Node::Not([0, 0])], 1),
        told
    ));
    assert!(refused(Expression::new(vec![Node::Leaf(0), Node::Leaf(0)], 1), told));
    assert!(refused(
        Expression::new(vec![Node::Leaf(0), Node::And(vec![0]), Node::Or(vec![0, 1])], 1),
        told
    ));
}

#[test]
fn a_query_of_leaves_alone_is_flat_or_a_conjunction_and_one_nested_is_neither() -> Result<(), String> {
    let flat = expression(leaves(2), Node::Or(vec![0, 1]), 2)?;
    let conjunction = expression(leaves(2), Node::And(vec![0, 1]), 2)?;
    let alone = Expression::new(leaves(1), 1)?;
    let nested = expression(
        vec![Node::Leaf(0), Node::Leaf(1), Node::Not([0, 1]), Node::Leaf(0)],
        Node::Or(vec![2, 3]),
        2,
    )?;
    let found: Vec<(bool, bool)> = [&flat, &conjunction, &alone, &nested]
        .iter()
        .map(|each| (each.flat(), each.conjunction()))
        .collect();
    assert_eq!(found, [(true, false), (false, true), (false, false), (false, false)]);
    Ok(())
}

#[test]
fn a_query_may_hold_where_its_kept_parts_may_and_counts_only_the_leaves_that_hold() -> Result<(), String> {
    let present = [
        vec![true, true, false, false],
        vec![true, false, true, false],
        vec![true, true, true, true],
    ];
    let nodes = vec![
        Node::Leaf(0),
        Node::Leaf(1),
        Node::And(vec![0, 1]),
        Node::Leaf(2),
        Node::Leaf(1),
        Node::Not([3, 4]),
    ];
    let told = expression(nodes, Node::Or(vec![2, 5]), 3)?;
    assert_eq!(told.may_hold(&present), [true, true, true, true]);
    let (holds, counting) = told.counted(&present);
    assert_eq!(holds, [true, true, false, true]);
    let want = [
        (0, vec![true, false, false, false]),
        (1, vec![true, false, false, false]),
        (2, vec![false, true, false, true]),
    ];
    assert_eq!(counting, want);
    let all = expression(leaves(3), Node::And(vec![0, 1, 2]), 3)?;
    assert_eq!(all.may_hold(&present), [true, false, false, false]);
    assert_eq!(Expression::new(leaves(1), 1)?.may_hold(&[]), Vec::<bool>::new());
    Ok(())
}

#[test]
fn a_bound_at_the_threshold_or_below_it_by_the_order_of_a_sum_alone_may_reach_it() {
    let threshold = (0.1 + 0.2) + 0.3;
    let summed = 0.1 + (0.2 + 0.3);
    assert!(summed < threshold);
    assert!(may_reach(threshold, threshold) && may_reach(summed, threshold));
    assert!(!may_reach(threshold * (1.0 - 1e-6), threshold));
    assert_eq!((unquantized(0), unquantized(255)), (0.0, TOP));
}

#[test]
fn lazy_leaves_are_the_smallest_whose_sum_stays_below_the_threshold() {
    let told: [(&[f64], f64, usize); 5] = [
        (&[], 5.0, 0),
        (&[1.0, 2.0], 0.0, 0),
        (&[1.0, 2.0, 4.0], 3.5, 2),
        (&[1.0, 2.0], 10.0, 2),
        (&[3.0], 3.0, 0),
    ];
    for (maxima, threshold, lazy) in told {
        assert_eq!(unreached_alone(maxima, threshold), lazy);
    }
    let mut seed = 7u64;
    let mut drawn = || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        f64::from(u32::try_from(seed >> 40).unwrap_or(0)) / f64::from(1u32 << 24)
    };
    for _ in 0..2000 {
        let count = usize::try_from(drawn().to_bits() % 9).unwrap_or(0);
        let mut ascending: Vec<f64> = (0..count).map(|_| 100.0 * drawn()).collect();
        ascending.sort_by(f64::total_cmp);
        let threshold = 300.0 * drawn();
        let lazy = unreached_alone(&ascending, threshold);
        let sum = |upto: usize| ascending[..upto].iter().fold(0.0, |total, most| total + most);
        assert!(lazy == 0 || !may_reach(sum(lazy), threshold));
        assert!(lazy == ascending.len() || may_reach(sum(lazy + 1), threshold));
    }
}
