//! Times the forced-mate search on a fixed set of positions, on one thread.
//!
//! Usage: mate_bench <fen file> [reps] [depth,exhaustive,quiet ...]
//!        [--order A1,D1 or A10,D0,G4,V1 | --orders all] [--half tune|held|all]
//! (default: 5 reps of 5,0,0 2,2,0 3,0,1, the default move order, all positions).
//!
//! For each move order and setting it prints the mates found, the positions
//! searched (and how many of them lie where a mate is found), a checksum of
//! every answer, which must not change when the search is made faster or its
//! moves reordered, and the time per position over the reps (mean ± sd).
//! `--half tune` keeps the even-numbered positions and `held` the odd ones;
//! for a file of logged leaves (`game fen` per line), even and odd games.
//! With 0 reps it instead prints the index of every position with a mate.

use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::search::leaf_log::parse_corpus_line;
use kingfisher::search::mate_order::MoveOrder;
use kingfisher::search::mate_search::has_forced_mate_ordered;
use std::time::Instant;

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: mate_bench <fen file> [reps] [d,e,q ...] [--order A1,D1 or A10,D0,G4,V1 | --orders all] [--half tune|held|all]";
    let path = args.get(1).expect(usage);
    let positional: Vec<&String> = {
        let mut v = Vec::new();
        let mut i = 2;
        while i < args.len() {
            if args[i].starts_with("--") {
                i += 2;
            } else {
                v.push(&args[i]);
                i += 1;
            }
        }
        v
    };
    let reps: usize = positional.first().map_or(5, |r| r.parse().expect("reps"));
    let settings: Vec<(i32, i32, i32)> = if positional.len() > 1 {
        positional[1..]
            .iter()
            .map(|s| {
                let v: Vec<i32> = s.split(',').map(|x| x.parse().expect("d,e,q")).collect();
                (v[0], v[1], v[2])
            })
            .collect()
    } else {
        vec![(5, 0, 0), (2, 2, 0), (3, 0, 1)]
    };
    let orders: Vec<MoveOrder> = match (flag(&args, "--orders"), flag(&args, "--order")) {
        (Some("all"), _) => MoveOrder::all(),
        (Some(other), _) => panic!("--orders takes only \"all\", not {other:?}"),
        (None, Some(names)) => names
            .split(';')
            .map(|n| MoveOrder::parse(n).expect("--order like A1,D1 or A1,D1;A0,D0"))
            .collect(),
        (None, None) => vec![MoveOrder::DEFAULT],
    };
    let half = flag(&args, "--half").unwrap_or("all");
    let keep = |i: usize| match half {
        "tune" => i.is_multiple_of(2),
        "held" => !i.is_multiple_of(2),
        "all" => true,
        other => panic!("--half takes tune, held or all, not {other:?}"),
    };
    // A line may start with its game; then the halves split by game.
    let boards: Vec<Board> = std::fs::read_to_string(path)
        .expect("fen file")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(parse_corpus_line)
        .enumerate()
        .filter(|(i, (game, _))| keep(game.unwrap_or(*i)))
        .map(|(_, (_, fen))| Board::new_from_fen(fen))
        .collect();
    let mg = MoveGen::new();
    println!(
        "{} positions, {reps} reps; time per position in microseconds",
        boards.len()
    );
    if reps == 0 {
        for (depth, exhaustive, quiet) in settings {
            for (i, b) in boards.iter().enumerate() {
                if has_forced_mate_ordered(b, &mg, depth, exhaustive, quiet, orders[0]).0 {
                    println!("{depth},{exhaustive},{quiet} {i}");
                }
            }
        }
        return;
    }
    for order in orders {
        for &(depth, exhaustive, quiet) in &settings {
            let (mut mates, mut nodes, mut checksum) = (0u64, 0u64, 0u64);
            let mut nodes_in_mates = 0u64;
            let mut times = Vec::new();
            for rep in 0..reps {
                let start = Instant::now();
                for (i, b) in boards.iter().enumerate() {
                    let (found, n) =
                        has_forced_mate_ordered(b, &mg, depth, exhaustive, quiet, order);
                    if rep == 0 {
                        mates += found as u64;
                        nodes += n as u64;
                        if found {
                            nodes_in_mates += n as u64;
                            checksum = checksum.wrapping_mul(31).wrapping_add(i as u64 + 1);
                        }
                    }
                }
                times.push(start.elapsed().as_secs_f64() * 1e6 / boards.len() as f64);
            }
            let mean = times.iter().sum::<f64>() / reps as f64;
            let sd = if reps > 1 {
                (times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (reps - 1) as f64).sqrt()
            } else {
                0.0
            };
            println!(
                "order {} depth {depth} exhaustive {exhaustive} quiet {quiet}: {mates} mates, {nodes} positions searched ({nodes_in_mates} where a mate is found), checksum {checksum:016x}, {mean:.3} ± {sd:.3} us",
                order.name()
            );
        }
    }
}
