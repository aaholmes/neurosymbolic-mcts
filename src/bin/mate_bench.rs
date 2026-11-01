//! Times the forced-mate search on a fixed set of positions, on one thread.
//!
//! Usage: mate_bench <fen file> [reps] [depth,exhaustive,quiet ...]
//! (default: 5 reps of 5,0,0 2,2,0 3,0,1).
//!
//! For each setting it prints the mates found, the positions searched, and a
//! checksum of every answer, which must not change when the search is made
//! faster, then the time per position over the reps (mean ± sd). With 0 reps
//! it instead prints the index of every position where a mate is found.

use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::search::mate_search::has_forced_mate_quiet;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: mate_bench <fen file> [reps] [d,e,q ...]");
    let reps: usize = args.get(2).map_or(5, |r| r.parse().expect("reps"));
    let settings: Vec<(i32, i32, i32)> = if args.len() > 3 {
        args[3..]
            .iter()
            .map(|s| {
                let v: Vec<i32> = s.split(',').map(|x| x.parse().expect("d,e,q")).collect();
                (v[0], v[1], v[2])
            })
            .collect()
    } else {
        vec![(5, 0, 0), (2, 2, 0), (3, 0, 1)]
    };
    let boards: Vec<Board> = std::fs::read_to_string(path)
        .expect("fen file")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(Board::new_from_fen)
        .collect();
    let mg = MoveGen::new();
    println!(
        "{} positions, {reps} reps; time per position in microseconds",
        boards.len()
    );
    if reps == 0 {
        for (depth, exhaustive, quiet) in settings {
            for (i, b) in boards.iter().enumerate() {
                if has_forced_mate_quiet(b, &mg, depth, exhaustive, quiet).0 {
                    println!("{depth},{exhaustive},{quiet} {i}");
                }
            }
        }
        return;
    }
    for (depth, exhaustive, quiet) in settings {
        let (mut mates, mut nodes, mut checksum) = (0u64, 0u64, 0u64);
        let mut nodes_in_mates = 0u64;
        let mut times = Vec::new();
        for rep in 0..reps {
            let start = Instant::now();
            for (i, b) in boards.iter().enumerate() {
                let (found, n) = has_forced_mate_quiet(b, &mg, depth, exhaustive, quiet);
                if rep == 0 {
                    mates += found as u64;
                    nodes += n as u64;
                    if found {
                        nodes_in_mates += n as u64;
                    }
                    if found {
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
            "depth {depth} exhaustive {exhaustive} quiet {quiet}: {mates} mates, {nodes} positions searched ({nodes_in_mates} where a mate is found), checksum {checksum:016x}, {mean:.3} ± {sd:.3} us"
        );
    }
}
