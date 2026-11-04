//! Times the extended quiescence search on logged self-play leaves, on one thread.
//!
//! Usage: qsearch_bench <leaf file> [reps] [--half tune|held|all] [--impl new|ref|new,ref]
//!        [--dump <file>]
//! (default: 5 reps, all positions, the current search only).
//!
//! For each search it prints the time per call over the reps (mean ± sd), the
//! median, 99th percentile and largest time of a single call (each call's median
//! over the reps), the positions searched per call, how many calls finished
//! within the depth limit, and a checksum of every answer (score and completed
//! flag), which must not change when the search is made faster without changing
//! its results. With both searches it also reports how often they agree.
//! `--half tune` keeps the even-numbered games and `held` the odd ones (lines are
//! `game fen`, as logged with KF_LOG_MATE_LEAVES). `--dump` writes, per position,
//! the score, completed flag, positions searched and median microseconds.

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::search::leaf_log::parse_corpus_line;
use kingfisher::search::qsearch_bench::{answer_checksum, percentile};
use kingfisher::search::quiescence::forced_ext_pesto_balance_counted;
use kingfisher::search::quiescence_reference::forced_ext_pesto_balance_reference;
use std::io::Write;
use std::time::Instant;

type Search = fn(&mut BoardStack, &MoveGen, &PestoEval) -> (f32, bool, u32, u8);

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

struct Run {
    answers: Vec<(f32, bool)>,
    nodes: Vec<u32>,
    median_us: Vec<f64>,
}

fn run(
    name: &str,
    search: Search,
    boards: &[Board],
    reps: usize,
    mg: &MoveGen,
    pesto: &PestoEval,
) -> Run {
    let n = boards.len();
    let mut times = vec![Vec::with_capacity(reps); n];
    let mut per_rep_us = Vec::with_capacity(reps);
    let mut answers = Vec::new();
    let mut nodes = Vec::new();
    for rep in 0..reps {
        let start_rep = Instant::now();
        for (i, b) in boards.iter().enumerate() {
            let mut stack = BoardStack::with_board(b.clone());
            let t = Instant::now();
            let (score, completed, searched, _) = search(&mut stack, mg, pesto);
            times[i].push(t.elapsed().as_secs_f64() * 1e6);
            if rep == 0 {
                answers.push((score, completed));
                nodes.push(searched);
            }
        }
        per_rep_us.push(start_rep.elapsed().as_secs_f64() * 1e6 / n as f64);
    }
    let mut median_us: Vec<f64> = times.iter_mut().map(|t| percentile(t, 50.0)).collect();
    let mean = per_rep_us.iter().sum::<f64>() / reps as f64;
    let sd = if reps > 1 {
        (per_rep_us.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (reps - 1) as f64).sqrt()
    } else {
        0.0
    };
    let kept = median_us.clone();
    let (p50, p99, max) = (
        percentile(&mut median_us, 50.0),
        percentile(&mut median_us, 99.0),
        percentile(&mut median_us, 100.0),
    );
    let mean_nodes = nodes.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let completed = answers.iter().filter(|a| a.1).count();
    println!(
        "{name}: {mean:.1} ± {sd:.1} us per call, median {p50:.1}, p99 {p99:.0}, max {max:.0}; \
         {mean_nodes:.0} positions searched per call; {completed}/{n} completed; checksum {:016x}",
        answer_checksum(&answers)
    );
    Run {
        answers,
        nodes,
        median_us: kept,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: qsearch_bench <leaf file> [reps] [--half tune|held|all] [--impl new|ref|new,ref] [--dump <file>]";
    let path = args.get(1).expect(usage);
    let reps: usize = args
        .get(2)
        .filter(|a| !a.starts_with("--"))
        .map(|a| a.parse().expect(usage))
        .unwrap_or(5);
    let half = flag(&args, "--half").unwrap_or("all");
    let keep = |game: Option<usize>, i: usize| {
        let k = game.unwrap_or(i);
        match half {
            "tune" => k.is_multiple_of(2),
            "held" => !k.is_multiple_of(2),
            "all" => true,
            other => panic!("--half takes tune, held or all, not {other:?}"),
        }
    };
    let text = std::fs::read_to_string(path).expect("cannot read the leaf file");
    let boards: Vec<Board> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .filter_map(|(i, l)| {
            let (game, fen) = parse_corpus_line(l);
            keep(game, i).then(|| Board::new_from_fen(fen))
        })
        .collect();
    let (mg, pesto) = (MoveGen::new(), PestoEval::new());
    println!("{} positions ({half}), {reps} reps", boards.len());

    let impls = flag(&args, "--impl").unwrap_or("new");
    let mut runs = Vec::new();
    for name in impls.split(',') {
        let search: Search = match name {
            "new" => forced_ext_pesto_balance_counted,
            "ref" => forced_ext_pesto_balance_reference,
            other => panic!("--impl takes new and/or ref, not {other:?}"),
        };
        runs.push((name, run(name, search, &boards, reps, &mg, &pesto)));
    }
    if let [(_, a), (_, b)] = &runs[..] {
        let n = boards.len() as f64;
        let same_score = a
            .answers
            .iter()
            .zip(&b.answers)
            .filter(|(x, y)| x.0 == y.0)
            .count();
        let same_flag = a
            .answers
            .iter()
            .zip(&b.answers)
            .filter(|(x, y)| x.1 == y.1)
            .count();
        let mad = a
            .answers
            .iter()
            .zip(&b.answers)
            .map(|(x, y)| (x.0 - y.0).abs() as f64)
            .sum::<f64>()
            / n;
        println!(
            "agreement: same score {:.2}%, same completed flag {:.2}%, mean |score difference| {mad:.4} pawns",
            100.0 * same_score as f64 / n,
            100.0 * same_flag as f64 / n
        );
    }
    if let Some(dump) = flag(&args, "--dump") {
        let (_, r) = &runs[0];
        let mut f = std::fs::File::create(dump).expect("cannot write the dump");
        for i in 0..boards.len() {
            writeln!(
                f,
                "{} {} {} {:.2}",
                r.answers[i].0, r.answers[i].1 as u8, r.nodes[i], r.median_us[i]
            )
            .unwrap();
        }
    }
}
