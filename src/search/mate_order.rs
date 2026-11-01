//! Move orders for the forced-mate search.
//!
//! The search stops at the first mate at the attacker's move and at the first
//! refutation at the defender's, so the order in which moves are tried changes
//! how many positions it visits but never its answer. Each order here gives
//! every move a key, smaller first, and sorts the list stably by it; order 0
//! of each side leaves the list as generated.
//!
//! Attacker orders (each "closest" is the destination square's king distance
//! to the enemy king; where every move is searched, all but A0, A5 and A8 put
//! checks before quiet moves):
//! - A0: as generated.
//! - A1: double checks, captures, promotions, queen moves closest first,
//!   knight moves, then the rest closest first with ties bishop, rook, pawn.
//! - A2: A1 with captures of the most valuable piece first.
//! - A3: double checks, then everything closest first, ties Q, R, B, N, P.
//! - A4: double checks, captures, promotions, then Q, R, B, N, P, each closest first.
//! - A5: closest first.
//! - A6: A1 with knight moves closest first too.
//! - A7: fewest unattacked squares next to the enemy king after the move.
//! - A8: farthest first.
//! - A9: as A7, but estimated without making the move: the squares our
//!   pieces attack now (seen through the enemy king), plus those the moved
//!   piece attacks from its new square; ties closest first.
//! - A10: A9 for lists of checking moves, except at the attacker's last move
//!   (where order helps only if that move mates); other lists as generated.
//! - A11: as A10, but only queen moves are sorted, and put first; the rest
//!   keep their generated order.
//! - A12: as A11 for queen and rook moves.
//!
//! Defender orders (a block is any other move out of check; when not in check
//! it is any quiet move):
//! - D0: as generated (captures, blocks, king moves; pawns first within each).
//! - D1: captures, promotions, blocks, king moves.
//! - D2: captures, king moves, blocks.
//! - D3: king moves, captures, blocks.
//! - D4: blocks, captures, king moves.
//! - D5: captures by the most valuable piece first, blocks, king moves.
//! - D6: captures, blocks by the least valuable piece first, king moves
//!   closest to the checking piece first.
//!
//! Generation orders, which cost nothing because the generators emit moves
//! in them (named after the A and D orders, as in "A10,D0,G4,V1"):
//! - G0–G5, checking moves: G0 knights, bishops, rooks, queens (as
//!   generated); G1 queens, rooks, bishops, knights; G4 queens, knights,
//!   bishops, rooks; G2, G3 and G5 are G0, G1 and G4 with capturing checks
//!   first. The king's uncovered checks, then pawns, follow each list.
//! - V0–V2, moves out of check: V0 captures of the checker by pawns first
//!   (as generated), then blocks by pawns first; V1 captures by queens
//!   first; V2 captures and blocks by queens first.

use crate::board::Board;
use crate::move_generation::{CheckGenOrder, EvasionGenOrder, MoveGen};
use crate::move_types::Move;
use crate::piece_types::{BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttackerOrder(u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DefenderOrder(u8);

impl AttackerOrder {
    pub const ALL: [Self; 13] = [
        Self(0),
        Self(1),
        Self(2),
        Self(3),
        Self(4),
        Self(5),
        Self(6),
        Self(7),
        Self(8),
        Self(9),
        Self(10),
        Self(11),
        Self(12),
    ];

    /// True if the order applies only to lists of checking moves, and not at
    /// the attacker's last move.
    pub fn checking_moves_only(self) -> bool {
        self.0 >= 10
    }
}

impl DefenderOrder {
    pub const ALL: [Self; 7] = [
        Self(0),
        Self(1),
        Self(2),
        Self(3),
        Self(4),
        Self(5),
        Self(6),
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckOrder(u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvasionOrder(u8);

impl CheckOrder {
    pub const ALL: [Self; 6] = [Self(0), Self(1), Self(2), Self(3), Self(4), Self(5)];

    pub fn generation(self) -> CheckGenOrder {
        let pieces = match self.0 {
            0 | 2 => [KNIGHT, BISHOP, ROOK, QUEEN, KING, PAWN],
            1 | 3 => [QUEEN, ROOK, BISHOP, KNIGHT, KING, PAWN],
            _ => [QUEEN, KNIGHT, BISHOP, ROOK, KING, PAWN],
        };
        CheckGenOrder {
            pieces,
            captures_first: matches!(self.0, 2 | 3 | 5),
        }
    }
}

impl EvasionOrder {
    pub const ALL: [Self; 3] = [Self(0), Self(1), Self(2)];

    pub fn generation(self) -> EvasionGenOrder {
        const PAWNS: [usize; 5] = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN];
        const QUEENS: [usize; 5] = [QUEEN, ROOK, BISHOP, KNIGHT, PAWN];
        match self.0 {
            0 => EvasionGenOrder::PAWNS_FIRST,
            1 => EvasionGenOrder {
                captures: QUEENS,
                blocks: PAWNS,
            },
            _ => EvasionGenOrder {
                captures: QUEENS,
                blocks: QUEENS,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoveOrder {
    pub attacker: AttackerOrder,
    pub defender: DefenderOrder,
    pub checks: CheckOrder,
    pub evasions: EvasionOrder,
    /// Try first the move that last ended a search at the same ply (K).
    pub killer: bool,
    /// Remember results for positions reached again in one search (T).
    pub table: bool,
}

impl MoveOrder {
    /// Moves as generated, on both sides.
    pub const GENERATED: Self = Self {
        attacker: AttackerOrder(0),
        defender: DefenderOrder(0),
        checks: CheckOrder(0),
        evasions: EvasionOrder(0),
        killer: false,
        table: false,
    };

    /// The order the search uses unless told otherwise: A10 for the
    /// attacker, the defender's moves as generated with captures by the most
    /// valuable piece first (V1), and the table. Fastest of the orders timed
    /// on positions logged from self-play leaves.
    pub const DEFAULT: Self = Self {
        attacker: AttackerOrder(10),
        defender: DefenderOrder(0),
        checks: CheckOrder(0),
        evasions: EvasionOrder(1),
        killer: false,
        table: true,
    };

    /// Every pair of an attacker and a defender order, moves generated as G0, V0.
    pub fn all() -> Vec<Self> {
        AttackerOrder::ALL
            .iter()
            .flat_map(|&attacker| {
                DefenderOrder::ALL.iter().map(move |&defender| Self {
                    attacker,
                    defender,
                    ..Self::GENERATED
                })
            })
            .collect()
    }

    /// From a name such as "A1,D1" (generated as G0, V0), "A10,D0,G4,V1",
    /// or either followed by K and/or T.
    pub fn parse(name: &str) -> Option<Self> {
        let mut parts: Vec<&str> = name.split(',').collect();
        let (mut killer, mut table) = (false, false);
        while let Some(&last) = parts.last() {
            let flag = match last {
                "K" => &mut killer,
                "T" => &mut table,
                _ => break,
            };
            if *flag {
                return None;
            }
            *flag = true;
            parts.pop();
        }
        let field = |s: &str, prefix: char, n: usize| -> Option<u8> {
            let v: u8 = s.strip_prefix(prefix)?.parse().ok()?;
            ((v as usize) < n).then_some(v)
        };
        let (g, v) = match parts.len() {
            2 => (0, 0),
            4 => (
                field(parts[2], 'G', CheckOrder::ALL.len())?,
                field(parts[3], 'V', EvasionOrder::ALL.len())?,
            ),
            _ => return None,
        };
        Some(Self {
            attacker: AttackerOrder(field(parts[0], 'A', AttackerOrder::ALL.len())?),
            defender: DefenderOrder(field(parts[1], 'D', DefenderOrder::ALL.len())?),
            checks: CheckOrder(g),
            evasions: EvasionOrder(v),
            killer,
            table,
        })
    }

    pub fn name(self) -> String {
        let mut name = format!("A{},D{}", self.attacker.0, self.defender.0);
        if self.checks.0 != 0 || self.evasions.0 != 0 {
            name += &format!(",G{},V{}", self.checks.0, self.evasions.0);
        }
        if self.killer {
            name += ",K";
        }
        if self.table {
            name += ",T";
        }
        name
    }
}

/// King distance between two squares.
fn distance(a: usize, b: usize) -> u32 {
    let (fa, ra, fb, rb) = (a % 8, a / 8, b % 8, b / 8);
    fa.abs_diff(fb).max(ra.abs_diff(rb)) as u32
}

/// The piece of `color` on `sq`, if any.
fn piece_on(board: &Board, color: usize, sq: usize) -> Option<usize> {
    let bit = 1u64 << sq;
    (PAWN..=KING).find(|&p| board.pieces[color][p] & bit != 0)
}

/// Sorts `moves` stably by `key`, smaller first, without allocating: an
/// insertion sort, as the lists are short.
fn sort_by_key(moves: &mut [Move], mut key: impl FnMut(Move) -> u32) {
    const MAX: usize = 256;
    let n = moves.len().min(MAX);
    let mut keys = [0u32; MAX];
    for i in 0..n {
        keys[i] = key(moves[i]);
    }
    for i in 1..n {
        let (k, m) = (keys[i], moves[i]);
        let mut j = i;
        while j > 0 && keys[j - 1] > k {
            keys[j] = keys[j - 1];
            moves[j] = moves[j - 1];
            j -= 1;
        }
        keys[j] = k;
        moves[j] = m;
    }
}

/// A key from several small fields, the first most significant.
fn key(fields: &[u32]) -> u32 {
    fields.iter().fold(0, |k, &f| k * 64 + f.min(63))
}

/// Puts the attacker's (the side to move's) moves in `order`.
pub fn order_attacker_moves(
    board: &Board,
    move_gen: &MoveGen,
    order: AttackerOrder,
    moves: &mut [Move],
) {
    if order.0 == 0 || moves.len() < 2 {
        return;
    }
    let us = if board.w_to_move { WHITE } else { BLACK };
    let them = 1 - us;
    let their_king = board.pieces[them][KING];
    if their_king == 0 {
        return;
    }
    let ksq = their_king.trailing_zeros() as usize;
    if order.0 == 9 || order.0 == 10 {
        order_by_estimated_escapes(board, move_gen, us, ksq, moves, !0);
        return;
    }
    if order.0 == 11 || order.0 == 12 {
        let sorted = if order.0 == 11 {
            board.pieces[us][QUEEN]
        } else {
            board.pieces[us][QUEEN] | board.pieces[us][ROOK]
        };
        order_by_estimated_escapes(board, move_gen, us, ksq, moves, sorted);
        return;
    }
    let ep = board.en_passant.map(|e| e as usize);
    // Victims by value: queen 0 ... pawn 3, none 4.
    let victim_rank = |m: Move| match piece_on(board, them, m.to()) {
        Some(QUEEN) => 0,
        Some(ROOK) => 1,
        Some(BISHOP) | Some(KNIGHT) => 2,
        Some(_) => 3,
        None if Some(m.to()) == ep && board.pieces[us][PAWN] & (1u64 << m.from()) != 0 => 3,
        None => 4,
    };
    let rank_qrbnpk = |p: usize| match p {
        QUEEN => 0,
        ROOK => 1,
        BISHOP => 2,
        KNIGHT => 3,
        PAWN => 4,
        _ => 5,
    };
    let rank_brpk = |p: usize| match p {
        BISHOP => 0,
        ROOK => 1,
        PAWN => 2,
        _ => 3,
    };
    let escapes = |m: Move| {
        let after = board.apply_move_to_board(m);
        let around = move_gen.k_move_bitboard[ksq] & !after.pieces_occ[them];
        crate::bits::bits(&around)
            .filter(|&s| !after.is_square_attacked(s, us == WHITE, move_gen))
            .count() as u32
    };
    sort_by_key(moves, |m| {
        let piece = piece_on(board, us, m.from()).unwrap_or(KING);
        let (direct, uncovered) = board.check_kinds(m, move_gen);
        let quiet = !(direct || uncovered) as u32;
        let double = !(direct && uncovered) as u32;
        let victim = victim_rank(m);
        let capture = victim < 4;
        let promotion = m.promotion().is_some();
        let d = distance(m.to(), ksq);
        // A1, A2 and A6 share their blocks: 0 double check, 1 capture,
        // 2 promotion, 3 queen, 4 knight, 5 the rest.
        let a1_block = if double == 0 {
            0
        } else if capture {
            1
        } else if promotion {
            2
        } else if piece == QUEEN {
            3
        } else if piece == KNIGHT {
            4
        } else {
            5
        };
        match order.0 {
            1 | 2 | 6 => {
                let within = match a1_block {
                    1 if order.0 == 2 => victim,
                    3 => d,
                    4 if order.0 == 6 => d,
                    5 => d * 4 + rank_brpk(piece),
                    _ => 0,
                };
                key(&[quiet, a1_block, within])
            }
            3 => key(&[quiet, double, d, rank_qrbnpk(piece)]),
            4 => {
                let block = if double == 0 {
                    0
                } else if capture {
                    1
                } else if promotion {
                    2
                } else {
                    3 + rank_qrbnpk(piece)
                };
                key(&[quiet, block, d])
            }
            5 => d,
            7 => key(&[quiet, escapes(m), d]),
            _ => 7 - d,
        }
    });
}

/// The squares a `piece` of `color` on `sq` attacks with occupancy `occ`.
fn attacks(move_gen: &MoveGen, color: usize, piece: usize, sq: usize, occ: u64) -> u64 {
    match piece {
        PAWN if color == WHITE => move_gen.wp_capture_bitboard[sq],
        PAWN => move_gen.bp_capture_bitboard[sq],
        KNIGHT => move_gen.n_move_bitboard[sq],
        BISHOP => move_gen.gen_bishop_attacks_occ(sq, occ),
        ROOK => move_gen.gen_rook_attacks_occ(sq, occ),
        QUEEN => move_gen.gen_bishop_attacks_occ(sq, occ) | move_gen.gen_rook_attacks_occ(sq, occ),
        _ => move_gen.k_move_bitboard[sq],
    }
}

/// A9: fewest estimated escape squares for the enemy king on `ksq` first.
/// Only moves from a square in `from` are sorted, and put first; the rest
/// keep their order.
fn order_by_estimated_escapes(
    board: &Board,
    move_gen: &MoveGen,
    us: usize,
    ksq: usize,
    moves: &mut [Move],
    from: u64,
) {
    if !moves.iter().any(|m| from & (1u64 << m.from()) != 0) {
        return;
    }
    let them = 1 - us;
    let occ = board.pieces_occ[WHITE] | board.pieces_occ[BLACK];
    // Sliders see through the king, so it cannot step back along their line.
    let occ_without_king = occ & !(1u64 << ksq);
    let mut attacked = 0u64;
    for piece in PAWN..=KING {
        for sq in crate::bits::bits(&board.pieces[us][piece]) {
            attacked |= attacks(move_gen, us, piece, sq, occ_without_king);
        }
    }
    let around = move_gen.k_move_bitboard[ksq] & !board.pieces_occ[them];
    sort_by_key(moves, |m| {
        if from & (1u64 << m.from()) == 0 {
            return u32::MAX;
        }
        let piece = piece_on(board, us, m.from()).unwrap_or(KING);
        let piece = m.promotion().unwrap_or(piece);
        let occ_after = (occ_without_king & !(1u64 << m.from())) | (1u64 << m.to());
        let covered = attacked | attacks(move_gen, us, piece, m.to(), occ_after);
        let escapes = (around & !covered & !(1u64 << m.to())).count_ones();
        key(&[escapes, distance(m.to(), ksq)])
    });
}

/// Puts the defender's (the side to move's) moves in `order`.
pub fn order_defender_moves(
    board: &Board,
    move_gen: &MoveGen,
    order: DefenderOrder,
    moves: &mut [Move],
) {
    if order.0 == 0 || moves.len() < 2 {
        return;
    }
    let us = if board.w_to_move { WHITE } else { BLACK };
    let them = 1 - us;
    let ksq = board.pieces[us][KING].trailing_zeros() as usize;
    let ep = board.en_passant.map(|e| e as usize);
    // An enemy piece attacking our king, for D6.
    let checker = if order.0 == 6 && board.pieces[us][KING] != 0 {
        let occ = board.pieces_occ[WHITE] | board.pieces_occ[BLACK];
        let t = &board.pieces[them];
        let pawn_attackers = if us == WHITE {
            move_gen.wp_capture_bitboard[ksq]
        } else {
            move_gen.bp_capture_bitboard[ksq]
        };
        let attackers = (move_gen.n_move_bitboard[ksq] & t[KNIGHT])
            | (pawn_attackers & t[PAWN])
            | (move_gen.gen_bishop_attacks_occ(ksq, occ) & (t[BISHOP] | t[QUEEN]))
            | (move_gen.gen_rook_attacks_occ(ksq, occ) & (t[ROOK] | t[QUEEN]));
        (attackers != 0).then(|| attackers.trailing_zeros() as usize)
    } else {
        None
    };
    let value_rank = |p: usize| match p {
        QUEEN => 0,
        ROOK => 1,
        BISHOP => 2,
        KNIGHT => 3,
        PAWN => 4,
        _ => 5,
    };
    sort_by_key(moves, |m| {
        let piece = piece_on(board, us, m.from()).unwrap_or(KING);
        let capture =
            board.pieces_occ[them] & (1u64 << m.to()) != 0 || (piece == PAWN && Some(m.to()) == ep);
        let promotion = m.promotion().is_some();
        let king = piece == KING;
        // 0 capture, 1 promotion, 2 block (or quiet move), 3 king move.
        let class = if capture {
            0
        } else if promotion {
            1
        } else if king {
            3
        } else {
            2
        };
        match order.0 {
            1 => class,
            2 => [0, 2, 2, 1][class as usize],
            3 => [1, 2, 2, 0][class as usize],
            4 => [1, 0, 0, 2][class as usize],
            5 => match class {
                0 => value_rank(piece),
                3 => 7,
                _ => 6,
            },
            _ => match class {
                0 => 0,
                3 => 1 + 8 + checker.map_or(0, |c| distance(m.to(), c)),
                _ => 1 + (5 - value_rank(piece)).min(7),
            },
        }
    });
}
