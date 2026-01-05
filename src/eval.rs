//! Pesto evaluation function module
//!
//! This module implements the Pesto evaluation function, which uses tapered evaluation
//! to interpolate between piece-square tables for opening and endgame, optimized by Texel tuning.

use crate::bits::{bits, popcnt};
use crate::board::Board;
use crate::board_utils::flip_sq_ind_vertically;
use crate::board_utils::{
    get_adjacent_files_mask, get_file_mask, get_front_span_mask, get_king_attack_zone_mask,
    get_king_shield_zone_mask, get_passed_pawn_mask, get_rank_mask, sq_ind_to_bit, sq_to_file,
    sq_to_rank,
};
pub use crate::eval_constants::EvalWeights;
use crate::eval_constants::{
    EG_PESTO_TABLE, EG_VALUE, GAMEPHASE_INC, MG_PESTO_TABLE, MG_VALUE,
};
use crate::move_generation::MoveGen;
use crate::piece_types::{BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};
use std::cmp::min;

/// Struct representing the Pesto evaluation function
#[derive(Clone, Debug)]
pub struct PestoEval {
    mg_table: [[[i32; 64]; 6]; 2],
    eg_table: [[[i32; 64]; 6]; 2],
    pub weights: EvalWeights,
}

impl PestoEval {
    pub fn new() -> PestoEval {
        let mut mg_table = [[[0; 64]; 6]; 2];
        let mut eg_table = [[[0; 64]; 6]; 2];

        for p in 0..6 {
            for sq in 0..64 {
                mg_table[WHITE][p][sq] =
                    MG_VALUE[p] + MG_PESTO_TABLE[p][flip_sq_ind_vertically(sq)];
                eg_table[WHITE][p][sq] =
                    EG_VALUE[p] + EG_PESTO_TABLE[p][flip_sq_ind_vertically(sq)];
                mg_table[BLACK][p][sq] = MG_VALUE[p] + MG_PESTO_TABLE[p][sq];
                eg_table[BLACK][p][sq] = EG_VALUE[p] + EG_PESTO_TABLE[p][sq];
            }
        }

        let weights = EvalWeights::default();

        PestoEval {
            mg_table,
            eg_table,
            weights,
        }
    }

    pub fn with_weights(weights: EvalWeights) -> PestoEval {
        let mut mg_table = [[[0; 64]; 6]; 2];
        let mut eg_table = [[[0; 64]; 6]; 2];

        for p in 0..6 {
            for sq in 0..64 {
                mg_table[WHITE][p][sq] =
                    MG_VALUE[p] + MG_PESTO_TABLE[p][flip_sq_ind_vertically(sq)];
                eg_table[WHITE][p][sq] =
                    EG_VALUE[p] + EG_PESTO_TABLE[p][flip_sq_ind_vertically(sq)];
                mg_table[BLACK][p][sq] = MG_VALUE[p] + MG_PESTO_TABLE[p][sq];
                eg_table[BLACK][p][sq] = EG_VALUE[p] + EG_PESTO_TABLE[p][sq];
            }
        }

        PestoEval {
            mg_table,
            eg_table,
            weights,
        }
    }

    pub fn get_mg_score(&self, color: usize, piece: usize, square: usize) -> i32 {
        self.mg_table[color][piece][square]
    }

    pub fn get_eg_score(&self, color: usize, piece: usize, square: usize) -> i32 {
        self.eg_table[color][piece][square]
    }

    pub fn eval_plus_game_phase(&self, board: &Board, move_gen: &MoveGen) -> (i32, i32, i32) {
        let mut mg: [i32; 2] = [0, 0];
        let mut eg: [i32; 2] = [0, 0];
        let mut game_phase: i32 = 0;

        for color in 0..2 {
            for piece in 0..6 {
                let mut piece_bb = board.pieces[color][piece];
                while piece_bb != 0 {
                    let sq = piece_bb.trailing_zeros() as usize;
                    mg[color] += self.mg_table[color][piece][sq];
                    eg[color] += self.eg_table[color][piece][sq];
                    game_phase += GAMEPHASE_INC[piece];
                    piece_bb &= piece_bb - 1;
                }
            }
        }

        for color in [WHITE, BLACK] {
            let enemy_color = 1 - color;

            if popcnt(board.pieces[color][BISHOP]) >= 2 {
                mg[color] += self.weights.two_bishops_bonus[0];
                eg[color] += self.weights.two_bishops_bonus[1];
            }

            let friendly_pawns = board.pieces[color][PAWN];
            let enemy_pawns = board.pieces[enemy_color][PAWN];
            let mut chain_bonus_mg = 0;
            let mut chain_bonus_eg = 0;
            let mut duo_bonus_mg = 0;
            let mut duo_bonus_eg = 0;

            for sq in bits(&friendly_pawns) {
                let file = sq_to_file(sq);

                let passed_mask = get_passed_pawn_mask(color, sq);
                if (passed_mask & enemy_pawns) == 0 {
                    let rank = sq_to_rank(sq);
                    let bonus_rank = if color == WHITE { rank } else { 7 - rank };
                    mg[color] += self.weights.passed_pawn_bonus_mg[bonus_rank];
                    eg[color] += self.weights.passed_pawn_bonus_eg[bonus_rank];
                }

                let adjacent_mask = get_adjacent_files_mask(sq);
                if (adjacent_mask & friendly_pawns) == 0 {
                    mg[color] += self.weights.isolated_pawn_penalty[0];
                    eg[color] += self.weights.isolated_pawn_penalty[1];
                }

                let (defend1_sq_opt, defend2_sq_opt) = if color == WHITE {
                    (sq.checked_sub(9), sq.checked_sub(7))
                } else {
                    (sq.checked_add(7), sq.checked_add(9))
                };

                if let Some(defend1_sq) = defend1_sq_opt {
                    if defend1_sq < 64
                        && (sq_to_file(sq) as i8 - sq_to_file(defend1_sq) as i8).abs() == 1
                        && (friendly_pawns & sq_ind_to_bit(defend1_sq) != 0)
                    {
                        chain_bonus_mg += self.weights.pawn_chain_bonus[0];
                        chain_bonus_eg += self.weights.pawn_chain_bonus[1];
                    }
                }
                if let Some(defend2_sq) = defend2_sq_opt {
                    if defend2_sq < 64
                        && (sq_to_file(sq) as i8 - sq_to_file(defend2_sq) as i8).abs() == 1
                        && (friendly_pawns & sq_ind_to_bit(defend2_sq) != 0)
                    {
                        chain_bonus_mg += self.weights.pawn_chain_bonus[0];
                        chain_bonus_eg += self.weights.pawn_chain_bonus[1];
                    }
                }

                if file < 7 {
                    let neighbor_sq = sq + 1;
                    if (friendly_pawns & sq_ind_to_bit(neighbor_sq)) != 0 {
                        duo_bonus_mg += self.weights.pawn_duo_bonus[0];
                        duo_bonus_eg += self.weights.pawn_duo_bonus[1];
                    }
                }
            }
            mg[color] += chain_bonus_mg;
            eg[color] += chain_bonus_eg;
            mg[color] += duo_bonus_mg / 2;
            eg[color] += duo_bonus_eg / 2;

            let mut backward_penalty_mg = 0;
            let mut backward_penalty_eg = 0;
            for sq in bits(&friendly_pawns) {
                let adjacent_mask = get_adjacent_files_mask(sq);
                let front_span = get_front_span_mask(color, sq);
                let stop_sq = if color == WHITE { sq + 8 } else { sq.wrapping_sub(8) };

                let no_adjacent_support = (friendly_pawns & adjacent_mask & front_span) == 0;

                if no_adjacent_support && stop_sq < 64 {
                    if (enemy_pawns & sq_ind_to_bit(stop_sq)) != 0 {
                        backward_penalty_mg += self.weights.backward_pawn_penalty[0];
                        backward_penalty_eg += self.weights.backward_pawn_penalty[1];
                    }
                }
            }
            mg[color] += backward_penalty_mg;
            eg[color] += backward_penalty_eg;

            let king_sq = board.pieces[color][KING].trailing_zeros() as usize;
            if king_sq < 64 {
                let shield_zone_mask = get_king_shield_zone_mask(color, king_sq);
                let shield_pawns = popcnt(shield_zone_mask & friendly_pawns);
                mg[color] += shield_pawns as i32 * self.weights.king_safety_pawn_shield_bonus[0];
                eg[color] += shield_pawns as i32 * self.weights.king_safety_pawn_shield_bonus[1];

                let enemy_king_sq = board.pieces[enemy_color][KING].trailing_zeros() as usize;
                if enemy_king_sq < 64 {
                    let attack_zone = get_king_attack_zone_mask(enemy_color, enemy_king_sq);
                    let mut total_attack_weight = 0;

                    for piece_type in [KNIGHT, BISHOP, ROOK, QUEEN] {
                        let piece_bb = board.pieces[color][piece_type];
                        for sq in bits(&piece_bb) {
                            if get_king_attack_zone_mask(color, sq) & attack_zone != 0 {
                                total_attack_weight += self.weights.king_attack_weights[piece_type];
                            }
                        }
                    }
                    mg[color] += total_attack_weight;
                }
            }
            let friendly_rooks = board.pieces[color][ROOK];
            let seventh_rank = if color == WHITE { 6 } else { 1 };
            let seventh_rank_mask = get_rank_mask(seventh_rank);
            let rooks_on_seventh = friendly_rooks & seventh_rank_mask;

            for rook_sq in bits(&friendly_rooks) {
                let rank = sq_to_rank(rook_sq);
                let file = sq_to_file(rook_sq);

                let file_mask = get_file_mask(file);
                let friendly_pawns_on_file = friendly_pawns & file_mask;
                let enemy_pawns_on_file = enemy_pawns & file_mask;

                if friendly_pawns_on_file == 0 {
                    if enemy_pawns_on_file == 0 {
                        mg[color] += self.weights.rook_open_file_bonus[0];
                        eg[color] += self.weights.rook_open_file_bonus[1];
                    } else {
                        mg[color] += self.weights.rook_half_open_file_bonus[0];
                        eg[color] += self.weights.rook_half_open_file_bonus[1];
                    }
                }

                let friendly_file_pawns = friendly_pawns & get_file_mask(file);
                for pawn_sq in bits(&friendly_file_pawns) {
                    let passed_mask = get_passed_pawn_mask(color, pawn_sq);
                    if (passed_mask & enemy_pawns) == 0 {
                        let pawn_rank = sq_to_rank(pawn_sq);
                        if (color == WHITE && rank < pawn_rank)
                            || (color == BLACK && rank > pawn_rank)
                        {
                            mg[color] += self.weights.rook_behind_passed_pawn_bonus[0];
                            eg[color] += self.weights.rook_behind_passed_pawn_bonus[1];
                            break;
                        }
                    }
                }

                let enemy_file_pawns = enemy_pawns & get_file_mask(file);
                for pawn_sq in bits(&enemy_file_pawns) {
                    let passed_mask = get_passed_pawn_mask(enemy_color, pawn_sq);
                    if (passed_mask & friendly_pawns) == 0 {
                        let pawn_rank = sq_to_rank(pawn_sq);
                        if (color == WHITE && rank > pawn_rank)
                            || (color == BLACK && rank < pawn_rank)
                        {
                            mg[color] += self.weights.rook_behind_enemy_passed_pawn_bonus[0];
                            eg[color] += self.weights.rook_behind_enemy_passed_pawn_bonus[1];
                            break;
                        }
                    }
                }
            }

            if popcnt(rooks_on_seventh) >= 2 {
                mg[color] += self.weights.doubled_rooks_on_seventh_bonus[0];
                eg[color] += self.weights.doubled_rooks_on_seventh_bonus[1];
            }

            if color == WHITE {
                if board.castling_rights.white_kingside {
                    mg[color] += self.weights.castling_rights_bonus[0];
                }
                if board.castling_rights.white_queenside {
                    mg[color] += self.weights.castling_rights_bonus[0];
                }
            } else {
                if board.castling_rights.black_kingside {
                    mg[color] += self.weights.castling_rights_bonus[0];
                }
                if board.castling_rights.black_queenside {
                    mg[color] += self.weights.castling_rights_bonus[0];
                }
            }
        }

        let mut mobility_mg = [0; 2];
        let mut mobility_eg = [0; 2];
        let occupied = board.get_all_occupancy();

        for color in [WHITE, BLACK] {
            let friendly_occ = board.pieces_occ[color];

            let mut knight_moves = 0;
            for sq in bits(&board.pieces[color][KNIGHT]) {
                knight_moves += popcnt(move_gen.n_move_bitboard[sq] & !friendly_occ);
            }
            mobility_mg[color] += knight_moves as i32 * self.weights.mobility_weights_mg[0];
            mobility_eg[color] += knight_moves as i32 * self.weights.mobility_weights_eg[0];

            let mut bishop_moves = 0;
            for sq in bits(&board.pieces[color][BISHOP]) {
                bishop_moves += popcnt(move_gen.get_bishop_moves(sq, occupied) & !friendly_occ);
            }
            mobility_mg[color] += bishop_moves as i32 * self.weights.mobility_weights_mg[1];
            mobility_eg[color] += bishop_moves as i32 * self.weights.mobility_weights_eg[1];

            let mut rook_moves = 0;
            for sq in bits(&board.pieces[color][ROOK]) {
                rook_moves += popcnt(move_gen.get_rook_moves(sq, occupied) & !friendly_occ);
            }
            mobility_mg[color] += rook_moves as i32 * self.weights.mobility_weights_mg[2];
            mobility_eg[color] += rook_moves as i32 * self.weights.mobility_weights_eg[2];

            let mut queen_moves = 0;
            for sq in bits(&board.pieces[color][QUEEN]) {
                queen_moves += popcnt(move_gen.get_queen_moves(sq, occupied) & !friendly_occ);
            }
            mobility_mg[color] += queen_moves as i32 * self.weights.mobility_weights_mg[3];
            mobility_eg[color] += queen_moves as i32 * self.weights.mobility_weights_eg[3];

            mg[color] += mobility_mg[color];
            eg[color] += mobility_eg[color];
        }

        let mg_score = mg[0] - mg[1];
        let eg_score = eg[0] - eg[1];

        if board.w_to_move {
            (mg_score, eg_score, game_phase)
        } else {
            (-mg_score, -eg_score, game_phase)
        }
    }

    pub fn eval(&self, board: &Board, move_gen: &MoveGen) -> i32 {
        let (mg, eg, phase) = self.eval_plus_game_phase(board, move_gen);
        let mg_phase: i32 = min(24, phase);
        let eg_phase: i32 = 24 - mg_phase;
        let eg_phase_clamped = if eg_phase < 0 { 0 } else { eg_phase };
        let score = (mg * mg_phase + eg * eg_phase_clamped) / 24;
        if board.w_to_move { score } else { -score }
    }

    pub fn eval_update_board(&self, board: &mut Board, move_gen: &MoveGen) -> i32 {
        let (mg, eg, game_phase) = self.eval_plus_game_phase(board, move_gen);
        let mg_phase: i32 = min(24, game_phase);
        let eg_phase: i32 = 24 - mg_phase;
        let eg_phase_clamped = if eg_phase < 0 { 0 } else { eg_phase };
        let score = (mg * mg_phase + eg * eg_phase_clamped) / 24;
        board.eval = if board.w_to_move { score } else { -score };
        board.game_phase = game_phase;
        board.eval
    }

    pub fn move_eval(
        &self,
        board: &Board,
        _move_gen: &MoveGen, // Not fully utilized here, keeping signature
        from_sq_ind: usize,
        to_sq_ind: usize,
    ) -> i32 {
        let piece = match board.get_piece(from_sq_ind) {
            Some(p) => p,
            None => return 0,
        };
        let mut mg_score: i32 = self.mg_table[piece.0][piece.1][to_sq_ind]
            - self.mg_table[piece.0][piece.1][from_sq_ind];
        let eg_score: i32 = self.eg_table[piece.0][piece.1][to_sq_ind]
            - self.eg_table[piece.0][piece.1][from_sq_ind];

        if piece == (WHITE, KING) && from_sq_ind == 4 {
            if to_sq_ind == 6 {
                mg_score += self.mg_table[WHITE][ROOK][5] - self.mg_table[WHITE][ROOK][7];
            } else if to_sq_ind == 2 {
                mg_score += self.mg_table[WHITE][ROOK][3] - self.mg_table[WHITE][ROOK][0];
            }
        } else if piece == (BLACK, KING) && from_sq_ind == 60 {
            if to_sq_ind == 62 {
                mg_score += self.mg_table[BLACK][ROOK][61] - self.mg_table[BLACK][ROOK][63];
            } else if to_sq_ind == 58 {
                mg_score += self.mg_table[BLACK][ROOK][59] - self.mg_table[BLACK][ROOK][56];
            }
        }

        let mg_phase: i32 = min(24, board.game_phase);
        let eg_phase: i32 = 24 - mg_phase;
        (mg_score * mg_phase + eg_score * eg_phase) / 24
    }
}

/// Extrapolates a new value [ -1, 1 ] based on a parent value, material delta, and confidence k.
/// Uses the formula: v = tanh(arctanh(v0) + k * delta)
pub fn extrapolate_value(parent_value: f64, material_delta_cp: i32, k: f32) -> f64 {
    // 1. Clamp parent value to avoid infinity at +/- 1.0
    let v0 = (parent_value as f32).clamp(-0.999, 0.999);
    
    // 2. Solve for x0 (Parent Logit)
    // x = 2 * atanh(v)
    let x0 = 2.0 * v0.atanh();
    
    // 3. Apply Symbolic Shift
    // material_delta_cp is in centipawns. 100 cp = 1 pawn.
    let material_units = material_delta_cp as f32 / 100.0;
    let shift = k * material_units;
    let new_logit = x0 + shift;
    
    // 4. Return new Value [ -1, 1 ]
    // v = tanh(x / 2)
    (new_logit / 2.0).tanh() as f64
}
