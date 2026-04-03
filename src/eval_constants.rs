//! Constants for the Pesto evaluation function module
//! Values from Rofchade: <http://www.talkchess.com/forum3/viewtopic.php?f=2&t=68311&start=19>
//! All tables are standard RofChade values (unmodified).
//! Note that these apparently use a different indexing, so we need to flip the board vertically for white.

// Piece values in middlegame
pub const MG_VALUE: [i32; 6] = [82, 337, 365, 477, 1025, 0];

// Piece values in endgame
pub const EG_VALUE: [i32; 6] = [94, 281, 297, 512, 936, 0];

// Piece-square tables

/// Piece-square tables for middlegame pawns
const MG_PAWN_TABLE: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 98, 134, 61, 95, 68, 126, 34, -11, -6, 7, 26, 31, 65, 56, 25, -20, -14,
    13, 6, 21, 23, 12, 17, -23, -27, -2, -5, 12, 17, 6, 10, -25, -26, -4, -4, -10, 3, 3, 33, -12,
    -35, -1, -20, -23, -15, 24, 38, -22, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Piece-square tables for endgame pawns
const EG_PAWN_TABLE: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 178, 173, 158, 134, 147, 132, 165, 187, 94, 100, 85, 67, 56, 53, 82,
    84, 32, 24, 13, 5, -2, 4, 17, 17, 13, 9, -3, -7, -7, -8, 3, -1, 4, 7, -6, 1, 0, -5, -1, -8, 13,
    8, 8, 10, 13, 0, 2, -7, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Piece-square tables for middlegame knights
const MG_KNIGHT_TABLE: [i32; 64] = [
    -167, -89, -34, -49, 61, -97, -15, -107, -73, -41, 72, 36, 23, 62, 7, -17, -47, 60, 37, 65, 84,
    129, 73, 44, -9, 17, 19, 53, 37, 69, 18, 22, -13, 4, 16, 13, 28, 19, 21, -8, -23, -9, 12, 10,
    19, 17, 25, -16, -29, -53, -12, -3, -1, 18, -14, -19, -105, -21, -58, -33, -17, -28, -19, -23,
];

/// Piece-square tables for endgame knights
const EG_KNIGHT_TABLE: [i32; 64] = [
    -58, -38, -13, -28, -31, -27, -63, -99, -25, -8, -25, -2, -9, -25, -24, -52, -24, -20, 10, 9,
    -1, -9, -19, -41, -17, 3, 22, 22, 22, 11, 8, -18, -18, -6, 16, 25, 16, 17, 4, -18, -23, -3, -1,
    15, 10, -3, -20, -22, -42, -20, -10, -5, -2, -20, -23, -44, -29, -51, -23, -15, -22, -18, -50,
    -64,
];

/// Piece-square tables for middlegame bishops
const MG_BISHOP_TABLE: [i32; 64] = [
    -29, 4, -82, -37, -25, -42, 7, -8, -26, 16, -18, -13, 30, 59, 18, -47, -16, 37, 43, 40, 35, 50,
    37, -2, -4, 5, 19, 50, 37, 37, 7, -2, -6, 13, 13, 26, 34, 12, 10, 4, 0, 15, 15, 15, 14, 27, 18,
    10, 4, 15, 16, 0, 7, 21, 33, 1, -33, -3, -14, -21, -13, -12, -39, -21,
];

/// Piece-square tables for endgame bishops
const EG_BISHOP_TABLE: [i32; 64] = [
    -14, -21, -11, -8, -7, -9, -17, -24, -8, -4, 7, -12, -3, -13, -4, -14, 2, -8, 0, -1, -2, 6, 0,
    4, -3, 9, 12, 9, 14, 10, 3, 2, -6, 3, 13, 19, 7, 10, -3, -9, -12, -3, 8, 10, 13, 3, -7, -15,
    -14, -18, -7, -1, 4, -9, -15, -27, -23, -9, -23, -5, -9, -16, -5, -17,
];

/// Piece-square tables for middlegame rooks
const MG_ROOK_TABLE: [i32; 64] = [
    32, 42, 32, 51, 63, 9, 31, 43, 27, 32, 58, 62, 80, 67, 26, 44, -5, 19, 26, 36, 17, 45, 61, 16,
    -24, -11, 7, 26, 24, 35, -8, -20, -36, -26, -12, -1, 9, -7, 6, -23, -45, -25, -16, -17, 3, 0,
    -5, -33, -44, -16, -20, -9, -1, 11, -6, -71, -19, -13, 1, 17, 16, 7, -37, -26,
];

/// Piece-square tables for endgame rooks
const EG_ROOK_TABLE: [i32; 64] = [
    13, 10, 18, 15, 12, 12, 8, 5, 11, 13, 13, 11, -3, 3, 8, 3, 7, 7, 7, 5, 4, -3, -5, -3, 4, 3, 13,
    1, 2, 1, -1, 2, 3, 5, 8, 4, -5, -6, -8, -11, -4, 0, -5, -1, -7, -12, -8, -16, -6, -6, 0, 2, -9,
    -9, -11, -3, -9, 2, 3, -1, -5, -13, 4, -20,
];

/// Piece-square tables for middlegame queens
const MG_QUEEN_TABLE: [i32; 64] = [
    -28, 0, 29, 12, 59, 44, 43, 45, -24, -39, -5, 1, -16, 57, 28, 54, -13, -17, 7, 8, 29, 56, 47,
    57, -27, -27, -16, -16, -1, 17, -2, 1, -9, -26, -9, -10, -2, -4, 3, -3, -14, 2, -11, -2, -5, 2,
    14, 5, -35, -8, 11, 2, 8, 15, -3, 1, -1, -18, -9, 10, -15, -25, -31, -50,
];

/// Piece-square tables for endgame queens
const EG_QUEEN_TABLE: [i32; 64] = [
    -9, 22, 22, 27, 27, 19, 10, 20, -17, 20, 32, 41, 58, 25, 30, 0, -20, 6, 9, 49, 47, 35, 19, 9,
    3, 22, 24, 45, 57, 40, 57, 36, -18, 28, 19, 47, 31, 34, 39, 23, -16, -27, 15, 6, 9, 17, 10, 5,
    -22, -23, -30, -16, -16, -23, -36, -32, -33, -28, -22, -43, -5, -32, -20, -41,
];

/// Piece-square tables for middlegame king (standard RofChade values)
const MG_KING_TABLE: [i32; 64] = [
    -65, 23, 16, -15, -56, -34, 2, 13,
    29, -1, -20, -7, -8, -4, -38, -29,
    -9, 24, 2, -16, -20, 6, 22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49, -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
    1, 7, -8, -64, -43, -16, 9, 8,
    -15, 36, 12, -54, 8, -28, 24, 14,
];

// Piece-square tables for endgame king
const EG_KING_TABLE: [i32; 64] = [
    -74, -35, -18, -18, -11, 15, 4, -17, -12, 17, 14, 17, 17, 38, 23, 11, 10, 17, 23, 15, 20, 45,
    44, 13, -8, 22, 24, 27, 26, 33, 26, 3, -18, -4, 21, 24, 27, 23, 9, -11, -19, -3, 11, 21, 23,
    16, 7, -9, -27, -11, 4, 13, 14, 4, -5, -17, -53, -34, -21, -11, -28, -14, -24, -43,
];

/// Middlegame pesto tables
pub const MG_PESTO_TABLE: [[i32; 64]; 6] = [
    MG_PAWN_TABLE,
    MG_KNIGHT_TABLE,
    MG_BISHOP_TABLE,
    MG_ROOK_TABLE,
    MG_QUEEN_TABLE,
    MG_KING_TABLE,
];

/// Endgame pesto tables
pub const EG_PESTO_TABLE: [[i32; 64]; 6] = [
    EG_PAWN_TABLE,
    EG_KNIGHT_TABLE,
    EG_BISHOP_TABLE,
    EG_ROOK_TABLE,
    EG_QUEEN_TABLE,
    EG_KING_TABLE,
];

/// Values of pieces to determine the phase of the game
/// Weighted sum of all pieces except pawns and kings.
/// Starts at 24 when all are still on the board, and decreases to 0 when all are gone.
pub const GAMEPHASE_INC: [i32; 6] = [0, 1, 1, 2, 4, 0];

// --- Default Evaluation Weights (used for initialization) ---
// These were previously `pub const` but are now defaults for the EvalWeights struct.

// Two Bishops Bonus (MG/EG)
const DEFAULT_TWO_BISHOPS_BONUS: [i32; 2] = [25, 35]; // [MG, EG]

// Passed Pawn Bonus by Rank (Index 0 = Rank 1, Index 7 = Rank 8) - From White's perspective
// Rank 1 and 8 are impossible for passed pawns, but included for array size.
// Values increase significantly as the pawn advances.
const DEFAULT_PASSED_PAWN_BONUS_MG: [i32; 8] = [0, 10, 15, 25, 40, 60, 90, 0];
const DEFAULT_PASSED_PAWN_BONUS_EG: [i32; 8] = [0, 20, 30, 50, 100, 150, 200, 0];

// King Safety Bonus per Pawn in Shield Zone (MG/EG)
// Zone typically includes squares directly and diagonally one step in front of the king.
const DEFAULT_KING_SAFETY_PAWN_SHIELD_BONUS: [i32; 2] = [15, 4]; // [MG, EG]

// King Attack Score constants removed for now.

// Isolated Pawn Penalty (MG/EG) - Applied per isolated pawn
const DEFAULT_ISOLATED_PAWN_PENALTY: [i32; 2] = [-10, -15]; // [MG, EG]

// Pawn Chain Bonus (MG/EG) - Applied per pawn defended by another pawn diagonally
const DEFAULT_PAWN_CHAIN_BONUS: [i32; 2] = [10, 15]; // [MG, EG]

// Pawn Duo Bonus (MG/EG) - Applied per pair of pawns on adjacent files on the same rank
const DEFAULT_PAWN_DUO_BONUS: [i32; 2] = [8, 12]; // [MG, EG]

// Mobile Pawn Duo Bonus (MG/EG) - Bonus if squares in front are clear. Indexed by square of one of the pawns.
// Values increase by rank and are slightly higher for central files (c-f).
// Assumes table is from White's perspective (needs flipping for Black).
const MOBILE_DUO_BONUS_TABLE_MG: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, // Rank 1
    0, 0, 0, 0, 0, 0, 0, 0, // Rank 2
    2, 3, 4, 5, 5, 4, 3, 2, // Rank 3
    4, 6, 8, 10, 10, 8, 6, 4, // Rank 4
    6, 9, 12, 15, 15, 12, 9, 6, // Rank 5
    8, 12, 16, 20, 20, 16, 12, 8, // Rank 6
    10, 15, 20, 25, 25, 20, 15, 10, // Rank 7 - Added bonus
    0, 0, 0, 0, 0, 0, 0, 0, // Rank 8
];
const MOBILE_DUO_BONUS_TABLE_EG: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, // Rank 1
    0, 0, 0, 0, 0, 0, 0, 0, // Rank 2
    3, 4, 5, 6, 6, 5, 4, 3, // Rank 3
    5, 8, 10, 12, 12, 10, 8, 5, // Rank 4
    7, 11, 15, 18, 18, 15, 11, 7, // Rank 5
    9, 14, 19, 24, 24, 19, 14, 9, // Rank 6
    12, 18, 24, 30, 30, 24, 18, 12, // Rank 7 - Added bonus
    0, 0, 0, 0, 0, 0, 0, 0, // Rank 8
];

// Mobile Pawn Duo tables remain const as they are complex PSTs
pub const MOBILE_PAWN_DUO_BONUS_MG: [i32; 64] = MOBILE_DUO_BONUS_TABLE_MG;
pub const MOBILE_PAWN_DUO_BONUS_EG: [i32; 64] = MOBILE_DUO_BONUS_TABLE_EG;

// Export public constants for tests
pub const ISOLATED_PAWN_PENALTY: [i32; 2] = DEFAULT_ISOLATED_PAWN_PENALTY;
pub const PASSED_PAWN_BONUS_MG: [i32; 8] = DEFAULT_PASSED_PAWN_BONUS_MG;
pub const PASSED_PAWN_BONUS_EG: [i32; 8] = DEFAULT_PASSED_PAWN_BONUS_EG;
pub const PAWN_CHAIN_BONUS: [i32; 2] = DEFAULT_PAWN_CHAIN_BONUS;
pub const PAWN_DUO_BONUS: [i32; 2] = DEFAULT_PAWN_DUO_BONUS;
pub const KING_SAFETY_PAWN_SHIELD_BONUS: [i32; 2] = DEFAULT_KING_SAFETY_PAWN_SHIELD_BONUS;
pub const TWO_BISHOPS_BONUS: [i32; 2] = DEFAULT_TWO_BISHOPS_BONUS;

// Doubled Rooks on 7th Rank Bonus (MG/EG) - Additional bonus if two rooks are on 7th
const DEFAULT_DOUBLED_ROOKS_ON_SEVENTH_BONUS: [i32; 2] = [50, 25]; // [MG, EG]

// Rook on Open File Bonus (MG/EG) - No pawns of either color on the file
const DEFAULT_ROOK_OPEN_FILE_BONUS: [i32; 2] = [25, 15]; // [MG, EG]

// Rook on Half-Open File Bonus (MG/EG) - No friendly pawns on the file
const DEFAULT_ROOK_HALF_OPEN_FILE_BONUS: [i32; 2] = [15, 8]; // [MG, EG]

// Backward Pawn Penalty (MG/EG) - Applied per backward pawn
const DEFAULT_BACKWARD_PAWN_PENALTY: [i32; 2] = [-8, -12]; // [MG, EG]

// King Attack Score Constants (Simplified)
// Bonus per attacking piece type near the enemy king zone (MG only)
// Zone typically includes squares around the king (e.g., 3x3 area)
const DEFAULT_KING_ATTACK_WEIGHTS: [i32; 6] = [0, 7, 7, 10, 15, 0]; // P, N, B, R, Q, K
                                                                    // Example: KING_ATTACK_SCORE = sum(KING_ATTACK_WEIGHTS[piece] for piece attacking zone)

// --- Mobility Constants ---
// Bonus per legal move for each piece type [N, B, R, Q] (MG/EG)
// Knights and Bishops benefit more in open endgames? Queens less critical?
const DEFAULT_MOBILITY_WEIGHTS_MG: [i32; 4] = [3, 3, 2, 1]; // N, B, R, Q
const DEFAULT_MOBILITY_WEIGHTS_EG: [i32; 4] = [4, 4, 3, 2]; // N, B, R, Q
                                                            // --- Redundant King Safety Constants Removed ---
                                                            // (Using KING_ATTACK_WEIGHTS array defined above instead)

/// Bonus for a rook behind a passed pawn [mg, eg]
const DEFAULT_ROOK_BEHIND_PASSED_PAWN_BONUS: [i32; 2] = [20, 30];

/// Bonus for a rook behind enemy passed pawn [mg, eg]
const DEFAULT_ROOK_BEHIND_ENEMY_PASSED_PAWN_BONUS: [i32; 2] = [25, 30];

/// Bonus for castling rights [mg, eg]
const DEFAULT_CASTLING_RIGHTS_BONUS: [i32; 2] = [15, 0];

// --- Trainable Weights Structure ---

#[derive(Clone, Debug)]
pub struct EvalWeights {
    // Piece Values (MG/EG) - Keeping these const for now
    // pub mg_value: [i32; 6],
    // pub eg_value: [i32; 6],

    // Bonus/Penalty Terms (MG/EG pairs or arrays)
    pub two_bishops_bonus: [i32; 2],
    pub passed_pawn_bonus_mg: [i32; 8],
    pub passed_pawn_bonus_eg: [i32; 8],
    pub king_safety_pawn_shield_bonus: [i32; 2],
    pub isolated_pawn_penalty: [i32; 2],
    pub pawn_chain_bonus: [i32; 2],
    pub pawn_duo_bonus: [i32; 2],
    // Mobile pawn bonus uses PSTs, keep const for now
    pub doubled_rooks_on_seventh_bonus: [i32; 2],
    pub rook_behind_passed_pawn_bonus: [i32; 2],
    pub rook_behind_enemy_passed_pawn_bonus: [i32; 2],
    pub castling_rights_bonus: [i32; 2],
    pub rook_open_file_bonus: [i32; 2],
    pub rook_half_open_file_bonus: [i32; 2],
    pub backward_pawn_penalty: [i32; 2],
    pub king_attack_weights: [i32; 6],
    pub mobility_weights_mg: [i32; 4], // N, B, R, Q
    pub mobility_weights_eg: [i32; 4], // N, B, R, Q
}

impl Default for EvalWeights {
    fn default() -> Self {
        EvalWeights {
            two_bishops_bonus: DEFAULT_TWO_BISHOPS_BONUS,
            passed_pawn_bonus_mg: DEFAULT_PASSED_PAWN_BONUS_MG,
            passed_pawn_bonus_eg: DEFAULT_PASSED_PAWN_BONUS_EG,
            king_safety_pawn_shield_bonus: DEFAULT_KING_SAFETY_PAWN_SHIELD_BONUS,
            isolated_pawn_penalty: DEFAULT_ISOLATED_PAWN_PENALTY,
            pawn_chain_bonus: DEFAULT_PAWN_CHAIN_BONUS,
            pawn_duo_bonus: DEFAULT_PAWN_DUO_BONUS,
            doubled_rooks_on_seventh_bonus: DEFAULT_DOUBLED_ROOKS_ON_SEVENTH_BONUS,
            rook_behind_passed_pawn_bonus: DEFAULT_ROOK_BEHIND_PASSED_PAWN_BONUS,
            rook_behind_enemy_passed_pawn_bonus: DEFAULT_ROOK_BEHIND_ENEMY_PASSED_PAWN_BONUS,
            castling_rights_bonus: DEFAULT_CASTLING_RIGHTS_BONUS,
            rook_open_file_bonus: DEFAULT_ROOK_OPEN_FILE_BONUS,
            rook_half_open_file_bonus: DEFAULT_ROOK_HALF_OPEN_FILE_BONUS,
            backward_pawn_penalty: DEFAULT_BACKWARD_PAWN_PENALTY,
            king_attack_weights: DEFAULT_KING_ATTACK_WEIGHTS,
            mobility_weights_mg: DEFAULT_MOBILITY_WEIGHTS_MG, // Use DEFAULT_ prefix
            mobility_weights_eg: DEFAULT_MOBILITY_WEIGHTS_EG, // Use DEFAULT_ prefix
        }
    }
}
