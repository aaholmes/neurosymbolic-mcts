import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. The Squeeze-and-Excitation Block ---
# "The Volume Knob" for feature channels.
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Squeeze: Global Average Pooling (B, C, H, W) -> (B, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excite: Learn which channels matter
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # Flatten for Linear layer
        y = self.fc(y).view(b, c, 1, 1) # Reshape back to broadcast
        return x * y # Scale original features

# --- 2. The Residual Block (with SE) ---
# The standard "Workhorse" of the network
class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        # Insert SE Block here
        self.se = SEBlock(hidden_dim, reduction=16)

    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply SE Attention BEFORE adding the residual
        out = self.se(out) 
        
        out += residual
        return F.relu(out)

# --- 3. The Main Network (OracleNet) ---
class OracleNet(nn.Module):
    def __init__(self, num_blocks=6, hidden_dim=128, input_channels=17, policy_output_size=4672): 
        # Defaulting input_channels to 17 to match current Rust implementation
        # Defaulting hidden_dim to 128 to match previous capacity
        super().__init__()
        
        # Input Convolution
        self.start_conv = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.start_bn = nn.BatchNorm2d(hidden_dim)
        
        # Backbone: Stack of SE-ResBlocks
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # --- POLICY HEAD (73 Planes) ---
        # 1. Intermediate Conv
        self.p_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.p_bn = nn.BatchNorm2d(hidden_dim)
        # 2. Final Output Conv (Raw Logits, No Bias usually, but needed for Zero Init if no BN)
        # Note: We output 73 planes (AlphaZero standard)
        self.p_head = nn.Conv2d(hidden_dim, 73, kernel_size=1) 
        
        # --- VALUE HEAD (Neurosymbolic) ---
        # 1. Intermediate Conv (1x1 to reduce dimensions)
        self.v_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1) 
        self.v_bn = nn.BatchNorm2d(1)
        # 2. Flatten + Linear
        self.v_fc = nn.Linear(64, 256) # 8x8 image * 1 channel = 64 inputs
        self.v_out = nn.Linear(256, 1) # V_net (Residual)
        
        # --- DYNAMIC K HEAD (Confidence) ---
        # Handcrafted scalar features (8 values) + 5x5 king patches.
        # k is a position-type property: open vs closed, king safety, endgame.
        # Handcrafted features give k direct access to exactly the features
        # that determine material convertibility, without needing to rediscover
        # them from raw convolutions.
        self.k_stm_patch_fc = nn.Linear(12 * 25, 32)   # STM king 5x5 neighborhood
        self.k_opp_patch_fc = nn.Linear(12 * 25, 32)   # Opponent king 5x5 neighborhood
        self.k_combine = nn.Linear(8 + 32 + 32, 32)    # scalars + both king patches
        self.k_out = nn.Linear(32, 1)                   # k_logit -> K_net (Confidence)
        
        # Denominator Constant: 2 * ln(2)
        self.k_scale = 2 * math.log(2)

        # Apply Initialization
        self.apply(self._init_weights)
        
        # --- ZERO INITIALIZATION (Crucial) ---
        self._zero_init_heads()

    def _extract_k_scalars(self, x):
        """Extract 8 handcrafted scalar features from the raw input tensor.

        Args:
            x: [B, 17, 8, 8] input tensor (STM perspective)
        Returns:
            [B, 8] tensor of scalar features
        """
        B = x.size(0)

        # 1. Total pawns (STM + opponent)
        total_pawns = x[:, 0].sum(dim=(1, 2)) + x[:, 6].sum(dim=(1, 2))  # [B]

        # 2. STM non-pawn pieces (N, B, R, Q = planes 1-4)
        stm_pieces = x[:, 1:5].sum(dim=(1, 2, 3))  # [B]

        # 3. Opponent non-pawn pieces (N, B, R, Q = planes 7-10)
        opp_pieces = x[:, 7:11].sum(dim=(1, 2, 3))  # [B]

        # 4. STM queen present (plane 4)
        stm_queen = x[:, 4].sum(dim=(1, 2))  # [B] (0 or 1)

        # 5. Opponent queen present (plane 10)
        opp_queen = x[:, 10].sum(dim=(1, 2))  # [B] (0 or 1)

        # 6. Pawn contacts: STM pawn at row r, opponent pawn at row r-1
        # In STM perspective, STM pawns advance toward row 0
        stm_pawns = x[:, 0]   # [B, 8, 8]
        opp_pawns = x[:, 6]   # [B, 8, 8]
        contacts = (stm_pawns[:, 1:, :] * opp_pawns[:, :-1, :]).sum(dim=(1, 2))  # [B]

        # 7. Castling rights (planes 13-16, each is full-board 0 or 1)
        castling = x[:, 13:17, 0, 0].sum(dim=1)  # [B] (0-4)

        # 8. STM king rank (from plane 5)
        king_pos = x[:, 5].view(B, 64).argmax(dim=1)  # [B]
        stm_king_rank = (king_pos // 8).float()  # [B] (0-7)

        return torch.stack([
            total_pawns, stm_pieces, opp_pieces, stm_queen, opp_queen,
            contacts, castling, stm_king_rank
        ], dim=1)  # [B, 8]

    def _extract_king_patch(self, x, king_plane):
        """Extract 5x5 patch of piece planes centered on a king.

        Args:
            x: [B, 17, 8, 8] input tensor
            king_plane: plane index for the king (5=STM, 11=opponent)
        Returns:
            [B, 300] flattened patch (12 channels x 5 x 5)
        """
        B = x.size(0)
        piece_planes = x[:, :12]  # [B, 12, 8, 8]
        padded = F.pad(piece_planes, (2, 2, 2, 2), value=0)  # [B, 12, 12, 12]

        # Find king position from the specified plane
        king_pos = x[:, king_plane].view(B, 64).argmax(dim=1)  # [B]
        king_row = king_pos // 8  # [B]
        king_col = king_pos % 8   # [B]

        # Vectorized extraction using advanced indexing
        row_off = torch.arange(5, device=x.device)
        col_off = torch.arange(5, device=x.device)

        # Build index tensors [B, 12, 5, 5]
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1, 1).expand(B, 12, 5, 5)
        ch_idx = torch.arange(12, device=x.device).view(1, 12, 1, 1).expand(B, 12, 5, 5)
        row_idx = (king_row.view(B, 1, 1, 1) + row_off.view(1, 1, 5, 1)).expand(B, 12, 5, 5)
        col_idx = (king_col.view(B, 1, 1, 1) + col_off.view(1, 1, 1, 5)).expand(B, 12, 5, 5)

        patches = padded[batch_idx, ch_idx, row_idx, col_idx]  # [B, 12, 5, 5]
        return patches.view(B, -1)  # [B, 300]

    def forward(self, x, material_scalar):
        # K path: handcrafted scalars + king patches (independent of backbone)
        scalars = self._extract_k_scalars(x)                        # [B, 8]
        stm_patch = self._extract_king_patch(x, king_plane=5)       # [B, 300]
        opp_patch = self._extract_king_patch(x, king_plane=11)      # [B, 300]

        stm_feat = F.relu(self.k_stm_patch_fc(stm_patch))           # [B, 32]
        opp_feat = F.relu(self.k_opp_patch_fc(opp_patch))           # [B, 32]

        k_feat = torch.cat([scalars, stm_feat, opp_feat], dim=1)    # [B, 72]
        k_feat = F.relu(self.k_combine(k_feat))                     # [B, 32]
        k_logit = self.k_out(k_feat)                                # [B, 1]

        # Backbone
        x = F.relu(self.start_bn(self.start_conv(x)))

        for block in self.res_blocks:
            x = block(x)

        # Policy Path
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = self.p_head(p) # [B, 73, 8, 8]

        # Flatten correctly for AlphaZero mapping: [B, 8, 8, 73] -> [B, 4672]
        p = p.permute(0, 2, 3, 1) # [B, 8, 8, 73]
        p = p.contiguous().view(p.size(0), -1) # Flatten to [B, 4672]
        policy = F.log_softmax(p, dim=1)

        # Value Path (backbone-dependent)
        v_feat = F.relu(self.v_bn(self.v_conv(x)))
        v_feat = v_feat.view(v_feat.size(0), -1)
        v = F.relu(self.v_fc(v_feat))
        v_logit = self.v_out(v) # Logits
        
        # Calculate k (Confidence Scalar)
        # k = Softplus(k_logit) / (2 * ln2)
        k = F.softplus(k_logit) / self.k_scale
        
        # Ensure material_scalar matches shape [B, 1]
        if material_scalar.dim() == 1:
            material_scalar = material_scalar.unsqueeze(1)
            
        # Residual Recombination
        # V_final = Tanh( V_net + k * DeltaM )
        total_logit = v_logit + (k * material_scalar)
        value = torch.tanh(total_logit)

        if self.training:
            # Training: return tanh(v_logit + k * material) for loss computation
            return policy, value, k
        else:
            # Inference: return raw v_logit so Rust can compute
            # tanh(v_logit + k * delta_M) with enhanced material Q-search
            return policy, v_logit, k

    # Standard He Initialization for the body
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # --- THE MAGIC SAUCE ---
    def _zero_init_heads(self):
        """
        Force the final layers to output 0.0 at the start.
        """
        # 1. Policy Head: Zero weights & bias
        # Result: Softmax(0, 0, ...) = Uniform Distribution
        nn.init.constant_(self.p_head.weight, 0.0)
        nn.init.constant_(self.p_head.bias, 0.0)
        
        # 2. Value Head (V_net): Zero weights & bias
        # Result: V_net = 0. 
        # Total Value = Tanh(0 + k * Material) = Material Value
        nn.init.constant_(self.v_out.weight, 0.0)
        nn.init.constant_(self.v_out.bias, 0.0)
        
        # 3. K Head: Initialize to match k=0.5
        # Softplus(0) = ln(2). k = ln(2) / (2*ln(2)) = 0.5.
        nn.init.constant_(self.k_out.weight, 0.0)
        nn.init.constant_(self.k_out.bias, 0.0)
