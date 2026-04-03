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
        self.v_fc = nn.Linear(65, 256) # 8x8 image * 1 channel = 64 + 1 q_result input
        self.v_out = nn.Linear(256, 1) # V_net (Residual)
        
        # --- DYNAMIC K (Confidence Scalar) ---
        # Single learned scalar: k = 0.47 * softplus(k_logit).
        # Texel-calibrated: at init k_logit=0 → k = 0.47 * ln(2) ≈ 0.326.
        # Maps PeSTO centipawn evals through tanh to calibrated win probabilities.
        self.k_logit = nn.Parameter(torch.tensor(0.0))

        # Apply Initialization
        self.apply(self._init_weights)
        
        # --- ZERO INITIALIZATION (Crucial) ---
        self._zero_init_heads()

    def forward(self, x, scalars_input):
        # scalars_input: [B, 2] — column 0 = q_result (material delta), column 1 = qsearch_completed flag
        # Ensure shape [B, 2]
        if scalars_input.dim() == 1:
            scalars_input = scalars_input.unsqueeze(1)
        if scalars_input.size(1) == 1:
            # Backward compat: pad with 1.0 (assume completed) if only q_result provided
            scalars_input = torch.cat([scalars_input, torch.ones_like(scalars_input)], dim=1)

        q_result = scalars_input[:, 0:1]   # [B, 1]

        # K: global scalar confidence (position-independent)
        k = 0.47 * F.softplus(self.k_logit)  # scalar, Texel-calibrated
        k_batch = k.unsqueeze(0).expand(x.size(0), 1)  # [B, 1]

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

        # Value Path (backbone-dependent + q_result input)
        v_feat = F.relu(self.v_bn(self.v_conv(x)))
        v_feat = v_feat.view(v_feat.size(0), -1)          # [B, 64]
        v_feat = torch.cat([v_feat, q_result], dim=1)      # [B, 65]
        v = F.relu(self.v_fc(v_feat))
        v_logit = self.v_out(v) # Logits

        # Residual Recombination
        # V_final = Tanh( V_net + k * q_result )
        total_logit = v_logit + (k_batch * q_result)
        value = torch.tanh(total_logit)

        if self.training:
            # Training: return tanh(v_logit + k * q_result) for loss computation
            return policy, value, k_batch
        else:
            # Inference: return raw v_logit so Rust can compute
            # tanh(v_logit + k * delta_M) with enhanced material Q-search
            return policy, v_logit, k_batch

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
        
        # 3. K scalar: k_logit is initialized to 0.0 in __init__
        # Softplus(0) = ln(2). k = 0.47 * ln(2) ≈ 0.326.
