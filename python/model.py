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
        # Separate shallow input network — k is a position-type property,
        # not a position-specific one. Operating on raw input prevents
        # overfitting k to specific positions via deep backbone features.
        self.k_conv = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False)
        self.k_bn = nn.BatchNorm2d(32)
        self.k_pool = nn.AdaptiveAvgPool2d(1)  # [B, 32, 1, 1] -> [B, 32]
        self.k_fc = nn.Linear(32, 16)
        self.k_out = nn.Linear(16, 1)  # K_net (Confidence)
        
        # Denominator Constant: 2 * ln(2)
        self.k_scale = 2 * math.log(2)

        # Apply Initialization
        self.apply(self._init_weights)
        
        # --- ZERO INITIALIZATION (Crucial) ---
        self._zero_init_heads()

    def forward(self, x, material_scalar):
        # K path (from raw input, independent of backbone)
        k_feat = F.relu(self.k_bn(self.k_conv(x)))       # [B, 32, 8, 8]
        k_feat = self.k_pool(k_feat).view(x.size(0), -1) # [B, 32]
        k_feat = F.relu(self.k_fc(k_feat))                # [B, 16]
        k_logit = self.k_out(k_feat)                      # [B, 1]

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
