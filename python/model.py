import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# Shared Components
# ==========================================
class ResidualBlock(nn.Module):
    """Standard ResNet Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add -> ReLU"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

# ==========================================
# Symbolic Value Head
# ==========================================
class SymbolicValueHead(nn.Module):
    """
    A value head that learns a residual correction to a hard-coded material imbalance.
    Formula: V = tanh( V_net + k * DeltaM )
    """
    def __init__(self, input_channels, hidden_dim=256):
        super(SymbolicValueHead, self).__init__()
        
        # Standard Value Head components
        self.conv = nn.Conv2d(input_channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8 * 8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # Output V_net (logit)
        
        # The Learnable Symbolic Parameter k
        # Initialized to 0.5 as per specification
        self.k = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, material_scalar):
        """
        Args:
            x: Input feature map from the backbone [B, C, 8, 8]
            material_scalar: Material imbalance delta [B, 1] (or [B])
        """
        # Deep Value Calculation (V_net)
        v = F.relu(self.bn(self.conv(x)))
        v = v.view(v.size(0), -1) # Flatten [B, 64]
        v = F.relu(self.fc1(v))
        v_net = self.fc2(v) # Logit [B, 1]
        
        # Ensure material_scalar shape matches v_net
        if material_scalar.dim() == 1:
            material_scalar = material_scalar.unsqueeze(1) # [B] -> [B, 1]
            
        # Symbolic Residual Connection
        v_combined = v_net + (self.k * material_scalar)
        
        # Final Activation
        return torch.tanh(v_combined)

# ==========================================
# Caissawary Network
# ==========================================
class CaissawaryNet(nn.Module):
    """
    Neurosymbolic Chess Network
    Backbone: ResNet
    Policy: Standard
    Value: Symbolic Residual
    """
    def __init__(self, input_channels=12, filters=128, num_res_blocks=10, policy_output_size=4096):
        super(CaissawaryNet, self).__init__()
        
        # Input Block
        self.conv_input = nn.Conv2d(input_channels, filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(filters)
        
        # Residual Tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(filters) for _ in range(num_res_blocks)
        ])
        
        # Policy Head (Standard)
        self.policy_conv = nn.Conv2d(filters, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, policy_output_size)
        
        # Value Head (Symbolic)
        self.value_head = SymbolicValueHead(filters)

    def forward(self, x, material_scalar):
        # Input
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Tower
        for block in self.res_blocks:
            x = block(x)
            
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy = F.log_softmax(self.policy_fc(p), dim=1) # Log probabilities
        
        # Value Head
        value = self.value_head(x, material_scalar)
        
        return policy, value
