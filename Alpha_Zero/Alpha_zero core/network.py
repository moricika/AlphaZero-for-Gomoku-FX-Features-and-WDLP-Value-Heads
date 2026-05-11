import math
from typing import NamedTuple, Tuple
import torch
from torch import nn
import torch.nn.functional as F


class NetworkOutputs(NamedTuple):
    """
    Network outputs for AlphaZero.
    
    Extended to support WDLP (Win-Draw-Loss-Plies) value head.
    Backward compatible: wdl and plies are optional.
    """
    pi_prob: torch.Tensor
    value: torch.Tensor
    wdl: torch.Tensor = None    # INNOVATION: Win-Draw-Loss probabilities [batch, 3]
    plies: torch.Tensor = None  # INNOVATION: Predicted remaining plies [batch, 1]


def calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """Calculate output dimensions after conv2d operation."""
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


def initialize_weights(net: nn.Module) -> None:
    """Initialize weights using Kaiming initialization."""
    assert isinstance(net, nn.Module)
    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class ResNetBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, num_filters: int) -> None:
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)
        return out


class WDLPValueHead(nn.Module):
    """
    Win-Draw-Loss-Plies (WDLP) value head.
    
    INNOVATION: Based on "Representation Matters for Mastering Chess" (Czech et al., 2024)
    Adapted for 7×7 Gomoku.
    
    Predicts:
    - Win/Draw/Loss probabilities (3-way classification)
    - Remaining plies until game end (regression)
    - Combined value from WDL: value = P(win) - P(loss)
    
    This is a KEY INNOVATION adapting chess AI advances to Gomoku.
    """

    def __init__(self, in_channels: int, conv_out_size: int) -> None:
        super().__init__()
        
        # Initial convolution to reduce channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        
        # Shared feature extractor
        self.fc_shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # WDL head (Win-Draw-Loss classification)
        self.wdl_head = nn.Linear(64, 3)
        
        # Plies head (predict remaining moves)
        self.plies_head = nn.Linear(64, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Feature tensor [batch, in_channels, height, width]
            
        Returns:
            value: Position evaluation [batch, 1] computed as P(win) - P(loss)
            wdl: Win-Draw-Loss probabilities [batch, 3]
            plies: Predicted remaining plies [batch, 1]
        """
        x = self.conv(x)
        features = self.fc_shared(x)
        
        # WDL probabilities (softmax over 3 classes)
        wdl_logits = self.wdl_head(features)
        wdl = F.softmax(wdl_logits, dim=1)  # [batch, 3]
        
        # Predicted remaining plies
        plies = self.plies_head(features)  # [batch, 1]
        
        # Compute value from WDL: value = P(win) - P(loss)
        value = wdl[:, 0:1] - wdl[:, 2:3]  # [batch, 1]
        
        return value, wdl, plies


class AlphaZeroNet(nn.Module):
    """
    AlphaZero neural network with policy and value heads.
    
    INNOVATION: Supports two modes:
    1. Standard mode (use_wdlp=False): Original AlphaZero with MSE value head
    2. FX mode (use_wdlp=True): Enhanced with WDLP value head (Czech et al., 2024)
    
    Architecture:
    - Initial conv block
    - N residual blocks  
    - Policy head (outputs action probabilities)
    - Value head (MSE or WDLP based on use_wdlp flag)
    
    Optimized for 7×7 Gomoku when gomoku=True.
    """

    def __init__(
        self,
        input_shape: Tuple,
        num_actions: int,
        num_res_block: int = 6,
        num_filters: int = 64,
        num_fc_units: int = 64,
        gomoku: bool = False,
        use_wdlp: bool = False,  # INNOVATION: Enable WDLP value head
    ) -> None:
        super().__init__()
        
        c, h, w = input_shape
        self.use_wdlp = use_wdlp

        # Gomoku uses extra padding to handle edge cases better
        num_padding = 3 if gomoku else 1
        conv_out_hw = calc_conv2d_output((h, w), 3, 1, num_padding)
        conv_out = conv_out_hw[0] * conv_out_hw[1]

        # Initial convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(c, num_filters, kernel_size=3, stride=1, padding=num_padding, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )

        # Stack of residual blocks
        res_blocks = [ResNetBlock(num_filters) for _ in range(num_res_block)]
        self.res_blocks = nn.Sequential(*res_blocks)

        # Policy head: predicts action probabilities
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * conv_out, num_actions),
        )

        # Value head: MSE (standard) or WDLP (FX mode)
        if use_wdlp:
            # INNOVATION: WDLP value head for enhanced evaluation
            self.value_head = WDLPValueHead(num_filters, conv_out)
        else:
            # Standard MSE value head (original AlphaZero)
            self.value_head = nn.Sequential(
                nn.Conv2d(num_filters, 1, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(1 * conv_out, num_fc_units),
                nn.ReLU(),
                nn.Linear(num_fc_units, 1),
                nn.Tanh(),
            )

        initialize_weights(self)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        
        Args:
            x: Input state tensor [batch, channels, height, width]
            
        Returns:
            If use_wdlp=False (original mode):
                pi_logits: Raw policy logits [batch, num_actions]
                value: Position evaluation [batch, 1] in range [-1, 1]
            
            If use_wdlp=True (FX mode):
                pi_logits: Raw policy logits [batch, num_actions]
                value: Position evaluation [batch, 1]
                wdl: Win-Draw-Loss probabilities [batch, 3]
                plies: Predicted remaining plies [batch, 1]
        """
        conv_out = self.conv_block(x)
        features = self.res_blocks(conv_out)

        pi_logits = self.policy_head(features)
        
        if self.use_wdlp:
            value, wdl, plies = self.value_head(features)
            return pi_logits, value, wdl, plies
        else:
            value = self.value_head(features)
            return pi_logits, value
    
    def predict(self, x: torch.Tensor) -> NetworkOutputs:
        """
        Predict with named outputs (for compatibility).
        
        Returns:
            NetworkOutputs with pi_prob, value, and optionally wdl and plies
        """
        outputs = self.forward(x)
        
        if self.use_wdlp:
            pi_logits, value, wdl, plies = outputs
            return NetworkOutputs(pi_prob=pi_logits, value=value, wdl=wdl, plies=plies)
        else:
            pi_logits, value = outputs
            return NetworkOutputs(pi_prob=pi_logits, value=value)