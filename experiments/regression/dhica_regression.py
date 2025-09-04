import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from typing import Optional, Tuple, Dict, List

def exponential_linspace_int(start: int, end: int, num: int, divisible_by: int = 1) -> List[int]:
    """reference lines 328-334 enformer.py"""
    def round_to_divisible(x):
        return int(np.round(x / divisible_by) * divisible_by)
        
    if num == 1:
        return [end]
        
    base = np.exp(np.log(end / start) / (num - 1))
    return [round_to_divisible(start * base**i) for i in range(num)]


class RelativePositionalBias(nn.Module):
    def __init__(self, num_heads: int, max_distance: int = 1000,
                 num_bases: int = 16, feature_type: str = 'exponential',
                 min_half_life: float = 3.0, max_time: float = 10000.0):
      
        super(RelativePositionalBias, self).__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.num_bases = num_bases
        self.feature_type = feature_type
        self.min_half_life = min_half_life
        self.max_time = max_time

        # precompute relative positions [-max_distance, +max_distance]
        positions = torch.arange(-max_distance, max_distance + 1)
        self.register_buffer('positions', positions)

        # precompute basis features for all relative positions
        basis_features = self._compute_basis_features(positions)
        self.register_buffer('basis_features', basis_features)

        # learnable weights for basis functions
        self.basis_weights = nn.Parameter(
            torch.randn(num_heads, basis_features.size(-1))
        )

    def _compute_basis_features(self, positions: torch.Tensor) -> torch.Tensor:
        abs_pos = positions.abs().float()

        if self.feature_type == 'exponential':
            return self._exponential_basis(abs_pos)
        elif self.feature_type == 'gamma':
            return self._gamma_basis(abs_pos)
        elif self.feature_type == 'central_mask':
            return self._central_mask(abs_pos)
        elif self.feature_type == 'cosine':
            return self._cosine_basis(positions)
        elif self.feature_type == 'linear_masks':
            return self._linear_masks(abs_pos)
        elif self.feature_type == 'sin_cos':
            return self._sin_cos_basis(positions)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

    def _exponential_basis(self, abs_pos: torch.Tensor) -> torch.Tensor:
        max_range = math.log2(self.max_distance)
        half_lives = torch.pow(
            2.0, torch.linspace(self.min_half_life, max_range, self.num_bases)
        ).view(1, -1)
        features = torch.exp(-math.log(2.0) * abs_pos[:, None] / half_lives)
        return features

    def _gamma_basis(self, abs_pos: torch.Tensor) -> torch.Tensor:
        seq_length = float(self.max_distance)
        stddev = seq_length / (2 * self.num_bases)
        start_mean = seq_length / self.num_bases
        mean = torch.linspace(start_mean, seq_length, steps=self.num_bases).view(1, -1)
        concentration = (mean / stddev) ** 2
        rate = mean / (stddev ** 2)
        x = abs_pos[:, None] 
        log_unnormalized_prob = (concentration - 1) * torch.log(x + 1e-8) - rate * x
        log_normalization = (
            torch.lgamma(concentration) - concentration * torch.log(rate + 1e-8)
        )
        probs = torch.exp(log_unnormalized_prob - log_normalization)
        probs = probs / probs.max(dim=1, keepdim=True).values
        probs += 1e-8
        return probs

    def _central_mask(self, abs_pos: torch.Tensor) -> torch.Tensor:
        center_widths = torch.pow(2.0, torch.arange(1, self.num_bases + 1)).view(1, -1)
        features = (center_widths > abs_pos[:, None]).float()
        return features

    def _cosine_basis(self, positions: torch.Tensor) -> torch.Tensor:
        periodicity = 1.25 * torch.pow(2.0, torch.arange(self.num_bases)).view(1, -1)
        features = torch.cos(2 * math.pi * positions[:, None] / periodicity)
        return features

    def _linear_masks(self, abs_pos: torch.Tensor) -> torch.Tensor:
        distances = torch.arange(0, self.num_bases).view(1, -1)
        features = (distances == abs_pos[:, None]).float()
        return features

    def _sin_cos_basis(self, positions: torch.Tensor) -> torch.Tensor:
        if self.num_bases % 2 != 0:
            raise ValueError("num_bases must be even for sin/cos features.")
        i = torch.arange(0, self.num_bases, 2).float().view(1, -1)
        div_term = torch.pow(self.max_time, i / self.num_bases)
        pos_enc = torch.cat([
            torch.sin(positions[:, None] / div_term),
            torch.cos(positions[:, None] / div_term)
        ], dim=-1)
        return pos_enc

    def forward(self, query_len: int, key_len: int) -> torch.Tensor:
        relative_positions = torch.arange(key_len) - torch.arange(query_len)[:, None]
        relative_positions = relative_positions.clamp(-self.max_distance, self.max_distance)
        relative_positions += self.max_distance

        basis = self.basis_features[relative_positions]
        bias = torch.einsum('qkb,hb->hqk', basis, self.basis_weights)
        return bias


class GELU(nn.Module):
    """GELU activation function"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(1.702 * x) * x


class ConvBlock(nn.Module):
    """BatchNorm -> GELU -> Conv1D"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, 
                 padding: int = 0, dropout: float = 0.0):
        super(ConvBlock, self).__init__()
        
        self.bn = nn.BatchNorm1d(in_channels)
        self.activation = GELU()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class SoftmaxPooling1D(nn.Module):
    """Attention-based pooling"""
    def __init__(self, pool_size: int = 2, per_channel: bool = True, w_init_scale: float = 2.0):
        super().__init__()
        self.pool_size = pool_size
        self.per_channel = per_channel
        self.w_init_scale = w_init_scale
        self.logit_linear = None
    
    def _initialize(self, num_features: int, device: torch.device):
        if self.logit_linear is None:
            output_size = num_features if self.per_channel else 1
            self.logit_linear = nn.Linear(num_features, output_size, bias=False)
            
            with torch.no_grad():
                if self.per_channel:
                    self.logit_linear.weight.data = torch.eye(num_features) * self.w_init_scale
                else:
                    self.logit_linear.weight.data.fill_(self.w_init_scale)
            
            self.logit_linear = self.logit_linear.to(device)
            self.add_module('logit_linear', self.logit_linear)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_features, seq_len = x.shape
        
        if seq_len % self.pool_size != 0:
            trim_len = seq_len - (seq_len % self.pool_size)
            x = x[:, :, :trim_len]
            seq_len = trim_len
        
        self._initialize(num_features, x.device)
        
        x_reshaped = x.view(batch_size, num_features, seq_len // self.pool_size, self.pool_size)
        x_transposed = x_reshaped.permute(0, 2, 3, 1)
        
        logits = self.logit_linear(x_transposed)
        weights = F.softmax(logits, dim=-2)
        
        if self.per_channel:
            weighted = x_transposed * weights
        else:
            weighted = x_transposed * weights
        
        pooled = weighted.sum(dim=-2).permute(0, 2, 1)
        return pooled


class ResidualBlock(nn.Module):
    """Wrapper for residual connection"""
    def __init__(self, module: nn.Module):
        super(ResidualBlock, self).__init__()
        self.module = module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional bias"""
    def __init__(self, d_model: int, num_heads: int, 
                 key_size: int = 64, value_size: int = None,
                 attention_dropout: float = 0.05, output_dropout: float = 0.4):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or d_model // num_heads
        
        self.w_q = nn.Linear(d_model, num_heads * key_size, bias=False)
        self.w_k = nn.Linear(d_model, num_heads * key_size, bias=False)
        self.w_v = nn.Linear(d_model, num_heads * self.value_size, bias=False)
        self.w_o = nn.Linear(num_heads * self.value_size, d_model, bias=False)
        
        self.relative_bias = RelativePositionalBias(num_heads)
        
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout = nn.Dropout(output_dropout)
        
        with torch.no_grad():
            self.w_o.weight.zero_()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.value_size).transpose(1, 2)
        
        scale = math.sqrt(self.key_size)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        relative_bias = self.relative_bias(seq_len, seq_len)
        scores = scores + relative_bias.unsqueeze(0)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        out = torch.matmul(attention_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.value_size)
        out = self.w_o(out)
        out = self.output_dropout(out)
        
        return out, attention_weights


class TransformerBlock(nn.Module):
    """Transformer block with attention and FFN"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.4):
        super().__init__()
        
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            key_size=64,
            value_size=d_model // num_heads,
            attention_dropout=0.05,
            output_dropout=dropout
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.Dropout(dropout),
            GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attention_weights = self.attention(self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attention_weights

class FusionLayer(nn.Module):
    """Fusion layer combining DNA and DNase features"""
    def __init__(self, channels: int, dropout: float = 0.1, fusion_type: str = 'concat', 
                 d_attn: int = 128, entropy_lambda: float = 0.1):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.entropy_lambda = entropy_lambda
        
        if fusion_type == 'concat':
            # Original implementation
            self.projection = nn.Sequential(
                nn.BatchNorm1d(channels * 2),
                GELU(),
                nn.Conv1d(channels * 2, channels, kernel_size=1),
                nn.Dropout(dropout)
            )
            
        elif fusion_type == 'cross_attention':
            # Original implementation
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=channels,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.norm = nn.LayerNorm(channels)
        
        elif fusion_type == 'gated':
            # Original implementation
            self.gate = nn.Sequential(
                nn.Conv1d(channels * 2, channels, kernel_size=1),
                nn.Sigmoid()
            )
            
        elif fusion_type == 'mil':
            # Multiple Instance Learning with gated attention
            self.d_attn = d_attn
            
            # gated attention parameters for  dna
            self.U_dna = nn.Linear(channels, d_attn, bias=False) # for tanh
            self.V_dna = nn.Linear(channels, d_attn, bias=False) # for sigmoid
            self.b_dna = nn.Parameter(torch.zeros(d_attn)) # for tanh
            self.c_dna = nn.Parameter(torch.zeros(d_attn)) # for sigmoid
            

            # gated attention parameters for dnase
            self.U_dnase = nn.Linear(channels, d_attn, bias=False)
            self.V_dnase = nn.Linear(channels, d_attn, bias=False)
            self.b_dnase = nn.Parameter(torch.zeros(d_attn))
            self.c_dnase = nn.Parameter(torch.zeros(d_attn))
            
            # used to score each modality's gated output
            self.W = nn.Parameter(torch.randn(d_attn))
            
            self.attention_weights = None
    
    def compute_gated_attention(self, h_dna: torch.Tensor, h_dnase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gated attention weights for MIL fusion"""
        # h_dna, h_dnase: (batch, channels)
        
        # gated attention for DNA
        gate_dna = torch.sigmoid(self.U_dna(h_dna) + self.c_dna)
        attention_dna = torch.tanh(self.V_dna(h_dna) + self.b_dna)
        gated_dna = gate_dna * attention_dna  # (batch, d_attn)
        
        # gated attention for DNase
        gate_dnase = torch.sigmoid(self.U_dnase(h_dnase) + self.c_dnase)
        attention_dnase = torch.tanh(self.V_dnase(h_dnase) + self.b_dnase)
        gated_dnase = gate_dnase * attention_dnase  # (batch, d_attn)
        
        # attention scores
        score_dna = torch.matmul(gated_dna, self.W)  # (batch,)
        score_dnase = torch.matmul(gated_dnase, self.W)  # (batch,)
        
        # stack and apply softmax
        # these weights will be used to fuse modality outputs
        scores = torch.stack([score_dna, score_dnase], dim=1)  # (batch, 2)
        attention_weights = F.softmax(scores, dim=1)  # (batch, 2)
        
        return attention_weights[:, 0:1], attention_weights[:, 1:2]  # Split back
    
    def forward(self, dna_features: torch.Tensor, dnase_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dna_features: (batch, channels, seq_len)
            dnase_features: (batch, channels, seq_len)
        """
        
        if self.fusion_type == 'concat':
            concatenated = torch.cat([dna_features, dnase_features], dim=1)
            fused = self.projection(concatenated)
            
        elif self.fusion_type == 'cross_attention':
            dna_trans = dna_features.transpose(1, 2)
            dnase_trans = dnase_features.transpose(1, 2)
            attended_dna, _ = self.cross_attention(dna_trans, dnase_trans, dnase_trans)
            attended_dna = self.norm(attended_dna + dna_trans)
            fused = attended_dna.transpose(1, 2)
            
        elif self.fusion_type == 'gated':
            concatenated = torch.cat([dna_features, dnase_features], dim=1)
            gate = self.gate(concatenated)
            fused = gate * dna_features + (1 - gate) * dnase_features
            
        elif self.fusion_type == 'mil':
            # Multiple Instance Learning fusion from BioLangFusion Paper
            batch_size, channels, seq_len = dna_features.shape
            
            # Mean pool across sequence to get modality representations
            h_dna = dna_features.mean(dim=2)  # (batch, channels)
            h_dnase = dnase_features.mean(dim=2)  # (batch, channels)
            
            # Compute attention weights
            alpha_dna, alpha_dnase = self.compute_gated_attention(h_dna, h_dnase)
            
            # Store for entropy regularization
            self.attention_weights = torch.cat([alpha_dna, alpha_dnase], dim=1)
            
            # Expand attention weights to match sequence length
            alpha_dna = alpha_dna.unsqueeze(2).expand(-1, -1, seq_len)  # (batch, 1, seq_len)
            alpha_dnase = alpha_dnase.unsqueeze(2).expand(-1, -1, seq_len)  # (batch, 1, seq_len)
            
            # Weighted combination
            fused = alpha_dna * dna_features + alpha_dnase * dnase_features
            
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
        
        return fused
    
    def get_attention_entropy(self) -> torch.Tensor:
        """Compute entropy of attention weights for regularization"""
        if self.attention_weights is None:
            return torch.tensor(0.0)
        
        # Avoid log(0) by adding small epsilon
        eps = 1e-8
        entropy = -torch.sum(self.attention_weights * torch.log(self.attention_weights + eps), dim=1)
        return entropy.mean()


class SeparatePathwayModel(nn.Module):
    """dHICA-inspired model with separate DNA and DNase pathways - REGRESSION VERSION"""
    
    def __init__(self,
                 channels: int = 768,
                 num_transformer_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.4,
                 num_targets: int = 7,  # Changed from num_histones for regression
                 pooling_type: str = 'attention',
                 num_conv_blocks: int = 2,
                 fusion_type: str = 'concat'):
        super().__init__()
        
        print(f'Separate Pathway Model initialized (REGRESSION)')
        print(f'Channels: {channels}, Transformer layers: {num_transformer_layers}, Fusion: {fusion_type}')
        
        self.channels = channels
        self.fusion_type = fusion_type
        
        # separate pathways for DNA and DNase
        self.dna_pathway = self._build_pathway(
            input_channels=4,  # A, T, G, C
            output_channels=channels,
            pathway_name="DNA",
            pooling_type=pooling_type,
            num_conv_blocks=num_conv_blocks
        )
        
        self.dnase_pathway = self._build_pathway(
            input_channels=1,  # DNase signal
            output_channels=channels,
            pathway_name="DNase", 
            pooling_type=pooling_type,
            num_conv_blocks=num_conv_blocks
        )
        
        # fusion layer
        self.fusion = FusionLayer(channels, dropout=dropout/2, fusion_type=fusion_type)
        
        # shared transformer after fusion
        self.transformer = nn.ModuleList([
            TransformerBlock(channels, num_heads, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        self.output_bins = 16
        self.num_targets = num_targets
        # final prediction head for REGRESSION
        self.final_conv = ConvBlock(channels, channels * 2, kernel_size=1, dropout=dropout/8)
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels * 2, num_targets * self.output_bins),
            # NO SIGMOID for regression - continuous outputs
        )
        
        #self.regressor = nn.Conv1d(channels * 2, num_targets, kernel_size=1)

        self.attention_weights = []

    def get_fusion_entropy(self) -> torch.Tensor:
        """Get entropy from fusion layer for regularization"""
        if hasattr(self.fusion, 'get_attention_entropy'):
            return self.fusion.get_attention_entropy()
        return torch.tensor(0.0)
    
    def _build_pathway(self, input_channels: int, output_channels: int, 
                      pathway_name: str, pooling_type: str, num_conv_blocks: int):
        """Build individual pathway (DNA or DNase)"""
        
        # initial convolution and pooling
        # Use wider kernel for DNA, smaller for DNase
        stem_channels = output_channels // 2
        stem = self._make_pathway_stem(input_channels, stem_channels, pathway_name, pooling_type)
        
        # Get actual stem output channels
        actual_stem_out = self._get_stem_output_channels(stem)
        
        # Pathway-specific conv tower
        conv_tower = self._build_conv_tower(
            actual_stem_out, output_channels, pooling_type, num_conv_blocks
        )
        
        return nn.Sequential(stem, conv_tower)
    
    def _make_pathway_stem(self, input_channels: int, stem_out_channels: int, 
                          pathway_name: str, pooling_type: str):
        """Build pathway-specific stem"""
        layers = []
        
        if pathway_name == "DNA":
            # DNA-specific: wider kernel for motif detection
            layers.append(nn.Conv1d(input_channels, stem_out_channels, kernel_size=15, padding=7))
        else:  # DNase
            # DNase-specific: smaller kernel for accessibility patterns  
            layers.append(nn.Conv1d(input_channels, stem_out_channels, kernel_size=7, padding=3))
        
        
        layers.append(ResidualBlock(
            ConvBlock(stem_out_channels, stem_out_channels, kernel_size=1)
        ))
        
        if pooling_type == 'attention': # like in enformer
            layers.append(SoftmaxPooling1D(pool_size=2, w_init_scale=2.0))
        else:
            layers.append(nn.MaxPool1d(kernel_size=2, padding=0))
        
        return nn.Sequential(*layers)
    
    def _get_stem_output_channels(self, stem_module):
        """Get actual output channels from stem"""
        for module in reversed(list(stem_module.modules())):
            if isinstance(module, nn.Conv1d):
                return module.out_channels
            elif isinstance(module, ConvBlock):
                return module.conv.out_channels
        return None
    
    def _build_conv_tower(self, in_channels: int, out_channels: int, pooling_type: str, num_blocks: int = 2):
        """Build convolutional tower"""
        
        filter_list = exponential_linspace_int(
            start=in_channels, 
            end=out_channels, 
            num=num_blocks,
            divisible_by=128
        )
        
        blocks = []
        current_channels = in_channels
        
        for i, num_filters in enumerate(filter_list):
            block = nn.Sequential(
                ConvBlock(current_channels, num_filters, kernel_size=5, padding=2),
                ResidualBlock(ConvBlock(num_filters, num_filters, kernel_size=1)),
                SoftmaxPooling1D(pool_size=2, per_channel=True, w_init_scale=2.0) if pooling_type == 'attention' 
                else nn.MaxPool1d(kernel_size=2, padding=0)
            )
            blocks.append(block)
            current_channels = num_filters
        
        return nn.Sequential(*blocks)
    
    def forward(self, dna_input: torch.Tensor, dnase_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dna_input: (batch, 4, seq_len) - one-hot DNA sequence
            dnase_input: (batch, 1, seq_len) - DNase signal
        """
        
        # Process through separate pathways
        dna_features = self.dna_pathway(dna_input)
        dnase_features = self.dnase_pathway(dnase_input)
        
        # Fusion
        fused_features = self.fusion(dna_features, dnase_features)
        
        # Convert to transformer format
        x = fused_features.transpose(1, 2)  # (batch, seq_len, channels)
        
        # Transformer layers
        self.attention_weights = []
        for transformer_block in self.transformer:
            x, attn_weights = transformer_block(x)
            self.attention_weights.append(attn_weights)
        
        # Final processing
        x = x.transpose(1, 2)  # Back to conv format
        x = self.final_conv(x)
        
        # Regression output (continuous values)
        output = self.regressor(x)
        output = output.view(x.size(0), self.num_targets, self.output_bins)
        return output
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """Return stored attention weights"""
        return self.attention_weights


class DeepHistoneEnformer:
    """Training wrapper for separate pathway model - REGRESSION VERSION"""
    def __init__(self, 
                 use_gpu: bool = True,
                 learning_rate: float = 0.001,
                 channels: int = 768,
                 num_transformer_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.4,
                 num_targets: int = 7,  # Changed from num_histones
                 pooling_type: str = 'attention',
                 num_conv_blocks: int = 2,
                 fusion_type: str = 'concat'):
        
        self.forward_fn = SeparatePathwayModel(
            channels=channels,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_targets=num_targets,  # Changed from num_histones
            pooling_type=pooling_type,
            num_conv_blocks=num_conv_blocks,
            fusion_type=fusion_type
        )
        
        # CHANGED: Use MSE loss for regression instead of BCE
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.forward_fn.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.forward_fn = self.forward_fn.cuda()
            self.criterion = self.criterion.cuda()
    
    def updateLR(self, fold: float):
        """Update learning rate by multiplication factor"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= fold
    
    def _separate_inputs(self, seq_batch: np.ndarray, dns_batch: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate DNA and DNase inputs for the dual pathway model"""
        
        # Remove singleton dimensions first
        seq_batch = seq_batch.squeeze()  # (16, 1, 4, 1024) -> (16, 4, 1024)
        dns_batch = dns_batch.squeeze()  # (16, 1, 1, 1024) -> (16, 1024)
        
        # DNA: Now it's (16, 4, 1024) which is already correct format for conv1d
        dna_tensor = torch.tensor(seq_batch, dtype=torch.float32)
        
        # DNase: (16, 1024) -> (16, 1, 1024)
        dnase_tensor = torch.tensor(dns_batch, dtype=torch.float32).unsqueeze(1)
        
        return dna_tensor, dnase_tensor
    
    def train_on_batch(self, seq_batch: np.ndarray, dns_batch: np.ndarray, lab_batch: np.ndarray) -> float:
        """Train on a single batch - REGRESSION VERSION"""
        self.forward_fn.train()
        
        dna_input, dnase_input = self._separate_inputs(seq_batch, dns_batch)
        
        lab_batch = lab_batch.squeeze() 
        lab_batch = torch.tensor(lab_batch, dtype=torch.float32)  # Keep as float32 for regression
        
        if self.use_gpu:
            dna_input = dna_input.cuda()
            dnase_input = dnase_input.cuda()
            lab_batch = lab_batch.cuda()
        
        # Forward pass
        output = self.forward_fn(dna_input, dnase_input)
        
        # Main loss (MSE for regression)
        main_loss = self.criterion(output, lab_batch)
        
        # Add entropy regularization if using MIL fusion
        total_loss = main_loss
        if self.forward_fn.fusion_type == 'mil':
            entropy = self.forward_fn.get_fusion_entropy()
            entropy_reg = -self.forward_fn.fusion.entropy_lambda * entropy
            total_loss = main_loss + entropy_reg
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return main_loss.cpu().item()
    
    def eval_on_batch(self, seq_batch: np.ndarray, dns_batch: np.ndarray, lab_batch: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluate on a single batch - REGRESSION VERSION"""
        self.forward_fn.eval()
        
        with torch.no_grad():
            dna_input, dnase_input = self._separate_inputs(seq_batch, dns_batch)
            lab_batch = lab_batch.squeeze()
            lab_batch = torch.tensor(lab_batch, dtype=torch.float32)  # Keep as float32 for regression
            
            if self.use_gpu:
                dna_input = dna_input.cuda()
                dnase_input = dnase_input.cuda()
                lab_batch = lab_batch.cuda()
            
            output = self.forward_fn(dna_input, dnase_input)
            loss = self.criterion(output, lab_batch)
            
            return loss.cpu().item(), output.cpu().numpy()
    
    def test_on_batch(self, seq_batch: np.ndarray, dns_batch: np.ndarray) -> np.ndarray:
        """Test on a single batch (no labels) - REGRESSION VERSION"""
        self.forward_fn.eval()
        
        with torch.no_grad():
            dna_input, dnase_input = self._separate_inputs(seq_batch, dns_batch)
            
            if self.use_gpu:
                dna_input = dna_input.cuda()
                dnase_input = dnase_input.cuda()
            
            output = self.forward_fn(dna_input, dnase_input)
            return output.cpu().numpy()
    
    def get_attention_weights(self, seq_batch: np.ndarray, dns_batch: np.ndarray) -> List[torch.Tensor]:
        """Get attention weights for visualization"""
        self.forward_fn.eval()
        
        with torch.no_grad():
            dna_input, dnase_input = self._separate_inputs(seq_batch, dns_batch)
            
            if self.use_gpu:
                dna_input = dna_input.cuda()
                dnase_input = dnase_input.cuda()
            
            _ = self.forward_fn(dna_input, dnase_input)
            return self.forward_fn.get_attention_weights()
    
    def save_model(self, path: str):
        """Save model state"""
        torch.save(self.forward_fn.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model state"""
        self.forward_fn.load_state_dict(torch.load(path, map_location='cpu'))
        if self.use_gpu:
            self.forward_fn = self.forward_fn.cuda()