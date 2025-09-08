import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Tuple, Dict, List
import sys
import os

# Import from your existing regression model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'regression'))
from regression.dhica_regression import (
    exponential_linspace_int, RelativePositionalBias, GELU, ConvBlock,
    SoftmaxPooling1D, ResidualBlock, MultiHeadAttention, TransformerBlock
)

class TripleModalityFusion(nn.Module):
    """Fusion layer for DNA + DNase + 6 histone inputs"""
    def __init__(self, channels: int, dropout: float = 0.1, fusion_type: str = 'concat'):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            # DNA + DNase + 6 histones = 8 modalities
            self.projection = nn.Sequential(
                nn.BatchNorm1d(channels * 8),
                GELU(),
                nn.Conv1d(channels * 8, channels * 2, kernel_size=1),
                nn.BatchNorm1d(channels * 2),
                GELU(),
                nn.Conv1d(channels * 2, channels, kernel_size=1),
                nn.Dropout(dropout)
            )
            
        elif fusion_type == 'hierarchical':
            # First fuse DNA+DNase
            self.dna_dnase_fusion = nn.Sequential(
                nn.Conv1d(channels * 2, channels, kernel_size=1),
                nn.BatchNorm1d(channels),
                GELU()
            )
            
            # Progressive fusion for each histone
            self.histone_fusion_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(channels * 2, channels, kernel_size=1),
                    nn.BatchNorm1d(channels),
                    GELU()
                ) for _ in range(6)
            ])
            
            self.final_dropout = nn.Dropout(dropout)
            
        elif fusion_type == 'mil':
            # Extended MIL fusion for 8 modalities
            self.d_attn = 128
            
            # Gated attention for each modality
            self.modality_gates = nn.ModuleList([
                nn.Linear(channels, self.d_attn, bias=False) for _ in range(8)
            ])
            self.modality_attentions = nn.ModuleList([
                nn.Linear(channels, self.d_attn, bias=False) for _ in range(8)
            ])
            self.modality_biases = nn.ParameterList([
                nn.Parameter(torch.zeros(self.d_attn)) for _ in range(8)
            ])
            
            self.W = nn.Parameter(torch.randn(self.d_attn))
            self.attention_weights = None
            
    def forward(self, dna_features: torch.Tensor, dnase_features: torch.Tensor, 
                histone_features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            dna_features: (batch, channels, seq_len)
            dnase_features: (batch, channels, seq_len)
            histone_features_list: List of 6 (batch, channels, seq_len) tensors
        """
        
        if self.fusion_type == 'concat':
            all_features = [dna_features, dnase_features] + histone_features_list
            fused = torch.cat(all_features, dim=1)
            return self.projection(fused)
            
        elif self.fusion_type == 'hierarchical':
            # Start with DNA+DNase
            dna_dnase = torch.cat([dna_features, dnase_features], dim=1)
            combined = self.dna_dnase_fusion(dna_dnase)
            
            # Progressive fusion with histones
            for i, histone_features in enumerate(histone_features_list):
                combined_input = torch.cat([combined, histone_features], dim=1)
                combined = self.histone_fusion_layers[i](combined_input)
            
            return self.final_dropout(combined)
            
        elif self.fusion_type == 'mil':
            # MIL fusion for all 8 modalities
            all_features = [dna_features, dnase_features] + histone_features_list
            
            # Mean pool each modality
            pooled_features = [f.mean(dim=2) for f in all_features]  # (batch, channels)
            
            # Compute gated attention for each modality
            scores = []
            for i, h in enumerate(pooled_features):
                gate = torch.sigmoid(self.modality_gates[i](h) + self.modality_biases[i])
                attention = torch.tanh(self.modality_attentions[i](h))
                gated = gate * attention
                score = torch.matmul(gated, self.W)
                scores.append(score)
            
            # Compute attention weights
            scores = torch.stack(scores, dim=1)  # (batch, 8)
            attention_weights = F.softmax(scores, dim=1)  # (batch, 8)
            self.attention_weights = attention_weights
            
            # Weighted combination
            seq_len = dna_features.shape[2]
            fused = torch.zeros_like(dna_features)
            
            for i, (features, weight) in enumerate(zip(all_features, attention_weights.T)):
                weight_expanded = weight.unsqueeze(1).unsqueeze(2).expand(-1, features.shape[1], seq_len)
                fused += weight_expanded * features
            
            return fused


class PerHistoneModel(nn.Module):
    """Model that uses DNA + DNase + 6 histones to predict 1 target histone"""
    
    def __init__(self,
                 channels: int = 768,
                 num_transformer_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.4,
                 output_bins: int = 1024,  # Changed from 128 to 1024 for 128bp bins
                 pooling_type: str = 'attention',
                 num_conv_blocks: int = 2,
                 fusion_type: str = 'concat'):
        super().__init__()
        
        print(f'Per-Histone Model initialized')
        print(f'Channels: {channels}, Transformer layers: {num_transformer_layers}')
        print(f'Output bins: {output_bins}, Fusion: {fusion_type}')
        
        self.channels = channels
        self.output_bins = output_bins
        
        # DNA pathway (4 channels: A, T, G, C)
        self.dna_pathway = self._build_pathway(
            input_channels=4,
            output_channels=channels,
            pathway_name="DNA",
            pooling_type=pooling_type,
            num_conv_blocks=num_conv_blocks
        )
        
        # DNase pathway (1 channel)
        self.dnase_pathway = self._build_pathway(
            input_channels=1,
            output_channels=channels,
            pathway_name="DNase",
            pooling_type=pooling_type,
            num_conv_blocks=num_conv_blocks
        )
        
        # 6 histone pathways (1 channel each) - same as DNase processing
        self.histone_pathways = nn.ModuleList([
            self._build_pathway(
                input_channels=1,
                output_channels=channels,
                pathway_name=f"Histone_{i}",
                pooling_type=pooling_type,
                num_conv_blocks=num_conv_blocks
            ) for i in range(6)
        ])
        
        # Triple modality fusion
        self.fusion = TripleModalityFusion(
            channels=channels,
            dropout=dropout/2,
            fusion_type=fusion_type
        )
        
        # Transformer layers
        self.transformer = nn.ModuleList([
            TransformerBlock(channels, num_heads, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # Final prediction head - single target output
        self.final_conv = ConvBlock(channels, channels * 2, kernel_size=1, dropout=dropout//8)
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels * 2, output_bins)  # Single histone, 128 bins
        )
        
        self.attention_weights = []

    def _build_pathway(self, input_channels: int, output_channels: int, 
                      pathway_name: str, pooling_type: str, num_conv_blocks: int):
        """Build individual pathway - reuse from your existing model"""
        stem_channels = output_channels // 2
        stem = self._make_pathway_stem(input_channels, stem_channels, pathway_name, pooling_type)
        actual_stem_out = self._get_stem_output_channels(stem)
        conv_tower = self._build_conv_tower(actual_stem_out, output_channels, pooling_type, num_conv_blocks)
        
        return nn.Sequential(stem, conv_tower)
    
    def _make_pathway_stem(self, input_channels: int, stem_out_channels: int, 
                          pathway_name: str, pooling_type: str):
        """Build pathway-specific stem"""
        layers = []
        
        if pathway_name == "DNA":
            layers.append(nn.Conv1d(input_channels, stem_out_channels, kernel_size=15, padding=7))
        else:  # DNase and all histones
            layers.append(nn.Conv1d(input_channels, stem_out_channels, kernel_size=7, padding=3))
        
        layers.append(ResidualBlock(ConvBlock(stem_out_channels, stem_out_channels, kernel_size=1)))
        
        if pooling_type == 'attention':
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
    
    def forward(self, dna_input: torch.Tensor, dnase_input: torch.Tensor, 
                histone_inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dna_input: (batch, 4, seq_len) - one-hot DNA sequence
            dnase_input: (batch, 1, seq_len) - DNase signal
            histone_inputs: (batch, 6, seq_len) - 6 other histone ChIP-seq signals
        Returns:
            prediction: (batch, output_bins) - prediction for target histone
        """
        
        # Process DNA and DNase
        dna_features = self.dna_pathway(dna_input)
        dnase_features = self.dnase_pathway(dnase_input)
        
        # Process each histone through its pathway
        histone_features_list = []
        for i in range(6):
            # Extract single histone: (batch, 1, seq_len)
            single_histone = histone_inputs[:, i:i+1, :]
            histone_features = self.histone_pathways[i](single_histone)
            histone_features_list.append(histone_features)
        
        # Triple modality fusion
        fused_features = self.fusion(dna_features, dnase_features, histone_features_list)
        
        # Transformer processing
        x = fused_features.transpose(1, 2)  # (batch, seq_len, channels)
        
        self.attention_weights = []
        for transformer_block in self.transformer:
            x, attn_weights = transformer_block(x)
            self.attention_weights.append(attn_weights)
        
        # Final prediction
        x = x.transpose(1, 2)  # Back to conv format
        x = self.final_conv(x)
        output = self.regressor(x)  # (batch, output_bins)
        
        return output
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """Return stored attention weights"""
        return self.attention_weights


class PerHistoneTrainer:
    """Training wrapper for per-histone model"""
    
    def __init__(self, 
                 target_histone_idx: int,
                 histone_names: List[str] = None,
                 use_gpu: bool = True,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 **model_kwargs):
        
        if histone_names is None:
            histone_names = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 
                           'H3K9me3', 'H3K9ac', 'H3K27ac']
        
        self.target_histone_idx = target_histone_idx
        self.target_histone_name = histone_names[target_histone_idx]
        self.histone_names = histone_names
        
        print(f"Initializing trainer for {self.target_histone_name} (index {target_histone_idx})")
        
        self.model = PerHistoneModel(**model_kwargs)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.use_gpu = use_gpu
        if self.use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
    
    def _prepare_batch(self, seq_batch: np.ndarray, dns_batch: np.ndarray, 
                      all_histones: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch for training/evaluation"""
        
        # Process inputs
        seq_batch = seq_batch.squeeze()
        dns_batch = dns_batch.squeeze()
        
        dna_input = torch.tensor(seq_batch, dtype=torch.float32)
        dnase_input = torch.tensor(dns_batch, dtype=torch.float32).unsqueeze(1)  # Add channel dim
        
        # Extract target histone and other histones
        target_histone = torch.tensor(all_histones[:, self.target_histone_idx, :], dtype=torch.float32)
        
        # Get other 6 histones (exclude target)
        other_indices = [i for i in range(7) if i != self.target_histone_idx]
        other_histones = torch.tensor(all_histones[:, other_indices, :], dtype=torch.float32)
        
        return dna_input, dnase_input, other_histones, target_histone
    
    def train_on_batch(self, seq_batch: np.ndarray, dns_batch: np.ndarray, 
                      histone_inputs: np.ndarray, histone_targets: np.ndarray) -> float:
        """Train on a single batch"""
        self.model.train()
        
        dna_input, dnase_input, other_histones, target = self._prepare_batch(
            seq_batch, dns_batch, histone_inputs, histone_targets
        )
        
        if self.use_gpu:
            dna_input = dna_input.cuda()
            dnase_input = dnase_input.cuda()
            other_histones = other_histones.cuda()
            target = target.cuda()
        
        # Forward pass
        output = self.model(dna_input, dnase_input, other_histones)
        loss = self.criterion(output, target)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.cpu().item()
    
    def eval_on_batch(self, seq_batch: np.ndarray, dns_batch: np.ndarray, 
                     histone_inputs: np.ndarray, histone_targets: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluate on a single batch"""
        self.model.eval()
        
        with torch.no_grad():
            dna_input, dnase_input, other_histones, target = self._prepare_batch(
                seq_batch, dns_batch, histone_inputs, histone_targets
            )
            
            if self.use_gpu:
                dna_input = dna_input.cuda()
                dnase_input = dnase_input.cuda()
                other_histones = other_histones.cuda()
                target = target.cuda()
            
            output = self.model(dna_input, dnase_input, other_histones)
            loss = self.criterion(output, target)
            
            return loss.cpu().item(), output.cpu().numpy()
    
    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'target_histone_idx': self.target_histone_idx,
            'target_histone_name': self.target_histone_name
        }, path)
    
    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.use_gpu:
            self.model = self.model.cuda()