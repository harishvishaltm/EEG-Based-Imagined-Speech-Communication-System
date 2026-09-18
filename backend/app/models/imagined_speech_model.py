"""Imagined Speech Model Architecture.

Exact reproduction of the research notebook (project (1).ipynb):
LearnedChannelGate -> SharedBackbone (TemporalCNN + MultiHeadAttention + BiLSTM + FeatureFusion)
                   -> DatasetSpecificHead (Kara One, FEIS, Nguyen)
"""
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class SharedChannelEncoder(nn.Module):
    """Per-channel temporal encoder with weights shared across all channels.

    Input : (B, C, T)
    Output: (B, C, encoder_channels) - feature vector per channel after GAP over time.
    """

    def __init__(self, encoder_channels: int = 16, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, encoder_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(encoder_channels),
            nn.ELU(),
            nn.Conv1d(encoder_channels, encoder_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(encoder_channels),
            nn.ELU(),
        )
        self.encoder_channels = encoder_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, C, encoder_channels)"""
        B, C, T = x.shape
        x_flat = x.reshape(B * C, 1, T)
        feat = self.net(x_flat)
        feat = feat.mean(dim=-1)
        return feat.reshape(B, C, self.encoder_channels)


class LearnedChannelGate(nn.Module):
    """Learns an adaptive per-channel gate in [0, 1] multiplied by the input EEG."""

    def __init__(
        self,
        canonical_channels: int = 122,
        encoder_channels: int = 16,
        mlp_hidden_dim: int = 32,
        gate_type: str = "sigmoid",
        use_subject_embedding: bool = True,
        use_dataset_embedding: bool = True,
        embedding_dim: int = 8,
        n_subjects: int = 21,
        n_datasets: int = 3,
        gumbel_temperature: float = 1.0,
        gumbel_min_temperature: float = 0.3,
    ):
        super().__init__()
        assert gate_type in {"sigmoid", "gumbel_softmax"}, f"Unknown gate_type: {gate_type}"

        self.gate_type = gate_type
        self.canonical_channels = canonical_channels
        self.use_subject_embedding = use_subject_embedding
        self.use_dataset_embedding = use_dataset_embedding
        self.temperature = gumbel_temperature
        self.gumbel_min_temperature = gumbel_min_temperature

        self.encoder = SharedChannelEncoder(encoder_channels=encoder_channels)

        mlp_in_dim = encoder_channels
        if use_subject_embedding:
            self.subject_embedding = nn.Embedding(n_subjects, embedding_dim)
            mlp_in_dim += embedding_dim
        if use_dataset_embedding:
            self.dataset_embedding = nn.Embedding(n_datasets, embedding_dim)
            mlp_in_dim += embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def set_temperature(self, t: float) -> None:
        self.temperature = max(t, self.gumbel_min_temperature)

    def compute_logits(
        self,
        x: torch.Tensor,
        subject_id: Optional[torch.Tensor] = None,
        dataset_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = self.encoder(x)  # (B, C, encoder_channels)
        B, C, _ = feat.shape

        extras = []
        if self.use_subject_embedding:
            if subject_id is None:
                raise ValueError("subject_id required when use_subject_embedding=True")
            emb = self.subject_embedding(subject_id)
            extras.append(emb.unsqueeze(1).expand(B, C, -1))
        if self.use_dataset_embedding:
            if dataset_id is None:
                raise ValueError("dataset_id required when use_dataset_embedding=True")
            emb = self.dataset_embedding(dataset_id)
            extras.append(emb.unsqueeze(1).expand(B, C, -1))

        if extras:
            feat = torch.cat([feat] + extras, dim=-1)

        logits = self.mlp(feat).squeeze(-1)  # (B, C)
        return logits

    def apply_gate_activation(self, logits: torch.Tensor) -> torch.Tensor:
        if self.gate_type == "sigmoid":
            return torch.sigmoid(logits)

        if self.training:
            u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
            gumbel_noise = torch.log(u) - torch.log(1.0 - u)
            return torch.sigmoid((logits + gumbel_noise) / self.temperature)
        return torch.sigmoid(logits)

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor,
        subject_id: Optional[torch.Tensor] = None,
        dataset_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.compute_logits(x, subject_id, dataset_id)
        raw_gate = self.apply_gate_activation(logits)
        gate_weights = raw_gate * channel_mask
        gated_eeg = x * gate_weights.unsqueeze(-1)
        return gated_eeg, gate_weights


class TemporalCNNBackbone(nn.Module):
    """EEGNet-style separable temporal convolution backbone."""

    def __init__(
        self,
        in_channels: int = 122,
        temporal_filters: int = 32,
        depthwise_mult: int = 2,
        kernel_size: int = 25,
        pool_size: int = 4,
        dropout: float = 0.3,
        embed_dim: int = 64,
    ):
        super().__init__()
        padding = kernel_size // 2
        depthwise_out = in_channels * depthwise_mult

        self.depthwise = nn.Sequential(
            nn.Conv1d(
                in_channels,
                depthwise_out,
                kernel_size,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm1d(depthwise_out),
            nn.ELU(),
            nn.AvgPool1d(pool_size),
            nn.Dropout(dropout),
        )

        self.pointwise = nn.Sequential(
            nn.Conv1d(depthwise_out, temporal_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(temporal_filters),
            nn.ELU(),
            nn.AvgPool1d(pool_size),
            nn.Dropout(dropout),
        )

        self.project = (
            nn.Identity()
            if temporal_filters == embed_dim
            else nn.Conv1d(temporal_filters, embed_dim, kernel_size=1)
        )

        self.pool_size = pool_size
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.project(x)
        return x

    def output_seq_len(self, input_len: int) -> int:
        return input_len // (self.pool_size ** 2)


class TemporalSelfAttention(nn.Module):
    """Multi-Head Self-Attention over the reduced time sequence."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, embed_dim, T') -> (B, embed_dim, T')"""
        x_seq = x.transpose(1, 2)  # (B, T', embed_dim)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq, need_weights=False)
        x_seq = self.norm(x_seq + self.dropout(attn_out))
        return x_seq.transpose(1, 2)


class BiLSTMBlock(nn.Module):
    """Bidirectional 2-layer LSTM block."""

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_size * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, embed_dim, T') -> (B, T', 2*hidden_size)"""
        x_seq = x.transpose(1, 2)
        out, _ = self.lstm(x_seq)
        return out


class FeatureFusion(nn.Module):
    """Global average pooling of attention and BiLSTM representations followed by dense projection."""

    def __init__(
        self,
        embed_dim: int = 64,
        lstm_output_dim: int = 256,
        fusion_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        concat_dim = embed_dim + lstm_output_dim
        self.fc = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.fusion_dim = fusion_dim

    def forward(self, attn_out: torch.Tensor, lstm_out: torch.Tensor) -> torch.Tensor:
        attn_pooled = attn_out.mean(dim=-1)
        lstm_pooled = lstm_out.mean(dim=1)
        concat = torch.cat([attn_pooled, lstm_pooled], dim=-1)
        return self.fc(concat)


class SharedBackbone(nn.Module):
    """Assembled shared backbone: CNN -> Attention + BiLSTM -> Fusion."""

    def __init__(
        self,
        in_channels: int = 122,
        temporal_filters: int = 32,
        depthwise_mult: int = 2,
        kernel_size: int = 25,
        pool_size: int = 4,
        cnn_dropout: float = 0.3,
        embed_dim: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        bilstm_hidden: int = 128,
        bilstm_layers: int = 2,
        bilstm_dropout: float = 0.3,
        fusion_dim: int = 128,
        fusion_dropout: float = 0.3,
    ):
        super().__init__()
        self.cnn = TemporalCNNBackbone(
            in_channels=in_channels,
            temporal_filters=temporal_filters,
            depthwise_mult=depthwise_mult,
            kernel_size=kernel_size,
            pool_size=pool_size,
            dropout=cnn_dropout,
            embed_dim=embed_dim,
        )
        self.attention = TemporalSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout
        )
        self.bilstm = BiLSTMBlock(
            embed_dim=embed_dim,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            dropout=bilstm_dropout,
        )
        self.fusion = FeatureFusion(
            embed_dim=embed_dim,
            lstm_output_dim=bilstm_hidden * 2,
            fusion_dim=fusion_dim,
            dropout=fusion_dropout,
        )
        self.fusion_dim = self.fusion.fusion_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn(x)
        attn_out = self.attention(cnn_out)
        lstm_out = self.bilstm(attn_out)
        fused = self.fusion(attn_out, lstm_out)
        return fused

    def expected_seq_len(self, input_len: int = 1280) -> int:
        return self.cnn.output_seq_len(input_len)


class DatasetSpecificHead(nn.Module):
    """MLP classification head tailored to a specific dataset."""

    def __init__(
        self, fusion_dim: int = 128, n_classes: int = 11, hidden_dim: int = 64, dropout: float = 0.3
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImaginedSpeechModel(nn.Module):
    """End-to-end model combining LearnedChannelGate, SharedBackbone, and DatasetSpecificHeads."""

    def __init__(
        self,
        dataset_name_to_id: Dict[str, int],
        num_classes_per_dataset: Dict[str, int],
        n_subjects: int = 21,
        n_datasets: int = 3,
        canonical_channels: int = 122,
        encoder_channels: int = 16,
        mlp_hidden_dim: int = 32,
        embedding_dim: int = 8,
        temporal_filters: int = 32,
        depthwise_mult: int = 2,
        cnn_kernel_size: int = 25,
        pool_size: int = 4,
        cnn_dropout: float = 0.3,
        embed_dim: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        bilstm_hidden: int = 128,
        bilstm_layers: int = 2,
        bilstm_dropout: float = 0.3,
        fusion_dim: int = 128,
        fusion_dropout: float = 0.3,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.dataset_name_to_id = dataset_name_to_id
        self.id_to_dataset_name = {v: k for k, v in dataset_name_to_id.items()}

        self.gate = LearnedChannelGate(
            canonical_channels=canonical_channels,
            encoder_channels=encoder_channels,
            mlp_hidden_dim=mlp_hidden_dim,
            use_subject_embedding=True,
            use_dataset_embedding=True,
            embedding_dim=embedding_dim,
            n_subjects=n_subjects,
            n_datasets=n_datasets,
        )

        self.backbone = SharedBackbone(
            in_channels=canonical_channels,
            temporal_filters=temporal_filters,
            depthwise_mult=depthwise_mult,
            kernel_size=cnn_kernel_size,
            pool_size=pool_size,
            cnn_dropout=cnn_dropout,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            bilstm_hidden=bilstm_hidden,
            bilstm_layers=bilstm_layers,
            bilstm_dropout=bilstm_dropout,
            fusion_dim=fusion_dim,
            fusion_dropout=fusion_dropout,
        )

        self.heads = nn.ModuleDict(
            {
                name: DatasetSpecificHead(
                    fusion_dim=self.backbone.fusion_dim,
                    n_classes=num_classes_per_dataset[name],
                    hidden_dim=head_hidden_dim,
                    dropout=head_dropout,
                )
                for name in dataset_name_to_id
            }
        )

    def forward(
        self,
        eeg: torch.Tensor,
        channel_mask: torch.Tensor,
        subject_id: torch.Tensor,
        dataset_id: torch.Tensor,
        return_gate: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]]:
        gated_eeg, gate_weights = self.gate(eeg, channel_mask, subject_id, dataset_id)
        features = self.backbone(gated_eeg)

        logits_per_sample = []
        for i in range(features.shape[0]):
            ds_name = self.id_to_dataset_name[dataset_id[i].item()]
            head = self.heads[ds_name]
            logits_per_sample.append(head(features[i].unsqueeze(0)).squeeze(0))

        if return_gate:
            return logits_per_sample, gate_weights, gated_eeg
        return logits_per_sample

    def forward_single_dataset(
        self,
        eeg: torch.Tensor,
        channel_mask: torch.Tensor,
        subject_id: torch.Tensor,
        dataset_id: torch.Tensor,
        dataset_name: str,
        return_gate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Optimized forward method for batches from a single dataset."""
        gated_eeg, gate_weights = self.gate(eeg, channel_mask, subject_id, dataset_id)
        features = self.backbone(gated_eeg)
        logits = self.heads[dataset_name](features)

        if return_gate:
            return logits, gate_weights, gated_eeg
        return logits
