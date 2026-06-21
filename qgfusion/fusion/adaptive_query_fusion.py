"""
Adaptive Query Fusion -- Stage 3 (FUSION column in diagram)

Implements Option 2 from the proposal (recommended): concatenate modality queries and fuse
with a Transformer encoder so each fused query can attend across all three modalities.
A learned per-query, per-modality gate (the "adaptive" part) is applied before concatenation.

Option 3 (full cross-modality attention) can be added later by replacing the
TransformerEncoder with a custom cross-attention stack -- the interface below
(Qc, Ql, Qr) -> Qf stays the same either way.

Risk level (per proposal Sec 7): MEDIUM -- "query fusion stability"
"""
import torch
import torch.nn as nn


class AdaptiveQueryFusion(nn.Module):
    def __init__(self, embed_dim: int = 256, num_layers: int = 4, num_heads: int = 8, modality_embed: bool = True):
        super().__init__()
        self.modality_embed = modality_embed
        if modality_embed:
            # learnable per-modality bias so the fusion transformer can tell Qc/Ql/Qr apart
            self.modality_tokens = nn.Parameter(torch.randn(3, embed_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Simplest possible "adaptive" mechanism: a learned per-query softmax gate over the
        # 3 modalities (Option-1-style weighting) layered underneath Option 2's transformer.
        # TODO(risk=MEDIUM): validate this gate actually helps over equal weighting -- it's
        # cheap to ablate (just hardcode weights=[1/3,1/3,1/3]) and worth checking early.
        self.gate = nn.Sequential(nn.Linear(embed_dim * 3, 3), nn.Softmax(dim=-1))

    def forward(self, Qc: torch.Tensor, Ql: torch.Tensor, Qr: torch.Tensor):
        """
        Args:
            Qc, Ql, Qr: (B, N, embed_dim) per-modality queries (same N across modalities --
                        pad/truncate upstream if your QPNs use different num_queries)
        Returns:
            Qf: (B, N, embed_dim) fused scene queries
        """
        gate_in = torch.cat([Qc, Ql, Qr], dim=-1)
        weights = self.gate(gate_in)  # (B, N, 3)
        Qc_w = Qc * weights[..., 0:1]
        Ql_w = Ql * weights[..., 1:2]
        Qr_w = Qr * weights[..., 2:3]

        if self.modality_embed:
            Qc_w = Qc_w + self.modality_tokens[0]
            Ql_w = Ql_w + self.modality_tokens[1]
            Qr_w = Qr_w + self.modality_tokens[2]

        fused_seq = torch.cat([Qc_w, Ql_w, Qr_w], dim=1)  # (B, 3N, embed_dim)
        fused_seq = self.transformer(fused_seq)

        # pool back down to N fused scene queries (mean over the 3 modality copies per slot)
        N = Qc.shape[1]
        Qf = fused_seq.view(fused_seq.shape[0], 3, N, -1).mean(dim=1)
        return Qf
