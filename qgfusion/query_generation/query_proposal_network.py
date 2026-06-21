"""
Modality Query Generation -- Stage 2 (QUERY GENERATION column in diagram)

Implements Option C from the proposal (recommended): a small set of learnable query
embeddings that cross-attend to a modality's encoded features (DETR-style query proposal).

Shared module class -- instantiate one per modality (Qc, Ql, Qr) with the appropriate
input feature dim.

Risk level (per proposal Sec 7): LOW
"""
import torch
import torch.nn as nn


class QueryProposalNetwork(nn.Module):
    def __init__(self, feat_dim: int, embed_dim: int = 256, num_queries: int = 300, num_heads: int = 8):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.feat_proj = nn.Linear(feat_dim, embed_dim) if feat_dim != embed_dim else nn.Identity()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(inplace=True), nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, feats: torch.Tensor):
        """
        Args:
            feats: (B, L, feat_dim) flattened modality feature tokens
                   (flatten spatial dims of Fc/Fl, or point dim of Fr, before calling this)
        Returns:
            Q: (B, num_queries, embed_dim) modality queries
        """
        B = feats.shape[0]
        feats = self.feat_proj(feats)
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)

        attn_out, _ = self.cross_attn(query=queries, key=feats, value=feats)
        queries = self.norm1(queries + attn_out)
        queries = self.norm2(queries + self.ffn(queries))
        return queries
