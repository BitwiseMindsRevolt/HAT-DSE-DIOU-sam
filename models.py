import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 750, scale_factor: float = 1.0):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size * scale_factor)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class DualScaleTemporalEncoder(nn.Module):
    """
    Unified temporal encoder that captures both fine-grained local motion
    and long-range temporal dependencies without relying on sequence-length branching.

    When ``return_dual=True`` the local and global branches are returned separately
    so an upstream length-router can mix them per anchor (LR-DSE).
    """
    def __init__(self, embedding_dim, num_heads, dropout, return_dual=False):
        super(DualScaleTemporalEncoder, self).__init__()
        self.return_dual = return_dual

        # Local scale: Depthwise 1D Convolutions for short-term temporal dynamics
        self.local_encoder = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=5,
            padding=2,
            groups=embedding_dim  # Depthwise convolution for efficiency
        )
        self.local_norm = nn.LayerNorm(embedding_dim)

        # Global scale: Transformer Self-Attention for long-range dependencies
        self.global_encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )

        # Optional fusion projection (only used when return_dual=False)
        if not return_dual:
            self.fusion = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.GELU(),
                nn.LayerNorm(embedding_dim),
                nn.Dropout(dropout)
            )
        else:
            # Per-branch normalisation so the router operates on comparable scales.
            self.global_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """
        x shape: [seq_len, batch, dim]
        Returns:
            fused tensor [seq_len, batch, dim] if return_dual=False
            (local, global) tuple of [seq_len, batch, dim] tensors if return_dual=True
        """
        # Local processing
        local_x = x.permute(1, 2, 0)              # [batch, dim, seq_len]
        local_x = self.local_encoder(local_x)
        local_x = local_x.permute(2, 0, 1)        # [seq_len, batch, dim]
        local_x = self.local_norm(local_x + x)    # residual

        # Global processing on top of the locally-refined sequence
        global_x = self.global_encoder(local_x)

        if self.return_dual:
            return local_x, self.global_norm(global_x)

        fused = self.fusion(torch.cat([local_x, global_x], dim=-1))
        return fused


class LengthRouter(nn.Module):
    """
    Maps a per-anchor scalar (anchor length, in frames) to a mixing weight
    alpha ∈ (0, 1) controlling how much weight an anchor places on the local
    (depthwise-conv) branch versus the global (self-attention) branch of DSE.

    Short anchors are expected to converge toward alpha → 1 (local-heavy),
    long anchors toward alpha → 0 (global-heavy). The mapping is learned, not
    hand-tuned: the only inductive bias is the log-length input and a single
    hidden layer with GELU.
    """
    def __init__(self, hidden_dim=32):
        super(LengthRouter, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, anchor_lengths):
        # anchor_lengths: [K] float tensor of anchor lengths (frames)
        log_lens = torch.log(anchor_lengths.clamp(min=1e-6)).unsqueeze(-1)  # [K, 1]
        alpha = torch.sigmoid(self.mlp(log_lens)).squeeze(-1)               # [K]
        return alpha

class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature = opt["feat_dim"] 
        n_class = opt["num_of_class"]
        n_embedding_dim = opt["hidden_dim"]
        n_enc_layer = opt["enc_layer"]
        n_enc_head = opt["enc_head"]
        n_dec_layer = opt["dec_layer"]
        n_dec_head = opt["dec_head"]
        n_seglen = opt["segment_size"]
        self.anchors = opt["anchors"]
        self.anchors_stride = []
        dropout = 0.3
        self.best_loss = 1000000
        self.best_map = 0
        self.use_dse = bool(opt.get("DSE", False))
        self.use_lrdse = bool(opt.get("LRDSE", False))
        if self.use_lrdse and not self.use_dse:
            # LR-DSE is meaningless without the dual-branch DSE.
            self.use_dse = True

        # Enhanced feature reduction with dynamic dropout
        self.feature_reduction_rgb = nn.Sequential(
            nn.Linear(self.n_feature//2, n_embedding_dim//2),
            nn.GELU(),
            nn.LayerNorm(n_embedding_dim//2),
            nn.Dropout(dropout * 0.5)
        )
        self.feature_reduction_flow = nn.Sequential(
            nn.Linear(self.n_feature//2, n_embedding_dim//2),
            nn.GELU(),
            nn.LayerNorm(n_embedding_dim//2),
            nn.Dropout(dropout * 0.5)
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            n_embedding_dim, 
            dropout=dropout * 0.5,
            maxlen=400,
            scale_factor=0.5
        )
        
        # Unified Dual-Scale Temporal Encoder (only built when --DSE is enabled).
        # When --LRDSE is set the encoder returns its two branches separately so a
        # per-anchor mixing weight can route between them.
        if self.use_dse:
            self.temporal_encoder = DualScaleTemporalEncoder(
                embedding_dim=n_embedding_dim,
                num_heads=n_enc_head,
                dropout=dropout,
                return_dual=self.use_lrdse,
            )
        else:
            self.temporal_encoder = None

        # Length-Routed DSE components
        if self.use_lrdse:
            self.length_router = LengthRouter(hidden_dim=opt.get("router_hidden", 32))
            anchor_lengths = torch.tensor(self.anchors, dtype=torch.float32)
            self.register_buffer("anchor_lengths", anchor_lengths)
            # One pre-routing self-attention pass so the K anchor queries can
            # exchange information before being dispatched to per-anchor memories
            # (the per-anchor decoder pass below disables their joint self-attention).
            self.anchor_self_attn = nn.TransformerEncoderLayer(
                d_model=n_embedding_dim,
                nhead=n_dec_head,
                dropout=dropout,
                activation='gelu',
            )
        
        # Main encoder (adaptive layers)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embedding_dim, 
                nhead=n_enc_head, 
                dropout=dropout * (0.5 if i < 2 else 1.0),  # Lower dropout for initial layers
                activation='gelu'
            ) for i in range(n_enc_layer)
        ])
        self.encoder_norm = nn.LayerNorm(n_embedding_dim)
        
        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_embedding_dim, 
                nhead=n_dec_head, 
                dropout=dropout, 
                activation='gelu'
            ), 
            n_dec_layer, 
            nn.LayerNorm(n_embedding_dim)
        )
        

        
        # Enhanced classification and regression heads
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim),
            nn.GELU(),
            nn.LayerNorm(n_embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim, n_class)
        )
        self.regressor = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim),
            nn.GELU(),
            nn.LayerNorm(n_embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim, 2)
        )

        self.decoder_token = nn.Parameter(torch.Tensor(len(self.anchors), 1, n_embedding_dim))
        nn.init.normal_(self.decoder_token, std=0.01)
        
        # Additional normalization layers
        self.norm1 = nn.LayerNorm(n_embedding_dim)
        self.norm2 = nn.LayerNorm(n_embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # Enhanced feature processing
        inputs = inputs.float()
        base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2])
        base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:])
        base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)

        base_x = base_x.permute([1,0,2])  # seq_len x batch x featsize
        seq_len = base_x.shape[0]

        # Apply positional encoding
        pe_x = self.positional_encoding(base_x)

        # --- Length-Routed Dual-Scale path ---------------------------------
        if self.use_lrdse:
            m_local, m_global = self.temporal_encoder(pe_x)  # each [T, B, D]

            # Shared encoder stack applied to both branches independently so
            # local/global identity is preserved through the encoder.
            e_local, e_global = m_local, m_global
            for layer in self.encoder_layers:
                e_local = layer(e_local)
                e_global = layer(e_global)
            e_local = self.norm1(self.encoder_norm(e_local))
            e_global = self.norm1(self.encoder_norm(e_global))

            T, B, D = e_local.shape
            K = self.decoder_token.shape[0]

            # Per-anchor mixing weight from anchor length.
            alpha = self.length_router(self.anchor_lengths)             # [K]
            alpha = alpha.view(1, K, 1, 1)                              # broadcastable

            # Routed memory: [T, K, B, D] -> flatten (B, K) into batch.
            routed = alpha * e_local.unsqueeze(1) + (1.0 - alpha) * e_global.unsqueeze(1)
            routed = routed.permute(0, 2, 1, 3).reshape(T, B * K, D)    # [T, B*K, D]

            # Anchor tokens interact once, then are dispatched per-anchor.
            decoder_token = self.decoder_token.expand(-1, B, -1)        # [K, B, D]
            decoder_token = self.anchor_self_attn(decoder_token)        # peer mixing
            # Reorder to align with routed batch: anchor k of sample b -> row (b*K + k).
            tokens = decoder_token.permute(1, 0, 2).reshape(B * K, D).unsqueeze(0)  # [1, B*K, D]

            decoded_x = self.decoder(tokens, routed)                    # [1, B*K, D]
            decoded_x = self.norm2(decoded_x + self.dropout1(tokens))

            decoded_x = decoded_x.squeeze(0).reshape(B, K, D)           # [B, K, D]

            anc_cls = self.classifier(decoded_x)
            anc_reg = self.regressor(decoded_x)
            return anc_cls, anc_reg

        # --- Default path (baseline / DSE-only) ----------------------------
        if self.use_dse:
            encoded_x = self.temporal_encoder(pe_x)
        else:
            encoded_x = pe_x

        # Standard encoder processing
        for layer in self.encoder_layers:
            encoded_x = layer(encoded_x)

        # Apply encoder normalization
        encoded_x = self.encoder_norm(encoded_x)
        encoded_x = self.norm1(encoded_x)

        # Decoder processing
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)
        decoded_x = self.decoder(decoder_token, encoded_x)

        # Add residual connection and normalization
        decoded_x = self.norm2(decoded_x + self.dropout1(decoder_token))

        decoded_x = decoded_x.permute([1, 0, 2])

        anc_cls = self.classifier(decoded_x)
        anc_reg = self.regressor(decoded_x)

        return anc_cls, anc_reg


class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class=opt["num_of_class"]-1
        n_seglen=opt["segment_size"]
        n_embedding_dim=2*n_seglen
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
        # FC layers for the 2 streams
        
        self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2 = nn.Linear(n_embedding_dim, 1)
        self.norm = nn.InstanceNorm1d(n_class)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        #inputs - batch x seq_len x class
        
        base_x = inputs.permute([0,2,1])
        base_x = self.norm(base_x)
        x = self.relu(self.mlp1(base_x))
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        
        return x
