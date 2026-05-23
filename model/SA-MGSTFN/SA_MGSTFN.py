import importlib.util
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations

_SEMANTIC_UTILS_PATH = os.path.join(os.path.dirname(__file__), "semantic_utils.py")
_SEMANTIC_SPEC = importlib.util.spec_from_file_location("sa_mgstfn_semantic_utils", _SEMANTIC_UTILS_PATH)
_SEMANTIC_UTILS = importlib.util.module_from_spec(_SEMANTIC_SPEC)
_SEMANTIC_SPEC.loader.exec_module(_SEMANTIC_UTILS)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 4096):
        super().__init__()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        encoding = torch.zeros(max_len, embed_dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[:encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0).unsqueeze(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoding[:, :x.size(1)].expand(x.size(0), -1, x.size(2), -1)


class PatchEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, patch_len: int, stride: int, input_window: int):
        super().__init__()
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.stride = stride
        self.input_window = input_window
        padded_window = max(input_window, patch_len)
        self.num_patches = max((padded_window - patch_len) // stride + 1, 1)
        self.value_embedding = nn.Linear(input_dim * patch_len, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_nodes, input_dim = x.shape
        if input_dim != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {input_dim}.")

        patch_len = self.patch_len
        stride = self.stride
        if time_steps != self.input_window:
            scale = max(self.input_window // max(time_steps, 1), 1)
            patch_len = max(self.patch_len // scale, 1)
            stride = max(self.stride // scale, 1)

        if time_steps < patch_len:
            x = F.pad(x, (0, 0, 0, 0, patch_len - time_steps, 0))

        patches = x.permute(0, 2, 3, 1).unfold(dimension=-1, size=patch_len, step=stride)
        if patch_len != self.patch_len:
            patches = F.pad(patches, (0, self.patch_len - patch_len))
        patches = patches.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, num_nodes, self.input_dim * self.patch_len)
        embedded = self.value_embedding(patches)
        return embedded + self.position_encoding(embedded)


class TemporalContext(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.day_embedding = nn.Embedding(1441, embed_dim)
        self.week_embedding = nn.Embedding(8, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, output_dim: int) -> torch.Tensor:
        if x.size(-1) <= output_dim + 1:
            return torch.zeros(x.size(0), x.size(1), x.size(2), self.day_embedding.embedding_dim, device=x.device)
        day = x[..., output_dim].long().clamp(0, 1440)
        week = x[..., output_dim + 1].long().clamp(0, 7)
        return self.proj(torch.cat([self.day_embedding(day), self.week_embedding(week)], dim=-1))


class PatchTemporalContext(nn.Module):
    def __init__(self, embed_dim: int, patch_len: int, stride: int, input_window: int):
        super().__init__()
        self.temporal_context = TemporalContext(embed_dim)
        self.patch_len = patch_len
        self.stride = stride
        self.input_window = input_window

    def forward(self, x: torch.Tensor, output_dim: int) -> torch.Tensor:
        context = self.temporal_context(x, output_dim)
        time_steps = context.size(1)
        patch_len = self.patch_len
        stride = self.stride
        if time_steps != self.input_window:
            scale = max(self.input_window // max(time_steps, 1), 1)
            patch_len = max(self.patch_len // scale, 1)
            stride = max(self.stride // scale, 1)

        if time_steps < patch_len:
            context = F.pad(context, (0, 0, 0, 0, patch_len - time_steps, 0))

        patches = context.permute(0, 2, 3, 1).unfold(dimension=-1, size=patch_len, step=stride)
        return patches.mean(dim=-1).permute(0, 3, 1, 2)


class RMSNorm(nn.Module):
    def __init__(self, embed_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class GraphAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float, drop: float):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.feature_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.drop = nn.Dropout(drop)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_nodes, _ = x.shape
        flat_x = x.reshape(batch_size * time_steps, num_nodes, self.embed_dim)
        projected = self.feature_proj(flat_x).view(batch_size * time_steps, num_nodes, self.num_heads, self.head_dim)
        projected = projected.transpose(1, 2)

        source_scores = torch.sum(projected * self.attn_src.unsqueeze(0).unsqueeze(2), dim=-1)
        target_scores = torch.sum(projected * self.attn_dst.unsqueeze(0).unsqueeze(2), dim=-1)
        scores = self.leaky_relu(source_scores.unsqueeze(-1) + target_scores.unsqueeze(-2))
        edge_weight = adjacency.to(dtype=scores.dtype, device=x.device).clamp_min(0.0).unsqueeze(0).unsqueeze(0)
        mask = edge_weight > 0
        scores = scores + torch.log(edge_weight.clamp_min(1e-12))
        scores = scores.masked_fill(~mask, -1e9)
        attention = self.attn_drop(torch.softmax(scores, dim=-1))
        out = torch.matmul(attention, projected)
        out = F.gelu(out).transpose(1, 2).reshape(batch_size * time_steps, num_nodes, self.embed_dim)
        out = self.drop(out)
        return out.view(batch_size, time_steps, num_nodes, self.embed_dim)


class TimeAwareMultiGraphFusion(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float, drop: float, mlp_ratio: int):
        super().__init__()
        self.phy_gat = GraphAttention(embed_dim, num_heads, attn_drop, drop)
        self.flow_gat = GraphAttention(embed_dim, num_heads, attn_drop, drop)
        self.sem_gat = GraphAttention(embed_dim, num_heads, attn_drop, drop)
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )
        self.norm = RMSNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        time_context: torch.Tensor,
        physical_adj: torch.Tensor,
        flow_adj: torch.Tensor,
        semantic_adj: torch.Tensor,
    ) -> torch.Tensor:
        phy = self.phy_gat(x, physical_adj)
        flow = self.flow_gat(x, flow_adj)
        sem = self.sem_gat(x, semantic_adj)
        weights = torch.softmax(self.fusion_gate(time_context), dim=-1).unsqueeze(-1)
        fused = weights[..., 0, :] * phy + weights[..., 1, :] * flow + weights[..., 2, :] * sem
        return self.norm(fused + x)


class DecompositionRefiner(nn.Module):
    def __init__(self, embed_dim: int, drop: float):
        super().__init__()
        self.inv_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim, embed_dim),
        )
        self.spe_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim, embed_dim),
        )
        self.reconstruct = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, time_context: torch.Tensor):
        z_inv = self.inv_encoder(torch.cat([x, time_context], dim=-1))
        z_spe = self.spe_encoder(x)
        combined = torch.cat([z_inv, z_spe], dim=-1)
        reconstructed = self.reconstruct(z_inv + z_spe)
        return z_inv, z_spe, combined, reconstructed


class OrthogonalDomainMapper(nn.Module):
    def __init__(self, raw_dim: int, traffic_dim: int):
        super().__init__()
        self.projection = nn.Linear(raw_dim, traffic_dim)
        nn.init.orthogonal_(self.projection.weight)
        parametrizations.orthogonal(self.projection, "weight")

    def forward(self, semantic_embedding: torch.Tensor) -> torch.Tensor:
        return F.relu(self.projection(semantic_embedding))


class SAMGSTFN(nn.Module):
    def __init__(self, args, dataset_use, device, dim_in: int, dim_out: int):
        super().__init__()
        self.device = device
        self.dataset_use = dataset_use
        self.output_dim = dim_out
        self.input_base_dim = dim_in
        self.input_window = args.input_window
        self.output_window = args.output_window
        self.embed_dim = args.embed_dim
        self.semantic_dim = args.semantic_dim
        self.semantic_threshold = args.semantic_threshold
        self.patch_len = args.patch_len
        self.patch_stride = args.patch_stride
        self.flow_topk = args.flow_topk
        self.lambda_orth = args.lambda_orth
        self.lambda_recon = args.lambda_recon
        self.orth_loss = torch.tensor(0.0)
        self.recon_loss = torch.tensor(0.0)
        self.auxiliary_loss = torch.tensor(0.0)

        self.adj_mx_dict = args.adj_mx_dict
        self.raw_adj_mx_dict = args.raw_adj_mx_dict
        self.semantic_adjacency_buffer_names = {}
        self.semantic_embedding_dict = nn.ParameterDict()

        self.patch_embedding = PatchEmbedding(
            input_dim=dim_in,
            embed_dim=self.embed_dim,
            patch_len=self.patch_len,
            stride=self.patch_stride,
            input_window=self.input_window,
        )
        self.temporal_context = PatchTemporalContext(
            embed_dim=self.embed_dim,
            patch_len=self.patch_len,
            stride=self.patch_stride,
            input_window=self.input_window,
        )
        self.semantic_domain_mapper = OrthogonalDomainMapper(args.semantic_raw_dim, self.semantic_dim)
        self.semantic_feature_proj = nn.Linear(self.semantic_dim, self.embed_dim)
        self.encoder_layers = nn.ModuleList([
            TimeAwareMultiGraphFusion(
                embed_dim=self.embed_dim,
                num_heads=args.num_heads,
                attn_drop=args.attn_drop,
                drop=args.drop,
                mlp_ratio=args.mlp_ratio,
            )
            for _ in range(args.num_layers)
        ])
        self.refiner = DecompositionRefiner(self.embed_dim, args.drop)
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.Dropout(args.drop),
            nn.Linear(self.embed_dim, self.output_dim),
        )
        self.horizon_proj = nn.Linear(self.patch_embedding.num_patches, self.output_window)

        self._load_semantic_embeddings(args)
        self._init_static_semantic_adjacencies()

    def _load_semantic_embeddings(self, args):
        for dataset in self.dataset_use:
            texts = _SEMANTIC_UTILS.build_or_load_semantic_texts(
                dataset=dataset,
                meta_path=args.semantic_meta_path_dict[dataset],
                cache_dir=args.semantic_cache_dir_dict[dataset],
                radius=args.osm_radius,
                endpoint=args.osm_endpoint,
                timeout=args.osm_timeout,
                force_refresh=args.semantic_force_refresh,
                disable_osm=args.semantic_disable_osm,
            )
            embeddings = _SEMANTIC_UTILS.build_or_load_distilbert_embeddings(
                texts=texts,
                cache_dir=args.semantic_cache_dir_dict[dataset],
                model_name=args.semantic_model_name,
                batch_size=args.semantic_batch_size,
                max_length=args.semantic_max_length,
                device=self.device,
                force_refresh=args.semantic_force_refresh,
            )
            if embeddings.shape[1] != args.semantic_raw_dim:
                raise ValueError(
                    f"Expected DistilBERT embeddings with dim {args.semantic_raw_dim}, got {embeddings.shape[1]}."
                )
            semantic_embeddings = torch.FloatTensor(embeddings)
            self.semantic_embedding_dict[dataset] = nn.Parameter(semantic_embeddings, requires_grad=False)

    @staticmethod
    def _buffer_name(prefix: str, dataset: str) -> str:
        safe_dataset = "".join(character if character.isalnum() else "_" for character in dataset)
        return f"{prefix}_{safe_dataset}"

    def _init_static_semantic_adjacencies(self):
        for dataset in self.dataset_use:
            with torch.no_grad():
                semantic_raw = self.semantic_embedding_dict[dataset]
                semantic_features = self.semantic_domain_mapper(semantic_raw)
                semantic_adjacency = self._build_semantic_adjacency(semantic_features)
            buffer_name = self._buffer_name("semantic_adj", dataset)
            self.register_buffer(buffer_name, semantic_adjacency)
            self.semantic_adjacency_buffer_names[dataset] = buffer_name

    @staticmethod
    def _normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = adjacency.float()
        degree = adjacency.sum(dim=-1).clamp_min(1e-6)
        degree_inv_sqrt = torch.rsqrt(degree)
        return degree_inv_sqrt.unsqueeze(-1) * adjacency * degree_inv_sqrt.unsqueeze(0)

    def _build_semantic_adjacency(self, semantic_features: torch.Tensor) -> torch.Tensor:
        semantic_features = F.normalize(semantic_features, p=2, dim=-1)
        similarity = torch.matmul(semantic_features, semantic_features.transpose(0, 1)).clamp_min(0.0)
        semantic_adj = torch.where(similarity > self.semantic_threshold, similarity, torch.zeros_like(similarity))
        semantic_adj = semantic_adj + torch.eye(semantic_adj.size(0), device=semantic_adj.device)
        return self._normalize_adjacency(semantic_adj)

    def _semantic_features(self, select_dataset: str) -> torch.Tensor:
        semantic_raw = self.semantic_embedding_dict[select_dataset].to(self.device)
        return self.semantic_domain_mapper(semantic_raw)

    def _semantic_adjacency(self, select_dataset: str) -> torch.Tensor:
        buffer_name = self.semantic_adjacency_buffer_names[select_dataset]
        return getattr(self, buffer_name)

    def _flow_adjacency(self, source: torch.Tensor) -> torch.Tensor:
        traffic = source[..., :self.input_base_dim].mean(dim=(0, -1)).transpose(0, 1)
        traffic = traffic - traffic.mean(dim=-1, keepdim=True)
        traffic = traffic / traffic.std(dim=-1, keepdim=True).clamp_min(1e-6)
        correlation = torch.matmul(traffic, traffic.transpose(0, 1)) / max(traffic.size(-1), 1)
        correlation = correlation.clamp_min(0.0)
        if self.flow_topk > 0 and self.flow_topk < correlation.size(-1):
            topk_values, topk_indices = torch.topk(correlation, k=self.flow_topk, dim=-1)
            sparse = torch.zeros_like(correlation)
            sparse.scatter_(dim=-1, index=topk_indices, src=topk_values)
            correlation = sparse
        correlation = correlation + torch.eye(correlation.size(0), device=correlation.device)
        return self._normalize_adjacency(correlation)

    def _compute_auxiliary_losses(self, z_inv: torch.Tensor, z_spe: torch.Tensor, reconstructed: torch.Tensor, encoded: torch.Tensor):
        inv_by_patch = z_inv.permute(1, 0, 2, 3).reshape(z_inv.size(1), -1, z_inv.size(-1))
        spe_by_patch = z_spe.permute(1, 0, 2, 3).reshape(z_spe.size(1), -1, z_spe.size(-1))
        cross_covariance = torch.matmul(inv_by_patch.transpose(1, 2), spe_by_patch)
        inv_norm = torch.linalg.matrix_norm(inv_by_patch, ord="fro", dim=(1, 2)).clamp_min(1e-6)
        spe_norm = torch.linalg.matrix_norm(spe_by_patch, ord="fro", dim=(1, 2)).clamp_min(1e-6)
        normalized_cross = cross_covariance / (inv_norm * spe_norm).view(-1, 1, 1)
        self.orth_loss = torch.linalg.matrix_norm(normalized_cross, ord="fro", dim=(1, 2)).pow(2).mean()
        self.recon_loss = F.mse_loss(reconstructed, encoded)
        self.auxiliary_loss = self.lambda_orth * self.orth_loss + self.lambda_recon * self.recon_loss

    def forward(self, source: torch.Tensor, select_dataset: str) -> torch.Tensor:
        x = source[..., :self.input_base_dim]
        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        normalized = centered / stdev

        encoded = self.patch_embedding(normalized)
        time_context = self.temporal_context(source, self.output_dim)
        semantic_features = self._semantic_features(select_dataset)
        semantic_context = self.semantic_feature_proj(semantic_features).unsqueeze(0).unsqueeze(0)
        encoded = encoded + semantic_context

        physical_adj = self.adj_mx_dict[select_dataset].to(self.device)
        flow_adj = self._flow_adjacency(source)
        semantic_adj = self._semantic_adjacency(select_dataset)

        for layer in self.encoder_layers:
            encoded = layer(encoded, time_context, physical_adj, flow_adj, semantic_adj)

        z_inv, z_spe, combined, reconstructed = self.refiner(encoded, time_context)
        self._compute_auxiliary_losses(z_inv, z_spe, reconstructed, encoded)

        decoded = self.decoder(combined)
        decoded = self.horizon_proj(decoded.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        decoded = decoded * stdev[:, :1] + means[:, :1]
        return decoded
