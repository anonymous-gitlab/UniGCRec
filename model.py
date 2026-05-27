from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class _MLPLayers(nn.Module):
    def __init__(self, layers: Sequence[int], dropout: float = 0.0):
        super().__init__()
        mods: List[nn.Module] = []
        for idx in range(len(layers) - 1):
            mods.append(nn.Dropout(p=dropout))
            mods.append(nn.Linear(layers[idx], layers[idx + 1]))
            if idx != len(layers) - 2:
                mods.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_layers(x)


class _VectorQuantizer(nn.Module):
    def __init__(self, n_e: int, e_dim: int):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.embedding = nn.Embedding(n_e, e_dim)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        w = self.embedding.weight
        d = (torch.sum(x ** 2, dim=1, keepdim=True)
             + torch.sum(w ** 2, dim=1, keepdim=True).t()
             - 2.0 * torch.matmul(x, w.t()))
        return torch.argmin(d, dim=-1)


class _ResidualVectorQuantizer(nn.Module):
    def __init__(self, n_e_list: Sequence[int], e_dim: int):
        super().__init__()
        self.vq_layers = nn.ModuleList(
            [_VectorQuantizer(n_e, e_dim) for n_e in n_e_list]
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        indices: List[torch.Tensor] = []
        residual = x
        for q in self.vq_layers:
            idx = q.encode(residual)
            indices.append(idx)
            residual = residual - q.embedding(idx)
        return torch.stack(indices, dim=-1)


class RQVAE(nn.Module):
    def __init__(self, in_dim: int, e_dim: int,
                 num_emb_list: Sequence[int], layers: Sequence[int]):
        super().__init__()
        enc_dims = [in_dim] + list(layers) + [e_dim]
        self.encoder = _MLPLayers(enc_dims)
        self.rq = _ResidualVectorQuantizer(num_emb_list, e_dim)
        self.decoder = _MLPLayers(list(reversed(enc_dims)))

    @torch.no_grad()
    def get_indices(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.rq.encode(z)


class UserSTGate(nn.Module):
    def __init__(self, collab_dim: int, num_domains: int):
        super().__init__()
        self.agg_mlp = nn.Sequential(
            nn.Linear(2 * (num_domains - 1), 64),
            nn.ReLU(),
            nn.Linear(64, num_domains - 1),
        )
        self.gate_proj = nn.Linear(2 * collab_dim, collab_dim)

    def forward(self, c_anchor: torch.Tensor, c_others: torch.Tensor,
                sim_features: torch.Tensor, act_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = torch.stack([sim_features, act_features], dim=-1)
        feat = feat.view(feat.size(0), -1)
        w = self.agg_mlp(feat)
        if mask is not None:
            no_source = ~mask.any(dim=-1)
            w = w.masked_fill(~mask, float("-inf"))
            w = torch.softmax(w, dim=-1)
            w = torch.nan_to_num(w, nan=0.0)
        else:
            no_source = torch.zeros(w.size(0), dtype=torch.bool, device=w.device)
            w = torch.softmax(w, dim=-1)
        x_u = torch.sum(w.unsqueeze(-1) * c_others, dim=1)
        beta = torch.sigmoid(self.gate_proj(torch.cat([c_anchor, x_u], dim=-1)))
        if no_source.any():
            beta = torch.where(no_source.unsqueeze(-1), torch.ones_like(beta), beta)
        return beta * torch.tanh(c_anchor) + (1 - beta) * torch.tanh(x_u)


class UserFusion(nn.Module):
    def __init__(self, sem_dim: int, collab_dim: int, hidden_dim: int,
                 sem_floor: float = 0.35, collab_dropout: float = 0.2):
        super().__init__()
        self.sem_floor = sem_floor
        self.sem_branch = nn.Sequential(
            nn.Linear(sem_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.collab_branch = nn.Sequential(
            nn.Linear(collab_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(collab_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, s_u: torch.Tensor, c_bar: torch.Tensor) -> torch.Tensor:
        z_s = self.sem_branch(s_u)
        z_c = self.collab_branch(c_bar)
        gate = torch.sigmoid(self.gate(torch.cat([z_s, z_c], dim=-1)))
        sem_w = self.sem_floor + (1.0 - self.sem_floor) * gate
        return sem_w * z_s + (1.0 - sem_w) * z_c


class UniGCRec(nn.Module):
    def __init__(self, sem_dim: int, collab_dim: int, raw_collab_dim: int,
                 hidden_dim: int, num_domains: int, code_num: int,
                 code_depth: int, e_dim: int, sem_floor: float = 0.35,
                 collab_dropout: float = 0.2):
        super().__init__()
        self.collab_proj = (
            nn.Linear(raw_collab_dim, collab_dim)
            if raw_collab_dim != collab_dim else nn.Identity()
        )
        self.user_st_gate = UserSTGate(collab_dim, num_domains)
        self.user_fusion = UserFusion(
            sem_dim, collab_dim, hidden_dim, sem_floor, collab_dropout
        )
        self.item_fusion_mlp = nn.Sequential(
            nn.Linear(sem_dim + collab_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        rqvae_layers = [hidden_dim, hidden_dim // 2, e_dim * code_depth]
        num_emb_list = [code_num] * code_depth
        self.rqvae_user = RQVAE(hidden_dim, e_dim, num_emb_list, rqvae_layers)
        self.rqvae_item = RQVAE(hidden_dim, e_dim, num_emb_list, rqvae_layers)
        self.register_buffer(
            "_loaded", torch.zeros((), dtype=torch.bool), persistent=False
        )

    def encode_user(self, s_u: torch.Tensor, c_anchor: torch.Tensor,
                    c_others: torch.Tensor, sim_features: torch.Tensor,
                    act_features: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        c_anchor = self.collab_proj(c_anchor)
        c_others = self.collab_proj(c_others)
        c_bar = self.user_st_gate(
            c_anchor, c_others, sim_features, act_features, mask
        )
        return self.user_fusion(s_u, c_bar)

    def encode_item(self, s_i: torch.Tensor, c_i: torch.Tensor) -> torch.Tensor:
        c_i = self.collab_proj(c_i)
        return self.item_fusion_mlp(torch.cat([s_i, c_i], dim=-1))

    @staticmethod
    def _extract_state_dict(state):
        if isinstance(state, dict) and "model" in state:
            return state["model"]
        if isinstance(state, dict) and "state_dict" in state:
            return state["state_dict"]
        return state

    def _load_checked(self, sd) -> None:
        own_keys = set(self.state_dict().keys())
        missing = own_keys - set(sd.keys())
        if missing:
            sample = sorted(missing)[:3]
            raise RuntimeError(
                "Checkpoint is missing {} required parameter key(s) (e.g. {}). "
                "A compatible saved state is required.".format(
                    len(missing), sample
                )
            )
        self.load_state_dict(sd, strict=False)
        self._loaded.fill_(True)
        self.eval()

    def load_pretrained(self, ckpt_path: str, map_location: str = "cpu") -> None:
        state = torch.load(ckpt_path, map_location=map_location)
        self._load_checked(self._extract_state_dict(state))

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, map_location: str = "cpu") -> "UniGCRec":
        state = torch.load(ckpt_path, map_location=map_location)
        sd = cls._extract_state_dict(state)

        collab_dim = sd["user_st_gate.gate_proj.weight"].shape[0]
        num_domains = sd["user_st_gate.agg_mlp.0.weight"].shape[1] // 2 + 1
        hidden_dim, sem_dim = sd["user_fusion.sem_branch.0.weight"].shape
        if "collab_proj.weight" in sd:
            raw_collab_dim = sd["collab_proj.weight"].shape[1]
        else:
            raw_collab_dim = collab_dim
        code_depth = sum(
            1 for k in sd
            if k.startswith("rqvae_user.rq.vq_layers.")
            and k.endswith(".embedding.weight")
        )
        code_num, e_dim = sd["rqvae_user.rq.vq_layers.0.embedding.weight"].shape

        sem_floor = 0.35
        args = state.get("args") if isinstance(state, dict) else None
        if isinstance(args, dict):
            sem_floor = args.get("sem_floor", sem_floor)
        elif args is not None:
            sem_floor = getattr(args, "sem_floor", sem_floor)

        model = cls(
            sem_dim=int(sem_dim), collab_dim=int(collab_dim),
            raw_collab_dim=int(raw_collab_dim), hidden_dim=int(hidden_dim),
            num_domains=int(num_domains), code_num=int(code_num),
            code_depth=int(code_depth), e_dim=int(e_dim), sem_floor=float(sem_floor),
        )
        model._load_checked(sd)
        return model

    def _ensure_loaded(self) -> None:
        if not bool(self._loaded.item()):
            raise RuntimeError(
                "Model weights have not been loaded. Use "
                "UniGCRec.from_checkpoint(path) or load_pretrained(path) first."
            )

    @torch.no_grad()
    def user_csc_id(self, s_u: torch.Tensor, c_anchor: torch.Tensor,
                    c_others: torch.Tensor, sim_features: torch.Tensor,
                    act_features: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._ensure_loaded()
        h_u = self.encode_user(s_u, c_anchor, c_others, sim_features,
                               act_features, mask)
        return self.rqvae_user.get_indices(h_u)

    @torch.no_grad()
    def item_csc_id(self, s_i: torch.Tensor, c_i: torch.Tensor) -> torch.Tensor:
        self._ensure_loaded()
        h_i = self.encode_item(s_i, c_i)
        return self.rqvae_item.get_indices(h_i)
