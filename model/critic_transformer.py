from torch import nn
from einops import rearrange, pack
from .net import MLP
from typing import Sequence
import torch


class CriticTransformer(nn.Module):
    def __init__(
        self,
        context_hidden_size: int,
        hidden_size: int,
        n_heads: int,
        n_layers: int,
        dropout_p: float,
        add_turn_embedding: bool,
        pool_hidden_sizes: Sequence[int],
    ):
        super().__init__()

        self.src_proj = nn.Linear(context_hidden_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout_p,
            batch_first=True,
        )
        self.model = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=n_layers,
        )
        self.cls_emb = nn.Embedding(1, hidden_size)
        self.add_turn_embedding = add_turn_embedding
        self.pooler = MLP(
            input_dim=hidden_size,
            output_dim=1,
            hidden_sizes=pool_hidden_sizes,
            activation=nn.Tanh,
        )
        if add_turn_embedding:
            self.encoder_turn_embedding = nn.Embedding(22, hidden_size)

    def forward(
        self,
        last_usr_diaact_embeds: torch.Tensor = None,
        last_sys_diaact_embeds: torch.Tensor = None,
        belief_embeds: torch.Tensor = None,
        db_embeds: torch.Tensor = None,
        turns: torch.Tensor = None,
    ) -> torch.Tensor:
        """

        Args:
            last_usr_diaact_embeds (torch.Tensor, optional):
                shape: (B, D)
            last_sys_diaact_embeds (torch.Tensor, optional): _description_. Defaults to None.
                shape: (B, D)
            belief_embeds (torch.Tensor, optional): _description_. Defaults to None.
                shape: (B, D)
            db_embeds (torch.Tensor, optional): _description_. Defaults to None.
                shape: (B, D)
            turns (torch.Tensor, optional): _description_. Defaults to None.
                shape: (B)

        Returns:
            torch.Tensor: shape (B, 1)
        """
        device = db_embeds.device
        src_embeds = []
        if last_usr_diaact_embeds is not None:
            src_embeds.append(last_usr_diaact_embeds)
        if last_sys_diaact_embeds is not None:
            src_embeds.append(last_sys_diaact_embeds)
        if belief_embeds is not None:
            src_embeds.append(belief_embeds)
        if db_embeds is not None:
            src_embeds.append(db_embeds)
        assert len(src_embeds) > 0
        # D is the original depth
        src_embeds = rearrange(src_embeds, "S B D -> B S D")
        # shape (B, S, E)
        src_embeds = self.src_proj(src_embeds)
        if self.add_turn_embedding:
            src_embeds += rearrange(
                self.encoder_turn_embedding(turns), "B E -> B 1 E"
            )

        # shape (1, E)
        cls_embed = self.cls_emb(
            torch.tensor([0], dtype=torch.long, device=device)
        ).expand(src_embeds.shape[0], -1)
        # shape (B, *, E)
        src_embeds = pack([cls_embed, src_embeds], "B * E")[0]
        # shape (B, *, E)
        outputs = self.model(src=src_embeds)
        # shape (B, 1)
        v = self.pooler(outputs[:, 0])
        return v
