from torch import nn
from einops import rearrange, pack
from torch.distributions import Categorical
import torch


class DiaactTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_hidden_size: int,
        hidden_size: int,
        n_heads: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        add_encoder_turn_embedding: bool,
        add_encoder_type_embedding: bool,
        add_decoder_pos_embedding: bool,
        add_decoder_type_embedding: bool,
        tie_weights: bool = False,
        max_turn: int = 20,
        max_resp_len: int = 256,
        pad_id: int = 0,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_resp_len = max_resp_len
        self.tgt_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.pad_id = pad_id
        self.hidden_size = hidden_size
        self.src_proj = nn.Linear(context_hidden_size, hidden_size)
        self.add_encoder_turn_embedding = add_encoder_turn_embedding
        if add_encoder_turn_embedding:
            self.encoder_turn_embedding = nn.Embedding(max_turn, hidden_size)

        self.add_encoder_type_embedding = add_encoder_type_embedding
        if add_encoder_type_embedding:
            self.encoder_type_embedding = nn.Embedding(4, hidden_size)

        self.add_decoder_pos_embedding = add_decoder_pos_embedding
        if self.add_decoder_pos_embedding:
            self.decoder_pos_embedding = nn.Embedding(
                max_resp_len, hidden_size
            )

        self.add_decoder_type_embedding = add_decoder_type_embedding
        if self.add_decoder_type_embedding:
            self.decoder_type_embedding = nn.Embedding(3, hidden_size)

        self.tgt_proj = nn.Linear(hidden_size, vocab_size)
        if tie_weights:
            self.tgt_proj.weight = self.tgt_embeddings.weight
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=4 * hidden_size,
            dropout=dropout_p,
            batch_first=True,
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    def generate(
        self,
        tgt_ids: torch.Tensor,
        last_usr_diaact_embeds: torch.Tensor = None,
        last_sys_diaact_embeds: torch.Tensor = None,
        belief_embeds: torch.Tensor = None,
        db_embeds: torch.Tensor = None,
        turns: torch.Tensor = None,
        eos_token_id: int = None,
        do_sample: bool = False,
        domain_idx: torch.Tensor = None,
        intent_idx: torch.Tensor = None,
        slot_idx: torch.Tensor = None,
        start_idx: int = None,
    ):
        """Arguments similar to forward. Only change is that tgt_ids should
            be the decoder prefix.

        Args:
            tgt_ids (torch.Tensor): (B, T)
            tgt_key_padding_mask (torch.Tensor): (B, T)
            last_usr_diaact_embeds (torch.Tensor): (B, E)
            last_sys_diaact_embeds (torch.Tensor): (B, E)
            belief_embeds (torch.Tensor): (B, E)
            db_embeds (torch.Tensor): (B, E)
            turns (torch.Tensor): (B)
            eos_token_id (int): If given, it will terminate the generation
                once it was generated.
            pad_id (int): For padding the finished sequence.
            do_sample (bool): If True, sample token by softmax. Otherwise,
                do argmax.

        Returns:
            tgt_ids: (B, *).
        """
        # TODO: Verify correctness
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
        src_embeds = rearrange(src_embeds, "S B E -> B S E")
        src_embeds = self.src_proj(src_embeds)

        device = src_embeds.device
        if self.add_encoder_turn_embedding:
            src_embeds += rearrange(
                self.encoder_turn_embedding(turns), "B E -> B 1 E"
            )
        if self.add_encoder_type_embedding:
            src_embeds += rearrange(
                self.encoder_type_embedding(
                    torch.arange(src_embeds.shape[1], device=device)
                ),
                "S E -> 1 S E",
            )
        # shape (B, T, E)
        tgt_input_embeds = self.tgt_embeddings(tgt_ids)
        bz = tgt_input_embeds.shape[0]
        continue_mask = torch.ones(bz, dtype=torch.bool)
        tgt_type_ids = [0]
        status = -1
        while continue_mask.any() and tgt_ids.shape[1] < self.max_resp_len:
            tgt_embeds = tgt_input_embeds
            len_t = tgt_embeds.shape[1]
            if self.add_decoder_pos_embedding:
                # shape (T)
                tgt_pos_ids = torch.arange(
                    len_t, dtype=torch.long, device=device
                )
                # shape (1, T, E)
                tgt_pos_embeds = self.decoder_pos_embedding(tgt_pos_ids)
                tgt_embeds = tgt_embeds + tgt_pos_embeds
            if self.add_decoder_type_embedding:
                tgt_embeds = tgt_embeds + self.decoder_type_embedding(
                    torch.tensor(tgt_type_ids, dtype=torch.long, device=device)
                )
                tgt_type_ids.append((tgt_type_ids[-1] + 1) % 3)
            # shape (b, T, E), b = ongoing batch size.
            tgt_hiddens = self.transformer(
                src=src_embeds[continue_mask],
                tgt=tgt_embeds[continue_mask],
                tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                    len_t
                ).to(device),
            )
            # shape (b, V)
            logits = self.tgt_proj(tgt_hiddens[:, -1])
            # shape (V)
            mask = torch.ones_like(logits, dtype=torch.bool)[0]
            if status > -1:
                if status % 3 == 0:
                    mask[domain_idx] = 0
                elif status % 3 == 1:
                    mask[intent_idx] = 0
                elif status % 3 == 2:
                    mask[slot_idx] = 0
                status += 1
                logits[:, mask] = float("-inf")
            if do_sample:
                # shape (B)
                output_ids = Categorical(logits=logits).sample()
            else:
                output_ids = logits.max(dim=1)[1]
            # shape (B, T+1)
            tgt_ids = torch.cat(
                [
                    tgt_ids,
                    torch.ones((bz, 1), dtype=torch.long, device=device)
                    * self.pad_id,
                ],
                dim=1,
            )
            tgt_ids[continue_mask, -1] = output_ids
            # shape (B, T+1, E)
            tgt_input_embeds = pack(
                [tgt_input_embeds, self.tgt_embeddings(tgt_ids[:, -1])],
                "B * E",
            )[0]
            if domain_idx is not None and (output_ids == start_idx).any():
                status = 0
            if eos_token_id is not None:
                # shape (b), b = ongoing batch size
                global_idx = continue_mask.nonzero(as_tuple=True)[0]
                continue_mask[global_idx[output_ids == eos_token_id]] = 0
        return tgt_ids

    def forward(
        self,
        tgt_ids: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor = None,
        last_usr_diaact_embeds: torch.Tensor = None,
        last_sys_diaact_embeds: torch.Tensor = None,
        belief_embeds: torch.Tensor = None,
        db_embeds: torch.Tensor = None,
        turns: torch.Tensor = None,
    ):
        """

        Args:
            tgt_ids (torch.Tensor): (B, T)
            tgt_key_padding_mask (torch.Tensor): (B, T)
            last_usr_diaact_embeds (torch.Tensor): (B, E)
            last_sys_diaact_embeds (torch.Tensor): (B, E)
            belief_embeds (torch.Tensor): (B, E)
            db_embeds (torch.Tensor): (B, E)
            turns (torch.Tensor): (B)

        Returns:
            Dict with keys:
                `loss` (float).
                `logits` (torch.Tensor).

        """
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
        src_embeds = rearrange(src_embeds, "S B E -> B S E")
        src_embeds = self.src_proj(src_embeds)

        device = src_embeds.device
        if self.add_encoder_turn_embedding:
            src_embeds += rearrange(
                self.encoder_turn_embedding(turns), "B E -> B 1 E"
            )
        if self.add_encoder_type_embedding:
            src_embeds += rearrange(
                self.encoder_type_embedding(
                    torch.arange(src_embeds.shape[1], device=device)
                ),
                "S E -> 1 S E",
            )
        # shape (B, T, E)
        tgt_embeds = self.tgt_embeddings(tgt_ids)
        len_t = tgt_embeds.shape[1]
        if self.add_decoder_pos_embedding:
            input_ids = torch.arange(len_t, dtype=torch.long, device=device)
            tgt_embeds += self.decoder_pos_embedding(input_ids)
        if self.add_decoder_type_embedding:
            input_ids = (
                torch.arange(len_t, dtype=torch.long, device=device) % 3
            )
            tgt_embeds += self.decoder_type_embedding(input_ids)

        # shape (B, T, E)
        tgt_hiddens = self.transformer(
            src=src_embeds,
            tgt=tgt_embeds,
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(len_t).to(
                device
            ),
            tgt_key_padding_mask=~tgt_key_padding_mask.bool()
            if tgt_key_padding_mask is not None
            else None,
        )
        # shape (B, T, V), V=vocab size
        logits = self.tgt_proj(tgt_hiddens)
        loss = self.loss_fn(
            rearrange(logits[:, :-1, :], "B T V -> (B T) V"),
            rearrange(tgt_ids[:, 1:], "B T -> (B T)"),
        )
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": tgt_hiddens,
        }
