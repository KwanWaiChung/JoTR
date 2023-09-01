from transformers import GPT2LMHeadModel, PreTrainedTokenizer, GPT2Model
from torch import nn
from torch.distributions.categorical import Categorical
from util import top_k_top_p_filtering
from .net import MLP
from typing import List, Tuple, Dict, Sequence
import torch

categories = ["domain", "intent", "slot", "db"]
symbs = ["[d]", "[i]", "[s]", "[db]"]


class TransformerSingleHead(nn.Module):
    def __init__(
        self,
        model: GPT2LMHeadModel,
        add_cat_emb: bool = False,
        add_cat_head: bool = False,
        add_db_emb: bool = False,
        cat_hidden_sizes: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.model = model
        self.add_cat_emb = add_cat_emb
        self.add_db_emb = add_db_emb
        self.add_cat_head = add_cat_head
        if add_cat_head:
            self.cat_head = MLP(
                input_dim=model.config.n_embd,
                output_dim=4,
                hidden_sizes=cat_hidden_sizes,
            )
        if add_cat_emb:
            self.cat_emb = nn.Embedding(4, model.config.n_embd)
        if add_db_emb:
            self.db_emb = nn.Linear(24, model.config.n_embd)

    def forward(
        self,
        input_ids: torch.LongTensor,
        cat_ids: torch.LongTensor = None,
        db_ids: torch.LongTensor = None,
        token_type_ids: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:
        # shape (B, T, h)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if cat_ids is not None or self.add_cat_emb:
            cat_ids = cat_ids.clone()
            cat_ids[cat_ids == -100] = 0
            cat_embeds = self.cat_emb(cat_ids)
            inputs_embeds += cat_embeds
        if db_ids is not None or self.add_db_emb:
            db_embeds = self.db_emb(db_ids)
            inputs_embeds += db_embeds

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=True,
        )
        cat_logits = None
        last_hidden_states: torch.Tensor = outputs["hidden_states"][-1]
        if self.add_cat_head:
            cat_logits = self.cat_head(last_hidden_states)
        return {
            "cat_logits": cat_logits,
            "hidden_states": last_hidden_states,
            "logits": outputs["logits"],
        }
