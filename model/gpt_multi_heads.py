from transformers import GPT2Model, PreTrainedTokenizer
from torch import nn
from torch.distributions.categorical import Categorical
from util import top_k_top_p_filtering
from typing import List, Tuple, Dict
import torch

categories = ["domain", "intent", "slot", "db"]
symbs = ["[d]", "[i]", "[s]", "[db]"]


class GPT2MultiHead(nn.Module):
    def __init__(
        self,
        model: GPT2Model,
        n_domains: int,
        n_intents: int,
        n_slots: int,
        n_cats: int,
        add_cat_emb: bool = False,
        add_db_emb: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.cat_head = nn.Linear(model.config.n_embd, n_cats)
        self.domain_head = nn.Linear(model.config.n_embd, n_domains)
        self.intent_head = nn.Linear(model.config.n_embd, n_intents)
        self.slot_head = nn.Linear(model.config.n_embd, n_slots)
        self.add_cat_emb = add_cat_emb
        self.add_db_emb = add_db_emb
        if add_cat_emb:
            self.cat_emb = nn.Embedding(n_cats, model.config.n_embd)
        if add_db_emb:
            self.db_emb = nn.Linear(24, model.config.n_embd)

    def forward(
        self,
        input_ids: torch.Tensor,
        cat_ids: torch.Tensor = None,
        db_ids: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
    ):
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
            inputs_embeds=inputs_embeds, token_type_ids=token_type_ids
        )
        cat_logits = self.cat_head(outputs[0])
        domain_logits = self.domain_head(outputs[0])
        intent_logits = self.intent_head(outputs[0])
        slot_logits = self.slot_head(outputs[0])
        return {
            "cat_logits": cat_logits,
            "domain_logits": domain_logits,
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
            "hidden_states": outputs[0],
        }

    def _sample_or_greedy(
        self,
        logits,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k=0,
        top_p=1.0,
    ) -> Tuple[float, float, float]:
        # shape (dim)
        logits = logits[0, -1]
        log_prob = None
        entropy = None
        dist = Categorical(logits=logits)
        if do_sample:
            logits = top_k_top_p_filtering(
                logits=logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            dist = Categorical(logits=logits)
            id = dist.sample()
        else:
            id = logits.max(dim=0)[1]
        log_prob = dist.log_prob(id).item()
        entropy = dist.entropy().item()
        id = id.item()
        return id, log_prob, entropy

    def generate_one_step(
        self,
        input_ids: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        cat_ids: torch.Tensor = None,
        db_ids: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k=0,
        top_p=1.0,
        add_cat_str: bool = False,
    ) -> Dict[str, float]:
        outputs = self(input_ids, cat_ids, db_ids, token_type_ids)
        cat_id, cat_log_prob, cat_entropy = self._sample_or_greedy(
            logits=outputs["cat_logits"],
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if categories[cat_id] == "db":
            label_id = -100
            label_log_prob = 0.0
            label_entropy = 0.0
        else:
            symb: str = symbs[cat_id]
            if add_cat_str:
                new_input_id: int = tokenizer.convert_tokens_to_ids(symb)
                input_ids = torch.cat(
                    [
                        input_ids,
                        torch.tensor(
                            [[new_input_id]],
                            dtype=torch.long,
                            device=input_ids.device,
                        ),
                    ],
                    dim=1,
                )
                assert cat_ids is None
                if db_ids is not None:
                    last_db_id = db_ids[:, -1:, :]
                    db_ids = torch.cat(
                        [
                            db_ids,
                            last_db_id,
                        ],
                        dim=1,
                    )
                if token_type_ids is not None:
                    token_type_ids = torch.cat(
                        [
                            token_type_ids,
                            torch.tensor(
                                [[1]],
                                device=input_ids.device,
                                dtype=torch.long,
                            ),
                        ],
                        dim=1,
                    )
            outputs = self(input_ids, cat_ids, db_ids, token_type_ids)
            label_id, label_log_prob, label_entropy = self._sample_or_greedy(
                logits=outputs[f"{categories[cat_id]}_logits"],
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        # return cat_id, label_id
        return {
            "cat_id": cat_id,
            "cat_log_prob": cat_log_prob,
            "cat_entropy": cat_entropy,
            "label_id": label_id,
            "label_log_prob": label_log_prob,
            "label_entropy": label_entropy,
            "hidden_states": outputs["hidden_states"],
        }
