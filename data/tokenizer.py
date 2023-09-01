import json
import os
from typing import Tuple, Dict, Optional, List
from transformers import PreTrainedTokenizer


class FixedVocabTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab: List[str],
        model_max_length: int = None,
        bos_token=None,
        eos_token=None,
        unk_token="[UNK]",
        pad_token="[PAD]",
        add_token_type: bool = True,
        lower_case: bool = False,
    ):
        super().__init__(model_max_length=model_max_length)
        self.lower_case = lower_case
        if lower_case:
            vocab = [v.lower() for v in vocab]
        vocab = sorted(set([v for v in vocab if v]))
        if pad_token is not None:
            if lower_case:
                pad_token = pad_token.lower()
            self.pad_token = pad_token
            # pad token always come first
            vocab = [pad_token] + vocab
        self.bos_token2 = None
        if bos_token is not None:
            if lower_case:
                bos_token = bos_token.lower()
            self.bos_token2 = bos_token
            vocab.append(bos_token)
        if eos_token is not None:
            if lower_case:
                eos_token = eos_token.lower()
            self.eos_token = eos_token
            vocab.append(eos_token)
        if unk_token is not None:
            if lower_case:
                unk_token = unk_token.lower()
            self.unk_token = unk_token
            vocab.append(unk_token)
        if add_token_type:
            vocab = ["token_type_1", "token_type_2"] + vocab

        self.stoi = {token: i for i, token in enumerate(vocab)}
        self.itos = {i: token for i, token in enumerate(vocab)}

    def _convert_token_to_id(self, token: str) -> int:
        # backward compatable. Older version doesn't have lower case.
        if hasattr(self, "lower_case") and self.lower_case:
            token = token.lower()
        return (
            self.stoi[token.strip()]
            if token in self.stoi
            else self.unk_token_id
        )

    def _convert_id_to_token(self, index: int) -> str:
        return self.itos[index] if index in self.itos else self.unk_token

    def get_vocab(self) -> Dict[str, int]:
        return self.stoi.copy()

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return text.split()

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if filename_prefix is None:
            filename_prefix = ""
        vocab_path = os.path.join(
            save_directory, filename_prefix + "vocab.json"
        )
        json.dump(self.stoi, open(vocab_path, "w"))
        return (str(vocab_path),)

    def prepare_for_tokenization(
        self, text, is_split_into_words=False, **kwargs
    ):
        """To remain compatable with GPT2Tokenizer.
        Just pop the argument and do nothing.
        """
        if hasattr(self, "bos_token2") and self.bos_token2 is not None:
            text = f"{self.bos_token2} {text}"
        kwargs.pop("add_prefix_space", None)
        return (text, kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)
