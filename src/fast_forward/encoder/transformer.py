from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from fast_forward.encoder.base import Encoder

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class TransformerEncoder(Encoder):
    """Uses a pre-trained transformer model for encoding. Returns the pooler output."""

    def __init__(
        self, model: "str | Path", device: str = "cpu", **tokenizer_args: Any
    ) -> None:
        """Create a transformer encoder.

        :param model: Pre-trained transformer model (name or path).
        :param device: PyTorch device.
        :param **tokenizer_args: Additional tokenizer arguments.
        """
        super().__init__()
        self._model = AutoModel.from_pretrained(model)
        self._model.to(device)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._device = device
        self._tokenizer_args = tokenizer_args

    def _encode(self, texts: "Sequence[str]") -> np.ndarray:
        inputs = self._tokenizer(
            list(texts), return_tensors="pt", **self._tokenizer_args
        )
        inputs.to(self._device)
        with torch.no_grad():
            return self._model(**inputs).pooler_output.detach().cpu().numpy()


class TCTColBERTQueryEncoder(TransformerEncoder):
    """Query encoder for pre-trained TCT-ColBERT models.

    Adapted from Pyserini:
    https://github.com/castorini/pyserini/blob/310c828211bb3b9528cfd59695184c80825684a2/pyserini/encode/_tct_colbert.py#L72
    """

    def _encode(self, texts: "Sequence[str]") -> np.ndarray:
        max_length = 36
        inputs = self._tokenizer(
            ["[CLS] [Q] " + q + "[MASK]" * max_length for q in texts],
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
            **self._tokenizer_args,
        )
        inputs.to(self._device)
        with torch.no_grad():
            embeddings = self._model(**inputs).last_hidden_state.detach().cpu().numpy()
            return np.average(embeddings[:, 4:, :], axis=-2)


class TCTColBERTDocumentEncoder(TransformerEncoder):
    """Document encoder for pre-trained TCT-ColBERT models.

    Adapted from Pyserini:
    https://github.com/castorini/pyserini/blob/310c828211bb3b9528cfd59695184c80825684a2/pyserini/encode/_tct_colbert.py#L27
    """

    def _encode(self, texts: "Sequence[str]") -> np.ndarray:
        max_length = 512
        inputs = self._tokenizer(
            ["[CLS] [D] " + text for text in texts],
            max_length=max_length,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
            **self._tokenizer_args,
        )
        inputs.to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
            token_embeddings = outputs["last_hidden_state"][:, 4:, :]
            input_mask_expanded = (
                inputs.attention_mask[:, 4:]
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            return embeddings.detach().cpu().numpy()
