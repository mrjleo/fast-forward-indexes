""".. include:: docs/encoder.md"""  # noqa: D400, D415

import abc
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class Encoder(abc.ABC):
    """Base class for encoders."""

    @abc.abstractmethod
    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        pass

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a list of texts.

        :param texts: The texts to encode.
        :return: The resulting vector representations.
        """
        return self._encode(texts)


class TransformerEncoder(Encoder):
    """Uses a pre-trained transformer model for encoding. Returns the pooler output."""

    def __init__(
        self, model: str | Path, device: str = "cpu", **tokenizer_args: Any
    ) -> None:
        """Create a transformer encoder.

        :param model: Pre-trained transformer model (name or path).
        :param device: PyTorch device.
        :param **tokenizer_args: Additional tokenizer arguments.
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.device = device
        self.tokenizer_args = tokenizer_args

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.tokenizer(list(texts), return_tensors="pt", **self.tokenizer_args)
        inputs.to(self.device)
        with torch.no_grad():
            return self.model(**inputs).pooler_output.detach().cpu().numpy()


class LambdaEncoder(Encoder):
    """Encoder adapter class for arbitrary encoding functions."""

    def __init__(self, f: Callable[[str], np.ndarray]) -> None:
        """Create a lambda encoder.

        :param f: Function to encode a single piece of text.
        """
        super().__init__()
        self._f = f

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.array(list(map(self._f, texts)))


class TCTColBERTQueryEncoder(TransformerEncoder):
    """Query encoder for pre-trained TCT-ColBERT models.

    Adapted from Pyserini:
    https://github.com/castorini/pyserini/blob/310c828211bb3b9528cfd59695184c80825684a2/pyserini/encode/_tct_colbert.py#L72
    """

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        max_length = 36
        inputs = self.tokenizer(
            ["[CLS] [Q] " + q + "[MASK]" * max_length for q in texts],
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
            **self.tokenizer_args,
        )
        inputs.to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.detach().cpu().numpy()
            return np.average(embeddings[:, 4:, :], axis=-2)


class TCTColBERTDocumentEncoder(TransformerEncoder):
    """Document encoder for pre-trained TCT-ColBERT models.

    Adapted from Pyserini:
    https://github.com/castorini/pyserini/blob/310c828211bb3b9528cfd59695184c80825684a2/pyserini/encode/_tct_colbert.py#L27
    """

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        max_length = 512
        inputs = self.tokenizer(
            ["[CLS] [D] " + text for text in texts],
            max_length=max_length,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
            **self.tokenizer_args,
        )
        inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
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
