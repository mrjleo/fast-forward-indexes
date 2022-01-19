import abc
import logging
from pathlib import Path
from typing import Callable, Sequence, Union

import numpy as np
from transformers import AutoModel, AutoTokenizer


LOGGER = logging.getLogger(__name__)


class QueryEncoder(abc.ABC):
    """Base class for query encoders."""

    @abc.abstractmethod
    def encode(self, queries: Sequence[str]) -> np.ndarray:
        """Encode a list of queries.

        Args:
            queries (Sequence[str]): The queries to encode.

        Returns:
            np.ndarray: The query representations.
        """
        pass


class TransformerQueryEncoder(QueryEncoder):
    """Query encoder for transformer models. Returns the pooler output."""

    def __init__(
        self, model: Union[str, Path], device: str = "cpu", **tokenizer_args
    ) -> None:
        """Constructor.

        Args:
            model (Union[str, Path]): Pre-trained transformer model (name or path).
            device (str, optional): PyTorch device. Defaults to 'cpu'.
            **tokenizer_args: Additional tokenizer arguments.
        """
        if "tct_colbert" in model and not isinstance(self, TCTColBERTQueryEncoder):
            LOGGER.warn("consider using the TCTColBERTQueryEncoder class")

        super().__init__()
        self.model = AutoModel.from_pretrained(model)
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.device = device
        self.tokenizer_args = tokenizer_args

    def encode(self, queries: Sequence[str]) -> np.ndarray:
        inputs = self.tokenizer(queries, return_tensors="pt", **self.tokenizer_args)
        inputs.to(self.device)
        embeddings = self.model(**inputs).pooler_output.detach().cpu().numpy()
        return embeddings


class TCTColBERTQueryEncoder(TransformerQueryEncoder):
    """Query encoder for pre-trained TCT-ColBERT models.

    Adapted from Pyserini:
    https://github.com/castorini/pyserini/blob/70d79b420316d20302faaa000e08daa2416ed6e9/pyserini/dsearch/_dsearcher.py#L102
    """

    def encode(self, queries: Sequence[str]) -> np.ndarray:
        max_length = 36
        queries = ["[CLS] [Q] " + q + "[MASK]" * max_length for q in queries]
        inputs = self.tokenizer(
            queries,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
            **self.tokenizer_args
        )
        inputs.to(self.device)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.detach().cpu().numpy()
        return np.average(embeddings[:, 4:, :], axis=-2)


class LambdaQueryEncoder(QueryEncoder):
    """Query encoder adapter class for arbitrary encoding functions."""

    def __init__(self, func: Callable[[str], np.ndarray]) -> None:
        """Constructor.

        Args:
            func (Callable[[str], np.ndarray]): Function to encode a single query.
        """
        super().__init__()
        self._func = func

    def encode(self, queries: Sequence[str]) -> np.ndarray:
        return np.array(list(map(self._func, queries)))
