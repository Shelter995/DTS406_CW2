import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from part_two.utils.text_utils import normalize_text, sentence_split, word_tokens


@dataclass
class Chunk:
    chunk_id: int
    text: str
    start_sentence: int
    end_sentence: int
    word_count: int


class EmbeddingRetriever:
    """Lightweight local embedding retriever for article chunk selection."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.backend = "transformers"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(str(model_dir), device=str(self.device))
            self.backend = "sentence_transformers"
            self.tokenizer = None
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            self.model = AutoModel.from_pretrained(str(model_dir)).to(self.device)
            self.model.eval()

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        texts = [normalize_text(text) for text in texts]
        if self.backend == "sentence_transformers":
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.astype(np.float32)

        all_embeddings: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                embeddings = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
        return np.vstack(all_embeddings)


def make_query(article: str) -> str:
    sentences = sentence_split(article)
    return " ".join(sentences[: min(3, len(sentences))]) if sentences else normalize_text(article)


def sentence_aware_chunks(
    article: str,
    target_words: int,
    overlap_words: int,
) -> List[Chunk]:
    sentences = sentence_split(article)
    if not sentences:
        return []

    chunks: List[Chunk] = []
    current_sentences: List[str] = []
    current_start = 0
    current_words = 0

    for idx, sentence in enumerate(sentences):
        sentence_words = len(word_tokens(sentence))
        if current_sentences and current_words + sentence_words > target_words:
            chunks.append(
                _build_chunk(
                    chunk_id=len(chunks),
                    sentences=current_sentences,
                    start_sentence=current_start,
                    end_sentence=idx - 1,
                )
            )
            overlap_sentences = _overlap_sentences(current_sentences, overlap_words)
            current_start = idx - len(overlap_sentences)
            current_sentences = overlap_sentences
            current_words = sum(len(word_tokens(item)) for item in current_sentences)

        current_sentences.append(sentence)
        current_words += sentence_words

    if current_sentences:
        chunks.append(
            _build_chunk(
                chunk_id=len(chunks),
                sentences=current_sentences,
                start_sentence=current_start,
                end_sentence=len(sentences) - 1,
            )
        )
    return chunks


def retrieve_top_chunks(
    query: str,
    chunks: Sequence[Chunk],
    retriever: EmbeddingRetriever,
    top_k: int,
) -> List[Dict[str, object]]:
    if not chunks:
        return []
    if len(chunks) == 1:
        return [{"chunk": chunks[0], "score": 1.0}]

    texts = [query] + [chunk.text for chunk in chunks]
    embeddings = retriever.encode(texts)
    query_embedding = embeddings[0]
    chunk_embeddings = embeddings[1:]
    scores = chunk_embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][: min(top_k, len(chunks))]
    selected = [
        {"chunk": chunks[int(idx)], "score": float(scores[int(idx)])} for idx in top_indices
    ]
    selected.sort(key=lambda item: item["chunk"].chunk_id)
    return selected


def build_summary_prompt(prefix: str, content: str) -> str:
    return f"{prefix}\n\n{normalize_text(content)}"


def _build_chunk(
    chunk_id: int,
    sentences: Sequence[str],
    start_sentence: int,
    end_sentence: int,
) -> Chunk:
    text = normalize_text(" ".join(sentences))
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        start_sentence=start_sentence,
        end_sentence=end_sentence,
        word_count=len(word_tokens(text)),
    )


def _overlap_sentences(sentences: Sequence[str], overlap_words: int) -> List[str]:
    selected: List[str] = []
    total_words = 0
    for sentence in reversed(sentences):
        selected.insert(0, sentence)
        total_words += len(word_tokens(sentence))
        if total_words >= overlap_words:
            break
    return selected


def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts

