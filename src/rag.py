from functools import lru_cache
from pathlib import Path

import numpy as np
import ollama

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_PATHS = (
    ("README.md", PROJECT_ROOT / "README.md"),
    ("docs/sensor_caveats.md", PROJECT_ROOT / "docs" / "sensor_caveats.md"),
)
EMBEDDING_MODEL = "embeddinggemma"


def _split_markdown(source_name: str, text: str) -> list[dict[str, str]]:
    """Keep headings with their paragraphs and split each bullet into its own passage."""
    passages = []
    heading = ""
    paragraph_lines = []

    def save_paragraph():
        paragraph = " ".join(paragraph_lines).strip()
        if len(paragraph) > 40:
            content = f"{heading}\n{paragraph}" if heading else paragraph
            passages.append({"source": source_name, "text": content})
        paragraph_lines.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            save_paragraph()
            heading = line.lstrip("# ").strip()
        elif line.startswith("- "):
            save_paragraph()
            bullet = line[2:].strip()
            content = f"{heading}\n{bullet}" if heading else bullet
            if len(content) > 40:
                passages.append({"source": source_name, "text": content})
        elif not line:
            save_paragraph()
        else:
            paragraph_lines.append(line)
    save_paragraph()
    return passages


@lru_cache(maxsize=1)
def _document_index():
    passages = []
    for source_name, path in DOCUMENT_PATHS:
        if not path.is_file():
            raise FileNotFoundError(f"RAG document not found: {path}")
        passages.extend(_split_markdown(source_name, path.read_text(encoding="utf-8")))
    if not passages:
        raise ValueError("No searchable passages found in the project documents")

    vectors = []
    for item in passages:
        result = ollama.embed(model=EMBEDDING_MODEL, input=item["text"])
        vectors.append(np.asarray(result.embeddings[0], dtype=np.float32))
    return passages, np.vstack(vectors)


def retrieve_passages(query: str, top_k: int = 3) -> list[dict]:
    if not query.strip():
        return []
    passages, vectors = _document_index()
    result = ollama.embed(model=EMBEDDING_MODEL, input=query)
    query_vector = np.asarray(result.embeddings[0], dtype=np.float32)
    denominator = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
    similarities = (vectors @ query_vector) / np.maximum(denominator, 1e-12)
    indexes = np.argsort(similarities)[::-1][:max(1, top_k)]
    return [
        {
            "source": passages[i]["source"],
            "text": passages[i]["text"],
            "relevance": round(float(similarities[i]), 3),
        }
        for i in indexes
    ]
