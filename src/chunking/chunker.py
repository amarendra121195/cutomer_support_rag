"""Chunking: split documents into overlapping, retrieval-sized pieces."""
import re
from dataclasses import dataclass

from src.ingestion.loader import Document

# Places where text may be cut, from most to least preferred. Every pattern is zero-width,
# so the pieces always join back into exactly the original text.
_BOUNDARIES = [
    r"(?=\n#{2,3} )",     # before a heading
    r"(?<=\n\n)",         # after a paragraph
    r"(?<=\n)",           # after a line
    r"(?<=[.!?] )",       # after a sentence
    r"(?<= )",            # after a word
]


@dataclass
class Chunk:
    """A piece of a document, ready to embed and store."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedding model. The title gives each chunk its document's context."""
        return f"{self.metadata['title']}\n{self.text}"


def _split(text: str, size: int, boundaries: list[str] = _BOUNDARIES) -> list[str]:
    """Cut `text` into pieces of at most `size` characters at the best available boundary."""
    if len(text) <= size:
        return [text]
    if not boundaries:
        return [text[i:i + size] for i in range(0, len(text), size)]
    pieces = []
    for part in re.split(boundaries[0], text):
        if part:
            pieces.extend(_split(part, size, boundaries[1:]))
    return pieces


def _overlap_tail(pieces: list[str], limit: int) -> list[str]:
    """Trailing pieces of a finished chunk that fit in `limit` characters."""
    tail, length = [], 0
    for piece in reversed(pieces):
        if length + len(piece) > limit:
            break
        tail.insert(0, piece)
        length += len(piece)
    return tail


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split `text` into chunks of at most `chunk_size` characters.

    Pieces are packed greedily, so short sections share a chunk. When a chunk fills up, the
    next one starts by repeating up to `chunk_overlap` characters of whole trailing pieces,
    so a sentence cut off from its lead-in is still retrievable with it.
    """
    chunks, current = [], []
    for piece in _split(text, chunk_size):
        if current and sum(map(len, current)) + len(piece) > chunk_size:
            chunks.append("".join(current))
            current = _overlap_tail(current, min(chunk_overlap, chunk_size - len(piece)))
        current.append(piece)
    if current:
        chunks.append("".join(current))
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk every document, carrying its metadata onto each chunk."""
    chunks = []
    for doc in documents:
        for index, text in enumerate(split_text(doc.text, chunk_size, chunk_overlap)):
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}::{index}",
                doc_id=doc.doc_id,
                text=text,
                metadata={
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "source": doc.source,
                    "chunk_index": index,
                    **doc.metadata,
                },
            ))
    return chunks
