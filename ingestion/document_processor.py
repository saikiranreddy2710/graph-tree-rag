"""
Document Processor — Semantic chunking with metadata extraction.
Splits documents into overlapping chunks respecting sentence boundaries.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import tiktoken

from config.settings import settings


@dataclass
class ChunkMetadata:
    source: str
    chunk_index: int
    total_chunks: int
    section_header: Optional[str] = None
    document_type: str = "unknown"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    token_count: int = 0


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": {
                "source": self.metadata.source,
                "chunk_index": self.metadata.chunk_index,
                "total_chunks": self.metadata.total_chunks,
                "section_header": self.metadata.section_header,
                "document_type": self.metadata.document_type,
                "created_at": self.metadata.created_at,
                "token_count": self.metadata.token_count,
            },
        }


class DocumentProcessor:
    """Processes raw documents into semantically coherent, overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def _generate_chunk_id(self, source: str, index: int) -> str:
        raw = f"{source}::{index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _extract_section_headers(self, text: str) -> list[tuple[int, str]]:
        """Find markdown-style headers and their positions."""
        headers = []
        for match in re.finditer(r"^(#{1,4})\s+(.+)$", text, re.MULTILINE):
            headers.append((match.start(), match.group(2).strip()))
        return headers

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences, respecting abbreviations and decimals."""
        sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n")
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _detect_document_type(self, source: str, text: str) -> str:
        ext = Path(source).suffix.lower() if source else ""
        type_map = {
            ".md": "markdown",
            ".txt": "plaintext",
            ".pdf": "pdf",
            ".html": "html",
            ".json": "json",
            ".py": "code",
            ".js": "code",
            ".ts": "code",
        }
        return type_map.get(ext, "plaintext")

    def _find_current_section(self, position: int, headers: list[tuple[int, str]]) -> Optional[str]:
        """Find which section header a character position falls under."""
        current = None
        for hdr_pos, hdr_text in headers:
            if hdr_pos <= position:
                current = hdr_text
            else:
                break
        return current

    def process_text(
        self,
        text: str,
        source: str = "unknown",
    ) -> list[DocumentChunk]:
        """Split a single text into semantic chunks with metadata."""
        text = text.strip()
        if not text:
            return []

        headers = self._extract_section_headers(text)
        doc_type = self._detect_document_type(source, text)
        sentences = self._split_into_sentences(text)

        chunks: list[DocumentChunk] = []
        current_sentences: list[str] = []
        current_tokens = 0
        char_position = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # If single sentence exceeds chunk size, force-split it
            if sentence_tokens > self.chunk_size:
                if current_sentences:
                    chunks.append(
                        self._build_chunk(
                            current_sentences, source, len(chunks), headers, char_position, doc_type
                        )
                    )
                    current_sentences = []
                    current_tokens = 0

                # Force-split long sentence by tokens
                tokens = self._tokenizer.encode(sentence)
                for i in range(0, len(tokens), self.chunk_size):
                    sub_text = self._tokenizer.decode(tokens[i : i + self.chunk_size])
                    chunks.append(
                        self._build_chunk(
                            [sub_text], source, len(chunks), headers, char_position, doc_type
                        )
                    )
                char_position += len(sentence)
                continue

            if current_tokens + sentence_tokens > self.chunk_size:
                # Emit current chunk
                chunks.append(
                    self._build_chunk(
                        current_sentences, source, len(chunks), headers, char_position, doc_type
                    )
                )

                # Keep overlap sentences
                overlap_sentences: list[str] = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tok = self._count_tokens(s)
                    if overlap_tokens + s_tok > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tok

                current_sentences = overlap_sentences
                current_tokens = overlap_tokens

            current_sentences.append(sentence)
            current_tokens += sentence_tokens
            char_position += len(sentence) + 1  # +1 for space

        # Emit final chunk
        if current_sentences:
            chunks.append(
                self._build_chunk(
                    current_sentences, source, len(chunks), headers, char_position, doc_type
                )
            )

        # Backfill total_chunks
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)

        return chunks

    def _build_chunk(
        self,
        sentences: list[str],
        source: str,
        index: int,
        headers: list[tuple[int, str]],
        char_position: int,
        doc_type: str,
    ) -> DocumentChunk:
        text = " ".join(sentences)
        section = self._find_current_section(char_position, headers)
        token_count = self._count_tokens(text)
        chunk_id = self._generate_chunk_id(source, index)

        return DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            metadata=ChunkMetadata(
                source=source,
                chunk_index=index,
                total_chunks=0,  # Backfilled later
                section_header=section,
                document_type=doc_type,
                token_count=token_count,
            ),
        )

    def process_file(self, path: Path) -> list[DocumentChunk]:
        """Read and process a single file."""
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.process_text(text, source=str(path))

    def process_directory(self, dir_path: Path, glob: str = "**/*.*") -> list[DocumentChunk]:
        """Process all matching files in a directory."""
        all_chunks: list[DocumentChunk] = []
        supported = {".txt", ".md", ".html", ".json", ".py", ".js", ".ts", ".csv"}

        for path in sorted(dir_path.glob(glob)):
            if path.is_file() and path.suffix.lower() in supported:
                chunks = self.process_file(path)
                all_chunks.extend(chunks)

        return all_chunks
