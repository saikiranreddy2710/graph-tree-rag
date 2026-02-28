"""Tests for document processor."""

import pytest
from ingestion.document_processor import DocumentProcessor


class TestDocumentProcessor:
    def setup_method(self):
        self.processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)

    def test_empty_text(self):
        chunks = self.processor.process_text("")
        assert chunks == []

    def test_single_sentence(self):
        chunks = self.processor.process_text("This is a short sentence.", source="test.txt")
        assert len(chunks) >= 1
        assert chunks[0].text.strip() != ""
        assert chunks[0].metadata.source == "test.txt"

    def test_multiple_chunks(self):
        text = " ".join(["This is sentence number {}.".format(i) for i in range(100)])
        chunks = self.processor.process_text(text, source="test.md")
        assert len(chunks) > 1

    def test_chunk_ids_unique(self):
        text = " ".join(["Sentence {}.".format(i) for i in range(50)])
        chunks = self.processor.process_text(text, source="test.txt")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_metadata_populated(self):
        chunks = self.processor.process_text("# Header\n\nSome content here.", source="doc.md")
        assert len(chunks) >= 1
        assert chunks[0].metadata.document_type == "markdown"

    def test_section_header_detection(self):
        text = "# Introduction\n\nFirst paragraph.\n\n# Methods\n\nSecond paragraph."
        chunks = self.processor.process_text(text, source="paper.md")
        assert len(chunks) >= 1
