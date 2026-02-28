"""
Entity Extractor — LLM-based entity and relationship extraction for KG construction.
Uses the fast model (gpt-4o-mini) for cost-efficient extraction with confidence scores.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from ingestion.document_processor import DocumentChunk

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    Path(__file__).parent.parent / "config" / "prompts" / "entity_extraction.txt"
).read_text()


@dataclass
class Entity:
    name: str
    entity_type: str
    description: str
    source_chunks: list[str]  # chunk_ids where this entity was found

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.entity_type,
            "description": self.description,
            "source_chunks": self.source_chunks,
        }


@dataclass
class Relationship:
    source: str
    target: str
    relation_type: str
    description: str
    confidence: float
    source_chunks: list[str]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relation_type,
            "description": self.description,
            "confidence": self.confidence,
            "source_chunks": self.source_chunks,
        }


class EntityExtractor:
    """Extracts entities and relationships from document chunks using LLM."""

    def __init__(
        self,
        model: str = settings.fast_model,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def _call_llm(self, text: str) -> dict:
        """Call LLM for entity extraction with retry logic."""
        prompt = PROMPT_TEMPLATE.replace("{text}", text)

        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledge graph extraction engine. Return ONLY valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        return json.loads(content)

    async def extract_from_chunk(
        self, chunk: DocumentChunk
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract entities and relationships from a single chunk."""
        try:
            result = await self._call_llm(chunk.text)
        except Exception as e:
            logger.error(f"Entity extraction failed for chunk {chunk.chunk_id}: {e}")
            return [], []

        entities = []
        for ent in result.get("entities", []):
            entities.append(
                Entity(
                    name=self._normalize_name(ent.get("name", "")),
                    entity_type=ent.get("type", "CONCEPT"),
                    description=ent.get("description", ""),
                    source_chunks=[chunk.chunk_id],
                )
            )

        relationships = []
        for rel in result.get("relationships", []):
            confidence = float(rel.get("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))
            relationships.append(
                Relationship(
                    source=self._normalize_name(rel.get("source", "")),
                    target=self._normalize_name(rel.get("target", "")),
                    relation_type=rel.get("type", "RELATES_TO"),
                    description=rel.get("description", ""),
                    confidence=confidence,
                    source_chunks=[chunk.chunk_id],
                )
            )

        return entities, relationships

    async def extract_from_chunks(
        self,
        chunks: list[DocumentChunk],
        batch_size: int = 5,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract from multiple chunks, deduplicating entities across chunks."""
        import asyncio

        all_entities: list[Entity] = []
        all_relationships: list[Relationship] = []

        # Process in batches to respect rate limits
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            tasks = [self.extract_from_chunk(chunk) for chunk in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch extraction error: {result}")
                    continue
                ents, rels = result
                all_entities.extend(ents)
                all_relationships.extend(rels)

        # Deduplicate entities by canonical name
        entities = self._deduplicate_entities(all_entities)
        relationships = self._deduplicate_relationships(all_relationships)

        logger.info(
            f"Extracted {len(entities)} unique entities and "
            f"{len(relationships)} unique relationships from {len(chunks)} chunks"
        )
        return entities, relationships

    def _normalize_name(self, name: str) -> str:
        """Normalize entity names to canonical form."""
        name = name.strip()
        # Title case for proper nouns, but keep acronyms uppercase
        if name.isupper() and len(name) > 4:
            name = name.title()
        return name

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """Merge entities with the same canonical name."""
        merged: dict[str, Entity] = {}
        for ent in entities:
            key = ent.name.lower()
            if key in merged:
                # Merge source chunks
                existing = merged[key]
                existing.source_chunks = list(set(existing.source_chunks + ent.source_chunks))
                # Keep longer description
                if len(ent.description) > len(existing.description):
                    existing.description = ent.description
            else:
                merged[key] = Entity(
                    name=ent.name,
                    entity_type=ent.entity_type,
                    description=ent.description,
                    source_chunks=list(ent.source_chunks),
                )
        return list(merged.values())

    def _deduplicate_relationships(self, relationships: list[Relationship]) -> list[Relationship]:
        """Merge duplicate relationships, keeping highest confidence."""
        merged: dict[str, Relationship] = {}
        for rel in relationships:
            key = f"{rel.source.lower()}::{rel.relation_type}::{rel.target.lower()}"
            if key in merged:
                existing = merged[key]
                existing.confidence = max(existing.confidence, rel.confidence)
                existing.source_chunks = list(set(existing.source_chunks + rel.source_chunks))
            else:
                merged[key] = Relationship(
                    source=rel.source,
                    target=rel.target,
                    relation_type=rel.relation_type,
                    description=rel.description,
                    confidence=rel.confidence,
                    source_chunks=list(rel.source_chunks),
                )
        return list(merged.values())
