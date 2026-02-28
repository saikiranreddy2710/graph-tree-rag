"""
Graph Retriever — Entity-centric graph traversal with community expansion.
Extracts entities from query, walks the KG, and pulls connected context
including pre-generated community summaries.
Based on: GraphRAG (Microsoft, arXiv:2404.16130)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import litellm
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from ingestion.document_processor import DocumentChunk
from ingestion.graph_builder import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class GraphRetrievalResult:
    chunk_id: Optional[str]
    text: str
    score: float
    source_type: str  # "graph_walk", "community_summary", "entity_context"
    entities_matched: list[str] = field(default_factory=list)
    hop_distance: int = 0
    metadata: dict = field(default_factory=dict)


class GraphRetriever:
    """Retrieves context by traversing the knowledge graph from query entities."""

    def __init__(
        self,
        kg: KnowledgeGraph,
        chunks: list[DocumentChunk],
        model: str = settings.fast_model,
    ):
        self.kg = kg
        self.chunk_map = {c.chunk_id: c for c in chunks}
        self.model = model

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _extract_query_entities(self, query: str) -> list[str]:
        """Extract entity names from the query using LLM."""
        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract all entity names (people, organizations, concepts, "
                        "technologies, events, locations) from the query. "
                        'Return a JSON object: {"entities": ["name1", "name2"]}'
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("entities", [])

    def _fuzzy_match_entity(self, query_entity: str) -> list[str]:
        """Find matching graph nodes by fuzzy string matching."""
        query_lower = query_entity.lower().strip()
        matches = []

        for node in self.kg.graph.nodes:
            node_lower = node.lower()
            # Exact match
            if node_lower == query_lower:
                matches.append((node, 1.0))
            # Substring match
            elif query_lower in node_lower or node_lower in query_lower:
                matches.append((node, 0.8))
            # Word overlap
            else:
                query_words = set(query_lower.split())
                node_words = set(node_lower.split())
                overlap = len(query_words & node_words)
                if overlap > 0:
                    jaccard = overlap / len(query_words | node_words)
                    if jaccard > 0.3:
                        matches.append((node, jaccard))

        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches[:3]]

    def _walk_from_entity(
        self, entity: str, max_hops: int = settings.graph_max_hops
    ) -> list[GraphRetrievalResult]:
        """BFS walk from an entity, collecting connected chunk context."""
        neighbors = self.kg.get_neighbors(entity, max_hops=max_hops)
        results = []

        for neighbor in neighbors:
            node_name = neighbor["node"]
            depth = neighbor["depth"]
            source_chunks = neighbor.get("source_chunks", [])

            # Retrieve the actual text from source chunks
            for cid in source_chunks:
                if cid in self.chunk_map:
                    chunk = self.chunk_map[cid]
                    # Score decays with hop distance
                    score = 1.0 / (1.0 + depth * 0.5)
                    results.append(
                        GraphRetrievalResult(
                            chunk_id=cid,
                            text=chunk.text,
                            score=score,
                            source_type="graph_walk",
                            entities_matched=[entity, node_name],
                            hop_distance=depth,
                            metadata={
                                "source": chunk.metadata.source,
                                "entity": node_name,
                                "entity_type": neighbor.get("entity_type", "unknown"),
                            },
                        )
                    )

        return results

    def _get_community_context(self, entity: str) -> list[GraphRetrievalResult]:
        """Get community summary for the entity's community."""
        results = []

        summary = self.kg.get_community_summary(entity)
        if summary:
            comm_id = self.kg.entity_to_community.get(entity)
            comm = self.kg.communities.get(comm_id) if comm_id is not None else None
            results.append(
                GraphRetrievalResult(
                    chunk_id=None,
                    text=summary,
                    score=0.85,
                    source_type="community_summary",
                    entities_matched=[entity],
                    metadata={
                        "community_id": comm_id,
                        "community_size": comm.size if comm else 0,
                    },
                )
            )

        return results

    def _get_entity_edge_context(self, entity: str) -> list[GraphRetrievalResult]:
        """Build textual context from entity's direct edge descriptions."""
        results = []
        node_data = self.kg.get_node(entity)

        if not node_data:
            return results

        # Build context from direct edges
        edge_descriptions = []
        for _, target, data in self.kg.graph.out_edges(entity, data=True):
            conf = data.get("confidence", 0.5)
            rel_type = data.get("relation_type", "RELATES_TO")
            desc = data.get("description", "")
            edge_descriptions.append(
                f"{entity} --[{rel_type}]--> {target}: {desc} (confidence: {conf:.2f})"
            )

        for source, _, data in self.kg.graph.in_edges(entity, data=True):
            conf = data.get("confidence", 0.5)
            rel_type = data.get("relation_type", "RELATES_TO")
            desc = data.get("description", "")
            edge_descriptions.append(
                f"{source} --[{rel_type}]--> {entity}: {desc} (confidence: {conf:.2f})"
            )

        if edge_descriptions:
            entity_desc = node_data.get("description", "")
            context_text = (
                f"Entity: {entity} ({node_data.get('entity_type', 'unknown')})\n"
                f"Description: {entity_desc}\n"
                f"Relationships:\n" + "\n".join(edge_descriptions)
            )
            results.append(
                GraphRetrievalResult(
                    chunk_id=None,
                    text=context_text,
                    score=0.75,
                    source_type="entity_context",
                    entities_matched=[entity],
                    metadata={"num_edges": len(edge_descriptions)},
                )
            )

        return results

    async def retrieve(
        self,
        query: str,
        top_k: int = settings.graph_top_k,
    ) -> list[GraphRetrievalResult]:
        """Full graph retrieval pipeline: extract entities -> match -> walk -> expand."""
        # Step 1: Extract entities from query
        query_entities = await self._extract_query_entities(query)
        logger.info(f"Graph retriever extracted entities: {query_entities}")

        if not query_entities:
            return []

        all_results: list[GraphRetrievalResult] = []

        for entity in query_entities:
            # Step 2: Fuzzy match to graph nodes
            matched_nodes = self._fuzzy_match_entity(entity)
            if not matched_nodes:
                continue

            for matched in matched_nodes:
                # Step 3: Walk from matched entity
                walk_results = self._walk_from_entity(matched)
                all_results.extend(walk_results)

                # Step 4: Get community summary
                community_results = self._get_community_context(matched)
                all_results.extend(community_results)

                # Step 5: Get entity edge context
                edge_results = self._get_entity_edge_context(matched)
                all_results.extend(edge_results)

        # Deduplicate by chunk_id and sort by score
        seen_chunks = set()
        deduped = []
        for r in sorted(all_results, key=lambda x: x.score, reverse=True):
            key = r.chunk_id or r.text[:100]
            if key not in seen_chunks:
                seen_chunks.add(key)
                deduped.append(r)

        return deduped[:top_k]
