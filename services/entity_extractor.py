"""
Service for extracting and storing health entities from user input.
Coordinates between Gemini (extraction) and Neo4j (storage).
"""

from typing import Dict, Any, List
from core.llm_client import GeminiClient
from core.language_detector import LanguageCode
from database.neo4j_client import Neo4jClient
from database.models import Entity, Relationship, EntityType, RelationshipType
from utils.logger import setup_logger

logger = setup_logger(__name__)


class EntityExtractorService:
    """
    Extracts health entities from user text and stores in knowledge graph.
    """

    def __init__(self, gemini_client: GeminiClient, neo4j_client: Neo4jClient):
        self.gemini = gemini_client
        self.neo4j = neo4j_client

    async def extract_and_store(
        self,
        text: str,
        language: LanguageCode
    ) -> Dict[str, Any]:
        """
        Extract entities from text and store in Neo4j.

        Args:
            text: User text containing health information
            language: Text language

        Returns:
            Dictionary with extraction and storage results
        """
        logger.info(f"Extracting entities from text: {text[:100]}...")

        # Extract entities using Gemini
        extraction_result = await self.gemini.extract_entities(text, language)

        entities_data = extraction_result.get('entities', [])
        relationships_data = extraction_result.get('relationships', [])

        if not entities_data and not relationships_data:
            logger.warning("No entities or relationships extracted")
            return {
                'success': False,
                'message': 'No information could be extracted from the text',
                'entities_created': 0,
                'relationships_created': 0
            }

        # Store entities
        created_entities = []
        for entity_data in entities_data:
            try:
                entity = self._create_entity_from_data(entity_data)
                stored = self.neo4j.create_entity(entity)
                created_entities.append({
                    'type': entity.type.value,
                    'name': entity.name
                })
                logger.debug(f"Stored entity: {entity.type.value}.{entity.name}")
            except Exception as e:
                logger.error(f"Failed to store entity {entity_data}: {e}")

        # Store relationships
        created_relationships = []
        for rel_data in relationships_data:
            try:
                relationship = self._create_relationship_from_data(rel_data)
                stored = self.neo4j.create_relationship(relationship)
                created_relationships.append({
                    'from': relationship.from_entity,
                    'type': relationship.relationship_type.value,
                    'to': relationship.to_entity
                })
                logger.debug(
                    f"Stored relationship: {relationship.from_entity} "
                    f"-{relationship.relationship_type.value}-> {relationship.to_entity}"
                )
            except Exception as e:
                logger.error(f"Failed to store relationship {rel_data}: {e}")

        # Generate summary
        summary = self._generate_summary(created_entities, created_relationships, language)

        logger.info(
            f"Extraction complete: {len(created_entities)} entities, "
            f"{len(created_relationships)} relationships"
        )

        return {
            'success': True,
            'entities_created': len(created_entities),
            'relationships_created': len(created_relationships),
            'entities': created_entities,
            'relationships': created_relationships,
            'summary': summary,
            'language': language
        }

    def _create_entity_from_data(self, entity_data: Dict[str, Any]) -> Entity:
        """Create Entity model from extracted data."""
        entity_type = EntityType(entity_data['type'])
        return Entity(
            type=entity_type,
            name=entity_data['name'],
            properties=entity_data.get('properties', {})
        )

    def _create_relationship_from_data(self, rel_data: Dict[str, Any]) -> Relationship:
        """Create Relationship model from extracted data."""
        rel_type = RelationshipType(rel_data['type'])
        return Relationship(
            from_entity=rel_data['from'],
            relationship_type=rel_type,
            to_entity=rel_data['to'],
            properties=rel_data.get('properties', {})
        )

    def _generate_summary(
        self,
        entities: List[Dict[str, str]],
        relationships: List[Dict[str, str]],
        language: LanguageCode
    ) -> str:
        """Generate human-readable summary of what was stored."""
        if language == "ru":
            entity_summary = ", ".join([f"{e['name']} ({e['type']})" for e in entities[:3]])
            if len(entities) > 3:
                entity_summary += f" и ещё {len(entities) - 3}"

            if relationships:
                rel_examples = [
                    f"{r['from']} {r['type'].lower()} {r['to']}"
                    for r in relationships[:2]
                ]
                rel_summary = ", ".join(rel_examples)
                if len(relationships) > 2:
                    rel_summary += f" и ещё {len(relationships) - 2} связи"
                return f"Сохранил: {entity_summary}. Связи: {rel_summary}."
            else:
                return f"Сохранил: {entity_summary}."
        else:
            entity_summary = ", ".join([f"{e['name']} ({e['type']})" for e in entities[:3]])
            if len(entities) > 3:
                entity_summary += f" and {len(entities) - 3} more"

            if relationships:
                rel_examples = [
                    f"{r['from']} {r['type'].lower()} {r['to']}"
                    for r in relationships[:2]
                ]
                rel_summary = ", ".join(rel_examples)
                if len(relationships) > 2:
                    rel_summary += f" and {len(relationships) - 2} more"
                return f"Stored: {entity_summary}. Relationships: {rel_summary}."
            else:
                return f"Stored: {entity_summary}."