"""
Neo4j database client for health knowledge graph operations.
Handles entity and relationship CRUD with complex queries.
"""

import time
from typing import List, Dict, Any, Optional, LiteralString, cast
from neo4j import GraphDatabase, Driver

from config.settings import settings
from database.models import Entity, Relationship, EntityType, RelationshipType
from database.schema import Neo4jSchema
from utils.logger import setup_logger

logger = setup_logger(__name__)


class Neo4jClient:
    """
    Neo4j client for health knowledge graph operations.
    Provides CRUD operations and complex graph queries.
    """

    def __init__(self):
        self.driver: Optional[Driver] = None
        self.connected = False

    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            logger.info(f"Connecting to Neo4j at {settings.neo4j_uri}...")

            self.driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                max_connection_lifetime=3600,
                keep_alive=True
            )

            # Test connection
            self.driver.verify_connectivity()
            self.connected = True

            logger.info("Connected to Neo4j successfully")

            # Initialize schema
            Neo4jSchema.initialize(self.driver)

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self) -> None:
        """Close connection to Neo4j."""
        if self.driver:
            self.driver.close()
            self.connected = False
            logger.info("Closed Neo4j connection")

    def create_entity(self, entity: Entity) -> Dict[str, Any]:
        """
        Create or update an entity in the knowledge graph.

        Args:
            entity: Entity to create

        Returns:
            Created/updated entity data
        """
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j")

        start_time = time.time()

        with self.driver.session() as session:
            properties = entity.to_neo4j_properties()

            # MERGE creates if not exists, updates if exists
            query = cast(LiteralString,f"""
            MERGE (n:{entity.type.value} {{name: $name}})
            SET n += $properties
            RETURN n
            """)

            result = session.run(query, name=entity.name, properties=properties)
            node = result.single()["n"]

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Created/updated entity: {entity.type.value}.{entity.name}",
                extra={
                    'entity_type': entity.type.value,
                    'entity_name': entity.name,
                    'duration_ms': duration_ms
                }
            )

            return dict(node)

    def create_relationship(self, relationship: Relationship) -> Dict[str, Any]:
        """
        Create a relationship between two entities.

        Args:
            relationship: Relationship to create

        Returns:
            Created relationship data
        """
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j")

        start_time = time.time()

        with self.driver.session() as session:
            properties = relationship.to_neo4j_properties()

            # Find or create both nodes, then create relationship
            query = cast(LiteralString,f"""
            MERGE (from {{name: $from_name}})
            MERGE (to {{name: $to_name}})
            MERGE (from)-[r:{relationship.relationship_type.value}]->(to)
            SET r += $properties
            RETURN r, from, to
            """)

            result = session.run(
                query,
                from_name=relationship.from_entity,
                to_name=relationship.to_entity,
                properties=properties
            )

            record = result.single()

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Created relationship: {relationship.from_entity} -{relationship.relationship_type.value}-> {relationship.to_entity}",
                extra={
                    'from': relationship.from_entity,
                    'type': relationship.relationship_type.value,
                    'to': relationship.to_entity,
                    'duration_ms': duration_ms
                }
            )

            return {
                'relationship': dict(record["r"]),
                'from_entity': dict(record["from"]),
                'to_entity': dict(record["to"])
            }

    def get_entity(self, entity_type: EntityType, name: str) -> Optional[Dict[str, Any]]:
        """
        Get entity by type and name.

        Args:
            entity_type: Type of entity
            name: Entity name

        Returns:
            Entity data or None if not found
        """
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j")

        with self.driver.session() as session:
            query = cast(LiteralString,f"""
            MATCH (n:{entity_type.value} {{name: $name}})
            RETURN n
            """)

            result = session.run(query, name=name)
            record = result.single()

            if record:
                return dict(record["n"])
            return None

    def get_entity_relationships(
            self,
            entity_name: str,
            direction: str = "both",
            relationship_types: Optional[List[RelationshipType]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all relationships for an entity.

        Args:
            entity_name: Name of entity
            direction: 'outgoing', 'incoming', or 'both'
            relationship_types: Optional filter by relationship types

        Returns:
            List of relationships with connected entities
        """
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j")

        with self.driver.session() as session:
            # Build relationship pattern based on direction
            if direction == "outgoing":
                pattern = "(entity)-[r]->(connected)"
            elif direction == "incoming":
                pattern = "(entity)<-[r]-(connected)"
            else:  # both
                pattern = "(entity)-[r]-(connected)"

            # Build relationship type filter
            if relationship_types:
                type_filter = "|".join([rt.value for rt in relationship_types])
                pattern = pattern.replace("[r]", f"[r:{type_filter}]")

            query = cast(LiteralString,f"""
            MATCH {pattern}
            WHERE entity.name = $entity_name
            RETURN type(r) as relationship_type, r, connected
            """)

            result = session.run(query, entity_name=entity_name)

            relationships = []
            for record in result:
                relationships.append({
                    'type': record["relationship_type"],
                    'properties': dict(record["r"]),
                    'connected_entity': dict(record["connected"])
                })

            logger.debug(
                f"Found {len(relationships)} relationships for {entity_name}",
                extra={'entity': entity_name, 'count': len(relationships)}
            )

            return relationships

    def search_entities(
            self,
            entity_types: Optional[List[EntityType]] = None,
            name_pattern: Optional[str] = None,
            properties_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search entities with flexible filters.

        Args:
            entity_types: Filter by entity types
            name_pattern: Search by name (case-insensitive, supports wildcards)
            properties_filter: Filter by properties

        Returns:
            List of matching entities
        """
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j")

        with self.driver.session() as session:
            # Build label filter
            if entity_types:
                labels = "|".join([et.value for et in entity_types])
                match_clause = f"MATCH (n:{labels})"
            else:
                match_clause = "MATCH (n)"

            # Build WHERE clauses
            where_clauses = []
            params = {}

            if name_pattern:
                where_clauses.append("toLower(n.name) CONTAINS toLower($name_pattern)")
                params['name_pattern'] = name_pattern

            if properties_filter:
                for key, value in properties_filter.items():
                    param_name = f"prop_{key}"
                    where_clauses.append(f"n.{key} = ${param_name}")
                    params[param_name] = value

            where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

            query = cast(LiteralString,f"""
            {match_clause}
            {where_clause}
            RETURN n, labels(n) as labels
            LIMIT 100
            """)

            result = session.run(query, **params)

            entities = []
            for record in result:
                entity_data = dict(record["n"])
                entity_data['labels'] = record["labels"]
                entities.append(entity_data)

            logger.info(
                f"Search found {len(entities)} entities",
                extra={'count': len(entities)}
            )

            return entities

    def get_diet_info(self, diet_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a diet.

        Args:
            diet_name: Name of diet

        Returns:
            Dictionary with diet info, restrictions, allowed foods, etc.
        """
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j")

        with self.driver.session() as session:
            query = cast(LiteralString,"""
            MATCH (diet:DIET {name: $diet_name})
            OPTIONAL MATCH (diet)-[:RESTRICTS]->(restricted:FOOD)
            OPTIONAL MATCH (diet)-[:ALLOWS]->(allowed:FOOD)
            OPTIONAL MATCH (diet)-[:RECOMMENDED_FOR]->(condition)
            RETURN diet,
                   collect(DISTINCT restricted.name) as restrictions,
                   collect(DISTINCT allowed.name) as allowed_foods,
                   collect(DISTINCT condition.name) as recommended_for
            """)

            result = session.run(query, diet_name=diet_name)
            record = result.single()

            if not record:
                return {}

            return {
                'diet': dict(record["diet"]),
                'restrictions': [r for r in record["restrictions"] if r],
                'allowed_foods': [a for a in record["allowed_foods"] if a],
                'recommended_for': [c for c in record["recommended_for"] if c]
            }

    def find_path(
            self,
            from_entity: str,
            to_entity: str,
            max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find the shortest path between two entities.

        Args:
            from_entity: Source entity name
            to_entity: Target entity name
            max_depth: Maximum path length

        Returns:
            List of paths with nodes and relationships
        """
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j")

        with self.driver.session() as session:
            query = cast(LiteralString,"""
            MATCH path = shortestPath(
                (from {name: $from_name})-[*..%d]-(to {name: $to_name})
            )
            RETURN path
            LIMIT 5
            """ % max_depth)

            result = session.run(query, from_name=from_entity, to_name=to_entity)

            paths = []
            for record in result:
                path = record["path"]
                paths.append({
                    'nodes': [dict(node) for node in path.nodes],
                    'relationships': [
                        {'type': rel.type, 'properties': dict(rel)}
                        for rel in path.relationships
                    ],
                    'length': len(path.relationships)
                })

            return paths

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j")

        return Neo4jSchema.get_schema_info(self.driver)

    def __enter__(self):
        """Context manager entry."""
        if not self.connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()