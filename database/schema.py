"""
Neo4j schema definition and initialization.
Creates indexes and constraints for the health knowledge graph.
"""

from typing import cast, LiteralString
from neo4j import Driver
from utils.logger import setup_logger

logger = setup_logger(__name__)


class Neo4jSchema:
    """
    Manages Neo4j schema: indexes, constraints, and initial setup.
    """

    # Indexes for fast lookups
    INDEXES = [
        ("DIET", "name"),
        ("FOOD", "name"),
        ("NUTRIENT", "name"),
        ("HEALTH_METRIC", "name"),
        ("RECIPE", "name"),
        ("HEALTH_EFFECT", "name"),
        ("MEAL", "name"),
        ("INGREDIENT", "name"),
        ("CONDITION", "name"),
    ]

    # Unique constraints
    CONSTRAINTS = [
        ("DIET", "name"),
        ("FOOD", "name"),
        ("NUTRIENT", "name"),
        ("HEALTH_METRIC", "name"),
        ("RECIPE", "name"),
        ("HEALTH_EFFECT", "name"),
    ]

    @staticmethod
    def initialize(driver: Driver) -> None:
        """
        Initialize database schema with indexes and constraints.

        Args:
            driver: Neo4j driver instance
        """
        logger.info("Initializing Neo4j schema...")

        with driver.session() as session:
            # Create unique constraints (also creates indexes)
            for label, property_name in Neo4jSchema.CONSTRAINTS:
                constraint_name = f"unique_{label.lower()}_{property_name}"
                try:
                    session.run(cast(LiteralString,f"""
                        CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                        FOR (n:{label})
                        REQUIRE n.{property_name} IS UNIQUE
                    """))
                    logger.info(f"Created constraint: {constraint_name}")
                except Exception as e:
                    logger.warning(f"Constraint {constraint_name} already exists or failed: {e}")

            # Create additional indexes (for labels not in constraints)
            for label, property_name in Neo4jSchema.INDEXES:
                if (label, property_name) not in Neo4jSchema.CONSTRAINTS:
                    index_name = f"idx_{label.lower()}_{property_name}"
                    try:
                        session.run(cast(LiteralString,f"""
                            CREATE INDEX {index_name} IF NOT EXISTS
                            FOR (n:{label})
                            ON (n.{property_name})
                        """))
                        logger.info(f"Created index: {index_name}")
                    except Exception as e:
                        logger.warning(f"Index {index_name} already exists or failed: {e}")

            # Create full-text search indexes for descriptions
            try:
                session.run("""
                    CREATE FULLTEXT INDEX entity_descriptions IF NOT EXISTS
                    FOR (n:DIET|FOOD|RECIPE|HEALTH_EFFECT)
                    ON EACH [n.description, n.instructions, n.name]
                """)
                logger.info("Created full-text search index")
            except Exception as e:
                logger.warning(f"Full-text index already exists or failed: {e}")

        logger.info("Neo4j schema initialization complete")

    @staticmethod
    def get_schema_info(driver: Driver) -> dict:
        """
        Get current schema information from database.

        Args:
            driver: Neo4j driver instance

        Returns:
            Dictionary with schema information
        """
        with driver.session() as session:
            # Get constraints
            constraints_result = session.run("SHOW CONSTRAINTS")
            constraints = [record.data() for record in constraints_result]

            # Get indexes
            indexes_result = session.run("SHOW INDEXES")
            indexes = [record.data() for record in indexes_result]

            # Get node labels
            labels_result = session.run("CALL db.labels()")
            labels = [record["label"] for record in labels_result]

            # Get relationship types
            rels_result = session.run("CALL db.relationshipTypes()")
            relationship_types = [record["relationshipType"] for record in rels_result]

            # Get node counts
            node_counts = {}
            for label in labels:
                count_result = session.run(cast(LiteralString,f"MATCH (n:{label}) RETURN count(n) as count"))
                node_counts[label] = count_result.single()["count"]

            # Get relationship counts
            rel_counts = {}
            for rel_type in relationship_types:
                count_result = session.run(cast(LiteralString,f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"))
                rel_counts[rel_type] = count_result.single()["count"]

        return {
            'constraints': len(constraints),
            'indexes': len(indexes),
            'node_labels': labels,
            'relationship_types': relationship_types,
            'node_counts': node_counts,
            'relationship_counts': rel_counts,
            'total_nodes': sum(node_counts.values()),
            'total_relationships': sum(rel_counts.values())
        }

    @staticmethod
    def clear_all_data(driver: Driver) -> None:
        """
        Clear all data from database (keeps schema).
        USE WITH CAUTION!

        Args:
            driver: Neo4j driver instance
        """
        logger.warning("Clearing all data from Neo4j database...")

        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

        logger.warning("All data cleared from database")