"""
Data models for the health knowledge graph.
Defines entity types and relationship types with validation.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""
    DIET = "DIET"
    FOOD = "FOOD"
    NUTRIENT = "NUTRIENT"
    HEALTH_METRIC = "HEALTH_METRIC"
    RECIPE = "RECIPE"
    HEALTH_EFFECT = "HEALTH_EFFECT"
    MEAL = "MEAL"
    INGREDIENT = "INGREDIENT"
    CONDITION = "CONDITION"


class RelationshipType(str, Enum):
    """Types of relationships in the knowledge graph."""
    RESTRICTS = "RESTRICTS"
    ALLOWS = "ALLOWS"
    CONTAINS = "CONTAINS"
    AFFECTS = "AFFECTS"
    PART_OF = "PART_OF"
    CAUSES = "CAUSES"
    RECOMMENDED_FOR = "RECOMMENDED_FOR"
    CONFLICTS_WITH = "CONFLICTS_WITH"
    SIMILAR_TO = "SIMILAR_TO"
    PREPARED_BY = "PREPARED_BY"
    CONSUMED_IN = "CONSUMED_IN"


class Entity(BaseModel):
    """Base entity model."""
    type: EntityType
    name: str = Field(..., min_length=1, max_length=200)
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator('name')
    def normalize_name(cls, v):
        """Normalize entity name for consistent storage."""
        return v.strip().title()

    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties dictionary."""
        props = {
            'name': self.name,
            'created_at': (self.created_at or datetime.now()).isoformat(),
            'updated_at': (self.updated_at or datetime.now()).isoformat(),
            **self.properties
        }
        return props


class Relationship(BaseModel):
    """Relationship between entities."""
    from_entity: str = Field(..., description="Source entity name")
    relationship_type: RelationshipType
    to_entity: str = Field(..., description="Target entity name")
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None

    @field_validator('from_entity', 'to_entity')
    def normalize_entity_names(cls, v):
        """Normalize entity names."""
        return v.strip().title()

    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties dictionary."""
        props = {
            'created_at': (self.created_at or datetime.now()).isoformat(),
            **self.properties
        }
        return props


class DietEntity(Entity):
    """Diet-specific entity with common properties."""
    type: EntityType = EntityType.DIET
    description: Optional[str] = None
    goals: List[str] = Field(default_factory=list)  # weight loss, health, etc.

    def __init__(self, **data):
        super().__init__(**data)
        if self.description:
            self.properties['description'] = self.description
        if self.goals:
            self.properties['goals'] = self.goals


class FoodEntity(Entity):
    """Food-specific entity."""
    type: EntityType = EntityType.FOOD
    category: Optional[str] = None  # vegetable, meat, dairy, etc.
    calories: Optional[float] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.category:
            self.properties['category'] = self.category
        if self.calories:
            self.properties['calories'] = self.calories


class RecipeEntity(Entity):
    """Recipe-specific entity."""
    type: EntityType = EntityType.RECIPE
    servings: Optional[int] = None
    prep_time_minutes: Optional[int] = None
    instructions: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.servings:
            self.properties['servings'] = self.servings
        if self.prep_time_minutes:
            self.properties['prep_time_minutes'] = self.prep_time_minutes
        if self.instructions:
            self.properties['instructions'] = self.instructions


class HealthMetricEntity(Entity):
    """Health metric entity (blood sugar, weight, etc.)."""
    type: EntityType = EntityType.HEALTH_METRIC
    unit: Optional[str] = None
    normal_range_min: Optional[float] = None
    normal_range_max: Optional[float] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.unit:
            self.properties['unit'] = self.unit
        if self.normal_range_min:
            self.properties['normal_range_min'] = self.normal_range_min
        if self.normal_range_max:
            self.properties['normal_range_max'] = self.normal_range_max


class KnowledgeGraphQuery(BaseModel):
    """Query model for searching the knowledge graph."""
    entity_type: Optional[EntityType] = None
    entity_name: Optional[str] = None
    relationship_type: Optional[RelationshipType] = None
    properties_filter: Dict[str, Any] = Field(default_factory=dict)
    max_depth: int = Field(default=2, ge=1, le=5)


class GraphSearchResult(BaseModel):
    """Result from graph search."""
    entities: List[Entity]
    relationships: List[Relationship]
    query: KnowledgeGraphQuery
    result_count: int
    execution_time_ms: float