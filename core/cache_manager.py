"""
Redis-based caching for API responses and conversation history.
Implements semantic similarity checking to avoid redundant API calls.
"""

import json
import hashlib
from typing import Optional, List, Dict, Any
from datetime import datetime
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer
import numpy as np

from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class CacheManager:
    """
    Intelligent cache manager with semantic similarity checking.
    Stores responses and checks if similar queries were recently asked.
    """

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.ttl = settings.cache_ttl_seconds
        self.similarity_threshold = settings.semantic_similarity_threshold

        # Load embedding model for semantic similarity
        # Using lightweight multilingual model
        logger.info("Loading sentence transformer model...")
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        logger.info("Sentence transformer loaded successfully")

    def _get_cache_key(self, key_type: str, identifier: str) -> str:
        """Generate Redis cache key."""
        return f"cache:{key_type}:{identifier}"

    def _get_embedding_key(self, text: str) -> str:
        """Generate key for storing query embeddings."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding:{text_hash}"

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute sentence embedding for semantic similarity."""
        return self.embedder.encode(text, convert_to_numpy=True)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    async def get_response(self, query: str, check_semantic: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get cached response for query.
        Optionally checks for semantically similar queries.

        Args:
            query: User query
            check_semantic: Whether to check for similar queries

        Returns:
            Cached response dict or None if not found
        """
        # Check exact match first
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_key = self._get_cache_key("response", query_hash)

        cached = await self.redis.get(cache_key)
        if cached:
            logger.info("Cache hit (exact match)", extra={'query_length': len(query)})
            return json.loads(cached)

        # Check semantic similarity if enabled
        if check_semantic:
            similar_response = await self._find_similar_query(query)
            if similar_response:
                logger.info(
                    "Cache hit (semantic match)",
                    extra={
                        'query_length': len(query),
                        'similarity': similar_response.get('similarity', 0)
                    }
                )
                return similar_response

        logger.debug("Cache miss", extra={'query': query[:100]})
        return None

    async def _find_similar_query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Find cached response for semantically similar query.

        Args:
            query: User query

        Returns:
            Cached response if similar query found, None otherwise
        """
        try:
            # Compute embedding for current query
            query_embedding = self._compute_embedding(query)

            # Get all embedding keys from Redis
            embedding_keys = await self.redis.keys("embedding:*")

            if not embedding_keys:
                return None

            best_similarity = 0.0
            best_match_key = None

            # Compare with stored embeddings
            for emb_key in embedding_keys[:100]:  # Limit to avoid slowdown
                stored_data = await self.redis.get(emb_key)
                if not stored_data:
                    continue

                stored = json.loads(stored_data)
                stored_embedding = np.array(stored['embedding'])

                similarity = self._cosine_similarity(query_embedding, stored_embedding)

                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match_key = stored['response_key']

            if best_match_key:
                response = await self.redis.get(best_match_key)
                if response:
                    result = json.loads(response)
                    result['similarity'] = round(best_similarity, 3)
                    logger.info(
                        f"Found similar query with {best_similarity:.2f} similarity",
                        extra={'similarity': best_similarity}
                    )
                    return result

            return None

        except Exception as e:
            logger.error(f"Error finding similar query: {e}")
            return None

    async def set_response(
            self,
            query: str,
            response: Dict[str, Any],
            ttl: Optional[int] = None
    ):
        """
        Cache response for query with semantic indexing.

        Args:
            query: User query
            response: Response data to cache
            ttl: Optional TTL override
        """
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_key = self._get_cache_key("response", query_hash)

        # Store response
        response_data = {
            **response,
            'cached_at': datetime.now().isoformat(),
            'query': query
        }

        await self.redis.setex(
            cache_key,
            ttl or self.ttl,
            json.dumps(response_data)
        )

        # Store embedding for semantic search
        try:
            query_embedding = self._compute_embedding(query)
            embedding_key = self._get_embedding_key(query)

            embedding_data = {
                'embedding': query_embedding.tolist(),
                'response_key': cache_key,
                'query': query,
                'created_at': datetime.now().isoformat()
            }

            await self.redis.setex(
                embedding_key,
                ttl or self.ttl,
                json.dumps(embedding_data)
            )

            logger.info(
                "Cached response with semantic indexing",
                extra={'query_length': len(query)}
            )
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")

    async def get_conversation_history(
            self,
            conversation_id: str,
            max_turns: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for given conversation ID.

        Args:
            conversation_id: Unique conversation identifier
            max_turns: Maximum number of turns to retrieve

        Returns:
            List of conversation turns (user/assistant messages)
        """
        history_key = self._get_cache_key("conversation", conversation_id)

        # Get history from Redis list (LRANGE gets from newest to oldest)
        history_data = await self.redis.lrange(history_key, 0, max_turns - 1)

        if not history_data:
            return []

        history = [json.loads(item) for item in history_data]
        # Reverse to get chronological order (oldest first)
        history.reverse()

        logger.debug(
            f"Retrieved {len(history)} conversation turns",
            extra={'conversation_id': conversation_id, 'turns': len(history)}
        )

        return history

    async def add_conversation_turn(
            self,
            conversation_id: str,
            role: str,
            content: str,
            metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a turn to conversation history.

        Args:
            conversation_id: Unique conversation identifier
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata (language, duration, etc.)
        """
        history_key = self._get_cache_key("conversation", conversation_id)

        turn_data = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            **(metadata or {})
        }

        # Add to front of list (newest first)
        await self.redis.lpush(history_key, json.dumps(turn_data))

        # Keep only last 50 turns
        await self.redis.ltrim(history_key, 0, 49)

        # Set expiry (7 days)
        await self.redis.expire(history_key, 7 * 24 * 3600)

        logger.debug(
            f"Added conversation turn: {role}",
            extra={'conversation_id': conversation_id, 'role': role}
        )

    async def clear_conversation(self, conversation_id: str):
        """Clear conversation history for given ID."""
        history_key = self._get_cache_key("conversation", conversation_id)
        await self.redis.delete(history_key)
        logger.info(f"Cleared conversation: {conversation_id}")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        response_keys = await self._count_keys("cache:response:*")
        embedding_keys = await self._count_keys("embedding:*")
        conversation_keys = await self._count_keys("cache:conversation:*")

        # Fetch all Redis info in one call
        info = await self.redis.info()

        memory_info = info.get('memory', {})
        clients_info = info.get('clients', {})

        return {
            'cached_responses': response_keys,
            'cached_embeddings': embedding_keys,
            'active_conversations': conversation_keys,
            'redis_info': {
                'used_memory': memory_info.get('used_memory_human', 'N/A'),
                'connected_clients': clients_info.get('connected_clients', 0)
            }
        }

    async def _count_keys(self, pattern: str) -> int:
        count = 0
        async for _ in self.redis.scan_iter(match=pattern):
            count += 1
        return count