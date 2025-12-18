"""
Main orchestration service for the voice health assistant.
Coordinates voice input (Groq Whisper), Gemini chat, entity extraction, and storage.
"""

import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from core.voice_input import VoiceInput
from core.llm_client import GeminiClient
from core.language_detector import language_detector, LanguageCode
from core.cache_manager import CacheManager
from database.neo4j_client import Neo4jClient
from services.entity_extractor import EntityExtractorService
from utils.logger import setup_logger, log_conversation_turn
from config.prompts import CONFIRMATION_MESSAGES

logger = setup_logger(__name__)


class HealthAssistantService:
    """
    Main service orchestrating the health assistant functionality.
    """

    def __init__(
        self,
        voice_input: VoiceInput,
        gemini_client: GeminiClient,
        neo4j_client: Neo4jClient,
        cache_manager: CacheManager
    ):
        self.voice = voice_input
        self.gemini = gemini_client
        self.neo4j = neo4j_client
        self.cache = cache_manager
        self.entity_extractor = EntityExtractorService(gemini_client, neo4j_client)

    async def process_voice_input(
        self,
        audio_path: Path,
        conversation_id: Optional[str] = None,
        language_hint: Optional[LanguageCode] = None
    ) -> Dict[str, Any]:
        """
        Process voice input end-to-end: transcription → extraction → storage → response.

        Args:
            audio_path: Path to audio file
            conversation_id: Optional conversation ID (generates new if not provided)
            language_hint: Optional language hint for transcription

        Returns:
            Complete processing result with transcription, extraction, and response
        """
        start_time = time.time()
        conversation_id = conversation_id or str(uuid.uuid4())

        logger.info(
            f"Processing voice input: {audio_path.name}",
            extra={'conversation_id': conversation_id}
        )

        try:
            # Step 1: Transcribe audio
            transcription = await self.voice.transcribe(
                audio_path,
                language=language_hint
            )

            user_text = transcription['text']
            detected_language = transcription['language']

            logger.info(
                f"Transcribed: '{user_text}' (language={detected_language})",
                extra={
                    'text_length': len(user_text),
                    'language': detected_language,
                    'conversation_id': conversation_id
                }
            )

            # Step 2: Extract and store entities
            extraction_result = await self.entity_extractor.extract_and_store(
                user_text,
                detected_language
            )

            # Step 3: Generate conversational response
            # Prepare context about what was stored
            storage_context = ""
            if extraction_result['success'] and extraction_result['entities_created'] > 0:
                storage_context = f"\n\n[System: Stored {extraction_result['entities_created']} entities and {extraction_result['relationships_created']} relationships in knowledge graph]"

            # Get conversational response from Gemini
            chat_response = await self.gemini.chat(
                message=user_text + storage_context,
                language=detected_language,
                conversation_id=conversation_id,
                use_web_search=True,  # Enable web search for opinions/research
                use_cache=True
            )

            assistant_response = chat_response['response']

            # Step 4: Generate friendly confirmation if entities were stored
            if extraction_result['success'] and extraction_result['entities_created'] > 0:
                topics = ", ".join([e['name'] for e in extraction_result['entities'][:3]])
                confirmation = CONFIRMATION_MESSAGES[detected_language].format(
                    entity_count=extraction_result['entities_created'],
                    topics=topics,
                    summary=extraction_result['summary']
                )

                # Prepend confirmation to response
                assistant_response = f"{confirmation}\n\n{assistant_response}"

            duration_ms = (time.time() - start_time) * 1000

            # Log complete turn
            await log_conversation_turn(
                logger,
                conversation_id=conversation_id,
                turn_number=len(await self.cache.get_conversation_history(conversation_id)),
                user_message=user_text,
                assistant_message=assistant_response,
                language=detected_language,
                duration_ms=duration_ms
            )

            result = {
                'success': True,
                'conversation_id': conversation_id,
                'transcription': {
                    'text': user_text,
                    'language': detected_language,
                    'confidence': transcription['language_confidence'],
                    'duration_seconds': transcription['duration_seconds'],
                    'cost_usd': transcription['cost_usd']
                },
                'extraction': extraction_result,
                'response': {
                    'text': assistant_response,
                    'language': detected_language,
                    'web_search_used': chat_response.get('web_search_used', False),
                    'cached': chat_response.get('cached', False)
                },
                'processing_time_ms': round(duration_ms, 2)
            }

            logger.info(
                f"Voice input processed successfully in {duration_ms:.0f}ms",
                extra={
                    'conversation_id': conversation_id,
                    'duration_ms': duration_ms,
                    'entities_stored': extraction_result['entities_created']
                }
            )

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Error processing voice input: {e}",
                extra={'conversation_id': conversation_id, 'error': str(e)}
            )

            return {
                'success': False,
                'conversation_id': conversation_id,
                'error': str(e),
                'processing_time_ms': round(duration_ms, 2)
            }

    async def process_text_input(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        language: Optional[LanguageCode] = None,
        extract_entities: bool = True
    ) -> Dict[str, Any]:
        """
        Process text input (alternative to voice).

        Args:
            text: User text input
            conversation_id: Optional conversation ID
            language: Optional language (auto-detects if not provided)
            extract_entities: Whether to extract and store entities

        Returns:
            Processing result
        """
        start_time = time.time()
        conversation_id = conversation_id or str(uuid.uuid4())

        # Detect language if not provided
        if not language:
            language, confidence = language_detector.detect(text)
            logger.info(f"Detected language: {language} (confidence: {confidence:.2f})")

        logger.info(
            f"Processing text input: '{text[:100]}...'",
            extra={'conversation_id': conversation_id, 'language': language}
        )

        try:
            extraction_result = None

            # Extract entities if requested
            if extract_entities:
                extraction_result = await self.entity_extractor.extract_and_store(
                    text,
                    language
                )

            # Get conversational response
            storage_context = ""
            if extraction_result and extraction_result['success']:
                storage_context = f"\n\n[System: Stored {extraction_result['entities_created']} entities in knowledge graph]"

            chat_response = await self.gemini.chat(
                message=text + storage_context,
                language=language,
                conversation_id=conversation_id,
                use_web_search=True,
                use_cache=True
            )

            assistant_response = chat_response['response']

            # Add confirmation if entities were stored
            if (
                    isinstance(extraction_result, dict)
                    and extraction_result.get("success")
                    and isinstance(extraction_result.get("entities_created"), int)
                    and extraction_result["entities_created"] > 0
            ):
                entities = extraction_result.get("entities", [])

                safe_names = []
                if isinstance(entities, list):
                    for e in entities:
                        if isinstance(e, dict) and "name" in e:
                            safe_names.append(e["name"])

                topics = ", ".join(safe_names[:3])
                confirmation = CONFIRMATION_MESSAGES[language].format(
                    entity_count=extraction_result['entities_created'],
                    topics=topics,
                    summary=extraction_result['summary']
                )
                assistant_response = f"{confirmation}\n\n{assistant_response}"

            duration_ms = (time.time() - start_time) * 1000

            return {
                'success': True,
                'conversation_id': conversation_id,
                'input': {
                    'text': text,
                    'language': language
                },
                'extraction': extraction_result,
                'response': {
                    'text': assistant_response,
                    'language': language,
                    'web_search_used': chat_response.get('web_search_used', False),
                    'cached': chat_response.get('cached', False)
                },
                'processing_time_ms': round(duration_ms, 2)
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Error processing text input: {e}")

            return {
                'success': False,
                'conversation_id': conversation_id,
                'error': str(e),
                'processing_time_ms': round(duration_ms, 2)
            }

    async def query_knowledge(
        self,
        query: str,
        conversation_id: str,
        language: Optional[LanguageCode] = None
    ) -> Dict[str, Any]:
        """
        Query the knowledge graph conversationally.

        Args:
            query: User query
            conversation_id: Conversation ID
            language: Optional language

        Returns:
            Query result with assistant response
        """
        if not language:
            language, _ = language_detector.detect(query)

        # Gemini will use Neo4j data through conversation context
        # In Phase 3, we'll add MCP server for direct Neo4j queries

        response = await self.gemini.chat(
            message=query,
            language=language,
            conversation_id=conversation_id,
            use_web_search=True,
            use_cache=True
        )

        return response

    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            'database': self.neo4j.get_statistics(),
            'cache': await self.cache.get_cache_stats(),
            'whisper': self.voice.get_usage_stats(),
            'gemini': self.gemini.get_usage_stats()
        }