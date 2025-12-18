"""
Gemini API client with rate limiting, conversation history, and web search.
Implements conversational AI with personality and multi-language support.
"""

import time
import json
from typing import Dict, Any
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from config.settings import settings
from config.prompts import SYSTEM_PROMPTS, ENTITY_EXTRACTION_PROMPTS
from core.language_detector import LanguageCode
from core.cache_manager import CacheManager
from utils.logger import setup_logger, log_api_call
from utils.rate_limiter import GeminiRateLimiter

logger = setup_logger(__name__)


class GeminiClient:
    """
    Gemini API client with conversational capabilities and web search.
    Handles both regular chat and structured entity extraction.
    """

    def __init__(
        self,
        rate_limiter: GeminiRateLimiter,
        cache_manager: CacheManager
    ):
        # Configure Gemini
        genai.configure(api_key=settings.google_api_key)

        self.rate_limiter = rate_limiter
        self.cache = cache_manager

        # Model configuration
        self.model_name = "gemini-2.5-flash"

        # Generation config
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,
        }

        # Safety settings (allow most content for health discussions)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        logger.info(f"Initialized Gemini client with model: {self.model_name}")

    async def chat(
        self,
        message: str,
        language: LanguageCode,
        conversation_id: str,
        use_web_search: bool = False,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Send conversational message to Gemini with full context.

        Args:
            message: User message
            language: Message language ('en' or 'ru')
            conversation_id: Unique conversation identifier
            use_web_search: Whether Gemini can use web search (Google Search grounding)
            use_cache: Whether to check cache for similar queries

        Returns:
            Dictionary with response and metadata
        """
        # Check cache first
        if use_cache:
            cached = await self.cache.get_response(message, check_semantic=True)
            if cached:
                # Add to conversation history even if cached
                await self.cache.add_conversation_turn(
                    conversation_id, "user", message, {'language': language}
                )
                await self.cache.add_conversation_turn(
                    conversation_id, "assistant", cached['response'],
                    {'language': language, 'cached': True}
                )

                return {
                    'response': cached['response'],
                    'language': language,
                    'cached': True,
                    'conversation_id': conversation_id,
                    'similarity': cached.get('similarity')
                }

        # Check rate limit
        await self.rate_limiter.wait_if_needed()
        if not await self.rate_limiter.check():
            raise Exception("Gemini rate limit exceeded. Please wait a minute.")

        # Get conversation history
        history = await self.cache.get_conversation_history(conversation_id, max_turns=10)

        # System prompt in user's language
        system_prompt = SYSTEM_PROMPTS[language]

        # Build conversation for Gemini
        gemini_history = []
        for turn in history:
            role = "user" if turn['role'] == "user" else "model"
            gemini_history.append({
                "role": role,
                "parts": [turn['content']]
            })

        # Prepare API call
        start_time = time.time()

        try:
            logger.info(f"Sending chat message to Gemini (language={language}, web_search={use_web_search})")

            # Initialize model with or without tools
            tools = None
            if use_web_search:
                tools = ['google_search_retrieval']

            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.GenerationConfig(**self.generation_config),
                safety_settings=self.safety_settings,
                system_instruction=system_prompt,
                tools=tools
            )

            # Start chat with history
            chat = model.start_chat(history=gemini_history)

            # Send message
            response = chat.send_message(message)

            duration_ms = (time.time() - start_time) * 1000

            # Extract response text
            response_text = response.text

            # Check if web search was used (if grounding metadata exists)
            used_web_search = use_web_search and hasattr(response, 'grounding_metadata')

            # Get token counts if available
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage_metadata'):
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count

            log_api_call(
                logger,
                service="google",
                endpoint="gemini",
                duration_ms=duration_ms,
                status="success",
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                web_search_used=used_web_search
            )

            # Store in conversation history
            await self.cache.add_conversation_turn(
                conversation_id, "user", message, {'language': language}
            )
            await self.cache.add_conversation_turn(
                conversation_id, "assistant", response_text,
                {
                    'language': language,
                    'web_search_used': used_web_search,
                    'tokens': {
                        'input': input_tokens,
                        'output': output_tokens
                    }
                }
            )

            # Cache response
            await self.cache.set_response(
                message,
                {
                    'response': response_text,
                    'language': language,
                    'web_search_used': used_web_search
                }
            )

            logger.info(
                f"Chat response received: {len(response_text)} chars",
                extra={
                    'response_length': len(response_text),
                    'web_search_used': used_web_search,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens
                }
            )

            return {
                'response': response_text,
                'language': language,
                'conversation_id': conversation_id,
                'web_search_used': used_web_search,
                'cached': False,
                'tokens': {
                    'input': input_tokens,
                    'output': output_tokens
                }
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_api_call(
                logger,
                service="google",
                endpoint="gemini",
                duration_ms=duration_ms,
                status="error",
                error=str(e)
            )
            logger.error(f"Gemini API error: {e}")
            raise

    async def extract_entities(
        self,
        text: str,
        language: LanguageCode
    ) -> Dict[str, Any]:
        """
        Extract structured health entities from text using Gemini.

        Args:
            text: User text to analyze
            language: Text language

        Returns:
            Dictionary with extracted entities and relationships
        """
        # Check rate limit
        await self.rate_limiter.wait_if_needed()
        if not await self.rate_limiter.check():
            raise Exception("Gemini rate limit exceeded. Please wait a minute.")

        # Get extraction prompt in user's language
        extraction_prompt = ENTITY_EXTRACTION_PROMPTS[language].format(user_message=text)

        start_time = time.time()

        try:
            logger.info(f"Extracting entities from text (language={language})")

            # Initialize model for structured output
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,  # Lower temperature for structured output
                    top_p=0.95,
                    max_output_tokens=2048,
                ),
                safety_settings=self.safety_settings
            )

            response = model.generate_content(extraction_prompt)

            duration_ms = (time.time() - start_time) * 1000

            # Extract JSON from response
            response_text = response.text
            extracted_data = self._parse_json_response(response_text)

            # Get token counts
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage_metadata'):
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count

            log_api_call(
                logger,
                service="google",
                endpoint="gemini_extract",
                duration_ms=duration_ms,
                status="success",
                entities_found=len(extracted_data.get('entities', [])),
                relationships_found=len(extracted_data.get('relationships', [])),
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

            logger.info(
                f"Extracted {len(extracted_data.get('entities', []))} entities, "
                f"{len(extracted_data.get('relationships', []))} relationships"
            )

            return extracted_data

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_api_call(
                logger,
                service="google",
                endpoint="gemini_extract",
                duration_ms=duration_ms,
                status="error",
                error=str(e)
            )
            logger.error(f"Entity extraction failed: {e}")
            raise

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from Gemini response, handling Markdown code blocks and extra text."""
        # Remove Markdown code blocks if present
        text = text.strip()

        # Remove ```json and ``` markers
        if "```json" in text:
            text = text.split("```json")[1]
        if "```" in text:
            text = text.split("```")[0]

        text = text.strip()

        # Try to find JSON object in the text
        # Look for the first { and last }
        start_idx = text.find('{')
        end_idx = text.rfind('}')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = text[start_idx:end_idx + 1]

        try:
            parsed = json.loads(text)

            # Validate structure

            if not isinstance(parsed, dict):
                raise ValueError("Response is not a dictionary")

            # Ensure required keys exist
            if 'entities' not in parsed:
                parsed['entities'] = []
            if 'relationships' not in parsed:
                parsed['relationships'] = []
            if 'language' not in parsed:
                parsed['language'] = 'en'

            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nText: {text[:500]}")
            # Return empty structure
            return {

                'entities': [],
                'relationships': [],
                'language': 'en'
            }
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            return {
                'entities': [],
                'relationships': [],
                'language': 'en'
            }

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get Gemini usage statistics."""
        return await self.rate_limiter.get_stats()