"""
Claude API client with rate limiting, conversation history, web search, and cost optimization.
Implements conversational AI with personality, multi-language support, and smart caching.
"""

import time
import json
from typing import Dict, Any, Optional
from anthropic import Anthropic

from config.settings import settings
from config.prompts import SYSTEM_PROMPTS, ENTITY_EXTRACTION_PROMPTS
from core.language_detector import LanguageCode
from core.cache_manager import CacheManager
from utils.logger import setup_logger, log_api_call
from utils.rate_limiter import ClaudeRateLimiter

logger = setup_logger(__name__)


class ClaudeClient:
    """
    Claude API client with conversational capabilities, web search, and cost tracking.
    Optimized for minimal API calls while maintaining excellent quality.
    """

    def __init__(
        self,
        rate_limiter: ClaudeRateLimiter,
        cache_manager: CacheManager
    ):
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self.rate_limiter = rate_limiter
        self.cache = cache_manager
        self.model = "claude-sonnet-4-20250514"
        self.max_tokens = 4096

        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

        logger.info(f"Initialized Claude client with model: {self.model}")

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for API call."""
        input_cost = (input_tokens / 1_000_000) * settings.claude_cost_per_million_input_tokens
        output_cost = (output_tokens / 1_000_000) * settings.claude_cost_per_million_output_tokens
        return input_cost + output_cost

    async def chat(
        self,
        message: str,
        language: LanguageCode,
        conversation_id: str,
        use_web_search: bool = False,
        use_cache: bool = True,
        local_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send conversational message to Claude with full context.

        Args:
            message: User message
            language: Message language ('en' or 'ru')
            conversation_id: Unique conversation identifier
            use_web_search: Whether Claude can use web search
            use_cache: Whether to check cache for similar queries
            local_context: Optional context from local database (reduces need for web search)

        Returns:
            Dictionary with response and metadata
        """
        # Check cache first (no cost!)
        if use_cache:
            cached = await self.cache.get_response(message, check_semantic=True)
            if cached:
                logger.info("💰 Cache hit - saved API call!")
                # Add to conversation history
                await self.cache.add_conversation_turn(
                    conversation_id, "user", message, {'language': language}
                )
                await self.cache.add_conversation_turn(
                    conversation_id, "assistant", cached['response'],
                    {'language': language, 'cached': True, 'cost_usd': 0.0}
                )

                return {
                    'response': cached['response'],
                    'language': language,
                    'cached': True,
                    'conversation_id': conversation_id,
                    'similarity': cached.get('similarity'),
                    'cost_usd': 0.0,
                    'tokens': {'input': 0, 'output': 0}
                }

        # Check rate limit
        await self.rate_limiter.wait_if_needed()
        if not await self.rate_limiter.check():
            raise Exception("Claude rate limit exceeded. Please wait a minute.")

        # Get conversation history (last 5 turns to save tokens)
        history = await self.cache.get_conversation_history(conversation_id, max_turns=5)

        # Build messages array
        messages = []
        for turn in history:
            messages.append({
                'role': turn['role'],
                'content': turn['content']
            })

        # Add local context if provided (reduces need for web search)
        if local_context:
            enhanced_message = f"{message}\n\n[Context from your stored data: {local_context}]"
        else:
            enhanced_message = message

        # Add current message
        messages.append({
            'role': 'user',
            'content': enhanced_message
        })

        # System prompt in user's language
        system_prompt = SYSTEM_PROMPTS[language]

        # Prepare API call
        start_time = time.time()

        try:
            logger.info(f"🔵 Calling Claude API (language={language}, web_search={use_web_search})")

            api_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system_prompt,
                "messages": messages
            }

            # Add web search tool if requested
            if use_web_search:
                api_params["tools"] = [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search"
                    }
                ]

            # Call Claude API
            response = self.client.messages.create(**api_params)

            duration_ms = (time.time() - start_time) * 1000

            # Extract response text
            response_text = self._extract_response_text(response)

            # Check if web search was used
            used_web_search = self._check_web_search_usage(response)

            # Track costs
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost_usd = self._calculate_cost(input_tokens, output_tokens)

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost_usd += cost_usd

            # Track in rate limiter for budget monitoring
            await self.rate_limiter.track_cost(cost_usd)

            log_api_call(
                logger,
                service="anthropic",
                endpoint="messages",
                duration_ms=duration_ms,
                status="success",
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
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
                    'cost_usd': cost_usd,
                    'tokens': {'input': input_tokens, 'output': output_tokens}
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
                f"💰 Response received: {len(response_text)} chars, cost=${cost_usd:.4f}",
                extra={
                    'response_length': len(response_text),
                    'web_search_used': used_web_search,
                    'cost_usd': cost_usd,
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
                'cost_usd': cost_usd,
                'tokens': {'input': input_tokens, 'output': output_tokens}
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_api_call(
                logger,
                service="anthropic",
                endpoint="messages",
                duration_ms=duration_ms,
                status="error",
                error=str(e)
            )
            logger.error(f"Claude API error: {e}")
            raise

    async def extract_entities(
        self,
        text: str,
        language: LanguageCode
    ) -> Dict[str, Any]:
        """
        Extract structured health entities from text using Claude.
        Uses lower max_tokens to reduce costs for simple extraction.

        Args:
            text: User text to analyze
            language: Text language

        Returns:
            Dictionary with extracted entities and relationships
        """
        # Check rate limit
        await self.rate_limiter.wait_if_needed()
        if not await self.rate_limiter.check():
            raise Exception("Claude rate limit exceeded. Please wait a minute.")

        # Get extraction prompt in user's language
        extraction_prompt = ENTITY_EXTRACTION_PROMPTS[language].format(user_message=text)

        start_time = time.time()

        try:
            logger.info(f"🔵 Extracting entities (language={language})")

            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,  # Lower for extraction
                messages=[{
                    'role': 'user',
                    'content': extraction_prompt
                }]
            )

            duration_ms = (time.time() - start_time) * 1000

            # Extract JSON from response
            response_text = self._extract_response_text(response)
            extracted_data = self._parse_json_response(response_text)

            # Track costs
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost_usd = self._calculate_cost(input_tokens, output_tokens)

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost_usd += cost_usd

            await self.rate_limiter.track_cost(cost_usd)

            log_api_call(
                logger,
                service="anthropic",
                endpoint="extract_entities",
                duration_ms=duration_ms,
                status="success",
                entities_found=len(extracted_data.get('entities', [])),
                relationships_found=len(extracted_data.get('relationships', [])),
                cost_usd=cost_usd,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

            logger.info(
                f"💰 Extracted {len(extracted_data.get('entities', []))} entities, "
                f"{len(extracted_data.get('relationships', []))} relationships, cost=${cost_usd:.4f}"
            )

            return extracted_data

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_api_call(
                logger,
                service="anthropic",
                endpoint="extract_entities",
                duration_ms=duration_ms,
                status="error",
                error=str(e)
            )
            logger.error(f"Entity extraction failed: {e}")
            raise

    def _extract_response_text(self, response) -> str:
        """Extract text from Claude response, handling tool use blocks."""
        text_parts = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                logger.debug(f"Tool used: {block.name}")

        return "\n".join(text_parts).strip()

    def _check_web_search_usage(self, response) -> bool:
        """Check if web search tool was used in response."""
        for block in response.content:
            if block.type == "tool_use" and block.name == "web_search":
                return True
        return False

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from Claude response, handling markdown code blocks."""
        text = text.strip()

        # Remove markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1]
        if "```" in text:
            text = text.split("```")[0]

        text = text.strip()

        # Find JSON object
        start_idx = text.find('{')
        end_idx = text.rfind('}')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = text[start_idx:end_idx + 1]

        try:
            parsed = json.loads(text)

            # Validate and add defaults
            if 'entities' not in parsed:
                parsed['entities'] = []
            if 'relationships' not in parsed:
                parsed['relationships'] = []
            if 'language' not in parsed:
                parsed['language'] = 'en'

            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nText: {text[:500]}")
            return {
                'entities': [],
                'relationships': [],
                'language': 'en'
            }

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get Claude usage statistics with cost tracking."""
        stats = await self.rate_limiter.get_stats()
        stats['session_total_input_tokens'] = self.total_input_tokens
        stats['session_total_output_tokens'] = self.total_output_tokens
        stats['session_total_cost_usd'] = round(self.total_cost_usd, 4)
        return stats