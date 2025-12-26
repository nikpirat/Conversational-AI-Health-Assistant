#!/usr/bin/env python3
"""
Voice Health Assistant - Phase 1 CLI
Main entry point for testing the conversational AI health assistant.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
import redis.asyncio as redis

from config.settings import settings
from core.voice_input import VoiceInput
from core.llm_client import ClaudeClient
from core.cache_manager import CacheManager
from database.neo4j_client import Neo4jClient
from services.storage_service import HealthAssistantService
from utils.logger import setup_logger
from utils.rate_limiter import RateLimiter, ClaudeRateLimiter, GroqWhisperRateLimiter

logger = setup_logger(__name__)


class HealthAssistantCLI:
    """Command-line interface for the health assistant."""

    def __init__(self):
        self.service: Optional[HealthAssistantService] = None
        self.conversation_id: Optional[str] = None

    async def initialize(self):
        """Initialize all services."""
        print("🚀 Initializing Health Assistant...")
        print(f"📍 Language: {settings.default_language}")
        print()

        try:
            # Initialize Redis
            print("📦 Connecting to Redis...")
            redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=True
            )
            await redis_client.ping()
            print("✅ Redis connected")

            # Initialize rate limiters
            rate_limiter = RateLimiter(redis_client)
            claude_limiter = ClaudeRateLimiter(rate_limiter)
            groq_whisper_limiter = GroqWhisperRateLimiter(rate_limiter)

            # Initialize cache
            print("💾 Initializing cache...")
            cache_manager = CacheManager(redis_client)
            print("✅ Cache ready")

            # Initialize Neo4j
            print("🗄️  Connecting to Neo4j...")
            neo4j_client = Neo4jClient()
            neo4j_client.connect()
            print("✅ Neo4j connected")

            # Initialize voice input
            print("🎤 Initializing voice input (Groq Whisper)...")
            voice_input = VoiceInput(groq_whisper_limiter)
            print("✅ Voice input ready")

            # Initialize Claude client
            print("🧠 Initializing Claude...")
            claude_client = ClaudeClient(claude_limiter, cache_manager)
            print("✅ Claude ready")

            # Initialize main service
            self.service = HealthAssistantService(
                voice_input,
                claude_client,
                neo4j_client,
                cache_manager
            )

            print()
            print("=" * 60)
            print("✅ Health Assistant initialized successfully!")
            print("=" * 60)
            print()

        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            logger.error(f"Initialization error: {e}")
            raise

    async def process_voice_file(self, audio_path: Path):
        """Process a voice audio file."""
        if not audio_path.exists():
            print(f"❌ Audio file not found: {audio_path}")
            return

        print(f"🎤 Processing audio: {audio_path.name}")
        print()

        result = await self.service.process_voice_input(
            audio_path,
            conversation_id=self.conversation_id
        )

        if not result['success']:
            print(f"❌ Error: {result['error']}")
            return

        # Update conversation ID
        self.conversation_id = result['conversation_id']

        # Display results
        print("=" * 60)
        print("📝 TRANSCRIPTION")
        print("=" * 60)
        print(f"Text: {result['transcription']['text']}")
        print(f"Language: {result['transcription']['language']}")
        print(f"Confidence: {result['transcription']['confidence']:.2f}")
        print(f"Duration: {result['transcription']['duration_seconds']:.1f}s")
        print(f"Cost: ${result['transcription']['cost_usd']:.6f}")
        print()

        if result['extraction']['success']:
            print("=" * 60)
            print("📊 EXTRACTED KNOWLEDGE")
            print("=" * 60)
            print(f"Entities: {result['extraction']['entities_created']}")
            print(f"Relationships: {result['extraction']['relationships_created']}")

            if result['extraction']['entities']:
                print("\nEntities:")
                for entity in result['extraction']['entities']:
                    print(f"  - {entity['name']} ({entity['type']})")

            if result['extraction']['relationships']:
                print("\nRelationships:")
                for rel in result['extraction']['relationships'][:5]:
                    print(f"  - {rel['from']} -{rel['type']}-> {rel['to']}")
            print()

        print("=" * 60)
        print("💬 ASSISTANT RESPONSE")
        print("=" * 60)
        print(result['response']['text'])
        print()

        if result['response']['web_search_used']:
            print("🌐 Web search was used for this response")
        if result['response']['cached']:
            print("⚡ Response was cached")

        print()
        print(f"⏱️  Total processing time: {result['processing_time_ms']:.0f}ms")
        print()

    async def process_text(self, text: str, extract_entities: bool = True):
        """Process text input."""
        print(f"💬 Processing: '{text}'")
        print()

        result = await self.service.process_text_input(
            text,
            conversation_id=self.conversation_id,
            extract_entities=extract_entities
        )

        if not result['success']:
            print(f"❌ Error: {result['error']}")
            return

        # Update conversation ID
        self.conversation_id = result['conversation_id']

        # Display extraction results
        if result['extraction'] and result['extraction']['success']:
            print("=" * 60)
            print("📊 EXTRACTED KNOWLEDGE")
            print("=" * 60)
            print(f"Entities: {result['extraction']['entities_created']}")
            print(f"Relationships: {result['extraction']['relationships_created']}")
            print()

        # Display response
        print("=" * 60)
        print("💬 ASSISTANT RESPONSE")
        print("=" * 60)
        print(result['response']['text'])
        print()

        if result['response']['web_search_used']:
            print("🌐 Web search was used")
        if result['response']['cached']:
            print("⚡ Response was cached")

        print()

    async def show_statistics(self):
        """Display system statistics."""
        stats = await self.service.get_statistics()

        print("=" * 60)
        print("📊 SYSTEM STATISTICS")
        print("=" * 60)
        print()

        print("🗄️  DATABASE:")
        db_stats = stats['database']
        print(f"  Total nodes: {db_stats['total_nodes']}")
        print(f"  Total relationships: {db_stats['total_relationships']}")
        print(f"  Node types: {', '.join(db_stats['node_labels'])}")
        print()

        print("💾 CACHE:")
        cache_stats = stats['cache']
        print(f"  Cached responses: {cache_stats['cached_responses']}")
        print(f"  Cached embeddings: {cache_stats['cached_embeddings']}")
        print(f"  Active conversations: {cache_stats['active_conversations']}")
        print(f"  Memory used: {cache_stats['redis_info']['used_memory']}")
        print()

        print("🎤 GROQ WHISPER API:")
        whisper_stats = stats['whisper']
        print(f"  Requests this minute: {whisper_stats.get('requests_this_minute', 0)}")
        print(f"  Requests today: {whisper_stats['requests_today']}")
        print(f"  Requests remaining today: {whisper_stats.get('requests_remaining_today', 'N/A')}")
        print(f"  Cost today: $0.00 (FREE!)")
        print(f"  Percentage used: {whisper_stats.get('percentage_used', 0):.1f}%")
        print()

        print("🧠 CLAUDE API:")
        claude_stats = stats['claude']
        print(f"  Requests this minute: {claude_stats.get('requests_this_minute', 0)}")
        print(f"  Requests today: {claude_stats['requests_today']}")
        print(f"  Requests remaining today: {claude_stats.get('requests_remaining_today', 'N/A')}")
        print(f"  💰 Monthly cost: ${claude_stats.get('monthly_cost_usd', 0):.4f}")
        print(f"  💰 Budget remaining: ${claude_stats.get('budget_remaining', 0):.4f}")
        print(f"  💰 Budget used: {claude_stats.get('budget_percentage_used', 0):.1f}%")
        if 'session_total_cost_usd' in claude_stats:
            print(f"  💰 Session cost: ${claude_stats['session_total_cost_usd']:.4f}")
        print()

    async def interactive_mode(self):
        """Run interactive CLI mode."""
        print()
        print("=" * 60)
        print("🎯 INTERACTIVE MODE")
        print("=" * 60)
        print()
        print("Commands:")
        print("  Type text to chat")
        print("  'stats' - Show statistics")
        print("  'exit' or 'quit' - Exit")
        print()

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit']:
                    print("👋 Goodbye!")
                    break

                if user_input.lower() == 'stats':
                    await self.show_statistics()
                    continue

                # Process as text input
                await self.process_text(user_input, extract_entities=True)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Interactive mode error: {e}")


async def main():
    """Main entry point."""
    cli = HealthAssistantCLI()

    try:
        await cli.initialize()

        # Check command line arguments
        if len(sys.argv) > 1:
            command = sys.argv[1]

            if command == "voice" and len(sys.argv) > 2:
                # Process voice file
                audio_path = Path(sys.argv[2])
                await cli.process_voice_file(audio_path)

            elif command == "text" and len(sys.argv) > 2:
                # Process text
                text = " ".join(sys.argv[2:])
                await cli.process_text(text)

            elif command == "stats":
                # Show statistics
                await cli.show_statistics()

            else:
                print("Usage:")
                print("  python main.py voice <audio_file>")
                print("  python main.py text <your message>")
                print("  python main.py stats")
                print("  python main.py  (interactive mode)")

        else:
            # Interactive mode
            await cli.interactive_mode()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())