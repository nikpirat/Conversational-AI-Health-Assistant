"""
Voice input processing using Groq Whisper API.
Handles audio file transcription with rate limiting (FREE tier: 14,400 requests/day).
"""

import time
from pathlib import Path
from typing import Optional, Dict
from groq import Groq
from pydub import AudioSegment

from config.settings import settings
from core.language_detector import LanguageDetector, LanguageCode
from utils.logger import setup_logger, log_api_call
from utils.rate_limiter import GroqWhisperRateLimiter

logger = setup_logger(__name__)


class VoiceInput:
    """
    Voice input processor using Groq Whisper API.
    Supports multi-language transcription with automatic language detection.
    FREE tier: 14,400 requests/day, 30 requests/minute.
    """

    def __init__(self, rate_limiter: GroqWhisperRateLimiter):
        self.client = Groq(api_key=settings.groq_api_key)
        self.rate_limiter = rate_limiter
        self.language_detector = LanguageDetector(settings.default_language)
        self.model = settings.groq_whisper_model

    def get_audio_duration(self, audio_path: Path) -> float:
        """
        Get duration of audio file in seconds.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds
        """
        try:
            audio = AudioSegment.from_file(audio_path)
            duration_seconds = len(audio) / 1000.0
            logger.debug(f"Audio duration: {duration_seconds:.2f}s", extra={'file': str(audio_path)})
            return duration_seconds
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            # Estimate based on file size (rough estimate: 1MB ≈ 60 seconds for typical audio)
            file_size_mb = audio_path.stat().st_size / (1024 * 1024)
            estimated_duration = file_size_mb * 60
            logger.warning(f"Using estimated duration: {estimated_duration:.2f}s")
            return estimated_duration

    async def transcribe(
        self,
        audio_path: Path,
        language: Optional[LanguageCode] = None,
        prompt: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Transcribe audio file to text using Groq Whisper API.

        Args:
            audio_path: Path to audio file (mp3, wav, m4a, etc.)
            language: Optional language hint ('en' or 'ru')
            prompt: Optional prompt to guide transcription

        Returns:
            Dictionary with:
                - text: Transcribed text
                - language: Detected language
                - duration: Audio duration in seconds
                - cost: Always $0 (Groq is free)

        Raises:
            Exception if rate limit exceeded or API call fails
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Get audio duration for statistics
        duration_seconds = self.get_audio_duration(audio_path)

        # Check rate limit
        if not await self.rate_limiter.check():
            stats = self.rate_limiter.get_stats()
            raise Exception(
                f"Groq Whisper rate limit exceeded. Used: {stats['requests_today']} / {stats['per_day_limit']} today"
            )

        # Transcribe with Groq Whisper
        start_time = time.time()

        try:
            with open(audio_path, 'rb') as audio_file:
                logger.info(f"Transcribing audio with Groq: {audio_path.name}")

                # API call parameters
                api_params = {
                    "file": audio_file,
                    "model": self.model,
                    "response_format": "json",
                }

                # Add language hint if provided
                if language:
                    api_params["language"] = language
                    logger.debug(f"Using language hint: {language}")

                # Add prompt if provided (helps with context/terminology)
                if prompt:
                    api_params["prompt"] = prompt
                    logger.debug(f"Using prompt: {prompt[:50]}...")

                # Call Groq Whisper API
                response = self.client.audio.transcriptions.create(**api_params)

            duration_ms = (time.time() - start_time) * 1000

            # Extract transcription
            transcribed_text = response.text

            # Detect language from transcription
            detected_language, confidence = self.language_detector.detect(transcribed_text)

            log_api_call(
                logger,
                service="groq",
                endpoint="whisper",
                duration_ms=duration_ms,
                status="success",
                audio_duration_seconds=duration_seconds,
                detected_language=detected_language,
                confidence=confidence,
                cost_usd=0.0  # Groq is FREE!
            )

            logger.info(
                f"Transcription successful: {len(transcribed_text)} chars, language={detected_language}",
                extra={
                    'text_length': len(transcribed_text),
                    'language': detected_language,
                    'confidence': confidence,
                    'duration_seconds': duration_seconds,
                    'cost_usd': 0.0
                }
            )

            return {
                'text': transcribed_text,
                'language': detected_language,
                'language_confidence': confidence,
                'duration_seconds': duration_seconds,
                'cost_usd': 0.0,  # Groq is FREE!
                'audio_path': str(audio_path)
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_api_call(
                logger,
                service="groq",
                endpoint="whisper",
                duration_ms=duration_ms,
                status="error",
                error=str(e)
            )
            logger.error(f"Transcription failed: {e}")
            raise

    async def get_usage_stats(self) -> Dict[str, any]:
        """Get current Groq Whisper API usage statistics."""
        return await self.rate_limiter.get_stats()