"""
Language detection for user input.
Supports English and Russian with confidence scoring.
"""

from typing import Tuple, Literal
from langdetect import detect, detect_langs, LangDetectException
from utils.logger import setup_logger

logger = setup_logger(__name__)

LanguageCode = Literal["en", "ru"]


class LanguageDetector:
    """Detect language of user input with fallback to default."""

    def __init__(self, default_language: LanguageCode = "en"):
        self.default_language = default_language
        self.supported_languages = {"en", "ru"}

    def detect(self, text: str) -> Tuple[LanguageCode, float]:
        """
        Detect language of text.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (language_code, confidence_score)

        Examples:
            >>> detector = LanguageDetector()
            >>> detector.detect("Hello, how are you?")
            ('en', 0.99)
            >>> detector.detect("Привет, как дела?")
            ('ru', 0.99)
        """
        if not text or len(text.strip()) < 3:
            logger.warning("Text too short for language detection, using default")
            return self.default_language, 0.0

        try:
            # Get all detected languages with probabilities
            detected_langs = detect_langs(text)

            # Find the highest confidence supported language
            for lang_prob in detected_langs:
                lang_code = lang_prob.lang
                confidence = lang_prob.prob

                if lang_code in self.supported_languages:
                    logger.info(
                        f"Detected language: {lang_code} (confidence: {confidence:.2f})",
                        extra={'language': lang_code, 'confidence': confidence}
                    )
                    return lang_code, confidence

            # If no supported language found, use simple detect
            simple_detect = detect(text)
            if simple_detect in self.supported_languages:
                return simple_detect, 0.8  # Assume reasonable confidence

            logger.warning(
                f"Detected unsupported language: {simple_detect}, using default: {self.default_language}"
            )
            return self.default_language, 0.5

        except LangDetectException as e:
            logger.warning(
                f"Language detection failed: {e}, using default: {self.default_language}"
            )
            return self.default_language, 0.0

    def is_confident(self, confidence: float, threshold: float = 0.7) -> bool:
        """
        Check if confidence score is above threshold.

        Args:
            confidence: Confidence score from detection
            threshold: Minimum confidence threshold

        Returns:
            True if confidence is high enough
        """
        return confidence >= threshold

    def detect_with_fallback(self, text: str, fallback: LanguageCode) -> LanguageCode:
        """
        Detect language with custom fallback if confidence is low.

        Args:
            text: Input text
            fallback: Language to use if detection fails

        Returns:
            Detected or fallback language code
        """
        language, confidence = self.detect(text)

        if self.is_confident(confidence):
            return language

        logger.info(
            f"Low confidence ({confidence:.2f}), using fallback: {fallback}"
        )
        return fallback

    def detect_mixed(self, text: str) -> Tuple[LanguageCode, bool]:
        """
        Detect if text contains mixed languages.

        Args:
            text: Input text

        Returns:
            Tuple of (primary_language, is_mixed)
        """
        try:
            detected_langs = detect_langs(text)

            # Check if multiple supported languages detected with significant probability
            supported_langs = [
                (lang.lang, lang.prob)
                for lang in detected_langs
                if lang.lang in self.supported_languages
            ]

            if len(supported_langs) > 1 and supported_langs[1][1] > 0.3:
                primary = supported_langs[0][0]
                logger.info(
                    f"Mixed language detected: primary={primary}, secondary={supported_langs[1][0]}",
                    extra={'primary': primary, 'is_mixed': True}
                )
                return primary, True

            if supported_langs:
                return supported_langs[0][0], False

            return self.default_language, False

        except LangDetectException:
            return self.default_language, False


# Global instance
language_detector = LanguageDetector()