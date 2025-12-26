"""
Rate limiting utilities to prevent exceeding API quotas.
Tracks usage across different services and time windows.
"""
import asyncio
import time
from typing import Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import redis.asyncio as redis
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class UsageStats:
    """Track usage statistics for a service."""
    requests_today: int = 0
    requests_this_minute: int = 0
    total_cost_today: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)
    last_request: Optional[datetime] = None


class RateLimiter:
    """
    Rate limiter for API calls with Redis-backed persistence.
    Supports per-minute and per-day limits.
    """

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.local_stats: Dict[str, UsageStats] = {}

    def _get_key(self, service: str, window: str) -> str:
        """Generate Redis key for service and time window."""
        today = datetime.now().strftime("%Y-%m-%d")
        minute = datetime.now().strftime("%Y-%m-%d-%H-%M")

        if window == "day":
            return f"ratelimit:{service}:day:{today}"
        elif window == "minute":
            return f"ratelimit:{service}:minute:{minute}"
        else:
            raise ValueError(f"Unknown window: {window}")

    async def check_and_increment(
        self,
        service: str,
        cost: float = 0.0,
        per_minute_limit: Optional[int] = None,
        per_day_limit: Optional[int] = None
    ) -> bool:
        """
        Check if request is allowed and increment counter.

        Args:
            service: Service name (e.g., 'claude', 'whisper')
            cost: Cost of this request in USD
            per_minute_limit: Max requests per minute (None = no limit)
            per_day_limit: Max requests per day (None = no limit)

        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        # Check per-minute limit
        if per_minute_limit:
            minute_key = self._get_key(service, "minute")
            current_minute = await self.redis.get(minute_key)

            if current_minute and int(current_minute) >= per_minute_limit:
                logger.warning(
                    f"Rate limit exceeded for {service}: {per_minute_limit}/minute",
                    extra={'service': service, 'limit_type': 'per_minute'}
                )
                return False

            # Increment with 60-second expiry
            await self.redis.incr(minute_key)
            await self.redis.expire(minute_key, 60)

        # Check per-day limit
        if per_day_limit:
            day_key = self._get_key(service, "day")
            current_day = await self.redis.get(day_key)

            if current_day and int(current_day) >= per_day_limit:
                logger.warning(
                    f"Rate limit exceeded for {service}: {per_day_limit}/day",
                    extra={'service': service, 'limit_type': 'per_day'}
                )
                return False

            # Increment with expiry at end of day
            await self.redis.incr(day_key)
            seconds_until_midnight = (
                datetime.combine(datetime.now().date() + timedelta(days=1), datetime.min.time())
                - datetime.now()
            ).seconds
            await self.redis.expire(day_key, seconds_until_midnight)

        # Track cost
        if cost > 0:
            cost_key = f"cost:{service}:day:{datetime.now().strftime('%Y-%m-%d')}"
            await self.redis.incrbyfloat(cost_key, cost)
            seconds_until_midnight = (
                datetime.combine(datetime.now().date() + timedelta(days=1), datetime.min.time())
                - datetime.now()
            ).seconds
            await self.redis.expire(cost_key, seconds_until_midnight)

        logger.debug(
            f"Rate limit check passed for {service}",
            extra={'service': service, 'cost': cost}
        )
        return True

    async def get_usage_stats(self, service: str) -> Dict[str, any]:
        """
        Get current usage statistics for a service.

        Args:
            service: Service name

        Returns:
            Dictionary with usage stats
        """
        minute_key = self._get_key(service, "minute")
        day_key = self._get_key(service, "day")
        cost_key = f"cost:{service}:day:{datetime.now().strftime('%Y-%m-%d')}"

        requests_this_minute = int(await self.redis.get(minute_key) or 0)
        requests_today = int(await self.redis.get(day_key) or 0)
        cost_today = float(await self.redis.get(cost_key) or 0.0)

        return {
            'service': service,
            'requests_this_minute': requests_this_minute,
            'requests_today': requests_today,
            'cost_today': round(cost_today, 4),
            'timestamp': datetime.now().isoformat()
        }

    async def wait_if_needed(
        self,
        service: str,
        per_minute_limit: Optional[int] = None
    ) -> float:
        """
        Wait if rate limit would be exceeded, return wait time.

        Args:
            service: Service name
            per_minute_limit: Max requests per minute

        Returns:
            Seconds waited (0 if no wait needed)
        """
        if not per_minute_limit:
            return 0.0

        minute_key = self._get_key(service, "minute")
        current_minute = await self.redis.get(minute_key)

        if current_minute and int(current_minute) >= per_minute_limit:
            # Wait until next minute
            wait_time = 60 - (time.time() % 60)
            logger.info(
                f"Rate limit reached for {service}, waiting {wait_time:.1f}s",
                extra={'service': service, 'wait_seconds': wait_time}
            )
            await asyncio.sleep(wait_time)
            return wait_time

        return 0.0


class ClaudeRateLimiter:
    """Specialized rate limiter for Claude API with cost tracking."""

    def __init__(self, rate_limiter: RateLimiter):
        self.limiter = rate_limiter
        self.per_minute_limit = settings.claude_max_requests_per_minute
        self.per_day_limit = settings.claude_max_requests_per_day
        self.monthly_budget = settings.monthly_budget_usd

    async def check(self) -> bool:
        """Check if Claude request is allowed."""
        return await self.limiter.check_and_increment(
            service="claude",
            per_minute_limit=self.per_minute_limit,
            per_day_limit=self.per_day_limit
        )

    async def wait_if_needed(self) -> float:
        """Wait if Claude rate limit would be exceeded."""
        return await self.limiter.wait_if_needed(
            service="claude",
            per_minute_limit=self.per_minute_limit
        )

    async def track_cost(self, cost_usd: float):
        """Track API costs."""
        cost_key = f"cost:claude:month:{datetime.now().strftime('%Y-%m')}"
        await self.limiter.redis.incrbyfloat(cost_key, cost_usd)
        # Expire at end of month
        days_in_month = 30
        await self.limiter.redis.expire(cost_key, days_in_month * 24 * 3600)

    async def get_stats(self) -> Dict[str, any]:
        """Get Claude usage statistics with cost info."""
        stats = await self.limiter.get_usage_stats("claude")

        # Get monthly cost
        cost_key = f"cost:claude:month:{datetime.now().strftime('%Y-%m')}"
        monthly_cost = float(await self.limiter.redis.get(cost_key) or 0.0)

        stats['per_minute_limit'] = self.per_minute_limit
        stats['per_day_limit'] = self.per_day_limit
        stats['requests_remaining_today'] = self.per_day_limit - stats['requests_today']
        stats['monthly_cost_usd'] = round(monthly_cost, 4)
        stats['monthly_budget_usd'] = self.monthly_budget
        stats['budget_remaining'] = round(self.monthly_budget - monthly_cost, 4)
        stats['budget_percentage_used'] = round((monthly_cost / self.monthly_budget) * 100, 2) if self.monthly_budget > 0 else 0

        return stats


class GroqWhisperRateLimiter:
    """Specialized rate limiter for Groq Whisper API (FREE tier)."""

    def __init__(self, rate_limiter: RateLimiter):
        self.limiter = rate_limiter
        self.per_minute_limit = settings.groq_whisper_max_requests_per_minute
        self.per_day_limit = settings.groq_whisper_max_requests_per_day

    async def check(self) -> bool:
        """
        Check if Groq Whisper request is allowed (no cost tracking needed - it's free!).

        Returns:
            True if request allowed, False if rate limit exceeded
        """
        return await self.limiter.check_and_increment(
            service="groq_whisper",
            per_minute_limit=self.per_minute_limit,
            per_day_limit=self.per_day_limit
        )

    async def wait_if_needed(self) -> float:
        """Wait if Groq Whisper rate limit would be exceeded."""
        return await self.limiter.wait_if_needed(
            service="groq_whisper",
            per_minute_limit=self.per_minute_limit
        )

    async def get_stats(self) -> Dict[str, any]:
        """Get Groq Whisper usage statistics."""
        stats = await self.limiter.get_usage_stats("groq_whisper")
        stats['per_minute_limit'] = self.per_minute_limit
        stats['per_day_limit'] = self.per_day_limit
        stats['requests_remaining_today'] = self.per_day_limit - stats['requests_today']
        stats['cost_today'] = 0.0  # Groq is FREE!
        stats['percentage_used'] = round((stats['requests_today'] / self.per_day_limit) * 100, 2)
        return stats