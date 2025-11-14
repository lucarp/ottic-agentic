"""
SE Ranking API Client
Provides async methods to interact with SE Ranking Data API for SEO analysis.
API Documentation: https://seranking.com/api/data/
"""

import os
import logging
import asyncio
from pathlib import Path
from typing import Optional
import httpx
from dotenv import load_dotenv

# Load environment variables from project root
# This file is in backend/, so parent directory contains .env
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

# SE Ranking API Configuration
SERANKING_BASE_URL = "https://api.seranking.com/v1"
SERANKING_API_KEY = os.getenv("SERANKING_API_KEY")

# Rate limiting: SE Ranking allows 10 requests per second
RATE_LIMIT_DELAY = 0.1  # 100ms between requests


class SERankingClient:
    """Async client for SE Ranking Data API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize SE Ranking API client.

        Args:
            api_key: SE Ranking API key. If not provided, reads from environment.
        """
        self.api_key = api_key or SERANKING_API_KEY
        if not self.api_key:
            raise ValueError("SE Ranking API key not found. Set SERANKING_API_KEY environment variable.")

        self.base_url = SERANKING_BASE_URL
        self.headers = {
            "Authorization": f"Token {self.api_key}",
            "Accept": "application/json"
        }
        self._last_request_time = 0

    async def _rate_limit(self):
        """Implement rate limiting to stay within 10 req/sec."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < RATE_LIMIT_DELAY:
            await asyncio.sleep(RATE_LIMIT_DELAY - time_since_last)

        self._last_request_time = asyncio.get_event_loop().time()

    async def _make_request(self, endpoint: str, params: dict) -> dict:
        """
        Make an async HTTP request to SE Ranking API.

        Args:
            endpoint: API endpoint path (e.g., "/domain/overview/worldwide")
            params: Query parameters

        Returns:
            JSON response as dictionary

        Raises:
            httpx.HTTPError: If request fails
        """
        await self._rate_limit()

        url = f"{self.base_url}{endpoint}"
        params["output"] = "json"  # Always request JSON format

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                logger.info(f"SE Ranking API request: {endpoint} with params: {params}")
                response = await client.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()
                logger.info(f"SE Ranking API response: {endpoint} - Status {response.status_code}")
                return data
            except httpx.HTTPStatusError as e:
                logger.error(f"SE Ranking API error: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"SE Ranking API request failed: {str(e)}")
                raise

    async def get_domain_overview(self, domain: str, currency: str = "USD") -> dict:
        """
        Get worldwide domain overview with organic and paid search metrics.

        Args:
            domain: Domain to analyze (e.g., "stripe.com")
            currency: Currency for traffic value (default: "USD")

        Returns:
            Dictionary with domain metrics including:
            - organic_keywords_count
            - organic_traffic
            - organic_traffic_value
            - paid_keywords_count
            - paid_traffic
            - paid_traffic_value

        Example:
            >>> client = SERankingClient()
            >>> data = await client.get_domain_overview("stripe.com")
            >>> print(data["organic_keywords_count"])
        """
        endpoint = "/domain/overview/worldwide"
        params = {
            "domain": domain,
            "currency": currency
        }
        return await self._make_request(endpoint, params)

    async def get_competitors(
        self,
        domain: str,
        source: str = "us",
        type: str = "organic",
        limit: int = 10
    ) -> dict:
        """
        Get top organic or paid search competitors for a domain.

        Args:
            domain: Target domain (e.g., "stripe.com")
            source: Regional database (e.g., "us", "uk", "ca")
            type: "organic" or "paid"
            limit: Maximum number of competitors (default: 10, max: 100)

        Returns:
            Dictionary with competitor data including:
            - competitors: List of competitor domains with metrics
            - total: Total number of competitors found

        Example:
            >>> client = SERankingClient()
            >>> data = await client.get_competitors("stripe.com", source="us")
            >>> for comp in data["competitors"]:
            ...     print(f"{comp['domain']}: {comp['keywords_count']} common keywords")
        """
        endpoint = "/domain/competitors"
        params = {
            "domain": domain,
            "source": source,
            "type": type,
            "limit": min(limit, 100)  # Cap at API maximum
        }
        return await self._make_request(endpoint, params)

    async def get_keyword_comparison(
        self,
        domain: str,
        competitor: str,
        source: str = "us",
        type: str = "organic",
        diff: int = 1,
        limit: int = 100
    ) -> dict:
        """
        Compare keywords between two domains to find gaps.

        Args:
            domain: Competitor domain (what keywords do THEY rank for)
            competitor: Your domain (that you DON'T rank for)
            source: Regional database (e.g., "us", "uk")
            type: "organic" or "paid"
            diff: 1 = keywords only in domain (gaps), 0 = common keywords
            limit: Maximum keywords to return (default: 100, max: 1000)

        Returns:
            Dictionary with keyword gap data including:
            - keywords: List of keywords with volume, CPC, difficulty, positions
            - total: Total number of keywords found

        Example:
            >>> client = SERankingClient()
            >>> # Find keywords paypal.com ranks for that stripe.com doesn't
            >>> data = await client.get_keyword_comparison(
            ...     domain="paypal.com",
            ...     competitor="stripe.com",
            ...     diff=1
            ... )
        """
        endpoint = "/domain/keywords/comparison"
        params = {
            "domain": domain,
            "compare": competitor,
            "source": source,
            "type": type,
            "diff": diff,
            "limit": min(limit, 1000),
            "order_field": "volume",
            "order_type": "desc",
            "cols": "keyword,volume,cpc,competition,difficulty,position,price,traffic"
        }
        return await self._make_request(endpoint, params)

    async def get_similar_keywords(
        self,
        keyword: str,
        source: str = "us",
        limit: int = 50
    ) -> dict:
        """
        Find keywords semantically similar to a given keyword.

        Args:
            keyword: Seed keyword (e.g., "payment gateway")
            source: Regional database (e.g., "us", "uk")
            limit: Maximum keywords to return (default: 50, max: 100)

        Returns:
            Dictionary with similar keywords including:
            - keywords: List with keyword, volume, CPC, difficulty
            - total: Total number of similar keywords found

        Example:
            >>> client = SERankingClient()
            >>> data = await client.get_similar_keywords("credit card processing")
            >>> for kw in data["keywords"]:
            ...     print(f"{kw['keyword']}: {kw['volume']} monthly searches")
        """
        endpoint = "/keywords/similar"
        params = {
            "keyword": keyword,
            "source": source,
            "limit": min(limit, 100)
        }
        return await self._make_request(endpoint, params)

    async def get_related_keywords(
        self,
        keyword: str,
        source: str = "us",
        limit: int = 50
    ) -> dict:
        """
        Find keywords related to a given keyword (broader topic match).

        Args:
            keyword: Seed keyword (e.g., "SEO tools")
            source: Regional database (e.g., "us", "uk")
            limit: Maximum keywords to return (default: 50, max: 100)

        Returns:
            Dictionary with related keywords including:
            - keywords: List with keyword, volume, CPC, difficulty
            - total: Total number of related keywords found
        """
        endpoint = "/keywords/related"
        params = {
            "keyword": keyword,
            "source": source,
            "limit": min(limit, 100)
        }
        return await self._make_request(endpoint, params)

    async def check_subscription(self) -> dict:
        """
        Check current API subscription status and remaining credits.

        Returns:
            Dictionary with subscription info:
            - status: "active" or other
            - start_date: Subscription start
            - expiration_date: Subscription end
            - units_limit: Total API credits
            - units_left: Remaining API credits

        Example:
            >>> client = SERankingClient()
            >>> info = await client.check_subscription()
            >>> print(f"Credits remaining: {info['subscription_info']['units_left']}")
        """
        endpoint = "/account/subscription"
        params = {}
        return await self._make_request(endpoint, params)


# Convenience function for creating client instances
def get_seranking_client() -> SERankingClient:
    """
    Get a SE Ranking API client instance.

    This function explicitly reads the environment variable at call time
    to ensure it's available during async tool execution.

    Returns:
        Initialized SERankingClient
    """
    # Read API key at call time rather than relying on module-level variable
    api_key = os.getenv("SERANKING_API_KEY")

    # If not found, try loading .env again
    if not api_key:
        load_dotenv()
        api_key = os.getenv("SERANKING_API_KEY")

    return SERankingClient(api_key=api_key)
