"""Finnhub news retrieval and explicit offline input for sentiment analysis."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, field_validator
import requests

from tools.price_data import normalize_ticker


class NewsDataError(ValueError):
    """News could not be obtained or validated."""


class NewsArticle(BaseModel):
    model_config = ConfigDict(frozen=True)
    article_id: str = Field(min_length=1)
    headline: str = Field(min_length=1)
    summary: str = ''
    source: str = Field(min_length=1)
    url: HttpUrl
    published_at: AwareDatetime

    @field_validator('headline', 'source', 'article_id')
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError('Article fields must not be blank.')
        return value.strip()


class NewsBatch(BaseModel):
    ticker: str
    provider: str
    retrieved_at: AwareDatetime
    articles: list[NewsArticle]
    warnings: list[str] = Field(default_factory=list)


def fetch_company_news(ticker: str, *, as_of: datetime, days: int = 7) -> NewsBatch:
    ticker = normalize_ticker(ticker)
    if not 1 <= days <= 30 or as_of.tzinfo is None:
        raise NewsDataError('Use a timezone-aware as_of and a window of 1 to 30 days.')
    key = os.environ.get('FINNHUB_API_KEY', '').strip()
    if not key:
        raise NewsDataError('Set FINNHUB_API_KEY in .env before requesting company news.')
    as_of = as_of.astimezone(timezone.utc)
    start = as_of - timedelta(days=days)
    try:
        response = requests.get(
            'https://finnhub.io/api/v1/company-news',
            params={'symbol': ticker, 'from': start.date().isoformat(),
                    'to': as_of.date().isoformat()},
            headers={'X-Finnhub-Token': key}, timeout=30,
        )
        if response.status_code in (401, 403):
            raise NewsDataError('Finnhub denied news access; check your key and plan permissions.')
        if response.status_code == 429:
            raise NewsDataError('Finnhub rate limit reached; retry after the provider limit resets.')
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise NewsDataError('Finnhub did not return a news list.')
        articles = [NewsArticle(
            article_id=str(item['id']), headline=item['headline'], summary=item.get('summary') or '',
            source=item['source'], url=item['url'],
            published_at=datetime.fromtimestamp(item['datetime'], timezone.utc),
        ) for item in payload]
    except NewsDataError:
        raise
    except (requests.RequestException, ValueError, KeyError, TypeError, OverflowError) as error:
        # Never include provider response bodies or token-bearing headers in errors.
        raise NewsDataError(f'Could not retrieve valid Finnhub news ({type(error).__name__}).') from error
    return NewsBatch(ticker=ticker, provider='finnhub', retrieved_at=datetime.now(timezone.utc),
                     articles=articles, warnings=['Coverage is limited to news returned by Finnhub; it is not an exhaustive news archive.'])


def load_news_json(path: str | Path, ticker: str) -> NewsBatch:
    """Load a NewsBatch with explicit ticker identity; never silently relabel news."""
    batch = NewsBatch.model_validate(json.loads(Path(path).read_text(encoding='utf-8-sig')))
    if normalize_ticker(batch.ticker) != normalize_ticker(ticker):
        raise NewsDataError('News file ticker does not match the requested ticker.')
    return batch.model_copy(update={'warnings': batch.warnings + ['News file is user supplied; article relevance and provenance are not independently verified.']})
