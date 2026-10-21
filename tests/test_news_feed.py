from datetime import datetime, timezone
import os
import unittest
from unittest.mock import Mock, patch

from tools.news_feed import fetch_company_news, NewsDataError

NOW = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)


class NewsFeedTests(unittest.TestCase):
    def test_missing_key_fails_before_http(self):
        with patch.dict(os.environ, {'FINNHUB_API_KEY': ''}), patch('tools.news_feed.requests.get') as get:
            with self.assertRaisesRegex(NewsDataError, 'FINNHUB_API_KEY'):
                fetch_company_news('AAPL', as_of=NOW)
            get.assert_not_called()

    def test_header_auth_and_timestamp_parsing(self):
        response = Mock(status_code=200)
        response.json.return_value = [{'id': 123, 'headline': 'Revenue increases', 'source': 'Example',
                                       'url': 'https://example.com/news/123', 'datetime': NOW.timestamp(), 'summary': None}]
        with patch.dict(os.environ, {'FINNHUB_API_KEY': 'test-only-not-a-real-key'}), patch('tools.news_feed.requests.get', return_value=response) as get:
            batch = fetch_company_news(' brk.b ', as_of=NOW)
        self.assertEqual(batch.ticker, 'BRK.B')
        self.assertEqual(batch.articles[0].article_id, '123')
        self.assertEqual(batch.articles[0].published_at, NOW)
        self.assertEqual(batch.articles[0].summary, '')
        kwargs = get.call_args.kwargs
        self.assertEqual(kwargs['headers']['X-Finnhub-Token'], 'test-only-not-a-real-key')
        self.assertNotIn('token', kwargs['params'])
        self.assertEqual(kwargs['params']['symbol'], 'BRK.B')
        self.assertEqual(kwargs['timeout'], 30)

    def test_denials_and_malformed_responses_fail(self):
        for status, payload in [(401, []), (403, []), (429, []), (200, {'error': 'invalid'}), (200, [{}])]:
            response = Mock(status_code=status)
            response.json.return_value = payload
            with self.subTest(status=status, payload=payload), patch.dict(os.environ, {'FINNHUB_API_KEY': 'secret-value'}), patch('tools.news_feed.requests.get', return_value=response):
                with self.assertRaises(NewsDataError) as raised:
                    fetch_company_news('AAPL', as_of=NOW)
                self.assertNotIn('secret-value', str(raised.exception))

    def test_empty_provider_response_is_valid_news_batch(self):
        response = Mock(status_code=200)
        response.json.return_value = []
        with patch.dict(os.environ, {'FINNHUB_API_KEY': 'test-key'}), patch('tools.news_feed.requests.get', return_value=response):
            self.assertEqual(fetch_company_news('AAPL', as_of=NOW).articles, [])
