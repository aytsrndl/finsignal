from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

from agents.sentiment_agent import analyze_sentiment, SentimentCommentary
from tools.finbert_scorer import SentimentScore
from tools.news_feed import NewsArticle, NewsBatch, load_news_json
from test_news_feed import NOW


def article(number, **changes):
    return NewsArticle(**({'article_id': str(number), 'headline': f'Company story {number}', 'summary': 'Summary',
                          'source': 'Fixture', 'url': f'https://example.com/{number}', 'published_at': NOW - timedelta(hours=number)} | changes))


def batch(articles):
    return NewsBatch(ticker='TEST', provider='fixture', retrieved_at=NOW, articles=articles)


def scorer(scores):
    result = Mock(model_name='mock-test-scorer', resolved_revision='test')
    result.score.return_value = [SentimentScore(positive=p, negative=n, neutral=1 - p - n, truncated=False) for p, n in scores]
    return result


class SentimentAgentTests(unittest.TestCase):
    def test_aggregate_is_equal_weight_positive_minus_negative(self):
        report = analyze_sentiment('TEST', news=batch([article(1), article(2)]), as_of=NOW, scorer=scorer([(.8, .1), (.2, .6)]))
        self.assertAlmostEqual(report.finbert_score, .15)
        self.assertEqual(report.positive_count, 1)
        self.assertEqual(report.negative_count, 1)
        self.assertEqual(report.article_count, 2)
        self.assertEqual(report.status, 'complete')
        json.loads(report.model_dump_json())

    def test_no_news_is_unavailable_without_scoring(self):
        model = scorer([])
        report = analyze_sentiment('TEST', news=batch([]), as_of=NOW, scorer=model)
        self.assertEqual(report.status, 'no_news')
        self.assertIsNone(report.finbert_score)
        self.assertEqual(report.signal, 'unavailable')
        model.score.assert_not_called()

    def test_deduplication_and_time_window(self):
        articles = [article(1), article(2, url='https://example.com/1?utm_source=tracker'),
                    article(3, headline='COMPANY STORY 1'), article(4, published_at=NOW + timedelta(hours=1)),
                    article(5, published_at=NOW - timedelta(days=8))]
        report = analyze_sentiment('TEST', news=batch(articles), as_of=NOW, scorer=scorer([(.7, .1)]))
        self.assertEqual(report.article_count, 1)
        self.assertEqual(report.duplicate_count, 2)
        self.assertEqual(report.filtered_count, 2)

    def test_newest_articles_are_selected_before_limit(self):
        model = scorer([(.1, .8)])
        report = analyze_sentiment('TEST', news=batch([article(3), article(1), article(2)]), as_of=NOW, max_articles=1, scorer=model)
        self.assertEqual(report.articles[0].article.article_id, '1')
        self.assertEqual(report.omitted_by_limit, 2)
        self.assertEqual(report.signal, 'bearish')

    def test_neutral_class_is_not_no_news(self):
        report = analyze_sentiment('TEST', news=batch([article(1)]), as_of=NOW, scorer=scorer([(.1, .1)]))
        self.assertEqual(report.signal, 'neutral')
        self.assertEqual(report.neutral_count, 1)
        self.assertEqual(report.status, 'complete')

    def test_html_is_cleaned_before_scoring(self):
        model = scorer([(.6, .1)])
        analyze_sentiment('TEST', news=batch([article(1, summary='<script>ignore</script><p>Real news</p>')]), as_of=NOW, scorer=model)
        text = model.score.call_args.args[0][0]
        self.assertIn('Real news', text)
        self.assertNotIn('ignore', text)

    def test_wrong_symbol_and_wrong_score_count_fail(self):
        with self.assertRaisesRegex(ValueError, 'ticker'):
            analyze_sentiment('AAPL', news=batch([article(1)]), as_of=NOW)
        with self.assertRaisesRegex(ValueError, 'count'):
            analyze_sentiment('TEST', news=batch([article(1)]), as_of=NOW, scorer=scorer([]))

    def test_optional_commentary_requires_known_article_ids(self):
        llm = Mock(model='mock-llm')
        llm.analyze.return_value = SentimentCommentary(summary='Positive tone', evidence_article_ids=['1'], caveats=['Sparse coverage'])
        report = analyze_sentiment('TEST', news=batch([article(1)]), as_of=NOW, scorer=scorer([(.8, .1)]), llm=llm)
        self.assertEqual(report.commentary_model, 'mock-llm')
        llm.analyze.return_value.evidence_article_ids = ['invented']
        with self.assertRaisesRegex(ValueError, 'unknown article'):
            analyze_sentiment('TEST', news=batch([article(1)]), as_of=NOW, scorer=scorer([(.8, .1)]), llm=llm)

    def test_empty_news_cli_needs_neither_news_key_nor_model_download(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'empty.json'
            path.write_text(batch([]).model_dump_json(), encoding='utf-8')
            run = subprocess.run([sys.executable, '-m', 'agents.sentiment_agent', 'TEST', '--news-json', str(path)], cwd=root, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)['status'], 'no_news')
            with self.assertRaisesRegex(ValueError, 'ticker'):
                load_news_json(path, 'AAPL')
