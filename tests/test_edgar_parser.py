import importlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import requests
from tools import edgar_parser as parser


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.facts = json.loads((Path(__file__).parent / 'fixtures/company_facts.json').read_text())

    def test_import_does_not_make_requests(self):
        with patch.object(parser.requests, 'get') as get:
            importlib.reload(parser)
        get.assert_not_called()

    def test_missing_identity_fails_before_network(self):
        for value in ['', '   ']:
            with self.subTest(value=value), patch.dict(os.environ, {'SEC_USER_AGENT': value}), patch.object(parser.requests, 'get') as get:
                with self.assertRaisesRegex(ValueError, 'SEC_USER_AGENT'):
                    parser.get_cik_from_ticker('TEST')
                get.assert_not_called()

    def test_requests_use_identity_and_timeout(self):
        response = Mock()
        response.json.return_value = {'0': {'ticker': 'TEST', 'cik_str': 123}}
        with patch.dict(os.environ, {'SEC_USER_AGENT': 'Test Person test@example.com'}), patch.object(parser.requests, 'get', return_value=response) as get:
            self.assertEqual(parser.get_cik_from_ticker('test'), '0000000123')
        get.assert_called_once_with(parser.TICKERS_URL, headers={'User-Agent': 'Test Person test@example.com'}, timeout=30)
        response.raise_for_status.assert_called_once_with()

    def test_http_errors_propagate(self):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError('403')
        with patch.dict(os.environ, {'SEC_USER_AGENT': 'Test Person test@example.com'}), patch.object(parser.requests, 'get', return_value=response):
            with self.assertRaises(requests.HTTPError):
                parser.get_company_facts('0000000123')

    def test_seven_metrics_pipeline(self):
        ticker_response, facts_response = Mock(), Mock()
        ticker_response.json.return_value = {'0': {'ticker': 'TEST', 'cik_str': 123}}
        facts_response.json.return_value = self.facts
        with patch.dict(os.environ, {'SEC_USER_AGENT': 'Test Person test@example.com'}), patch.object(parser.requests, 'get', side_effect=[ticker_response, facts_response]) as get:
            result = parser.build_sec_metrics('TEST')
        self.assertEqual(get.call_count, 2)
        self.assertIn('CIK0000000123.json', get.call_args.args[0])
        self.assertEqual(result['entity_name'], 'Fixture Company')
        expected = {'revenue_growth': .2, 'net_income_growth': .8, 'net_margin': .15, 'debt_to_equity': 1., 'debt_to_assets': .5, 'current_ratio': 2., 'roe': .15}
        for key, value in expected.items():
            with self.subTest(metric=key):
                self.assertAlmostEqual(result[key]['value'], value)
                self.assertEqual(result[key]['year'], '2025')

    def test_annual_values_filter_sort_and_deduplicate(self):
        entries = self.facts['facts']['us-gaap']['Revenues']['units']['USD']
        entries.reverse()
        entries.extend([
            {'end': '2025-06-30', 'val': 40, 'fp': 'Q2', 'form': '10-Q'},
            {'end': '2025-12-31', 'val': 120, 'fp': 'FY', 'form': '10-K'},
        ])
        self.assertEqual(parser.get_annual_values(self.facts, 'Revenues'), [
            {'year': '2024', 'date': '2024-12-31', 'value': 100},
            {'year': '2025', 'date': '2025-12-31', 'value': 120},
        ])

    def test_tag_fallbacks(self):
        gaap = self.facts['facts']['us-gaap']
        gaap['ProfitLoss'] = gaap.pop('NetIncomeLoss')
        self.assertEqual(parser.get_net_income(self.facts)[-1]['value'], 18)
        self.assertEqual(parser.get_revenue(self.facts)[-1]['value'], 120)

    def test_missing_metrics_return_null(self):
        del self.facts['facts']['us-gaap']['AssetsCurrent']
        with patch.object(parser, 'get_cik_from_ticker', return_value='0000000123'), patch.object(parser, 'get_company_facts', return_value=self.facts):
            result = parser.build_sec_metrics('TEST')
        self.assertEqual(result['current_ratio'], {'value': None, 'year': None})
        self.assertAlmostEqual(result['net_margin']['value'], .15)

    def test_zero_denominators_are_skipped(self):
        self.facts['facts']['us-gaap']['LiabilitiesCurrent']['units']['USD'][0]['val'] = 0
        self.assertEqual(parser.calculate_current_ratio(self.facts), [])
        growth = parser.calculate_growth([{'year': '2024', 'date': '2024-12-31', 'value': 0}, {'year': '2025', 'date': '2025-12-31', 'value': 10}])
        self.assertIsNone(growth[-1]['growth'])

    def test_html_extraction(self):
        self.assertEqual(parser.html_to_text('<style>hidden</style><script>hidden</script><p>Revenue grew.</p>'), 'Revenue grew.')

    def test_latest_filing_download(self):
        responses = [Mock(), Mock()]
        responses[0].json.return_value = {'filings': {'recent': {
            'form': ['10-Q', '10-K'], 'filingDate': ['2026-05-01', '2026-02-01'],
            'accessionNumber': ['123-26-2', '123-26-1'], 'primaryDocument': ['quarter.htm', 'annual.htm']}}}
        responses[1].text = '<p>Annual filing</p>'
        with patch.dict(os.environ, {'SEC_USER_AGENT': 'Test Person test@example.com'}), patch.object(parser.requests, 'get', side_effect=responses) as get:
            self.assertEqual(parser.get_latest_10k_html('0000000123'), '<p>Annual filing</p>')
        self.assertEqual(get.call_args.args[0], 'https://www.sec.gov/Archives/edgar/data/123/123261/annual.htm')

    def test_no_annual_filing(self):
        with patch.object(parser, 'get_recent_filings', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'No 10-K'):
                parser.get_latest_10k_html('0000000123')


if __name__ == '__main__':
    unittest.main()
