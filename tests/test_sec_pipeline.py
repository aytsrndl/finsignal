import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agents.sec_filing_agent import analyze_sec_filing, TenKAnalysis, main
from clients.llm import AnalysisError, StructuredLLM
from tools import edgar_parser as edgar


class SECPipelineTests(unittest.TestCase):
    def setUp(self):
        self.facts = json.loads((Path(__file__).parent / 'fixtures/company_facts.json').read_text())
        self.analysis = TenKAnalysis(business_summary='Example company', major_risks=[{
            'title': 'Supplier risk', 'explanation': 'Supplier concentration.',
            'evidence': 'We depend on a single supplier.'}], business_trends=[],
            management_outlook='Not available.', red_flags=[])
        self.llm = Mock(model='test-model')
        self.llm.analyze.return_value = self.analysis
        self.stack = __import__('contextlib').ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(edgar, 'get_cik_from_ticker', return_value='0000000123'))
        self.stack.enter_context(patch.object(edgar, 'get_company_facts', return_value=self.facts))
        self.filings = self.stack.enter_context(patch.object(edgar, 'get_recent_filings', return_value=[{
            'filing_date': '2026-02-01', 'accession_number': '123-26-1', 'primary_document': 'annual.htm'}]))
        self.html = self.stack.enter_context(patch.object(edgar, 'get_filing_html', return_value='<p>We depend on a single supplier.</p>'))

    def test_complete_report_and_prompt(self):
        report = analyze_sec_filing(' test ', llm=self.llm)
        self.assertEqual(report.ticker, 'TEST')
        self.assertAlmostEqual(report.metrics.net_margin.value, .15)
        self.assertEqual(report.source.filing_date, '2026-02-01')
        self.assertEqual(report.model, 'test-model')
        self.assertIn('123261/annual.htm', report.source.document_url)
        payload = json.loads(self.llm.analyze.call_args.kwargs['payload'])
        self.assertEqual(payload['metrics']['roe']['year'], '2025')
        self.assertEqual(payload['filing_text'], 'We depend on a single supplier.')
        self.assertEqual(self.llm.analyze.call_args.kwargs['schema'], TenKAnalysis)
        json.loads(report.model_dump_json())

    def test_missing_metrics_are_explicit(self):
        del self.facts['facts']['us-gaap']['AssetsCurrent']
        report = analyze_sec_filing('TEST', llm=self.llm)
        self.assertIsNone(report.metrics.current_ratio.value)
        self.assertIn('current_ratio is unavailable.', report.warnings)

    def test_no_filing(self):
        self.filings.return_value = []
        with self.assertRaisesRegex(ValueError, 'No 10-K'):
            analyze_sec_filing('TEST', llm=self.llm)
        self.llm.analyze.assert_not_called()

    def test_empty_or_oversized_text_is_not_sent(self):
        for html, limit in [('<script>hidden</script>', 100), ('<p>long filing</p>', 2)]:
            with self.subTest(html=html):
                self.html.return_value = html
                with self.assertRaises(ValueError):
                    analyze_sec_filing('TEST', llm=self.llm, max_filing_chars=limit)
        self.llm.analyze.assert_not_called()

    def test_fabricated_evidence_rejected(self):
        self.analysis.major_risks[0].evidence = 'Invented quotation'
        with self.assertRaises(AnalysisError):
            analyze_sec_filing('TEST', llm=self.llm)

    def test_provider_failure_propagates(self):
        self.llm.analyze.side_effect = TimeoutError('provider unavailable')
        with self.assertRaises(TimeoutError):
            analyze_sec_filing('TEST', llm=self.llm)

    def test_bad_ticker_rejected(self):
        with self.assertRaises(ValueError):
            analyze_sec_filing('not a ticker', llm=self.llm)
        self.llm.analyze.assert_not_called()

    def test_cli_failure_has_nonzero_status_without_provider_details(self):
        from io import StringIO
        with patch('sys.argv', ['sec-agent', 'TEST']), patch('agents.sec_filing_agent.analyze_sec_filing', side_effect=RuntimeError('sensitive request')), patch('sys.stderr', new_callable=StringIO) as stderr:
            self.assertEqual(main(), 1)
            self.assertNotIn('sensitive request', stderr.getvalue())


class StructuredLLMTests(unittest.TestCase):
    def test_structured_request_and_model_override(self):
        client = Mock()
        analysis = TenKAnalysis(business_summary='Business', major_risks=[], business_trends=[], management_outlook='Unknown', red_flags=[])
        client.responses.parse.return_value = SimpleNamespace(status='completed', output_parsed=analysis)
        with patch.dict(os.environ, {'FINSIGNAL_LLM_MODEL': 'configured-model'}):
            llm = StructuredLLM(client=client)
        self.assertEqual(llm.analyze(instructions='Analyze', payload='data', schema=TenKAnalysis), analysis)
        client.responses.parse.assert_called_once_with(model='configured-model', instructions='Analyze', input='data', text_format=TenKAnalysis, store=False)

    def test_refusal_and_incomplete_output_fail(self):
        client = Mock()
        for status, parsed in [('completed', None), ('incomplete', None)]:
            client.responses.parse.return_value = SimpleNamespace(status=status, output_parsed=parsed)
            with self.subTest(status=status), self.assertRaises(AnalysisError):
                StructuredLLM(client=client).analyze(instructions='Analyze', payload='data', schema=TenKAnalysis)

    def test_missing_key_fails_without_creating_client(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}), patch('clients.llm.OpenAI') as constructor:
            with self.assertRaisesRegex(ValueError, 'OPENAI_API_KEY'):
                StructuredLLM().analyze(instructions='Analyze', payload='data', schema=TenKAnalysis)
            constructor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
