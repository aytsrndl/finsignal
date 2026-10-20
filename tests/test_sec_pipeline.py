import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agents.sec_filing_agent import analyze_sec_filing, TenKAnalysis, CitedTenKAnalysis, filing_passages, main
from clients.llm import AnalysisError, StructuredLLM
from tools import edgar_parser as edgar


class SECPipelineTests(unittest.TestCase):
    def setUp(self):
        self.facts = json.loads((Path(__file__).parent / 'fixtures/company_facts.json').read_text())
        self.analysis = CitedTenKAnalysis(business_summary='Example company', major_risks=[{
            'title': 'Supplier risk', 'explanation': 'Supplier concentration.',
            'evidence_id': 'P000001'}], business_trends=[],
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
        self.assertEqual(payload['filing_passages'], {'P000001': 'We depend on a single supplier.'})
        self.assertEqual(self.llm.analyze.call_args.kwargs['schema'], CitedTenKAnalysis)
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

    def test_unknown_passage_id_rejected(self):
        self.analysis.major_risks[0].evidence_id = 'Invented quotation'
        with self.assertRaisesRegex(AnalysisError, 'after one corrective retry'):
            analyze_sec_filing('TEST', llm=self.llm)
        self.assertEqual(self.llm.analyze.call_count, 2)

    def test_citation_can_be_corrected_once(self):
        invalid = self.analysis.model_copy(deep=True)
        invalid.major_risks[0].evidence_id = 'A paraphrase, not a quote.'
        self.llm.analyze.side_effect = [invalid, self.analysis]
        report = analyze_sec_filing('TEST', llm=self.llm)
        self.assertEqual(report.analysis.major_risks[0].evidence, 'We depend on a single supplier.')
        self.assertIn('Citation validation required one corrective LLM retry.', report.warnings)
        repair = json.loads(self.llm.analyze.call_args.kwargs['payload'])
        self.assertEqual(repair['validation_feedback']['unsupported_fields'], ['major_risks[0].evidence_id'])
        self.assertEqual(repair['filing_passages'], {'P000001': 'We depend on a single supplier.'})

    def test_valid_evidence_needs_no_retry(self):
        analyze_sec_filing('TEST', llm=self.llm)
        self.llm.analyze.assert_called_once()

    def test_evidence_is_copied_from_selected_passage(self):
        source = ('Financial information. ' * 80) + 'Supplier’s terms include “advance payment”.'
        self.html.return_value = f'<p>{source}</p>'
        passages = filing_passages(edgar.html_to_text(self.html.return_value))
        selected_id = list(passages)[-1]
        self.analysis.major_risks[0].evidence_id = selected_id
        report = analyze_sec_filing('TEST', llm=self.llm)
        self.assertEqual(report.analysis.major_risks[0].evidence, passages[selected_id])
        self.assertIn('“advance payment”', report.analysis.major_risks[0].evidence)
        self.assertNotIn('evidence_id', report.analysis.major_risks[0].model_dump())

    def test_retry_checks_all_finding_categories(self):
        invalid = self.analysis.model_copy(deep=True)
        invalid.major_risks[0].evidence_id = 'Invented quote'
        corrected = self.analysis.model_copy(deep=True)
        corrected.red_flags = [invalid.major_risks[0]]
        self.llm.analyze.side_effect = [invalid, corrected]
        with self.assertRaisesRegex(AnalysisError, r'red_flags\[0\]'):
            analyze_sec_filing('TEST', llm=self.llm)
        self.assertEqual(self.llm.analyze.call_count, 2)

    def test_unsupported_finding_can_be_removed(self):
        invalid = self.analysis.model_copy(deep=True)
        invalid.major_risks[0].evidence_id = ''
        corrected = self.analysis.model_copy(deep=True)
        corrected.major_risks = []
        self.llm.analyze.side_effect = [invalid, corrected]
        self.assertEqual(analyze_sec_filing('TEST', llm=self.llm).analysis.major_risks, [])

    def test_cli_shows_evidence_error(self):
        from io import StringIO
        with patch('sys.argv', ['sec-agent', 'TEST']), patch('agents.sec_filing_agent.analyze_sec_filing', side_effect=AnalysisError('Evidence still does not match.')), patch('sys.stderr', new_callable=StringIO) as stderr:
            self.assertEqual(main(), 1)
            self.assertIn('Evidence still does not match.', stderr.getvalue())

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


class FilingPassageTests(unittest.TestCase):
    def test_passages_preserve_entire_source(self):
        for source in ('', 'One short sentence.', 'word ' * 900, 'x' * 2501,
                       'Supplier’s terms\ninclude “advance payment”. ' * 100):
            with self.subTest(length=len(source)):
                passages = filing_passages(source)
                self.assertEqual(''.join(passages.values()), source)
                self.assertTrue(all(0 < len(value) <= 1200 for value in passages.values()))
                self.assertEqual(passages, filing_passages(source))

    def test_invalid_size_rejected(self):
        with self.assertRaises(ValueError):
            filing_passages('source', max_chars=0)


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
