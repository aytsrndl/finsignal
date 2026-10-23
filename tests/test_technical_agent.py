from contextlib import redirect_stderr
from datetime import date
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from agents.technical_agent import analyze_technical, main
from test_price_data import make_history


class TechnicalAgentTests(unittest.TestCase):
    def test_complete_report_uses_known_linear_series(self):
        history = make_history()
        with patch('agents.technical_agent.fetch_price_history') as fetch:
            report = analyze_technical('TEST', history=history)
        fetch.assert_not_called()
        self.assertEqual(report.indicators.sma_50, 314.5)
        self.assertEqual(report.indicators.sma_200, 239.5)
        self.assertEqual(report.signals.trend, 'uptrend')
        self.assertEqual(report.signals.moving_average_cross, 'none')
        self.assertEqual(report.indicators.rsi_14, 100)
        self.assertEqual(report.indicators.relative_volume, 1)
        self.assertEqual(report.indicators.trailing_20_session_low, 319)
        self.assertEqual(report.indicators.trailing_20_session_high, 340)
        self.assertEqual(json.loads(report.model_dump_json())['ticker'], 'TEST')

    def test_shared_fetch_occurs_once(self):
        with patch('agents.technical_agent.fetch_price_history', return_value=make_history()) as fetch:
            analyze_technical('TEST')
        fetch.assert_called_once_with('TEST')

    def test_falling_and_flat_price_trends(self):
        falling = analyze_technical('TEST', history=make_history(closes=list(range(339, 99, -1))))
        self.assertEqual(falling.signals.trend, 'downtrend')
        self.assertEqual(falling.signals.rsi_state, 'oversold')
        flat = analyze_technical('TEST', history=make_history(closes=[100] * 240))
        self.assertEqual(flat.signals.trend, 'mixed')
        self.assertEqual(flat.signals.rsi_state, 'neutral')
        self.assertEqual(flat.signals.moving_average_cross, 'none')

    def test_stale_history_is_flagged(self):
        report = analyze_technical('TEST', history=make_history())
        self.assertTrue(any('stale' in warning for warning in report.warnings))

    def test_reported_sma_cross_requires_actual_cross(self):
        # A long flat window followed by a drop keeps SMA50 below SMA200;
        # the final jump is sufficient to cross both averages on that bar.
        history = make_history(closes=[100] * 200 + [99] * 20 + [200])
        report = analyze_technical('TEST', history=history)
        self.assertEqual(report.signals.moving_average_cross, 'bullish_cross')

    def test_volume_baseline_excludes_current_session(self):
        history = make_history(21)
        history.bars[-1] = history.bars[-1].model_copy(update={'volume': 5000})
        report = analyze_technical('TEST', history=history)
        self.assertEqual(report.indicators.prior_20_session_average_volume, 1000)
        self.assertEqual(report.indicators.relative_volume, 5)

    def test_adjusted_prices_prevent_split_signal(self):
        history = make_history(closes=[100] * 100 + [50] * 140)
        history.bars = [bar.model_copy(update={'adjusted_close': 50}) for bar in history.bars]
        report = analyze_technical('TEST', history=history)
        self.assertEqual(report.indicators.sma_200, 50)
        self.assertEqual(report.indicators.rsi_14, 50)
        self.assertEqual(report.signals.sma_50_vs_200, 'equal')

    def test_short_history_returns_nulls_and_warnings(self):
        report = analyze_technical('TEST', history=make_history(1))
        self.assertIsNone(report.indicators.rsi_14)
        self.assertIsNone(report.indicators.macd_signal)
        self.assertIsNone(report.indicators.sma_200)
        self.assertEqual(report.signals.trend, 'unavailable')
        self.assertTrue(report.warnings)

    def test_zero_volume_not_infinite(self):
        history = make_history(21)
        history.bars = [bar.model_copy(update={'volume': 0}) for bar in history.bars]
        report = analyze_technical('TEST', history=history)
        self.assertIsNone(report.indicators.relative_volume)
        self.assertIsNotNone(report.model_dump_json())

    def test_mismatched_symbol_and_future_dates_fail(self):
        with self.assertRaisesRegex(ValueError, 'ticker'):
            analyze_technical('AAPL', history=make_history())
        history = make_history(1)
        history.bars = [history.bars[0].model_copy(update={'date': date(2099, 1, 1)})]
        with self.assertRaisesRegex(ValueError, 'future'):
            analyze_technical('TEST', history=history)

    def test_cli_failure_is_nonzero(self):
        with patch('sys.argv', ['technical', 'TEST']), patch('agents.technical_agent.fetch_price_history', side_effect=ValueError('No prices')), redirect_stderr(StringIO()) as stderr:
            self.assertEqual(main(), 1)
            self.assertIn('No prices', stderr.getvalue())

    def test_offline_cli_emits_json(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, '-m', 'agents.technical_agent', 'TEST',
                                 '--prices-csv', 'tests/fixtures/prices.csv', '--currency', 'USD'],
                                cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['sessions'], 240)
        self.assertEqual(report['signals']['trend'], 'uptrend')
