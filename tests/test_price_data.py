from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import pandas as pd
from pydantic import ValidationError

from tools.price_data import PriceBar, PriceHistory, PriceDataError, fetch_price_history, load_price_csv


def make_history(size=240, *, ticker='TEST', closes=None):
    closes = closes if closes is not None else list(range(100, 100 + size))
    first = date(2025, 1, 1)
    bars = [PriceBar(date=first + timedelta(days=i), open=value, high=value + 1,
                     low=value - 1, close=value, adjusted_close=value, volume=1000)
            for i, value in enumerate(closes)]
    return PriceHistory(ticker=ticker, provider='csv', provider_symbol=ticker, source='synthetic',
                        currency='USD', retrieved_at=datetime.now(timezone.utc), bars=bars)


class PriceDataTests(unittest.TestCase):
    def test_invalid_ohlcv_rejected(self):
        baseline = make_history(1).bars[0].model_dump()
        for changes in ({'close': float('nan')}, {'open': float('inf')}, {'high': 50},
                        {'low': 150}, {'volume': -1}, {'volume': 1.5}, {'adjusted_close': 0}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                PriceBar(**(baseline | changes))

    def test_duplicate_or_reversed_dates_rejected(self):
        history = make_history(2).model_dump()
        for bars in ([history['bars'][0]] * 2, list(reversed(history['bars']))):
            with self.subTest(bars=bars), self.assertRaises(ValidationError):
                PriceHistory.model_validate(history | {'bars': bars})

    def provider(self):
        today = pd.Timestamp.now(tz='America/New_York').normalize()
        index = pd.DatetimeIndex([today - pd.Timedelta(days=1), today])
        frame = pd.DataFrame({'Open': [100, 101], 'High': [102, 103], 'Low': [99, 100],
                              'Close': [101, 102], 'Adj Close': [50.5, 51], 'Volume': [1000, 2000]}, index=index)
        instrument = Mock(history_metadata={'currency': 'USD'})
        instrument.history.return_value = frame
        return instrument

    def test_provider_preserves_adjustments_and_excludes_today(self):
        instrument = self.provider()
        with patch('yfinance.Ticker', return_value=instrument) as ticker, patch('yfinance.set_tz_cache_location'):
            history = fetch_price_history(' brk.b ')
        ticker.assert_called_once_with('BRK-B')
        self.assertEqual(history.ticker, 'BRK.B')
        self.assertEqual(len(history.bars), 1)
        self.assertEqual(history.bars[0].close, 101)
        self.assertEqual(history.bars[0].adjusted_close, 50.5)
        self.assertEqual(history.currency, 'USD')
        self.assertTrue(history.warnings)
        self.assertTrue(instrument.history.call_args.kwargs['keepna'])
        self.assertFalse(instrument.history.call_args.kwargs['auto_adjust'])

    def test_empty_missing_and_nonfinite_provider_data_fail(self):
        instrument = self.provider()
        original = instrument.history.return_value
        for frame in (original.iloc[:0], original.drop(columns=['Adj Close']),
                      original.assign(Close=float('nan')), original.reset_index(drop=True)):
            instrument.history.return_value = frame
            with self.subTest(columns=list(frame.columns)), patch('yfinance.Ticker', return_value=instrument), patch('yfinance.set_tz_cache_location'), self.assertRaises(PriceDataError):
                fetch_price_history('TEST')

    def test_provider_error_is_not_a_report(self):
        with patch('yfinance.Ticker', side_effect=TimeoutError('connection timed out')), patch('yfinance.set_tz_cache_location'):
            with self.assertRaisesRegex(PriceDataError, 'TimeoutError'):
                fetch_price_history('TEST')

    def test_bad_ticker_fails_before_provider_call(self):
        with patch('yfinance.Ticker') as ticker, self.assertRaises(PriceDataError):
            fetch_price_history('AAPL MSFT')
        ticker.assert_not_called()

    def test_csv_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prices.csv'
            path.write_text('date,open,high,low,close,adjusted_close,volume\n2025-01-01,10,12,9,11,5.5,100\n')
            history = load_price_csv(path, 'TEST', currency='USD')
            self.assertEqual(history.bars[0].adjusted_close, 5.5)
            path.write_text('date,close\n2025-01-01,11\n')
            with self.assertRaises(PriceDataError):
                load_price_csv(path, 'TEST')
