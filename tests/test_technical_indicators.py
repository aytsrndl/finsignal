import unittest

from tools.technical_indicators import sma, ema, rsi, macd, crossover


class IndicatorTests(unittest.TestCase):
    def test_sma_values_and_warmup(self):
        self.assertEqual(sma([1, 2, 3, 4, 5], 3), [None, None, 2, 3, 4])

    def test_ema_is_sma_seeded(self):
        self.assertEqual(ema([2, 4, 6, 10, 12], 3), [None, None, 4, 7, 9.5])
        self.assertEqual(ema([1, 2], 3), [None, None])

    def test_wilder_rsi_known_example(self):
        # Published Wilder worksheet sequence; first RSI is 70.4641.
        closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
                  45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00]
        result = rsi(closes)
        self.assertTrue(all(value is None for value in result[:14]))
        self.assertAlmostEqual(result[14], 70.464135, places=5)
        self.assertAlmostEqual(result[15], 66.249619, places=5)

    def test_rsi_flat_rising_and_falling(self):
        self.assertEqual(rsi([10] * 16)[-1], 50)
        self.assertEqual(rsi(list(range(1, 17)))[-1], 100)
        self.assertEqual(rsi(list(range(16, 0, -1)))[-1], 0)
        self.assertIsNone(rsi([10] * 14)[-1])

    def test_macd_linear_prices_and_warmup(self):
        # For a linear ramp, the seeded EMA difference is exactly 7.
        line, signal, histogram = macd(list(range(1, 41)))
        self.assertIsNone(line[24])
        self.assertAlmostEqual(line[25], 7)
        self.assertIsNone(signal[32])
        self.assertAlmostEqual(signal[33], 7)
        self.assertAlmostEqual(histogram[-1], 0)
        for size in (1, 25, 26, 33):
            parts = macd([10] * size)
            self.assertTrue(all(len(part) == size for part in parts))
            self.assertIsNone(parts[1][-1])

    def test_crossovers_are_events_not_positions(self):
        self.assertEqual(crossover([1, 3], [2, 2]), 'bullish_cross')
        self.assertEqual(crossover([3, 1], [2, 2]), 'bearish_cross')
        self.assertEqual(crossover([3, 4], [2, 2]), 'none')
        self.assertEqual(crossover([2, 2], [2, 2]), 'none')
        self.assertEqual(crossover([None, 3], [2, 2]), 'unavailable')

    def test_invalid_periods(self):
        for function in (sma, ema, rsi):
            with self.subTest(function=function.__name__), self.assertRaises(ValueError):
                function([1, 2], 0)
