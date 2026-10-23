"""Deterministic indicators with explicit warm-up and smoothing conventions."""
from collections.abc import Sequence


def sma(values: Sequence[float], period: int) -> list[float | None]:
    if period < 1:
        raise ValueError('period must be positive')
    return [None if i + 1 < period else sum(values[i + 1 - period:i + 1]) / period
            for i in range(len(values))]


def ema(values: Sequence[float], period: int) -> list[float | None]:
    """Seed with the first period's SMA, then use alpha=2/(period+1)."""
    if period < 1:
        raise ValueError('period must be positive')
    result = [None] * len(values)
    if len(values) >= period:
        value = sum(values[:period]) / period
        result[period - 1] = value
        alpha = 2 / (period + 1)
        for i in range(period, len(values)):
            value += alpha * (values[i] - value)
            result[i] = value
    return result


def rsi(values: Sequence[float], period: int = 14) -> list[float | None]:
    """Wilder RSI; flat prices yield 50, only gains 100, only losses 0."""
    if period < 1:
        raise ValueError('period must be positive')
    result = [None] * len(values)
    if len(values) <= period:
        return result
    changes = [b - a for a, b in zip(values, values[1:])]
    gain = sum(max(change, 0) for change in changes[:period]) / period
    loss = sum(max(-change, 0) for change in changes[:period]) / period
    for i in range(period, len(values)):
        if i > period:
            change = changes[i - 1]
            gain = (gain * (period - 1) + max(change, 0)) / period
            loss = (loss * (period - 1) + max(-change, 0)) / period
        result[i] = 50.0 if gain == loss == 0 else (100.0 if loss == 0 else 100 - 100 / (1 + gain / loss))
    return result


def macd(values: Sequence[float]) -> tuple[list, list, list]:
    """MACD(12,26,9), using SMA-seeded EMAs; signal starts at bar 34."""
    fast, slow = ema(values, 12), ema(values, 26)
    line = [None if a is None or b is None else a - b for a, b in zip(fast, slow)]
    signal = [None] * min(25, len(values)) + ema([x for x in line if x is not None], 9)
    histogram = [None if a is None or b is None else a - b for a, b in zip(line, signal)]
    return line, signal, histogram


def crossover(first: Sequence[float | None], second: Sequence[float | None]) -> str:
    """Report an event only when the previous and latest valid bars cross."""
    if len(first) < 2 or len(second) < 2 or any(x is None for x in (*first[-2:], *second[-2:])):
        return 'unavailable'
    previous, current = first[-2] - second[-2], first[-1] - second[-1]
    if previous <= 0 < current:
        return 'bullish_cross'
    if previous >= 0 > current:
        return 'bearish_cross'
    return 'none'
