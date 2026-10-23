"""Technical analysis from shared, validated daily prices; no LLM required."""
from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict

from tools.price_data import PriceHistory, fetch_price_history, load_price_csv, normalize_ticker
from tools.technical_indicators import crossover, macd, rsi, sma

Relation = Literal['above', 'below', 'equal', 'unavailable']
Cross = Literal['bullish_cross', 'bearish_cross', 'none', 'unavailable']


class TechnicalIndicators(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    rsi_14: float | None
    sma_50: float | None
    sma_200: float | None
    macd: float | None
    macd_signal: float | None
    macd_histogram: float | None
    prior_20_session_average_volume: float | None
    relative_volume: float | None
    trailing_20_session_low: float | None
    trailing_20_session_high: float | None


class TechnicalSignals(BaseModel):
    rsi_state: Literal['overbought', 'oversold', 'neutral', 'unavailable']
    price_vs_sma_50: Relation
    price_vs_sma_200: Relation
    sma_50_vs_200: Relation
    moving_average_cross: Cross
    macd_vs_signal: Relation
    macd_cross: Cross
    trend: Literal['uptrend', 'downtrend', 'mixed', 'unavailable']


class TechnicalReport(BaseModel):
    ticker: str
    as_of: date
    retrieved_at: datetime
    generated_at: datetime
    provider: str
    provider_symbol: str
    source: str
    currency: str | None
    sessions: int
    history_start: date
    latest_close: float
    latest_adjusted_close: float
    price_basis: Literal['adjusted_for_splits_and_dividends']
    indicators: TechnicalIndicators
    signals: TechnicalSignals
    interpretation: str
    warnings: list[str]


def relation(first, second) -> Relation:
    if first is None or second is None:
        return 'unavailable'
    return 'above' if first > second else 'below' if first < second else 'equal'


def analyze_technical(ticker: str, *, history: PriceHistory | None = None) -> TechnicalReport:
    """Fetch history once or reuse it across agents by passing history explicitly."""
    ticker = normalize_ticker(ticker)
    history = fetch_price_history(ticker) if history is None else history
    # Revalidate shared inputs in case a caller mutated a model after construction.
    history = PriceHistory.model_validate(history.model_dump())
    if normalize_ticker(history.ticker) != ticker:
        raise ValueError('Price history ticker does not match the requested ticker.')
    bars = history.bars
    closes = [bar.adjusted_close for bar in bars]
    ma50, ma200 = sma(closes, 50), sma(closes, 200)
    line, signal, histogram = macd(closes)
    momentum = rsi(closes)[-1]
    prior_volume = sum(bar.volume for bar in bars[-21:-1]) / 20 if len(bars) >= 21 else None
    relative_volume = bars[-1].volume / prior_volume if prior_volume is not None and prior_volume > 0 else None
    lows = [bar.low * bar.adjusted_close / bar.close for bar in bars[-20:]]
    highs = [bar.high * bar.adjusted_close / bar.close for bar in bars[-20:]]
    indicators = TechnicalIndicators(
        rsi_14=momentum, sma_50=ma50[-1], sma_200=ma200[-1], macd=line[-1],
        macd_signal=signal[-1], macd_histogram=histogram[-1],
        prior_20_session_average_volume=prior_volume, relative_volume=relative_volume,
        trailing_20_session_low=min(lows) if len(bars) >= 20 else None,
        trailing_20_session_high=max(highs) if len(bars) >= 20 else None,
    )
    price_short, price_long = relation(closes[-1], ma50[-1]), relation(closes[-1], ma200[-1])
    ma_relation = relation(ma50[-1], ma200[-1])
    trend = ('unavailable' if 'unavailable' in (price_short, price_long, ma_relation)
             else 'uptrend' if price_short == price_long == ma_relation == 'above'
             else 'downtrend' if price_short == price_long == ma_relation == 'below' else 'mixed')
    signals = TechnicalSignals(
        rsi_state='unavailable' if momentum is None else 'overbought' if momentum >= 70 else 'oversold' if momentum <= 30 else 'neutral',
        price_vs_sma_50=price_short, price_vs_sma_200=price_long, sma_50_vs_200=ma_relation,
        moving_average_cross=crossover(ma50, ma200), macd_vs_signal=relation(line[-1], signal[-1]),
        macd_cross=crossover(line, signal), trend=trend,
    )
    warnings = list(history.warnings)
    if len(bars) < 200:
        warnings.append('Fewer than 200 sessions: long-term moving-average trend is unavailable.')
    if len(bars) < 34:
        warnings.append('Fewer than 34 sessions: MACD signal and histogram are unavailable.')
    if len(bars) < 15:
        warnings.append('Fewer than 15 sessions: RSI(14) is unavailable.')
    if relative_volume is None:
        warnings.append('Relative volume requires 20 prior sessions with positive average volume.')
    if len(bars) < 20:
        warnings.append('Fewer than 20 sessions: trailing range is unavailable.')
    if history.currency is None:
        warnings.append('Price currency is unavailable.')
    today = datetime.now(timezone.utc).date()
    if bars[-1].date > today:
        raise ValueError('Price history contains future dates.')
    if (today - bars[-1].date).days > 7:
        warnings.append('Latest price is more than seven calendar days old; signals may be stale.')
    warnings.append('Technical signals describe historical prices and are not calibrated return forecasts.')
    interpretation = (
        f'Moving-average trend: {trend}. RSI(14): {signals.rsi_state}. '
        f'MACD is {signals.macd_vs_signal} its signal line; latest crossover: {signals.macd_cross}. '
        'Overbought/oversold readings alone do not establish a reversal. '
        'Trailing high/low levels describe the observed range, not guaranteed support or resistance.'
    )
    return TechnicalReport(
        ticker=ticker, as_of=bars[-1].date, retrieved_at=history.retrieved_at,
        generated_at=datetime.now(timezone.utc), provider=history.provider,
        provider_symbol=history.provider_symbol, source=history.source, currency=history.currency,
        sessions=len(bars), history_start=bars[0].date, latest_close=bars[-1].close,
        latest_adjusted_close=closes[-1], price_basis='adjusted_for_splits_and_dividends',
        indicators=indicators, signals=signals, interpretation=interpretation, warnings=warnings,
    )


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Daily technical indicators and signals as JSON.')
    parser.add_argument('ticker')
    parser.add_argument('--prices-csv', help='Use offline daily OHLCV data instead of Yahoo Finance.')
    parser.add_argument('--currency', help='Currency label for supplied CSV data.')
    args = parser.parse_args()
    if args.currency and not args.prices_csv:
        parser.error('--currency is only used with --prices-csv')
    try:
        history = load_price_csv(args.prices_csv, args.ticker, currency=args.currency) if args.prices_csv else None
        report = analyze_technical(args.ticker, history=history)
    except (ValueError, OSError) as error:
        print(f'Technical analysis failed: {error}', file=sys.stderr)
        return 1
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
