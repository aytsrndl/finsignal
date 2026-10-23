"""Validated daily OHLCV history shared by technical and risk agents."""
import csv
from datetime import date, datetime, timezone
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tools.price_http import price_session


class PriceDataError(ValueError):
    """Price history is missing, malformed, or unavailable."""


def normalize_ticker(ticker: str) -> str:
    symbol = ticker.strip().upper()
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,14}', symbol):
        raise PriceDataError('Provide one stock ticker, such as AAPL or BRK.B.')
    return symbol


class PriceBar(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)
    date: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    adjusted_close: float = Field(gt=0)
    volume: int = Field(ge=0)

    @model_validator(mode='after')
    def check_range(self):
        if self.high < max(self.open, self.close, self.low) or self.low > min(self.open, self.close):
            raise ValueError('OHLC prices are inconsistent with the daily high/low.')
        return self


class PriceHistory(BaseModel):
    ticker: str
    provider: Literal['yahoo_finance', 'csv']
    provider_symbol: str
    source: str
    retrieved_at: datetime
    currency: str | None = None
    bars: list[PriceBar] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def check_dates(self):
        dates = [bar.date for bar in self.bars]
        if any(a >= b for a, b in zip(dates, dates[1:])):
            raise ValueError('Price dates must be unique and strictly increasing.')
        return self


def fetch_price_history(ticker: str) -> PriceHistory:
    """Fetch two years of daily data, excluding the exchange's current date.

    Current-date bars are conservatively omitted even after close, so partial
    sessions cannot enter calculations. No missing OHLCV values are filled.
    """
    import pandas as pd
    import yfinance as yf

    symbol = normalize_ticker(ticker)
    provider_symbol = symbol.replace('.', '-')
    try:
        cache_dir = Path(__file__).resolve().parents[1] / 'data' / 'cache' / 'yfinance'
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir))
        with price_session() as session:
            instrument = yf.Ticker(provider_symbol, **({'session': session} if session is not None else {}))
            frame = instrument.history(period='2y', interval='1d', auto_adjust=False,
                                       back_adjust=False, actions=False, timeout=30,
                                       keepna=True, raise_errors=True)
            metadata = (instrument.history_metadata or {}) if not frame.empty else {}
        if frame.empty:
            raise PriceDataError(f'No daily prices returned for {symbol}.')
        required = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'}
        if not required.issubset(frame.columns):
            raise PriceDataError('Provider response lacks required OHLCV or adjusted-close columns.')
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise PriceDataError('Provider response lacks exchange-local timestamps.')
        today = pd.Timestamp.now(tz=frame.index.tz).date()
        complete = frame.loc[frame.index.date < today]
        bars = [PriceBar(date=stamp.date(), open=row['Open'], high=row['High'],
                         low=row['Low'], close=row['Close'], adjusted_close=row['Adj Close'],
                         volume=row['Volume']) for stamp, row in complete.iterrows()]
        warnings = []
        if len(complete) < len(frame):
            warnings.append('Current-date bars excluded to avoid using an incomplete session.')
        return PriceHistory(ticker=symbol, provider='yahoo_finance',
                            provider_symbol=provider_symbol,
                            source=f'https://finance.yahoo.com/quote/{provider_symbol}/history/',
                            retrieved_at=datetime.now(timezone.utc), currency=metadata.get('currency'),
                            bars=bars, warnings=warnings)
    except PriceDataError:
        raise
    except Exception as error:
        raise PriceDataError(f'Could not retrieve valid daily history for {symbol} ({type(error).__name__}).') from error


def load_price_csv(path: str | Path, ticker: str, *, currency: str | None = None) -> PriceHistory:
    """Read explicit offline inputs; do not infer missing adjustments or volume."""
    symbol = normalize_ticker(ticker)
    path = Path(path)
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        required = {'date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume'}
        if not required.issubset(reader.fieldnames or []):
            raise PriceDataError('CSV requires date, open, high, low, close, adjusted_close, volume.')
        bars = [PriceBar.model_validate({key: row[key] for key in required}) for row in reader]
    return PriceHistory(ticker=symbol, provider='csv', provider_symbol=symbol,
                        source=str(path), retrieved_at=datetime.now(timezone.utc),
                        currency=currency, bars=bars,
                        warnings=['CSV data is user supplied; symbol, adjustment accuracy, and session completeness are not verified.'])
